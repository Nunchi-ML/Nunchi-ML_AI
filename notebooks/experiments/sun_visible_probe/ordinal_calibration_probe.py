"""Group-OOF probability calibration and a separate nested decision control.

Frozen data and outer folds match ordinal_compare.py. Sigmoid calibration,
scaling and balancing use outer TRAINING rows only. No outer labels are
passed to model fit/calibration/threshold selection. No model is exported.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ordinal_compare as O
from nunchi_ml.glare.ordinal import CumulativeOrdinalClassifier, cumulative_to_class_probabilities

SPECS = ["lightgbm", "logistic", "logistic_severity_g", "logistic_severity_g_A_none"]
THRESHOLDS = [.25, .35, .45, .5, .55, .65, .75]


def restore_balanced_prior(q, priors, balanced):
    """Undo inverse-frequency weighting algebraically; not a calibration claim."""
    q = np.asarray(q, dtype=float)
    priors = np.asarray(priors, dtype=float)
    if (q.ndim != 2 or q.shape[1] != 2 or priors.shape != (2,)
            or len(balanced) != 2 or not np.isfinite(q).all()
            or ((q < 0) | (q > 1)).any() or not np.isfinite(priors).all()
            or ((priors < 0) | (priors > 1)).any()):
        raise ValueError("Expected valid binary probabilities and training priors")
    result = q.copy()
    for k, weighted in enumerate(balanced):
        if not weighted:
            continue
        prior = priors[k]
        if prior in (0, 1):
            result[:, k] = prior
        else:
            numerator = q[:, k] * prior
            result[:, k] = numerator / (numerator + (1-q[:, k]) * (1-prior))
    return result


def threshold_decision(p, alert_threshold, severity_threshold):
    alert = p[:, 1:].sum(axis=1)
    severity = p[:, 2] / np.maximum(alert, 1e-12)
    return np.where(alert >= alert_threshold, np.where(severity >= severity_threshold, 2, 1), 0)


def select_thresholds(y, inner_oof_p):
    """Input must be group-held-out predictions on the outer training rows."""
    # Prefer default thresholds when tied; never choose thresholds from outer test scores.
    best = None
    for alert in THRESHOLDS:
        for severe in THRESHOLDS:
            pred = threshold_decision(inner_oof_p, alert, severe)
            f = f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)
            key = (f, -abs(alert-.5)-abs(severe-.5), -alert, -severe)
            if best is None or key > best[0]:
                best = (key, alert, severe)
    return best[1], best[2], float(best[0][0])


def inner_reference_predictions(train):
    p = np.zeros((len(train), 3))
    for tr, va in O.inner_splits(train.group.to_numpy()):
        if set(train.iloc[tr].group) & set(train.iloc[va].group):
            raise AssertionError("Inner reference group leakage")
        p[va] = O.P.two_stage(O.P.REL_R)(train.iloc[tr], train.iloc[va])
    return p


def model_for(spec, calibration):
    if spec not in SPECS:
        raise ValueError(f"Unknown model spec: {spec}")
    family = "lightgbm" if spec == "lightgbm" else "logistic"
    A, B = O.make_estimator(family, 2), O.make_estimator(family, 2)
    weighting, balanced = "balanced", [True, True]
    if "severity_g" in spec:
        B = make_pipeline(ColumnTransformer([("G", "passthrough", list(range(len(O.G_COLUMNS))))]),
                          StandardScaler(), LogisticRegression(max_iter=3000))
    if spec.endswith("A_none"):
        B[-1].set_params(class_weight="balanced")
        weighting, balanced = None, [False, True]
    return CumulativeOrdinalClassifier(A, B, class_weight=weighting, calibration=calibration), balanced


def probability_metrics(y, q):
    return {f"ge{k}_{name}": value for k in (1, 2) for name, value in
            O.binary_metrics((y >= k).astype(int), q[:, k-1] >= .5, q[:, k-1]).items()}


def evaluate(data, args):
    train = data[data.camera.ne("AI Hub")].reset_index(drop=True)
    stress = data[data.camera.eq("AI Hub")].reset_index(drop=True)
    batches = []
    for seed in range(args.repeats):
        folds = [(train.iloc[tr], train.iloc[va], k) for k, (tr, va) in enumerate(
            GroupKFold(5, shuffle=True, random_state=seed).split(train, train.label, train.group))]
        batches.append(("group_cv", seed, train, folds))
    if not args.only_cv:
        for camera in train.camera.unique():
            mask = train.camera.eq(camera)
            batches.append((f"held_camera:{camera}", -1, train[mask], [(train[~mask], train[mask], 0)]))
        if not args.augment_aihub:
            batches.append(("reused_AIHub", -1, stress, [(train, stress, 0)]))
    scores, binary, diagnostics, predictions, thresholds, splits, audits = [], [], [], [], [], [], []
    started = time.monotonic()
    for kind, repeat, test, folds in batches:
        store, q_store, prediction_store, fold_ids = {}, {}, {}, {}
        def put(name, validation, p, q=None, pred=None):
            store.setdefault(name, {}).update(dict(zip(validation.id, p)))
            if q is not None:
                q_store.setdefault(name, {}).update(dict(zip(validation.id, q)))
            if pred is not None:
                prediction_store.setdefault(name, {}).update(dict(zip(validation.id, pred)))
        for train_fold, validation, fold in folds:
            added = len(stress) if args.augment_aihub else 0
            if args.augment_aihub:
                train_fold = pd.concat([train_fold, stress], ignore_index=True)
            if set(train_fold.id) & set(validation.id):
                raise AssertionError("Training/validation image ID overlap")
            if set(train_fold.group) & set(validation.group):
                raise AssertionError("Outer capture-group leakage")
            meta = dict(test=kind, repeat=repeat, outer_fold=fold)
            for row in validation.itertuples():
                splits.append({**meta, "id": row.id, "group": row.group})
                fold_ids[row.id] = fold
            inner = O.inner_splits(train_fold.group.to_numpy())
            for k, (tr, va) in enumerate(inner):
                a, b = train_fold.iloc[tr], train_fold.iloc[va]
                audits.append({**meta, "inner_fold": k, "train_groups": json.dumps(sorted(map(str, a.group.unique()))),
                               "validation_groups": json.dumps(sorted(map(str, b.group.unique()))),
                               "train_rows": len(a), "validation_rows": len(b),
                               "train_ge1": int(a.label.ge(1).sum()), "train_ge2": int(a.label.ge(2).sum()),
                               "validation_ge1": int(b.label.ge(1).sum()), "validation_ge2": int(b.label.ge(2).sum())})
            legacy = O.P.two_stage(O.P.REL_R)(train_fold, validation)
            put("reference:fixed_0.5", validation, legacy, pred=threshold_decision(legacy, .5, .5))
            # Separate decision experiment: this is the OLD conditional two-stage structure.
            inner_p = inner_reference_predictions(train_fold)
            ta, ts, inner_f1 = select_thresholds(train_fold.label.to_numpy(int), inner_p)
            put("reference:nested_thresholds", validation, legacy, pred=threshold_decision(legacy, ta, ts))
            thresholds.append({**meta, "alert": ta, "severity": ts, "inner_macro_f1": inner_f1,
                               "training_rows": len(train_fold), "added_aihub_rows": added,
                               "training_normal": int(train_fold.label.eq(0).sum()),
                               "training_glare": int(train_fold.label.eq(1).sum()),
                               "training_severe": int(train_fold.label.eq(2).sum())})
            X = train_fold[O.SETS["g_rel"]]
            Xt = validation[O.SETS["g_rel"]]
            y = train_fold.label.to_numpy(int)
            groups = train_fold.group.to_numpy()
            priors = [(y >= k).mean() for k in (1, 2)]
            def fit_spec(spec):
                raw, balanced = model_for(spec, None)
                raw.fit(X, y)
                q_raw = raw.predict_cumulative_proba(Xt, corrected=False)
                calibrated, _ = model_for(spec, "sigmoid")
                calibrated.fit(X, y, groups=groups)
                q_cal = calibrated.predict_cumulative_proba(Xt, corrected=False)
                return spec, q_raw, restore_balanced_prior(q_raw, priors, balanced), q_cal
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for spec, q_raw, q_prior, q_cal in pool.map(fit_spec, args.specs):
                    for variant, q in [("raw", q_raw), ("restore_prior", q_prior), ("group_sigmoid", q_cal)]:
                        put(f"{spec}/{variant}", validation, cumulative_to_class_probabilities(q), q=q)
        y = test.label.to_numpy(int)
        for name, values in store.items():
            meta = dict(test=kind, repeat=repeat, method=name)
            p = np.stack([values[i] for i in test.id])
            pred = (np.array([prediction_store[name][i] for i in test.id], int)
                    if name in prediction_store else p.argmax(axis=1))
            if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
                raise AssertionError("Invalid probabilities")
            scores.append({**meta, **O.metrics(y, pred)})
            binary.append({**meta, **O.binary_metrics(y, pred >= 1, p[:, 1:].sum(1))})
            q = np.stack([q_store[name][i] for i in test.id]) if name in q_store else None
            if q is not None:
                diagnostics.append({**meta, "violation_rate": float((q[:, 1] > q[:, 0]).mean()),
                                    **probability_metrics(y, q)})
            for i, row in enumerate(test.itertuples()):
                predictions.append({**meta, "id": row.id, "group": row.group, "label": int(y[i]),
                                    "outer_fold": fold_ids[row.id], "pred": int(pred[i]),
                                    **{f"p{k}": float(p[i, k]) for k in range(3)},
                                    "q1_before_projection": float(q[i, 0]) if q is not None else None,
                                    "q2_before_projection": float(q[i, 1]) if q is not None else None})
        for filename, rows in [("scores.csv", scores), ("binary_scores.csv", binary), ("threshold_scores.csv", diagnostics),
                               ("predictions.csv", predictions), ("thresholds.csv", thresholds), ("splits.csv", splits),
                               ("calibration_audit.csv", audits)]:
            pd.DataFrame(rows).to_csv(args.out / filename, index=False, encoding="utf-8-sig")
        print(f"{kind} repeat={repeat} complete; elapsed={time.monotonic()-started:.1f}s", flush=True)
    return pd.DataFrame(scores), pd.DataFrame(binary), pd.DataFrame(diagnostics)


def report(scores, binary, diagnostics, out):
    cv = scores[scores.test.eq("group_cv")]
    columns = ["macro_f1", "weighted_f1", "accuracy", "qwk", "mae", "f1_1", "recall_2", "count_0_to_2", "count_2_to_0"]
    summary = cv.groupby("method")[columns].mean()
    summary["split_sd"] = cv.groupby("method").macro_f1.std()
    summary.to_csv(out / "summary.csv", encoding="utf-8-sig")
    prob = diagnostics[diagnostics.test.eq("group_cv")].groupby("method").mean(numeric_only=True).drop(columns="repeat")
    prob.to_csv(out / "probability_summary.csv", encoding="utf-8-sig")
    binary_summary = binary[binary.test.eq("group_cv")].groupby("method").mean(numeric_only=True).drop(columns="repeat")
    binary_summary.to_csv(out / "binary_summary.csv", encoding="utf-8-sig")
    per_test = scores.groupby(["test", "method"])[columns].mean()
    per_test.to_csv(out / "per_test_summary.csv", encoding="utf-8-sig")
    key = ["violation_rate", "ge1_brier", "ge2_brier", "ge1_log_loss", "ge2_log_loss"]
    fmt = lambda frame: frame.to_html(float_format=lambda x: f"{x:.4f}", border=0)
    augmented = json.loads((out / "method.json").read_text(encoding="utf-8"))["config"].get("augment_aihub", False)
    regime = ("별도 학습 조건: 각 외부 학습 fold에 AI Hub304장(심함42장)을 추가했다. 검증320장과 원래 그룹분할은 같다. "
              "AI Hub는 이 조건에서 학습 데이터이며 시험 점수를 계산하지 않는다. 향상분은 데이터 추가와 알고리즘 효과를 구분해야 한다."
              if augmented else "원래 학습 조건: AI Hub를 학습에서 제외하고 기존320장 그룹분할을 유지했다.")
    text = f"""<!doctype html><meta charset="utf-8"><title>그룹별 확률 보정</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;max-width:1250px;margin:30px auto;padding:20px;color:#182335;background:#f5f6f8}}section{{background:white;padding:24px;margin:20px 0;border-radius:12px;overflow:auto}}table{{font-size:13px;border-collapse:collapse;white-space:nowrap}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}</style>
<h1>눈부심 후속 실험 — 그룹 내부 확률 보정</h1><p>2026-11-04 · 딥러닝 없음 · 최종 모델 결정 보류</p>
<section><h2>비교 조건</h2><p>{regime}</p><p>기존320장·24촬영그룹과 같은5겹×{cv['repeat'].nunique()}회. sigmoid는 각 외부 학습 표 안의3개 그룹 fold에서 얻은 OOF 예측으로만 학습했다. 스케일링과 balanced 가중치는 각 내부 학습 fold에서 다시 fit/계산했다. 바깥 검증 라벨은 모델·보정·임계값 선택에 들어가지 않는다.</p><p>raw=기존 누적분류, restore_prior=학습 클래스 비율로 balanced 가중치를 대수적으로 되돌리는 대조(확률 보정 성공 보장 아님), group_sigmoid=그룹 OOF sigmoid 후 단조성 보정·argmax. reference:nested_thresholds는 누적분류와 별개의 기존 조건부 두단계 판정 대조로, 내부 OOF에서49개 임계값 조합만 비교했다.</p></section>
<section><h2>3등급 점수</h2>{fmt(summary)}<p>split_sd는 같은 데이터의 분할 변동이며 독립 신뢰구간이 아니다. 극단 오류 건수는320장당 평균이다. <a href="scores.csv">각 회차의 등급별 P/R/F1·혼동행렬</a></p></section>
<section><h2>확률 오차와 단조성</h2>{fmt(prob[key])}<p>Brier/log loss는 작을수록 좋지만 확률 보정과 구분 능력의 영향을 함께 받는다. 점수가 낮아졌다고 분류 F1 개선이나 보정 성공을 단독 확정하지 않는다. 표는 투영 전 q의 오차다. 모든 분류에는 q1≥q2 투영을 적용한다.</p></section>
<section><h2>눈부심 알림</h2>{fmt(binary_summary[['binary_f1','balanced_accuracy','alert_recall','false_alert_rate']])}</section>
<section><h2>카메라 제외·추가 시험</h2>{fmt(per_test.loc[per_test.index.get_level_values('test')!='group_cv'])}<p>AI Hub의 학습/시험 역할은 위 조건을 따른다. 단일 그룹에 몰린 심함 라벨과 새 독립시험 부족은 보정으로 해소되지 않는다. 다른 촬영의 심함 표본과 사람 검증이 필요하며, 4등급 라벨이나 새 정답을 만들지 않았다.</p></section>"""
    (out / "report.html").write_text(text, encoding="utf-8")
    print(summary.round(4).to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=Path("data/analysis_20261104/model2_snapshot.csv"))
    parser.add_argument("--rel", type=Path, default=Path("data/sun_visible/rel_features.csv"))
    parser.add_argument("--cache", type=Path, default=Path("data/analysis_20261104/step3_visibility/image_features.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--specs", nargs="+", choices=SPECS, default=SPECS)
    parser.add_argument("--only-cv", action="store_true")
    parser.add_argument("--augment-aihub", action="store_true", help="Add AIHub to outer training only; never score AIHub in this regime")
    args = parser.parse_args()
    if args.repeats < 1 or args.workers not in range(1, 5):
        parser.error("Positive repeats and workers1..4 required")
    if args.out.exists() or not args.out.resolve().is_relative_to(args.data_root.resolve()):
        parser.error("Use a new private output directory within data-root")
    args.out.mkdir(parents=True)
    data = O.load_data(args.snapshot, args.rel, args.data_root, args.cache)
    source_paths = [Path(__file__), Path(O.__file__), Path(O.P.__file__), Path(O.C.__file__),
                    Path(O.sys.modules[CumulativeOrdinalClassifier.__module__].__file__),
                    O.HERE / "ordinal_image_features.py", O.HERE / "rel_features.py",
                    O.HERE.parent / "glare_video_source/g_weather_time_probe.py"]
    saved = args.out / "source_snapshot"
    saved.mkdir()
    for path in source_paths:
        (saved / path.name).write_bytes(path.read_bytes())
    info = dict(status="running", config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                files={str(p): O.fingerprint(p) for p in [args.snapshot, args.rel, args.cache]},
                sources={str(p): O.fingerprint(p) for p in source_paths},
                versions={n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scikit-learn", "lightgbm"]},
                weather_sha256={p.name: O.fingerprint(p) for p in sorted((args.data_root / "asos").glob("*.csv"))},
                input_features=O.SETS["g_rel"], calibration="sigmoid, ensemble=False, 3 inner GroupKFold folds",
                threshold_grid=THRESHOLDS, limitation="Repeated development CV and reused AIHub; not independent confirmation. "
                "Model/class-count decisions and deployment remain deferred. Raw sigmoid estimates are projected before ordinal argmax.")
    method = args.out / "method.json"
    method.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    # Rare/missing inner classes are recorded with counts in calibration_audit.csv.
    warnings.filterwarnings("ignore", message="Number of classes in training fold.*", category=RuntimeWarning)
    try:
        scores, binary, diagnostics = evaluate(data, args)
        report(scores, binary, diagnostics, args.out)
    except (Exception, KeyboardInterrupt) as error:
        info.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=type(error).__name__)
        method.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    info.update(status="complete", split_sha256=O.fingerprint(args.out / "splits.csv"),
                calibration_audit_sha256=O.fingerprint(args.out / "calibration_audit.csv"))
    method.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
