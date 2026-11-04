"""Offline, grouped ordinal/multiclass/binary comparison. No deployment/export.

The .623 reference is legacy G+relative12 TWO-STAGE logistic with .5/.5
decision on 320 rows, GroupKFold(5, shuffle=True, seeds=0..29). All new
three-class candidates use argmax. Also report legacy argmax to separate
the decision-rule change. AI Hub is reused stress data, not a fresh test.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "glare_video_source"))
sys.path.insert(0, str(HERE))

import cv2
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             cohen_kappa_score, confusion_matrix, f1_score, log_loss,
                             mean_absolute_error, precision_recall_fscore_support,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

import codex_protocol_compare as P
import g_visibility_probe as C
from g_weather_time_probe import G_COLUMNS, replay
from nunchi_ml.glare.features import FEATURES
from nunchi_ml.glare.ordinal import CumulativeOrdinalClassifier, cumulative_to_class_probabilities, fit_with_sample_weight
from ordinal_image_features import EXTRA, extract_array as extra_array
from rel_features import REL, rel_array

SETS = {"g_rel": G_COLUMNS + P.REL_R, "g_screen": G_COLUMNS + FEATURES,
        "g_rel_extra": G_COLUMNS + P.REL_R + EXTRA, "g_only": G_COLUMNS}


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(snapshot, rel_path, data_root, cache):
    snap = pd.read_csv(snapshot, dtype={"id": str})
    if snap.id.isna().any() or not snap.id.is_unique or not snap.label.isin([0, 1, 2]).all():
        raise ValueError("Frozen IDs/labels must be valid and unique")
    if snap.group.isna().any() or snap.captured_at_kst.isna().any():
        raise ValueError("Every sample needs a capture group and timestamp")
    snap = replay(snap, data_root / "asos", "previous")
    paths = C.image_paths(snap, data_root)
    data = C.features(snap, paths, cache)
    rel = pd.read_csv(rel_path, dtype={"id": str})[["id", *REL]]
    if rel.id.isna().any() or not rel.id.is_unique:
        raise ValueError("Duplicate/missing relative feature ID")
    # Recompute and compare the preexisting cache instead of silently trusting it.
    def one(path):
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        return {**rel_array(image), **extra_array(image)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        fresh = pd.DataFrame(pool.map(one, paths))
    check = snap[["id"]].merge(rel, on="id", how="left", validate="one_to_one", sort=False)
    if not np.allclose(check[REL].to_numpy(float), fresh[REL].to_numpy(float), rtol=1e-6, atol=1e-7):
        raise ValueError("Relative features do not match the current image bytes/extractor")
    fresh = fresh.rename(columns={c: "r_" + c for c in REL})
    data = pd.concat([data.reset_index(drop=True), fresh], axis=1)
    if not np.isfinite(data[list(dict.fromkeys(sum(SETS.values(), [])))].to_numpy(float)).all():
        raise ValueError("Missing/nonfinite image/numerical features; do not drop rows")
    if data.image_sha256.duplicated().any():
        raise ValueError("Duplicate image bytes; review group assignments before comparison")
    return data


def make_estimator(family, classes, params=None):
    params = params or {}
    if family == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=params.get("C", 1.), max_iter=3000))
    if family == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=params.get("trees", 200), learning_rate=params.get("lr", .05),
                              num_leaves=params.get("leaves", 7), min_child_samples=params.get("min_child", 5),
                              random_state=0, n_jobs=1, verbosity=-1)
    if family == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=params.get("trees", 200), learning_rate=params.get("lr", .05),
                                  depth=params.get("depth", 3), l2_leaf_reg=params.get("l2", 3.),
                                  loss_function="Logloss" if classes == 2 else "MultiClass",
                                  random_seed=0, thread_count=1, verbose=False, allow_writing_files=False)
    if family == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=params.get("trees", 200), learning_rate=params.get("lr", .05),
                             max_depth=params.get("depth", 3), min_child_weight=params.get("min_child", 1),
                             reg_lambda=params.get("l2", 1.), tree_method="hist", n_jobs=1, random_state=0,
                             objective="binary:logistic" if classes == 2 else "multi:softprob",
                             **({"num_class": classes} if classes > 2 else {}))
    raise ValueError(f"Unknown family: {family}")


def task_probabilities(family, X, y, test, balanced, classes, params=None):
    unique, encoded = np.unique(y, return_inverse=True)
    weights = compute_sample_weight("balanced", encoded) if balanced else np.ones(len(y))
    model = (DummyClassifier(strategy="constant", constant=0) if len(unique) == 1
             else make_estimator(family, len(unique), params))
    fit_with_sample_weight(model, np.asarray(X), encoded, weights)
    result = np.zeros((len(test), classes))
    result[:, unique] = model.predict_proba(np.asarray(test))
    return result


def inner_splits(groups):
    if len(np.unique(groups)) < 2:
        raise ValueError("Nested tuning needs at least two training capture groups")
    return list(GroupKFold(min(3, len(np.unique(groups)))).split(groups, groups=groups))


def tune(family, X, y, groups, balanced, trials, classes):
    if trials == 0 or np.unique(y).size == 1:
        return {}
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = inner_splits(groups)
    def objective(trial):
        params = ({"C": trial.suggest_categorical("C", [.1, 1., 10.])} if family == "logistic" else
                  {"trees": trial.suggest_categorical("trees", [100, 200]),
                   "lr": trial.suggest_categorical("lr", [.03, .05, .1])})
        if family == "logistic":
            pass
        elif family == "lightgbm":
            params.update(leaves=trial.suggest_categorical("leaves", [3, 7, 15]),
                          min_child=trial.suggest_categorical("min_child", [5, 10, 20]))
        else:
            params.update(depth=trial.suggest_int("depth", 2, 4),
                          l2=trial.suggest_float("l2", 1., 10., log=True))
        pred = np.zeros(len(y), int)
        for tr, va in folds:
            assert not set(groups[tr]) & set(groups[va])
            pred[va] = task_probabilities(family, X[tr], y[tr], X[va], balanced, classes, params).argmax(1)
        return f1_score(y, pred, labels=list(range(classes)), average="macro", zero_division=0)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.enqueue_trial({"C": 1.} if family == "logistic" else {"trees": 200, "lr": .05,
                         **({"leaves": 7, "min_child": 5} if family == "lightgbm"
                            else {"depth": 3, "l2": 3. if family == "catboost" else 1.})})
    study.optimize(objective, n_trials=trials, n_jobs=1)
    return study.best_params


def metrics(y, pred):
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    result = dict(macro_f1=float(f1.mean()), weighted_f1=float(f1_score(y, pred, average="weighted", zero_division=0)),
                  accuracy=float(accuracy_score(y, pred)), qwk=float(cohen_kappa_score(y, pred, labels=[0, 1, 2], weights="quadratic")),
                  mae=float(mean_absolute_error(y, pred)), count_0_to_2=int(((y == 0) & (pred == 2)).sum()),
                  count_2_to_0=int(((y == 2) & (pred == 0)).sum()),
                  confusion_matrix=json.dumps(confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()))
    for k in range(3):
        result.update({f"precision_{k}": float(precision[k]), f"recall_{k}": float(recall[k]),
                       f"f1_{k}": float(f1[k]), f"support_{k}": int(support[k])})
    return result


def binary_metrics(y, pred, probability=None):
    truth = np.asarray(y) >= 1
    pred = np.asarray(pred, dtype=bool)
    result = dict(binary_f1=float(f1_score(truth, pred, zero_division=0)),
                  binary_macro_f1=float(f1_score(truth, pred, labels=[False, True], average="macro", zero_division=0)),
                  accuracy=float(accuracy_score(truth, pred)), balanced_accuracy=float(balanced_accuracy_score(truth, pred)),
                  alert_precision=float(precision_score(truth, pred, zero_division=0)),
                  alert_recall=float(recall_score(truth, pred, zero_division=0)),
                  false_alert_rate=float(pred[~truth].mean()) if (~truth).any() else None,
                  confusion_matrix=json.dumps(confusion_matrix(truth, pred, labels=[False, True]).tolist()))
    if probability is not None:
        result.update(brier=float(brier_score_loss(truth, probability)),
                      log_loss=float(log_loss(truth, np.column_stack([1-probability, probability]), labels=[False, True])),
                      auc=float(roc_auc_score(truth, probability)) if np.unique(truth).size == 2 else None)
    return result


def evaluate(data, args):
    train = data[data.camera.ne("AI Hub")].reset_index(drop=True)
    stress = data[data.camera.eq("AI Hub")].reset_index(drop=True)
    # Split once; every model/feature set receives exactly these row indices.
    splits = []
    for repeat in range(args.repeats):
        folds = []
        for fold, (tr, va) in enumerate(GroupKFold(5, shuffle=True, random_state=repeat).split(train, train.label, train.group)):
            if set(train.iloc[tr].group) & set(train.iloc[va].group):
                raise AssertionError("Capture-group leakage")
            folds.append((train.iloc[tr], train.iloc[va], fold))
        splits.append(("group_cv", repeat, train, folds))
    if not args.only_cv:
        for camera in train.camera.unique():
            mask = train.camera.eq(camera)
            splits.append((f"held_camera:{camera}", -1, train[mask], [(train[~mask], train[mask], 0)]))
        splits.append(("reused_AIHub", -1, stress, [(train, stress, 0)]))
    scores, binary_scores, diagnostics, predictions, split_rows, parameters = [], [], [], [], [], []
    started = time.monotonic()
    for kind, repeat, test, folds in splits:
        cache, raw_q, fold_ids = {}, {}, {}
        def put(name, va, p):
            cache.setdefault(name, {}).update(dict(zip(va.id, p)))
        for tr, va, fold in folds:
            for d in va.itertuples():
                split_rows.append(dict(test=kind, repeat=repeat, fold=fold, id=d.id, group=d.group))
                fold_ids[d.id] = fold
            legacy = P.two_stage(P.REL_R)(tr, va)
            put("reference:two_stage_legacy", va, legacy)
            put("reference:two_stage_argmax", va, legacy)
            def fit_pair(spec):
                feature_set, family, weighting = spec
                cols = SETS[feature_set]
                X, target, Xt = tr[cols].to_numpy(float), tr.label.to_numpy(int), va[cols].to_numpy(float)
                balanced = weighting == "balanced"
                prefix = f"{feature_set}/{family}/{weighting}"
                params = tune(family, X, target, tr.group.to_numpy(), balanced, args.trials, 3)
                multi = task_probabilities(family, X, target, Xt, balanced, 3, params)
                pa = tune(family, X, (target >= 1).astype(int), tr.group.to_numpy(), balanced, args.trials, 2)
                pb = tune(family, X, (target >= 2).astype(int), tr.group.to_numpy(), balanced, args.trials, 2)
                model = CumulativeOrdinalClassifier(make_estimator(family, 2, pa), make_estimator(family, 2, pb),
                                                    class_weight="balanced" if balanced else None).fit(X, target)
                return prefix, multi, model.predict_cumulative_proba(Xt, corrected=False), params, pa, pb
            specs = [(fs, family, weight) for fs in args.features for family in args.families for weight in args.weighting]
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for prefix, multi, q, params, pa, pb in pool.map(fit_pair, specs):
                    put(prefix + "/multiclass", va, multi)
                    name = prefix + "/ordinal"
                    put(name, va, cumulative_to_class_probabilities(q))
                    raw_q.setdefault(name, {}).update(dict(zip(va.id, q)))
                    parameters.append(dict(test=kind, repeat=repeat, fold=fold, method=prefix,
                                           multiclass=params, ge1=pa, ge2=pb))
        # Equal ensemble after converting each model into valid class probabilities.
        for fs in args.features:
            for weighting in args.weighting:
                members = [f"{fs}/{f}/{weighting}/ordinal" for f in args.families]
                name = f"{fs}/equal_ensemble/{weighting}/ordinal"
                cache[name] = {i: np.mean([cache[m][i] for m in members], axis=0) for i in test.id}
        y = test.label.to_numpy(int)
        for name, values in cache.items():
            p = np.stack([values[i] for i in test.id])
            if (p < -1e-12).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-12):
                raise AssertionError("Invalid class probabilities")
            pred = C.decision(p) if name.endswith("two_stage_legacy") else p.argmax(1)
            meta = dict(test=kind, repeat=repeat, method=name)
            scores.append({**meta, **metrics(y, pred)})
            binary_scores.append({**meta, "decision": "collapse_three_grades", **binary_metrics(y, pred >= 1, p[:, 1:].sum(1))})
            q = np.stack([raw_q[name][i] for i in test.id]) if name in raw_q else None
            if q is not None:
                diagnostics.append({**meta, "rows": len(q), "violations": int((q[:, 1] > q[:, 0]).sum()),
                                    "violation_rate": float((q[:, 1] > q[:, 0]).mean()),
                                    **{f"ge{k}_{metric}": value for k in (1, 2) for metric, value in
                                       binary_metrics((y >= k).astype(int), q[:, k-1] >= .5, q[:, k-1]).items()}})
                binary_scores.append({**meta, "decision": "direct_binary_A", **binary_metrics(y, q[:, 0] >= .5, q[:, 0])})
            for row, d in enumerate(test.itertuples()):
                predictions.append({**meta, "id": d.id, "group": d.group, "camera": d.camera,
                                    "fold": fold_ids[d.id], "label": int(y[row]), "pred": int(pred[row]),
                                    **{f"p{k}": float(p[row, k]) for k in range(3)},
                                    "raw_q1": float(q[row, 0]) if q is not None else None,
                                    "raw_q2": float(q[row, 1]) if q is not None else None})
        # Incremental artifacts survive an interruption; completion is separate in method.json.
        for filename, rows in [("scores.csv", scores), ("binary_scores.csv", binary_scores),
                               ("threshold_scores.csv", diagnostics), ("predictions.csv", predictions), ("splits.csv", split_rows)]:
            pd.DataFrame(rows).to_csv(args.out / filename, index=False, encoding="utf-8-sig")
        (args.out / "parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{kind} repeat={repeat} done; elapsed={time.monotonic()-started:.1f}s", flush=True)
    return pd.DataFrame(scores), pd.DataFrame(binary_scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("data/analysis_20261104/model2_snapshot.csv"))
    parser.add_argument("--rel", type=Path, default=Path("data/sun_visible/rel_features.csv"))
    parser.add_argument("--cache", type=Path, default=Path("data/analysis_20261104/step3_visibility/image_features.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--features", nargs="+", choices=list(SETS), default=["g_rel"])
    parser.add_argument("--families", nargs="+", choices=["catboost", "xgboost", "lightgbm", "logistic"], default=["catboost", "xgboost", "lightgbm"])
    parser.add_argument("--weighting", nargs="+", choices=["balanced", "none"], default=["balanced", "none"])
    parser.add_argument("--trials", type=int, default=0, help="0=fixed parameters; 2..8=independent nested group tuning per task")
    parser.add_argument("--workers", type=int, default=3, help="Independent model fits, each estimator uses one thread")
    parser.add_argument("--only-cv", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.trials not in [0, *range(2, 9)] or args.workers not in range(1, 5):
        parser.error("Positive repeats and trials=0 or 2..8 required")
    if not args.out.resolve().is_relative_to(args.data_root.resolve()) or args.out.exists():
        parser.error("Use a NEW private output directory inside data-root")
    args.out.mkdir(parents=True)
    data = load_data(args.snapshot, args.rel, args.data_root, args.cache)
    data[["id", "label", "camera", "group", "image_sha256", *list(dict.fromkeys(sum(SETS.values(), []))) ]].to_csv(
        args.out / "feature_manifest.csv", index=False, encoding="utf-8-sig")
    info = dict(status="running", primary="mean per-repeat 3-class macro_f1; same 5-fold GroupKFold seeds 0..29 for full run",
                config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, feature_sets=SETS,
                rows=len(data), cv_rows=int(data.camera.ne("AI Hub").sum()),
                cv_groups=int(data.loc[data.camera.ne("AI Hub"), "group"].nunique()),
                files={str(p): fingerprint(p) for p in [args.snapshot, args.rel, args.cache]},
                sources={str(p): fingerprint(p) for p in [Path(__file__), HERE / "ordinal_image_features.py", HERE / "rel_features.py",
                         Path(C.__file__), Path(P.__file__), HERE.parent / "glare_video_source/g_weather_time_probe.py",
                         Path(sys.modules[CumulativeOrdinalClassifier.__module__].__file__)]},
                versions={p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "catboost", "xgboost", "lightgbm", "optuna"]},
                weather_sha256={p.name: fingerprint(p) for p in sorted((args.data_root / "asos").glob("*.csv"))},
                fixed_parameters={"trees": 200, "lr": .05, "catboost_depth": 3, "xgboost_depth": 3, "lightgbm_leaves": 7},
                probability_correction="Equal-mean isotonic projection for q1<q2 before class conversion; argmax",
                limitation="Repeated CV is development evidence, repeats are correlated, not an independent CI. AIHub is reused. "
                           "Pixel values are not physical luminance. Weather observation time only, publication availability unknown. "
                           "No calibration/threshold search on outer folds, no final model selection/export.")
    (args.out / "method.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    scores, binary = evaluate(data, args)
    numeric = scores.select_dtypes(include="number").columns.drop("repeat")
    summary = scores.groupby(["test", "method"], sort=False)[numeric].agg(["mean", "std"])
    summary.to_csv(args.out / "summary.csv", encoding="utf-8-sig")
    binary.groupby(["test", "method", "decision"], sort=False).mean(numeric_only=True).to_csv(args.out / "binary_summary.csv", encoding="utf-8-sig")
    info["status"] = "complete"
    info["splits_sha256"] = fingerprint(args.out / "splits.csv")
    (args.out / "method.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(scores.groupby(["test", "method"])[["macro_f1", "weighted_f1", "accuracy", "qwk", "mae", "f1_1"]].mean().round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
