"""새 영상 평가 — 사진만 보는 ML vs 사진만 보는 규칙. 라벨 근거를 명시한다.

학습: 지금까지의 624장 전부(②-v4). 시험: new_camera_frames.py 가 만든 라벨 묶음의 사람 라벨 프레임 (학습 · 설계에 안 쓴 카메라·장소).
비교 (경계값 · 모델은 학습 624장에서만 정함, 시험 라벨을 보고 고르지 않음):
    규칙 R1  밝기 규칙(전체 밝기)       pilot_eval.brightness_rule — 임계값 2개
    규칙 R2  하얀 픽셀 비율 규칙         같은 방식
    규칙 R3  상대 특징 12 중 학습에서 가장 좋은 1개의 임계값 규칙  (규칙에게도 최선의 기회)
    ML 1    기존 모델 ② 형태 LightGBM(화면 숫자 12)
    ML 2    로지스틱 (상대 특징 12)       P(눈부심 1+) ≥ .5 알림, 심함은 P(2) ≥ P(1)
통계: 영상(sequence) 단위 군집 부트스트랩. human_consensus는 최소 2표와 엄격한 과반수 필요.
AI 임시 라벨·사람 1인·사람 합의를 구분한다. 영상별 카메라 독립성은 확인되지 않았다.

    python notebooks/experiments/sun_visible_probe/new_camera_eval.py --pack data/new_cameras --labels data/new_cameras/labels_a.csv [labels_b.csv ...] \
        --label-kind human_consensus --out data/new_cameras/human_evaluation \
        --snapshot data/analysis_20261104/model2_snapshot.csv --rel data/sun_visible/rel_features.csv --cache <image_features.csv>
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "glare_video_source"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import cv2
import sklearn
import g_visibility_probe as C
from g_weather_time_probe import replay
from nunchi_ml.glare.features import FEATURES, extract_array
from nunchi_ml.glare.pilot_eval import lightgbm_model
from rel_features import REL, rel_array
from new_camera_audit import load_labels, score, bootstrap_accuracy, compare_reference


def fit_single_rule(tr, col):
    v, y = tr[col].to_numpy(float), tr.label.to_numpy()
    qs = np.unique(np.quantile(v, np.linspace(.05, .95, 19)))
    best = (-1, None)
    for i, a in enumerate(qs):
        for b in qs[i + 1:]:
            for sign in (1, -1):
                f = f1_score(y, np.digitize(sign * v, [sign * a, sign * b]) if sign == 1 else np.digitize(-v, [-b, -a]), average="macro")
                if f > best[0]:
                    best = (f, (a, b, sign))
    return best


def apply_rule(te, col, params):
    a, b, sign = params
    v = te[col].to_numpy(float)
    return np.digitize(v, [a, b]) if sign == 1 else np.digitize(-v, [-b, -a])


def decide(p):
    return np.where(p[:, 1:].sum(1) >= .5, np.where(p[:, 2] >= p[:, 1], 2, 1), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--labels", type=Path, nargs="+", required=True)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--rel", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--label-kind", required=True, choices=["ai_provisional", "human_single", "human_consensus"])
    ap.add_argument("--out", type=Path, required=True, help="Ignored local data directory for audit artifacts")
    ap.add_argument("--bootstrap", type=int, default=3000)
    ap.add_argument("--compare-ai", type=Path, help="Optional separate AI label comparison; never a human consensus vote")
    a = ap.parse_args()
    if a.bootstrap < 1:
        ap.error("--bootstrap must be positive")
    if not a.out.resolve().is_relative_to(a.data_root.resolve()):
        ap.error("--out must be inside --data-root (private evaluation artifacts)")
    sheet_path = a.pack / "label_pack" / "label_sheet_blind.csv"
    sheet = pd.read_csv(sheet_path, dtype={"id": str})[["id", "file"]]
    labels, label_review, label_info = load_labels(a.labels, sheet.id, a.label_kind)
    agreement, agreement_info = None, None
    if a.compare_ai:
        if a.label_kind == "ai_provisional":
            ap.error("--compare-ai requires human labels")
        reference, _, _ = load_labels([a.compare_ai], sheet.id, "ai_provisional")
        agreement, agreement_info = compare_reference(labels, reference)

    snap = replay(pd.read_csv(a.snapshot, dtype={"id": str}), a.data_root / "asos", "previous")
    if not snap.id.is_unique or snap.id.isna().any():
        raise ValueError("Snapshot IDs must be unique and present")
    tr = C.features(snap, C.image_paths(snap, a.data_root), a.cache)
    rel = pd.read_csv(a.rel, dtype={"id": str})[["id"] + REL]
    tr = tr.merge(rel, on="id", how="left", suffixes=("", "_rel"), validate="one_to_one")
    if not np.isfinite(tr[FEATURES + REL].to_numpy(float)).all():
        raise ValueError("Missing/nonfinite training features; do not drop rows")

    map_path = a.pack / "private_frame_map.csv"
    fmap = pd.read_csv(map_path, dtype={"id": str})
    if not fmap.id.is_unique or fmap.id.isna().any() or fmap.video.isna().any():
        raise ValueError("Frame map IDs and video groups must be valid")
    te = sheet.merge(labels.reset_index(), on="id", validate="one_to_one").merge(fmap, on="id", how="left", suffixes=("", "_m"), validate="one_to_one")
    if te.video.isna().any() or len(te) != len(labels):
        raise ValueError("Every accepted label needs exactly one mapped video")
    rows = []
    train_hashes = set(tr.image_sha256)
    for r in te.itertuples():
        path = (a.pack / "label_pack" / r.file).resolve()
        if not path.is_relative_to((a.pack / "label_pack").resolve()):
            raise ValueError("Image path escapes blind pack")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest in train_hashes:
            raise ValueError("Test photo also exists in training data")
        im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        rows.append({**extract_array(im), **rel_array(im), "image_sha256": digest})
    te = pd.concat([te.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    print(f"시험 프레임 {len(te)}장 · 영상 {te.video.nunique()}개 · 라벨 분포 정상 {(te.label == 0).sum()} / 눈부심 {(te.label == 1).sum()} / 심함 {(te.label == 2).sum()}")
    print(te.groupby("video").label.agg(n="size", glare=lambda s: int((s >= 1).sum())).to_string())

    y = te.label.to_numpy()
    preds = {"B0 always normal": np.zeros(len(y), dtype=int), "B1 always alert": np.ones(len(y), dtype=int)}
    rules = {}
    for name, col in (("R1 밝기 규칙(전체 밝기)", "mean_all"), ("R2 하얀 픽셀 비율 규칙", "white_ratio")):
        f, params = fit_single_rule(tr, col)
        preds[name] = apply_rule(te, col, params)
        rules[name] = dict(feature=col, training_macro_f1=float(f), parameters=list(map(float, params)))
    best = max(((fit_single_rule(tr, c), c) for c in REL), key=lambda t: t[0][0])
    (_, params), col = best
    preds[f"R3 최선의 단일 특징 규칙({col})"] = apply_rule(te, col, params)
    rules["R3"] = dict(feature=col, training_macro_f1=float(best[0][0]), parameters=list(map(float, params)))
    m = lightgbm_model().fit(tr[FEATURES], tr.label)
    preds["ML1 LightGBM(화면 12, 기존 모델 ② 형태)"] = m.predict(te[FEATURES])
    lg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(tr[REL], tr.label)
    p = np.zeros((len(te), 3)); p[:, lg.classes_] = lg.predict_proba(te[REL])
    preds["ML2 로지스틱(상대 12)"] = decide(p)

    scores = [dict(method=n, **score(y, pp)) for n, pp in preds.items()]
    intervals, differences = bootstrap_accuracy(y, preds, te.video.to_numpy(), a.bootstrap)
    per_video = []
    for video in te.video.unique():
        mask = te.video.eq(video).to_numpy()
        per_video.extend(dict(video=video, method=n, **score(y[mask], pp[mask])) for n, pp in preds.items())
    a.out.mkdir(parents=True, exist_ok=True)
    label_review.to_csv(a.out / "label_review.csv", encoding="utf-8-sig")
    if agreement is not None:
        agreement.to_csv(a.out / "human_ai_comparison.csv", encoding="utf-8-sig")
    predictions = te[["id", "file", "video", "label", "image_sha256"]].copy()
    for name, pp in preds.items():
        predictions[name] = pp
    predictions.to_csv(a.out / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(scores).to_csv(a.out / "scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_video).to_csv(a.out / "per_video.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(differences).to_csv(a.out / "paired_accuracy.csv", index=False, encoding="utf-8-sig")
    sources = [Path(__file__), HERE / "new_camera_audit.py", HERE / "rel_features.py",
               Path(C.__file__), HERE.parent / "glare_video_source/g_weather_time_probe.py",
               Path(sys.modules[extract_array.__module__].__file__),
               Path(sys.modules[lightgbm_model.__module__].__file__)]
    fingerprint = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = dict(label_evidence=label_info, human_ai_agreement=agreement_info, train_rows=len(tr), test_rows=len(te),
                  test_videos=int(te.video.nunique()), bootstrap_repeats=a.bootstrap, seed=0,
                  rules=rules, scores=scores, accuracy_intervals=intervals, paired_accuracy=differences,
                  files={str(p): fingerprint(p) for p in [a.snapshot, a.rel, a.cache, sheet_path, map_path, *a.labels, *([a.compare_ai] if a.compare_ai else [])]},
                  sources={str(p): fingerprint(p) for p in sources},
                  versions=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__,
                                sklearn=sklearn.__version__, opencv=cv2.__version__,
                                lightgbm=sys.modules["lightgbm"].__version__),
                  limitation="Video groups are not verified independent cameras. AI labels are provisional; no G inputs or deployment change. No threshold/model selection on this test.")
    (a.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"\n라벨 근거: {a.label_kind}; 수락 {len(labels)}/{len(sheet)}장 (AI 임시 결과는 사람 검증 아님)")
    if agreement_info:
        print("AI 라벨과 별도 비교 (사람 정답에 AI 표를 섞지 않음):", json.dumps(agreement_info, ensure_ascii=False))
    print(pd.DataFrame(scores).drop(columns="confusion_matrix").round(4).to_string(index=False))
    print("\n영상 군집 부트스트랩: ML − 기준선 알림정확도 차이")
    print(pd.DataFrame(differences).round(4).to_string(index=False))
    print(f"\n재현 기록: {a.out / 'report.json'}")


if __name__ == "__main__":
    main()
