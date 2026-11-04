"""새 카메라 독립 시험 — 사진만 보는 ML vs 사진만 보는 규칙 (영상 프레임, 라벨은 사람).

학습: 지금까지의 624장 전부(②-v4). 시험: new_camera_frames.py 가 만든 라벨 묶음의 사람 라벨 프레임 (학습 · 설계에 안 쓴 카메라·장소).
비교 (경계값 · 모델은 학습 624장에서만 정함, 시험 라벨을 보고 고르지 않음):
    규칙 R1  밝기 규칙(전체 밝기)       pilot_eval.brightness_rule — 임계값 2개
    규칙 R2  하얀 픽셀 비율 규칙         같은 방식
    규칙 R3  상대 특징 12 중 학습에서 가장 좋은 1개의 임계값 규칙  (규칙에게도 최선의 기회)
    ML 1    기존 모델 ② 형태 LightGBM(화면 숫자 12)
    ML 2    로지스틱 (상대 특징 12)       P(눈부심 1+) ≥ .5 알림, 심함은 P(2) ≥ P(1)
통계: 영상(sequence) 단위 군집 부트스트랩. 라벨이 2인 이상이면 카파 · 다수결.

    python notebooks/experiments/sun_visible_probe/new_camera_eval.py --pack data/new_cameras --labels data/new_cameras/labels_a.csv [labels_b.csv ...] \
        --snapshot data/analysis_20261104/model2_snapshot.csv --rel data/sun_visible/rel_features.csv --cache <image_features.csv>
"""
import argparse
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
import g_visibility_probe as C
from g_weather_time_probe import replay
from nunchi_ml.glare.features import FEATURES, extract_array
from nunchi_ml.glare.pilot_eval import lightgbm_model
from rel_features import REL, rel_array


def load_labels(paths):
    ls = [pd.read_csv(p, dtype={"id": str}).set_index("id") for p in paths]
    lab = pd.concat([l["label"].replace({"x": np.nan, "X": np.nan}).astype(float).rename(f"l{i}") for i, l in enumerate(ls)], axis=1)
    if lab.shape[1] > 1:
        from itertools import combinations
        for i, j in combinations(range(lab.shape[1]), 2):
            m = lab[[f"l{i}", f"l{j}"]].dropna()
            po = float((m.iloc[:, 0] == m.iloc[:, 1]).mean())
            pe = sum(float((m.iloc[:, 0] == c).mean()) * float((m.iloc[:, 1] == c).mean()) for c in (0, 1, 2))
            print(f"라벨러 {i}–{j}: 일치 {po:.2f} · 카파 {(po - pe) / (1 - pe):.2f} (n={len(m)})")
    maj = lab.mode(axis=1)
    out = lab.median(axis=1).round() if lab.shape[1] == 1 else maj.iloc[:, 0].where(maj.notna().sum(axis=1) == 1)
    return out.dropna().astype(int).rename("label")


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
    a = ap.parse_args()

    snap = replay(pd.read_csv(a.snapshot, dtype={"id": str}), a.data_root / "asos", "previous")
    tr = C.features(snap, C.image_paths(snap, a.data_root), a.cache)
    rel = pd.read_csv(a.rel, dtype={"id": str})[["id"] + REL]
    tr = tr.merge(rel, on="id", how="left", suffixes=("", "_rel"))

    labels = load_labels(a.labels)
    fmap = pd.read_csv(a.pack.parent / "private_frame_map.csv" if (a.pack.parent / "private_frame_map.csv").exists() else a.pack / "private_frame_map.csv", dtype={"id": str})
    sheet = pd.read_csv(a.pack / "label_pack" / "label_sheet_blind.csv", dtype={"id": str})[["id", "file"]]
    te = sheet.merge(labels.reset_index(), on="id").merge(fmap, on="id", suffixes=("", "_m"))
    rows = []
    for r in te.itertuples():
        im = cv2.imdecode(np.fromfile(str(a.pack / "label_pack" / r.file), np.uint8), cv2.IMREAD_COLOR)
        rows.append({**extract_array(im), **rel_array(im)})
    te = pd.concat([te.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    print(f"시험 프레임 {len(te)}장 · 영상 {te.video.nunique()}개 · 라벨 분포 정상 {(te.label == 0).sum()} / 눈부심 {(te.label == 1).sum()} / 심함 {(te.label == 2).sum()}")
    print(te.groupby("video").label.agg(n="size", glare=lambda s: int((s >= 1).sum())).to_string())

    y = te.label.to_numpy()
    preds = {}
    for name, col in (("R1 밝기 규칙(전체 밝기)", "mean_all"), ("R2 하얀 픽셀 비율 규칙", "white_ratio")):
        f, params = fit_single_rule(tr, col)
        preds[name] = apply_rule(te, col, params)
    best = max(((fit_single_rule(tr, c), c) for c in REL), key=lambda t: t[0][0])
    (_, params), col = best
    preds[f"R3 최선의 단일 특징 규칙({col})"] = apply_rule(te, col, params)
    m = lightgbm_model().fit(tr[FEATURES], tr.label)
    preds["ML1 LightGBM(화면 12, 기존 모델 ② 형태)"] = m.predict(te[FEATURES])
    lg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(tr[REL], tr.label)
    p = np.zeros((len(te), 3)); p[:, lg.classes_] = lg.predict_proba(te[REL])
    preds["ML2 로지스틱(상대 12)"] = decide(p)

    def acc(yy, pp): return float(np.mean((yy >= 1) == (pp >= 1)))
    def f1(yy, pp): return float(f1_score(yy, pp, average="macro", labels=[0, 1, 2]))
    print("\n방법 | 알림정확도 | macro F1 | 헛알림(정상 중) | 알림재현율(눈부심 중)")
    for n, pp in preds.items():
        fa = float(np.mean(pp[y == 0] >= 1)) if (y == 0).any() else float("nan")
        rc = float(np.mean(pp[y >= 1] >= 1)) if (y >= 1).any() else float("nan")
        print(f"  {n:42s} {acc(y, pp):.3f} · {f1(y, pp):.3f} · {fa:.3f} · {rc:.3f}")

    rng = np.random.default_rng(0); cl = te.video.to_numpy(); u = np.unique(cl); idx = {c: np.where(cl == c)[0] for c in u}
    B = {n: [] for n in preds}
    for _ in range(3000):
        ii = np.concatenate([idx[c] for c in rng.choice(u, len(u))])
        for n, pp in preds.items():
            B[n].append(acc(y[ii], pp[ii]))
    print(f"\n영상 {len(u)}개 군집 부트스트랩 — 알림정확도 95% 구간, 그리고 ML − 방법 차이")
    for n in preds:
        b = np.array(B[n]); line = f"  {n:42s} [{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]"
        for ml in [k for k in preds if k.startswith("ML")]:
            if n != ml:
                d = np.array(B[ml]) - b; line += f" | {ml[:3]}−이 방법 {d.mean():+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]"
        print(line)


if __name__ == "__main__":
    main()
