"""사진 상대 특징(해가 보이는가) 이 화면 숫자 12개보다 카메라가 바뀔 때 덜 무너지는가.

같은 624장 (②-v4: 도로 광각 97 · 블랙박스 163 · 서울 폰 60 · AI Hub 304) · 같은 G 6개 · 같은 검증:
    ① 세 카메라(AI Hub 제외) 촬영 단위 GroupKFold 5 × 30
    ② 세 카메라 하나 빼고 시험 (눈부심 여부)
    ③ 세 카메라로 학습 → AI Hub 시험 (처음 보는 카메라, 외부 시험)
방법:
    G 만                         로지스틱 (화각 반영 6개)   ← 최종 결정(PM)
    두 단계 (기존 화면 12)         여부 = G + 화면 12 로지스틱 → 심함 = G
    두 단계 (상대 12)             여부 = G + 상대 12 로지스틱 → 심함 = G
    상대 12 만 (여부)             해가 보이는가만으로 눈부심 여부
    G + 상대 4 (로지스틱)          G + peak_med · blob_circ · ring2 · glow_con 만 한 모델
라벨 · 해 위치를 특징 만들 때 쓰지 않는다. 앱도 사진 한 장으로 똑같이 계산 가능.

    python notebooks/experiments/sun_visible_probe/sun_visible_probe.py
"""
import os
import sys
from pathlib import Path

os.environ["G_AIHUB"] = "1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "glare_video_source"))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import g_model_probe as G
import retrain_korea as R
from nunchi_ml.glare.features import FEATURES
from rel_features import REL, rel_file

ROOT = G.F.ROOT / "data"
CACHE = ROOT / "sun_visible" / "rel_features.csv"
G_COLS = G.SETS["화각 반영"]
REL4 = ["peak_med", "blob_circ", "ring2", "glow_con"]
THREE = ("도로 광각", "블랙박스", "서울 폰")


def index_daegu():
    idx = {}
    for dp, _, fs in os.walk(ROOT / "glare_mapillary"):
        for f in fs:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                idx.setdefault(Path(f).stem, Path(dp) / f)
    return idx


def path_of(r, daegu):
    if r.camera == "서울 폰":
        return ROOT / "glare_mapillary_seoul" / r.file
    if r.camera == "AI Hub":
        return ROOT / "aihub_special" / r.file
    return daegu.get(str(r.id))


def data():
    d = G.data()
    if CACHE.exists():
        f = pd.read_csv(CACHE, dtype={"id": str})
    else:
        daegu, rows = index_daegu(), []
        for r in d.itertuples():
            p = path_of(r, daegu)
            if p is None or not Path(p).exists():
                print("없음", r.camera, r.id)
                continue
            rows.append({"id": str(r.id), **rel_file(p)})
        f = pd.DataFrame(rows)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        f.to_csv(CACHE, index=False, encoding="utf-8-sig")
        print(f"상대 특징 {len(f)}장 → {CACHE}")
    d["id"] = d["id"].astype(str)
    s = d.camera.eq("서울 폰")      # 서울 사진 화면 숫자는 g_fusion_probe 가 캐시해 둔 파일에서 (없으면 못 비교)
    sc = pd.read_csv(G.SEOUL / "screen_features.csv", dtype={"id": str}).set_index("id")
    d.loc[s, FEATURES] = sc.loc[d.id[s], FEATURES].to_numpy()
    d = d.merge(f, on="id", how="inner", suffixes=("", "_rel"))
    return d[d[FEATURES + REL].notna().all(axis=1)].reset_index(drop=True)


def logit():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))


def proba(tr, te, cols):
    m = logit().fit(tr[cols], tr.label)
    p = np.zeros((len(te), 3))
    p[:, m.classes_] = m.predict_proba(te[cols])
    return p


def two_stage(tr, te, extra):
    pg = proba(tr, te, G_COLS)
    pf = proba(tr, te, G_COLS + extra)
    return np.where(pf[:, 1:].sum(1) > pf[:, 0], np.where(pg[:, 2] > pg[:, 1], 2, 1), 0)


def binary(tr, te, cols):
    """눈부심 여부만 (0 vs 1+) — 3단계 예측 형태로 맞춤 (1 = 있음)."""
    m = logit().fit(tr[cols], (tr.label >= 1).astype(int))
    return m.predict(te[cols])


def methods(tr, te):
    return {"G 만": proba(tr, te, G_COLS).argmax(1),
            "두 단계 (기존 화면 12)": two_stage(tr, te, FEATURES),
            "두 단계 (상대 12)": two_stage(tr, te, REL),
            "G + 상대 4 (한 모델)": proba(tr, te, G_COLS + REL4).argmax(1),
            "상대 12 만 (여부)": binary(tr, te, REL),
            "기존 화면 12 만 (여부)": binary(tr, te, FEATURES)}


def extra(y, p):
    """정상(0) 중 알림 낸 비율(헛알림) · 눈부심(1+) 중 알림 낸 비율(재현율)."""
    return np.mean(p[y == 0] >= 1) if (y == 0).any() else np.nan, np.mean(p[y >= 1] >= 1) if (y >= 1).any() else np.nan


def main():
    d = data()
    print(f"사진 {len(d)}장 · " + " · ".join(f"{c} {int(d.camera.eq(c).sum())}" for c in THREE + ("AI Hub",)))
    d3 = d[d.camera.isin(THREE)].reset_index(drop=True)
    y3 = d3.label.to_numpy()

    print("\n② 카메라 하나 빼고 학습 → 뺀 카메라로 시험 (눈부심 여부 정확도 · 도로 광각 / 블랙박스 / 서울 폰)")
    res = {}
    for c in THREE:
        te = d3.camera.eq(c).to_numpy()
        for n, p in methods(d3[~te], d3[te]).items():
            res.setdefault(n, {})[c] = R.scores(y3[te], p)["눈부심 여부 정확도"]
    for n, r in res.items():
        a = [r[c] for c in THREE]
        print(f"  {n:24s} " + " · ".join(f"{x:.2f}" for x in a) + f" → 평균 {np.mean(a):.2f} · 최저 {min(a):.2f}")

    print("\n③ 세 카메라로 학습 → AI Hub 시험 (처음 보는 카메라): 눈부심 여부 · 3단계 F1 · 헛알림(정상 중) · 재현율(눈부심 중)")
    ah = d[d.camera.eq("AI Hub")].reset_index(drop=True)
    ya = ah.label.to_numpy()
    for n, p in methods(d3, ah).items():
        s = R.scores(ya, p)
        fa, rc = extra(ya, p)
        print(f"  {n:24s} 여부 {s['눈부심 여부 정확도']:.2f} · F1 {s['macro F1']:.2f} · 헛알림 {fa:.2f} · 재현율 {rc:.2f}")

    print("\n① 세 카메라 촬영 단위 GroupKFold 5 × 30 (평균 ± 표준편차)")
    rows = []
    for r in range(30):
        pred = {}
        for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=r).split(d3, y3, d3.group):
            for n, p in methods(d3.iloc[tr], d3.iloc[te]).items():
                pred.setdefault(n, np.zeros(len(d3), int))[te] = p
        rows += [{"방법": n, **R.scores(y3, p)} for n, p in pred.items()]
    s = pd.DataFrame(rows).groupby("방법", sort=False)
    m, sd = s.mean(numeric_only=True), s.std(numeric_only=True)
    for n in m.index:
        print(f"  {n:24s} F1 {m.loc[n, 'macro F1']:.2f}±{sd.loc[n, 'macro F1']:.2f} · 여부 {m.loc[n, '눈부심 여부 정확도']:.2f}±{sd.loc[n, '눈부심 여부 정확도']:.2f}")


if __name__ == "__main__":
    main()
