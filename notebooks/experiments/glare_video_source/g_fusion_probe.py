"""G (해 위치 + 날씨) 를 주 판단으로 두고 화면 모델 ② 를 보조로 붙이면 나아지는가.

같은 320장 (g_model_probe.data(): 대구 260 + 서울 운전 시점 60, 촬영 24개) · 같은 검증으로 비교:
    G 만          로지스틱, 화각 반영 6개 (채택 후보)
    화면 만        LightGBM, 화면 숫자 12개 (지금 모델 ②)
    한 모델로      G 6개 + 화면 12개를 한 모델에 (로지스틱 / LightGBM)
    확률 섞기      G 확률 × w + 화면 확률 × (1 − w), w = 0.8 · 0.6 (고정, 고르지 않음)
    G + 화면 2개   G 6개 + 가장 밝은 점 가로 위치 · 밝은 덩어리 (시범 모델 중요 변수) 만 더함
    두 단계        눈부심 여부(0 vs 1+)는 'G + 화면 12' 로지스틱, 눈부심이면 심함(1 vs 2)은 G 확률로
검증: ① 카메라 하나 빼고 시험 (도로 광각 · 블랙박스 · 서울 폰)  ② 촬영 단위 GroupKFold 5 × 30

    python notebooks/experiments/glare_video_source/g_fusion_probe.py
    G_AIHUB=1 · G_SCREEN_NORM=1 로 AI Hub 추가 · 화면 숫자 카메라별 표준화 (둘 다 환경변수)
서울 사진 화면 숫자는 처음 한 번 뽑아 data/glare_mapillary_seoul/screen_features.csv 에 둠.
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nunchi_ml.glare.features import FEATURES, extract
from nunchi_ml.glare.pilot_eval import lightgbm_model

import g_model_probe as G
import retrain_korea as R

G_COLS = G.SETS["화각 반영"]
SCREEN2 = ["brightest_x", "bright_blob"]
CAMS = ("도로 광각", "블랙박스", "서울 폰") + (("AI Hub",) if G.AIHUB else ())
SCREEN_NORM = os.getenv("G_SCREEN_NORM") == "1"


def data():
    d = G.data()
    s = d.camera.eq("서울 폰")
    cache = G.SEOUL / "screen_features.csv"
    if cache.exists():
        f = pd.read_csv(cache)
    else:
        f = pd.DataFrame([{"id": i, **extract(str(G.SEOUL / p))} for i, p in zip(d.id[s], d.file[s])])[["id"] + FEATURES]
        f.to_csv(cache, index=False, encoding="utf-8-sig")
    f = f.set_index("id")
    d.loc[s, FEATURES] = f.loc[d.id[s], FEATURES].to_numpy()
    d = d[d[FEATURES].notna().all(axis=1)].reset_index(drop=True)
    if SCREEN_NORM:      # 카메라마다 자기 평균 · 편차로 (라벨 안 씀 — 앱은 그 기기 최근 화면들로 똑같이 가능)
        g = d.groupby("camera")[FEATURES]
        d[FEATURES] = (d[FEATURES] - g.transform("mean")) / g.transform("std").replace(0, 1)
    return d


def logit():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))


def proba(model, tr, te, cols):
    """클래스 0 · 1 · 2 확률 (학습 폴드에 없는 클래스는 0)."""
    m = model().fit(tr[cols], tr.label)
    p = np.zeros((len(te), 3))
    p[:, m.classes_] = m.predict_proba(te[cols])
    return p


def methods(tr, te):
    """방법별 예측 라벨."""
    pg = proba(logit, tr, te, G_COLS)
    ps = proba(lightgbm_model, tr, te, FEATURES)
    pf = proba(logit, tr, te, G_COLS + FEATURES)
    glare = pf[:, 1:].sum(1) > pf[:, 0]
    out = {"G 만 (로지스틱)": pg.argmax(1),
           "화면 만 (LightGBM)": ps.argmax(1),
           "한 모델 · 로지스틱 (G + 화면 12)": pf.argmax(1),
           # 눈부심 여부는 G + 화면, 심함은 G 확률로 (눈부심일 때만)
           "두 단계 (여부 G+화면 → 심함 G)": np.where(glare, np.where(pg[:, 2] > pg[:, 1], 2, 1), 0),
           "한 모델 · LightGBM (G + 화면 12)": proba(lightgbm_model, tr, te, G_COLS + FEATURES).argmax(1),
           "G + 화면 2개 (로지스틱)": proba(logit, tr, te, G_COLS + SCREEN2).argmax(1)}
    for w in (0.8, 0.6):
        out[f"확률 섞기 G {w:.1f} · 화면 {1 - w:.1f}"] = (w * pg + (1 - w) * ps).argmax(1)
    return out


def fmt(s):
    return f"F1 {s['macro F1']:.2f} · 심함 {s['심함 재현율']:.2f} · 여부 {s['눈부심 여부 정확도']:.2f}"


def main():
    d = data()
    y = d.label.to_numpy()
    print(f"사진 {len(d)}장 · " + " · ".join(f"{c} {int(d.camera.eq(c).sum())}" for c in CAMS) + f" · 촬영 {d.group.nunique()}개\n")

    print("① 카메라 하나 빼고 학습 → 뺀 카메라로 시험 (눈부심 여부 · 세 카메라 평균)")
    res = {}
    for c in CAMS:
        te = d.camera.eq(c).to_numpy()
        for name, p in methods(d[~te], d[te]).items():
            res.setdefault(name, {})[c] = R.scores(y[te], p)
    for name, r in res.items():
        acc = [r[c]["눈부심 여부 정확도"] for c in CAMS]
        f1 = [r[c]["macro F1"] for c in CAMS]
        print(f"  {name:30s} 여부 " + " · ".join(f"{a:.2f}" for a in acc) + f" → 평균 {np.mean(acc):.2f} · 최저 {min(acc):.2f}"
              f" | F1 평균 {np.mean(f1):.2f}")

    print("\n② 촬영 단위 GroupKFold 5 × 30 (평균 ± 표준편차)")
    rows = []
    for r in range(30):
        pred = {}
        for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=r).split(d, y, d.group):
            for name, p in methods(d.iloc[tr], d.iloc[te]).items():
                pred.setdefault(name, np.zeros(len(d), int))[te] = p
        rows += [{"방법": n, **R.scores(y, p)} for n, p in pred.items()]
    s = pd.DataFrame(rows).groupby("방법", sort=False)
    m, sd = s.mean(numeric_only=True), s.std(numeric_only=True)
    for n in m.index:
        print(f"  {n:30s} " + " · ".join(f"{k} {m.loc[n, c]:.2f}±{sd.loc[n, c]:.2f}"
                                         for k, c in (("F1", "macro F1"), ("심함", "심함 재현율"), ("여부", "눈부심 여부 정확도"))))


if __name__ == "__main__":
    main()
