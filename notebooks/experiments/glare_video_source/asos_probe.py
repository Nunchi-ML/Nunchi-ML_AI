"""구름양 · 일조(ASOS 대구) 를 붙이면 '해는 정면인데 흐려서 눈부심 0' 을 가를 수 있나.

1) 라벨 사진의 눈부심 비율을 날씨(맑음 / 구름 / 흐림) · 해가 화면 안인지(in_fov) 로 나눠 봄
2) 모델 비교 (fov_probe.py · v2_candidate.py 와 같은 사진 · 촬영 단위 검증)
    한국만: 화면 12 + 해 위치  vs  + 날씨(cloud · sunshine)
    GLARE+한국 v2: 화면 12 + 해 위치  vs  + 날씨 (GLARE 는 빈칸)

    python notebooks/experiments/glare_video_source/asos_probe.py --glare ../../data/glare/pilot_glare
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from nunchi_ml.glare.features import FEATURES
from nunchi_ml.glare.pilot_eval import load, lightgbm_model

import fetch_asos as A
import find_sun_frames as F
import retrain_korea as R
import v2_candidate as V

OUT = F.OUT
SUN, WX = V.SUN, ["cloud", "sunshine"]


def cv(d, cols, n=5):
    y, p = d.label.to_numpy(), np.zeros(len(d), int)
    for tr, te in GroupKFold(n_splits=n).split(d, y, d.group):
        p[te] = lightgbm_model().fit(d.iloc[tr][cols], y[tr]).predict(d.iloc[te][cols])
    return p


def show(name, d, p, cams):
    y = d.label.to_numpy()
    out = []
    for cam in cams:
        m = d.camera.eq(cam).to_numpy()
        s = R.scores(y[m], p[m])
        out.append(f"{cam} F1 {s['macro F1']:.2f} · 심함 {s['심함 재현율']:.2f} · 여부 {s['눈부심 여부 정확도']:.2f}")
    print(f"  {name:28s} " + " | ".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glare", required=True)
    a = ap.parse_args()
    k = V.korea_with_sun()
    t = pd.concat([pd.read_csv(OUT / n)[["id", "captured_at_kst"]] for n in ("labels_drive.csv", "labels_dash.csv")]).drop_duplicates("id")
    k = A.attach(k.merge(t, on="id"))
    k["날씨"] = pd.cut(k.cloud, [-1, 2, 7, 10], labels=["맑음(0~2)", "구름(3~7)", "흐림(8~10)"])
    agg = dict(사진=("label", "size"), 눈부심=("label", lambda x: (x >= 1).mean()), 심함=("label", lambda x: (x == 2).mean()))
    print("1) 날씨별 눈부심 비율 (한국 라벨 284장)")
    print(k.groupby(["camera", "날씨"], observed=True).agg(**agg).round(2).to_string())
    print("\n   해가 화면 안(in_fov=1) 사진만 · 날씨별")
    print(k[k.in_fov == 1].groupby(["camera", "날씨"], observed=True).agg(**agg).round(2).to_string())
    print("\n   일조(그 한 시간 해가 비친 비율)와 눈부심 여부 순위 상관:",
          round(k[["sunshine"]].assign(g=k.label >= 1).corr("spearman").iloc[0, 1], 2),
          "· 전운량:", round(k[["cloud"]].assign(g=k.label >= 1).corr("spearman").iloc[0, 1], 2))

    print("\n2) 모델 비교 — 한국만 (촬영 단위 GroupKFold 5)")
    cams = ("블랙박스", "도로 광각")
    for name, cols in (("화면 12 + 해 위치", FEATURES + SUN), ("화면 12 + 해 위치 + 날씨", FEATURES + SUN + WX),
                       ("해 위치 + 날씨 (화면 없이)", SUN + WX)):
        show(name, k, cv(k, cols), cams)

    g = load(Path(a.glare) / "glare_features.csv", Path(a.glare) / "pilot_labels.csv")
    g = g.assign(group="us:" + g.vid, camera="GLARE", **{c: np.nan for c in SUN + WX})
    cols = ["group", "camera", "label"] + FEATURES + SUN + WX
    d = pd.concat([g[cols], k[cols]], ignore_index=True)
    print("\n   GLARE+한국 v2 (촬영 단위 GroupKFold 6)")
    for name, c in (("v2 화면 12 + 해 위치", FEATURES + SUN), ("v2 + 날씨", FEATURES + SUN + WX)):
        show(name, d, cv(d, c, 6), cams + ("GLARE",))


if __name__ == "__main__":
    main()
