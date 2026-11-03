"""블랙박스 추가 세트(한 날 · 한 차 · 해 높이 20~24°) 라벨 분석 — 해 높이가 거의 같으니 Δ 만의 효과를 본다.

    python notebooks/experiments/glare_video_source/analyze_dashcam.py --labels labels_dash_성진.csv
(fix_metadata.py 를 먼저 돌려 heading_suspect 가 있어야 함)
결과: data/glare_mapillary/labels_dash.csv (라벨 · Δ · 화면 특징 12개)
"""
import argparse

import pandas as pd
from scipy.stats import spearmanr

from nunchi_ml.glare.features import FEATURES, extract

import find_sun_frames as F

OUT = F.OUT
BINS = [0, 5, 10, 15, 20, 25, 35, 60, 180]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels_dash_성진.csv")
    a = ap.parse_args()
    d = pd.read_csv(OUT / "dashcam_set.csv").merge(pd.read_csv(OUT / a.labels)[["id", "label"]], on="id")
    d = d[d.label.isin([0, 1, 2])].reset_index(drop=True)
    d = pd.concat([d, pd.DataFrame([extract(str(OUT / f)) for f in d.file])], axis=1)
    d["glare"], d["severe"] = d.label >= 1, d.label == 2
    d.to_csv(OUT / "labels_dash.csv", index=False, encoding="utf-8-sig")
    ok = d[~d.heading_suspect]
    print(f"라벨 {d.label.value_counts().sort_index().to_dict()} · 해 높이 {d.sun_alt.min():.1f}~{d.sun_alt.max():.1f}° · "
          f"방향 의심 {int(d.heading_suspect.sum())}장 제외 → {len(ok)}장")
    agg = dict(사진=("label", "size"), 눈부심=("glare", "mean"), 심함=("severe", "mean"), 평균=("label", "mean"))
    print("\nΔ 구간별"); print(ok.groupby(pd.cut(ok.gap, BINS, include_lowest=True), observed=True).agg(**agg).round(2).to_string())
    r = pd.Series({c: spearmanr(ok[c], ok.label)[0] for c in FEATURES + ["gap"]})
    print("\n라벨과 순위 상관"); print(r.sort_values(key=abs, ascending=False).round(2).to_string())


if __name__ == "__main__":
    main()
