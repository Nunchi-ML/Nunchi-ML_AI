"""두 사람 라벨 비교 → 일치도(카파) · 합의 라벨 · 해 위치와의 관계.

    python notebooks/experiments/glare_video_source/compare_labels.py labels_성진.csv labels_범수.csv
    (파일은 label_tool.html 의 'CSV 저장' 결과. 경로를 안 주면 data/glare_mapillary/labels_*.csv 두 개를 씀)
결과 (data/glare_mapillary/, git 제외):
    labels_merged.csv     사진별 두 사람 라벨 · 합의 라벨(같으면 그 값, 다르면 빈칸 → 같이 다시 보기) + 해 높이 · Δ · 그룹
    labels_disagree.csv   서로 다르게 붙인 사진만 (재검토용)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import find_sun_frames as F

OUT = F.OUT
K = [0, 1, 2]


def kappa(a, b, weighted=False):
    """코헨 카파. weighted=True 면 2차 가중 (0↔2 불일치를 0↔1 보다 크게 침)."""
    m = np.zeros((3, 3))
    for x, y in zip(a, b):
        m[x, y] += 1
    n = m.sum()
    w = np.array([[(i - j) ** 2 / 4 if weighted else float(i != j) for j in K] for i in K])
    exp = np.outer(m.sum(1), m.sum(0)) / n
    return 1 - (w * m).sum() / (w * exp).sum()


def load(path):
    d = pd.read_csv(path, dtype={"label": str})
    who = d.labeler.dropna().iloc[0] if d.labeler.notna().any() else Path(path).stem
    return d[["id", "label", "unsure", "note"]].rename(columns=lambda c: c if c == "id" else f"{c}_{who}"), who


def main():
    paths = sys.argv[1:] or sorted(str(p) for p in OUT.glob("labels_*.csv") if p.stem not in ("labels_merged", "labels_disagree"))
    if len(paths) != 2:
        sys.exit(f"라벨 파일 2개가 필요해요 (찾은 것: {paths})")
    (a, wa), (b, wb) = load(paths[0]), load(paths[1])
    d = pd.read_csv(OUT / "label_set.csv").merge(a, on="id").merge(b, on="id")
    la, lb = d[f"label_{wa}"], d[f"label_{wb}"]

    excl = (la == "x") | (lb == "x")
    both = la.isin(["0", "1", "2"]) & lb.isin(["0", "1", "2"]) & ~excl
    print(f"사진 {len(d)}장 · 둘 다 라벨 {int(both.sum())}장 · 한 명이라도 제외 {int(excl.sum())}장 "
          f"(둘 다 제외 {int(((la == 'x') & (lb == 'x')).sum())}) · 빈칸 {int((~both & ~excl).sum())}장")
    if both.sum() < 10:
        sys.exit("둘 다 라벨한 사진이 너무 적어요")

    x, y = la[both].astype(int), lb[both].astype(int)
    print(f"\n일치율 {np.mean(x == y):.1%} · 카파 {kappa(x, y):.2f} · 가중 카파 {kappa(x, y, True):.2f}  "
          f"(0.6 이상이면 기준이 잘 맞는 편, 지난번 GLARE 는 0.20)")
    print(f"\n          {wb}: 0    1    2")
    for i in K:
        print(f"  {wa} {i}   " + "".join(f"{int(((x == i) & (y == j)).sum()):5d}" for j in K))

    d["label"] = np.where(both & (la == lb), la, np.where(excl & (la == lb), "x", ""))
    d.to_csv(OUT / "labels_merged.csv", index=False, encoding="utf-8-sig")
    d[(la != lb)].to_csv(OUT / "labels_disagree.csv", index=False, encoding="utf-8-sig")
    print(f"\n합의 {int((d.label != '').sum())}장 · 다시 볼 것 {int((la != lb).sum())}장 → labels_merged.csv · labels_disagree.csv")

    # 해 위치와 관계 (합의된 0/1/2 만)
    g = d[d.label.isin(["0", "1", "2"])].assign(glare=lambda t: t.label.astype(int) >= 1)
    if len(g):
        print("\n그룹별 눈부심(1 이상) 비율 — 정면 저각도일수록 높아야 해 위치 기준이 맞는 것")
        print(g.groupby("group").glare.agg(["count", "mean"]).rename(columns={"count": "사진", "mean": "눈부심 비율"}).round(2).to_string())
        g["Δ구간"] = pd.cut(g.gap, [0, 25, 45, 90, 180], include_lowest=True)
        print("\n해 높이 0~25° 사진의 Δ 구간별 눈부심 비율")
        print(g[g.sun_alt <= 25].groupby("Δ구간", observed=True).glare.agg(["count", "mean"]).round(2).to_string())


if __name__ == "__main__":
    main()
