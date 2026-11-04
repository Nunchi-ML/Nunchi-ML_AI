"""라벨러 여러 명 일치도 · 다수결 라벨 (AI Hub 역광 308장, 서울 운전 시점 60장 등).

    python notebooks/experiments/glare_video_source/compare_aihub_labels.py
    python notebooks/experiments/glare_video_source/compare_aihub_labels.py --dir data/glare_mapillary_seoul --set label_set.csv
입력: <dir>/labels_<이름>.csv (label_tool.html 'CSV 저장' 결과, 2명 이상)
결과 (<dir>, 기본 data/aihub_special/, git 제외):
    labels_merged.csv     사진별 각자 라벨 · 다수결 라벨(majority, 동률이면 빈칸) · 만장일치 여부 + 해 높이 · 주행
    labels_disagree.csv   만장일치가 아닌 사진 (재검토용)
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from compare_labels import kappa

OUT = Path(__file__).resolve().parents[3] / "data" / "aihub_special"


def fleiss(m):
    """플라이스 카파. m: 사진 × 클래스 표 (각 칸 = 그 클래스를 고른 사람 수)."""
    n = m.sum(1)[0]
    p = m.sum(0) / m.sum()
    P = ((m ** 2).sum(1) - n) / (n * (n - 1))
    return (P.mean() - (p ** 2).sum()) / (1 - (p ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=OUT)
    ap.add_argument("--set", default="aihub_label_set.csv", help="사진 정보 표 (해 높이 · 묶음 등)")
    args = ap.parse_args()
    out = args.dir if args.dir.is_absolute() else OUT.parents[1] / args.dir
    files = sorted(out.glob("labels_*.csv"))
    files = [f for f in files if f.stem not in ("labels_merged", "labels_disagree")]
    who = [f.stem.removeprefix("labels_") for f in files]
    d = pd.read_csv(out / args.set)
    for w, f in zip(who, files):
        d = d.merge(pd.read_csv(f, dtype={"label": str})[["id", "label"]].rename(columns={"label": w}), on="id")
    d = d[d[who].isin(["0", "1", "2"]).all(axis=1)].copy()          # X · 빈칸 빼기
    d[who] = d[who].astype(int)
    print(f"사진 {len(d)}장 · 라벨러 {', '.join(who)}\n")

    print("두 사람씩       카파   가중   완전일치  0↔2")
    for a, b in combinations(who, 2):
        print(f"{a}–{b:6s}  {kappa(d[a], d[b]):.2f}   {kappa(d[a], d[b], True):.2f}   "
              f"{(d[a] == d[b]).mean():.0%}     {((d[a] - d[b]).abs() == 2).sum()}")
    m = np.stack([(d[who] == k).sum(axis=1).to_numpy() for k in (0, 1, 2)], axis=1)
    print(f"전체 플라이스 카파 {fleiss(m):.2f} · 만장일치 {(m.max(1) == len(who)).mean():.0%}\n")

    top = m.argmax(1)
    d["majority"] = np.where((m == m.max(1, keepdims=True)).sum(1) == 1, top, np.nan)
    d["unanimous"] = m.max(1) == len(who)
    print("라벨러별 분포", {w: d[w].value_counts().sort_index().to_dict() for w in who})
    print("다수결 분포  ", d.majority.value_counts(dropna=False).sort_index().to_dict(), "\n")
    print(pd.crosstab(d.group, d.majority, margins=True).to_string(), "\n")

    keep = [c for c in ["id", "file", "group", "sequence", "frame", "sun_alt", "sun_az", "gap"] if c in d] + [*who, "majority", "unanimous"]
    d[keep].to_csv(out / "labels_merged.csv", index=False, encoding="utf-8-sig")
    d.loc[~d.unanimous, keep].to_csv(out / "labels_disagree.csv", index=False, encoding="utf-8-sig")
    print(f"→ labels_merged.csv · labels_disagree.csv ({(~d.unanimous).sum()}장)")


if __name__ == "__main__":
    main()
