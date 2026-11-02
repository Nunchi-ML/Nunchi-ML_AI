"""운전 시점 사진만 남긴 라벨 데이터 + 해 위치 · 화면 특징 12개 + 첫 분석.

인도 · 보행자 시점 묶음은 sequence_view.csv (묶음마다 한 장씩 눈으로 보고 나눔: dashcam / road / sidewalk / unsure) 로 뺀다.

    python notebooks/experiments/glare_video_source/make_drive_set.py                       # 기본: labels_성진.csv
    python notebooks/experiments/glare_video_source/make_drive_set.py --labels labels_성진.csv --pack
결과 (data/glare_mapillary/, git 제외):
    labels_drive.csv           운전 시점 사진 · 라벨 · 해 높이 · Δ · 화면 특징 12개
    drive_pack/ (--pack)       두 번째 라벨러용: 운전 시점 사진만 + 새로 섞은 label_sheet_blind.csv → 압축 drive_pack.zip
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from nunchi_ml.glare.features import FEATURES, extract

import find_sun_frames as F

OUT = F.OUT
HERE = Path(__file__).parent
DRIVE = ["dashcam", "road"]


def rate(d, by):
    return d.groupby(by, observed=True).agg(사진=("label", "size"), 눈부심=("glare", "mean"),
                                            심함=("label", lambda x: (x == 2).mean())).round(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels_성진.csv")
    ap.add_argument("--pack", action="store_true", help="두 번째 라벨러용 묶음 만들기")
    a = ap.parse_args()

    view = pd.read_csv(HERE / "sequence_view.csv")
    lab = pd.read_csv(OUT / a.labels, dtype={"label": str})[["id", "label"]]
    d = pd.read_csv(OUT / "label_set.csv").merge(view[["sequence", "view"]], on="sequence").merge(lab, on="id")
    print("시점별 사진:", d.view.value_counts().to_dict())
    d = d[d.view.isin(DRIVE) & d.label.isin(["0", "1", "2"])].reset_index(drop=True)
    d["label"] = d.label.astype(int)
    d["glare"] = d.label >= 1
    d = pd.concat([d, pd.DataFrame([extract(str(OUT / f)) for f in d.file])], axis=1)
    d.to_csv(OUT / "labels_drive.csv", index=False, encoding="utf-8-sig")
    print(f"운전 시점 {len(d)}장 (묶음 {d.sequence.nunique()}개) · 라벨 {d.label.value_counts().sort_index().to_dict()} → labels_drive.csv")

    print("\n시점별"); print(rate(d, "view").to_string())
    print("\n정면 저각도 (fix_metadata.py 보정 후)"); print(rate(d, "front_low").to_string())
    print("\n해 높이별"); print(rate(d, pd.cut(d.sun_alt, [0, 10, 25, 90]).rename("해 높이")).to_string())
    print("\nΔ 별 (해 높이 0~25° · 방향 의심 사진 제외)")
    low = d[(d.sun_alt <= 25) & ~d.get("heading_suspect", False)]
    print(rate(low, pd.cut(low.gap, [0, 25, 45, 90, 180], include_lowest=True).rename("Δ")).to_string())
    r = pd.Series({c: spearmanr(d[c], d.label)[0] for c in FEATURES + ["sun_alt", "gap"]})
    print("\n라벨과 순위 상관 (|r| 큰 순)"); print(r.sort_values(key=abs, ascending=False).round(2).to_string())

    if a.pack:
        p = OUT / "drive_pack"
        shutil.rmtree(p, ignore_errors=True)
        for f in d.file:
            (p / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(OUT / f, p / f)
        blind = d[["id", "file", "sequence"]].sample(frac=1, random_state=2).reset_index(drop=True)
        blind.insert(0, "no", range(1, len(blind) + 1))
        blind.assign(label="", confidence="", note="").to_csv(p / "label_sheet_blind.csv", index=False, encoding="utf-8-sig")
        pd.read_csv(OUT / "attribution.csv").merge(d[["id"]]).to_csv(p / "attribution.csv", index=False, encoding="utf-8-sig")
        shutil.copy2(HERE / "label_tool.html", p / "label_tool.html")      # 압축 하나만 받으면 바로 시작
        z = shutil.make_archive(str(OUT / "drive_pack"), "zip", p)
        print(f"\n두 번째 라벨러용 {len(blind)}장 → {z} ({Path(z).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
