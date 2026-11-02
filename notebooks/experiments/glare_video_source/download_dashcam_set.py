"""블랙박스 여행(2019-02-24 오후, fix_metadata.py 로 +9시간 보정한 묶음) 사진 추가 라벨 세트.

1~2초 간격 연속 사진이라 묶음마다 3초 이상 떨어진 것만 고른다 (거의 같은 장면 반복 방지).
이미 라벨 세트(label_set.csv)에 있는 사진 · 방향 의심 사진은 뺀다.

    python notebooks/experiments/glare_video_source/download_dashcam_set.py
결과 (data/glare_mapillary/, git 제외):
    images/<묶음>/<id>.jpg              사진 (기존 폴더에 같이)
    dashcam_set.csv                     분석용: 해 높이 · Δ · front_low
    dash_pack/ · dash_pack.zip          라벨러용: 사진 + 무작위 순서 label_sheet_blind.csv + label_tool.html
"""
import shutil
from pathlib import Path

import pandas as pd

import find_sun_frames as F
from download_label_set import fetch

OUT = F.OUT
HERE = Path(__file__).parent
GAP_S = 3


def pick(c, done):
    t = c[(c.offset_h == 9) & ~c.id.isin(done) & ~c.get("heading_suspect", False)].copy()
    t = t.sort_values(["sequence", "captured_at_ms"])
    keep, last = [], {}
    for r in t.itertuples():
        if r.sequence not in last or r.captured_at_ms - last[r.sequence] >= GAP_S * 1000:
            keep.append(r.Index); last[r.sequence] = r.captured_at_ms
    return t.loc[keep]


def main():
    c = pd.read_csv(OUT / "candidates.csv")
    sel = pick(c, set(pd.read_csv(OUT / "label_set.csv").id))
    print(f"고른 사진 {len(sel)}장 (정면 저각도 {int(sel.front_low.sum())}) · 묶음 {sel.sequence.nunique()}개")
    sel = sel.merge(fetch(sel, F.token()), on="id")
    sel = sel[sel.file.notna()]
    sel.assign(group="dashcam")[["id", "file", "group", "sequence", "captured_at_kst", "lat", "lon", "heading",
                                 "sun_alt", "sun_az", "gap", "front_low", "creator"]] \
        .to_csv(OUT / "dashcam_set.csv", index=False, encoding="utf-8-sig")

    p = OUT / "dash_pack"
    shutil.rmtree(p, ignore_errors=True)
    for f in sel.file:
        (p / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT / f, p / f)
    blind = sel[["id", "file", "sequence"]].sample(frac=1, random_state=3).reset_index(drop=True)
    blind.insert(0, "no", range(1, len(blind) + 1))
    blind.assign(label="", confidence="", note="").to_csv(p / "label_sheet_blind.csv", index=False, encoding="utf-8-sig")
    sel.assign(url="https://www.mapillary.com/app/?pKey=" + sel.id.astype(str))[["id", "creator", "captured_at_kst", "url"]] \
        .to_csv(p / "attribution.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(HERE / "label_tool.html", p / "label_tool.html")
    z = shutil.make_archive(str(OUT / "dash_pack"), "zip", p)
    print(f"받은 사진 {len(sel)}장 → dashcam_set.csv · {z} ({Path(z).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
