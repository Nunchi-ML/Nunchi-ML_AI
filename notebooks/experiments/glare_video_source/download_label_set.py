"""라벨용 사진 내려받기 + 블라인드 라벨 시트 만들기.

고르는 법 (시각 확인된 묶음만, 시드 0):
    front  정면 저각도 (해 높이 0~25° · Δ ±25°) 전부
    near   해 낮고 Δ 25~45° 전부
    other  비교군 (해가 높거나 옆 · 등 뒤) — 묶음마다 고르게 150장

    python notebooks/experiments/glare_video_source/download_label_set.py
결과 (data/glare_mapillary/, git 제외):
    images/<묶음>/<id>.jpg      1024px 사진
    label_set.csv               사진별 해 높이 · Δ · 그룹 (분석용 — 라벨러에게는 보여 주지 않음)
    label_sheet_blind.csv       라벨러용: 무작위 순서 · 파일 이름만 · label 칸 비움
    attribution.csv             사진별 촬영자 · Mapillary 주소 (CC BY-SA 4.0 출처 표기용)
"""
import time

import numpy as np
import pandas as pd
import requests

import find_sun_frames as F

OUT = F.OUT
N_OTHER = 150


def pick(d):
    ok = d[d.clock_ok & (d.sun_alt > 0)].copy()
    front = ok[ok.front_low].assign(group="front")
    near = ok[(ok.sun_alt <= 25) & (ok.gap > 25) & (ok.gap <= 45)].assign(group="near")
    rest = ok.drop(front.index.union(near.index))
    per = max(1, N_OTHER // rest.sequence.nunique())
    # 묶음마다 per 장씩 (groupby.apply 는 pandas 3 에서 묶음 열을 빼서 head 로)
    other = rest.sample(frac=1, random_state=0).groupby("sequence").head(per).head(N_OTHER).assign(group="other")
    return pd.concat([front, near, other])


def main():
    tok = F.token()
    sel = pick(pd.read_csv(OUT / "candidates.csv"))
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    meta = []
    for n, r in enumerate(sel.itertuples(), 1):
        path = OUT / "images" / str(r.sequence) / f"{r.id}.jpg"
        info = {}
        for attempt in range(3):
            try:
                info = requests.get(f"https://graph.mapillary.com/{r.id}", timeout=60,
                                    params={"access_token": tok, "fields": "thumb_1024_url,creator"}).json()
                if not path.exists():
                    img = requests.get(info["thumb_1024_url"], timeout=60); img.raise_for_status()
                    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(img.content)
                break
            except (requests.RequestException, KeyError) as e:
                print("  재시도:", r.id, e); time.sleep(3 * (attempt + 1))
        meta.append({"id": r.id, "file": path.relative_to(OUT).as_posix() if path.exists() else None,
                     "creator": (info.get("creator") or {}).get("username")})
        if n % 50 == 0 or n == len(sel):
            print(f"  {n}/{len(sel)}", flush=True)
        time.sleep(0.1)
    m = pd.DataFrame(meta)
    sel = sel.merge(m, on="id")
    sel = sel[sel.file.notna()]
    cols = ["id", "file", "group", "sequence", "captured_at_kst", "lat", "lon", "heading", "sun_alt", "sun_az", "gap", "front_low"]
    sel[cols].to_csv(OUT / "label_set.csv", index=False, encoding="utf-8-sig")
    blind = sel[["id", "file", "sequence"]].sample(frac=1, random_state=1).reset_index(drop=True)
    blind.insert(0, "no", range(1, len(blind) + 1))
    blind = blind.assign(label="", confidence="", note="")     # label: 0 정상 / 1 눈부심 / 2 심한 눈부심
    blind.to_csv(OUT / "label_sheet_blind.csv", index=False, encoding="utf-8-sig")
    sel.assign(url="https://www.mapillary.com/app/?pKey=" + sel.id.astype(str))[["id", "creator", "captured_at_kst", "url"]] \
        .to_csv(OUT / "attribution.csv", index=False, encoding="utf-8-sig")
    size = sum(p.stat().st_size for p in (OUT / "images").rglob("*.jpg")) / 1e6
    print(f"\n받은 사진 {len(sel)}장 ({size:.1f} MB) · {sel.group.value_counts().to_dict()} · 묶음 {sel.sequence.nunique()}개")
    print(f"→ {OUT / 'label_sheet_blind.csv'} (라벨러용) · label_set.csv (분석용) · attribution.csv (출처)")


if __name__ == "__main__":
    main()
