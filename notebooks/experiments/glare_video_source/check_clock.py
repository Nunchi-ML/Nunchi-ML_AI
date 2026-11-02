"""촬영 시각 시간대 점검: 연속 촬영 묶음마다 대표 사진 1장(작은 썸네일)의 밝기로 '낮 사진인가'를 보고,
기록 시각 그대로(UTC 기준)와 9시간 당긴 시각(카메라가 한국시간을 UTC로 저장한 경우) 중 해 높이가 맞는 쪽을 고른다.

    python notebooks/experiments/glare_video_source/check_clock.py
결과: data/glare_mapillary/clock_check.csv (묶음별 판정) · candidates.csv 에 offset_h · 보정한 해 위치 · front_low 다시 계산
"""
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

from nunchi_ml.sun import heading_gap, is_front_low, sun_position

import find_sun_frames as F

OUT = F.OUT
DARK, BRIGHT = 60, 95            # 썸네일 평균 밝기(0~255): 이보다 어두우면 밤, 밝으면 낮


def thumb_brightness(image_id, tok):
    r = requests.get(f"https://graph.mapillary.com/{image_id}", params={"access_token": tok, "fields": "thumb_256_url"}, timeout=60)
    r.raise_for_status()
    img = requests.get(r.json()["thumb_256_url"], timeout=60)
    img.raise_for_status()
    g = np.asarray(Image.open(io.BytesIO(img.content)).convert("L"), float)
    return float(g[: int(g.shape[0] * 0.6)].mean())                 # 위쪽 60% (하늘 · 먼 풍경) — 보닛 · 차 안 영향 줄임


def main():
    tok = F.token()
    d = pd.read_csv(OUT / "candidates.csv")
    utc = pd.to_datetime(d.captured_at_ms, unit="ms", utc=True)
    rows = []
    for seq, g in d.groupby("sequence"):
        i = g.index[len(g) // 2]
        b = thumb_brightness(d.at[i, "id"], tok); time.sleep(0.2)
        alt = {o: sun_position((utc[[i]] + pd.Timedelta(hours=o)).dt.tz_convert("Asia/Seoul"),
                               d.at[i, "lat"], d.at[i, "lon"]).altitude.iloc[0] for o in (0, -9, 9)}
        # 낮 사진이면 해가 떠 있는 후보, 밤 사진이면 진 후보 중 앞 순서(0 → −9 → +9). 애매한 밝기(흐림 · 해 질 녘)는 쓰지 않음
        ok = [o for o in (0, -9, 9) if (alt[o] > 0 if b >= BRIGHT else alt[o] <= 0)] if (b >= BRIGHT or b <= DARK) else []
        offset = ok[0] if ok else None
        rows.append({"sequence": seq, "photos": len(g), "sample_id": d.at[i, "id"], "brightness": round(b, 1),
                     "alt_as_recorded": round(alt[0], 1), "alt_minus9h": round(alt[-9], 1), "alt_plus9h": round(alt[9], 1),
                     "offset_h": offset})
        print(f"  {seq[:10]}… 사진 {len(g):4d} 밝기 {b:5.1f} 해높이 기록 {alt[0]:6.1f}° / -9h {alt[-9]:6.1f}° / +9h {alt[9]:6.1f}° → {offset}", flush=True)
    ck = pd.DataFrame(rows)
    ck.to_csv(OUT / "clock_check.csv", index=False, encoding="utf-8-sig")

    d = d.merge(ck[["sequence", "offset_h"]], on="sequence", how="left")
    ok = d.offset_h.notna()
    t = (utc + pd.to_timedelta(d.offset_h.fillna(0), unit="h")).dt.tz_convert("Asia/Seoul")   # offset_h 는 부호 그대로 (−9 · +9)
    pos = sun_position(t, d.lat, d.lon)
    d["captured_at_kst"] = t.dt.strftime("%Y-%m-%d %H:%M:%S")
    d["sun_alt"], d["sun_az"] = pos.altitude.round(2).to_numpy(), pos.azimuth.round(2).to_numpy()
    d["gap"] = heading_gap(d.sun_az, d.heading).round(1)
    d["front_low"] = is_front_low(d.sun_alt, d.gap) & ok
    d["clock_ok"] = ok
    d.sort_values(["front_low", "sun_alt"], ascending=[False, True]).to_csv(OUT / "candidates.csv", index=False, encoding="utf-8-sig")

    day = d[ok & (d.sun_alt > 0)]
    print(f"\n시각 확인된 묶음 {int(ck.offset_h.notna().sum())}/{len(ck)} (9시간 보정 {int((ck.offset_h == -9).sum())}개) · 사진 {int(ok.sum()):,}장")
    print(f"  낮 {len(day):,}장 · 해 높이 0~25° {int((day.sun_alt <= 25).sum()):,}장 · 정면 저각도 {int(d.front_low.sum()):,}장 "
          f"(묶음 {d[d.front_low].sequence.nunique()}개)")
    print(f"  Δ(해–카메라 방향) 분포 (해 높이 0~25°): {np.histogram(day[day.sun_alt <= 25].gap, bins=[0, 25, 45, 90, 135, 180])[0].tolist()}  (0~25 · 25~45 · 45~90 · 90~135 · 135~180°)")


if __name__ == "__main__":
    main()
