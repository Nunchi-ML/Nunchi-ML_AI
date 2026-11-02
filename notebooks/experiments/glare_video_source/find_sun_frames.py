"""도로를 따라 찍힌 Mapillary 사진(촬영 시각 · GPS · 카메라 방향)에서 '해가 정면에 낮게 있던' 장면 찾기.

쓰임: 모델 ② 학습 사진을 한국 도로로 늘리기 + 모델 ③ 눈부심 점수 G가 맞는지 확인

실행 (AI 폴더에서, .env 에 MAPILLARY_TOKEN — https://www.mapillary.com/dashboard/developers 에서 무료 발급):
    python notebooks/experiments/glare_video_source/find_sun_frames.py --route ../FE/src/data/dalgu.json
결과:
    data/glare_mapillary/candidates.csv   사진마다 촬영 시각 · 위치 · 방향 · 해 높이 · 방향 차이 · 정면 저각도 여부
사진 저작권: Mapillary 이용자 기여 (CC BY-SA 4.0) — 쓰면 출처 표기
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from nunchi_ml.sun import heading_gap, is_front_low, sun_position

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "glare_mapillary"
API = "https://graph.mapillary.com/images"
FIELDS = "id,captured_at,compass_angle,computed_compass_angle,geometry,computed_geometry,camera_type,sequence,make,model"
TILE = 0.009                     # Mapillary bbox 검색은 0.01° 보다 작아야 함


def token():
    if os.getenv("MAPILLARY_TOKEN"):
        return os.environ["MAPILLARY_TOKEN"]
    env = ROOT / ".env"
    for line in env.read_text(encoding="utf-8").splitlines() if env.exists() else []:
        if line.startswith("MAPILLARY_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("MAPILLARY_TOKEN 이 없어요 — AI/.env 에 MAPILLARY_TOKEN=... 한 줄 (git 에 안 올라감)")


def load_route(path):
    """FE dalgu.json({"poly": [[lon, lat], ...]}) 또는 [[lon, lat], ...] 목록"""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.asarray(d["poly"] if isinstance(d, dict) else d, float)


def tiles_along(route, buffer_deg=0.002):
    """도로선이 지나는 0.009° 칸만 (도로에서 먼 칸은 안 물음)"""
    keys = set()
    for (x0, y0), (x1, y1) in zip(route, route[1:]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / (TILE / 3)))
        for t in np.linspace(0, 1, n + 1):
            x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            for dx in (-buffer_deg, 0, buffer_deg):
                for dy in (-buffer_deg, 0, buffer_deg):
                    keys.add((math.floor((x + dx) / TILE), math.floor((y + dy) / TILE)))
    return sorted((i * TILE, j * TILE, (i + 1) * TILE, (j + 1) * TILE) for i, j in keys)


def search(box, tok, cache, depth=0):
    """칸 하나의 사진 목록. 2,000장(한 번 최대)에 닿으면 4칸으로 나눠 다시"""
    key = ",".join(f"{v:.5f}" for v in box)
    if key in cache:
        return cache[key]
    for attempt in range(4):
        try:
            r = requests.get(API, params={"access_token": tok, "bbox": key, "fields": FIELDS, "limit": 2000}, timeout=60)
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1)); continue
            r.raise_for_status(); data = r.json().get("data", []); break
        except requests.RequestException as e:
            print("  재시도:", e); time.sleep(3 * (attempt + 1))
    else:
        raise SystemExit("Mapillary 응답 없음")
    if len(data) >= 2000 and depth < 3:
        x0, y0, x1, y1 = box; mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        data = [d for b in [(x0, y0, mx, my), (mx, y0, x1, my), (x0, my, mx, y1), (mx, my, x1, y1)] for d in search(b, tok, cache, depth + 1)]
    cache[key] = data
    return data


def dist_to_route_m(lon, lat, route):
    """사진 위치 → 도로선까지 거리(m), 가까운 근사"""
    k = 111320 * math.cos(math.radians(route[:, 1].mean()))
    px, py = (np.asarray(lon) - route[0, 0]) * k, (np.asarray(lat) - route[0, 1]) * 110940
    rx, ry = (route[:, 0] - route[0, 0]) * k, (route[:, 1] - route[0, 1]) * 110940
    best = np.full(len(px), np.inf)
    for (ax, ay), (bx, by) in zip(zip(rx, ry), zip(rx[1:], ry[1:])):
        dx, dy = bx - ax, by - ay; L = dx * dx + dy * dy or 1e-9
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / L, 0, 1)
        best = np.minimum(best, np.hypot(px - ax - dx * t, py - ay - dy * t))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--route", required=True, help="도로선 JSON (FE src/data/dalgu.json)")
    ap.add_argument("--max-dist", type=float, default=40, help="도로선에서 이 거리(m) 안 사진만")
    a = ap.parse_args()

    tok, route = token(), load_route(a.route)
    OUT.mkdir(parents=True, exist_ok=True)
    cache_file = OUT / ".search_cache.json"
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    boxes = tiles_along(route)
    rows = {}
    for i, b in enumerate(boxes, 1):
        for d in search(b, tok, cache):
            rows[d["id"]] = d
        if i % 20 == 0 or i == len(boxes):
            cache_file.write_text(json.dumps(cache), encoding="utf-8")
            print(f"  칸 {i}/{len(boxes)} · 사진 {len(rows):,}장", flush=True)

    recs = []
    for d in rows.values():
        g = (d.get("computed_geometry") or d.get("geometry") or {}).get("coordinates")
        head = d.get("computed_compass_angle", d.get("compass_angle"))
        if not g or head is None or not d.get("captured_at") or d.get("camera_type") in ("spherical", "equirectangular"):
            continue
        recs.append({"id": d["id"], "sequence": d.get("sequence"), "captured_at_ms": d["captured_at"],
                     "lon": g[0], "lat": g[1], "heading": float(head), "make": d.get("make"), "model": d.get("model")})
    df = pd.DataFrame(recs)
    if df.empty:
        raise SystemExit("조건에 맞는 사진이 없어요")
    df["dist_m"] = dist_to_route_m(df.lon, df.lat, route)
    df = df[df.dist_m <= a.max_dist].copy()
    t = pd.to_datetime(df.captured_at_ms, unit="ms", utc=True).dt.tz_convert("Asia/Seoul")
    df["captured_at_kst"] = t.dt.strftime("%Y-%m-%d %H:%M:%S")
    pos = sun_position(t, df.lat, df.lon)
    df["sun_alt"], df["sun_az"] = pos.altitude.round(2).to_numpy(), pos.azimuth.round(2).to_numpy()
    df["gap"] = heading_gap(df.sun_az, df.heading).round(1)
    df["front_low"] = is_front_low(df.sun_alt, df.gap)
    df.sort_values(["front_low", "sun_alt"], ascending=[False, True]).to_csv(OUT / "candidates.csv", index=False, encoding="utf-8-sig")

    day = df[df.sun_alt > 0]
    print(f"\n도로 위 사진 {len(df):,}장 (연속 촬영 묶음 {df.sequence.nunique():,}개) · 낮 {len(day):,}장")
    print(f"  해 높이 0~25°: {int(((day.sun_alt <= 25)).sum()):,}장 · 그중 정면(±25°): {int(df.front_low.sum()):,}장 "
          f"(묶음 {df[df.front_low].sequence.nunique():,}개)")
    print(f"  촬영 연도: {t.dt.year.value_counts().sort_index().to_dict()}")
    print(f"→ {OUT / 'candidates.csv'}")


if __name__ == "__main__":
    main()
