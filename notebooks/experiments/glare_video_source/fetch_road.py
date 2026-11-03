"""OpenStreetMap 에서 도로 이름으로 (대구 · 서울) 도로선 받기 → find_sun_frames.py --route 에 넣는 JSON.

    python notebooks/experiments/glare_video_source/fetch_road.py 신천대로 --tag sincheon
    python notebooks/experiments/glare_video_source/fetch_road.py 올림픽대로 --tag olympic --region seoul
결과: data/roads/<tag>.json  {"name": ..., "lines": [[[lon, lat], ...], ...]}  (git 제외, © OpenStreetMap 기여자 ODbL)
"""
import argparse
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[3]
OVERPASS = "https://overpass-api.de/api/interpreter"
REGIONS = {"daegu": (35.75, 128.40, 36.05, 128.80),   # 남 · 서 · 북 · 동
           "seoul": (37.42, 126.76, 37.71, 127.19)}


def fetch(name, bbox=REGIONS["daegu"]):
    s, w, n, e = bbox
    q = f'[out:json][timeout:90];way["highway"]["name"="{name}"]({s},{w},{n},{e});out geom;'
    for attempt in range(4):
        try:
            r = requests.post(OVERPASS, data={"data": q}, timeout=120,
                              headers={"User-Agent": "nunchi-ml/0.1 (Nunchi-ML student project)"})
            r.raise_for_status()
            return [[[p["lon"], p["lat"]] for p in el["geometry"]] for el in r.json()["elements"] if el.get("geometry")]
        except requests.RequestException as ex:
            print("  재시도:", ex); time.sleep(5 * (attempt + 1))
    raise SystemExit("Overpass 응답 없음")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="도로 이름 (예: 신천대로)")
    ap.add_argument("--tag", required=True, help="파일 이름 (예: sincheon)")
    ap.add_argument("--region", default="daegu", choices=REGIONS)
    a = ap.parse_args()
    lines = fetch(a.name, REGIONS[a.region])
    if not lines:
        raise SystemExit(f"'{a.name}' 도로선이 없어요")
    out = ROOT / "data" / "roads" / f"{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"name": a.name, "lines": lines}, ensure_ascii=False), encoding="utf-8")
    pts = sum(len(l) for l in lines)
    print(f"{a.name}: 도로선 {len(lines)}개 · 점 {pts:,}개 → {out}")


if __name__ == "__main__":
    main()
