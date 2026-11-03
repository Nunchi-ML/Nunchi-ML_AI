"""AI Hub 「특수환경 자율주행 3D 이미지」 라벨 JSON → 사진별 촬영 정보 표 (원천 사진 없이).

라벨 zip 34개(Training 23 · Validation 11)에서 시각 · 날씨 · GPS 를 뽑고,
같은 주행(drive) 안 GPS 이동 경로로 진행 방향을 구해 해 높이 · Δ 를 붙인다.
→ 어느 지역 원천 zip 을 받을지 고르는 데 씀.

    python aihub_index.py "<라벨 폴더>"            # Training/ · Validation/ 가 있는 폴더
    → AI/data/aihub_special/meta.csv
"""
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from nunchi_ml.sun import sun_position, heading_gap, is_front_low

OUT = Path(__file__).resolve().parents[3] / "data" / "aihub_special" / "meta.csv"
MIN_MOVE_M = 3.0                  # 이보다 덜 움직이면 진행 방향 계산 안 함 (정차 · 서행)
BAD = []                          # 읽지 못한 JSON


def load_info(raw):
    """IMAGE_INFO 만 필요. 일부 파일은 cp949 이거나 끝 쉼표(`, }`) 때문에 깨져 있어 고쳐 읽음."""
    try:
        s = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        s = raw.decode("cp949", errors="replace")
    try:
        return json.loads(s)["IMAGE_INFO"]
    except json.JSONDecodeError:
        try:
            return json.loads(re.sub(r",(\s*[}\]])", r"\1", s))["IMAGE_INFO"]
        except (json.JSONDecodeError, KeyError):
            return None


def read_zip(path, split):
    area = path.stem.replace("[라벨]", "")
    rows = []
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if not n.lower().endswith(".json"):
                continue
            info = load_info(z.read(n))
            if info is None:
                BAD.append(f"{split}/{area}/{n}")
                continue
            loc = (info.get("DATA_COLLECTION_AREA") or [{}])[0]
            parts = n.split("/")
            rows.append({
                "split": split, "area": area,
                "drive": f"{split}/{area}/" + "/".join(parts[:-2]),     # .../drive_xxxx
                "file": parts[-1].removesuffix(".json"),
                "camera": info.get("CAMETA_NAME"),
                "time": info.get("DATA_COLLECTION_TIME"),
                "weather": info.get("WEATHER"),
                "acq_time": info.get("ACQUISITION_TIME"),
                "road_type": info.get("ROAD_TYPE"),
                "lat": loc.get("LATITUDE"),
                "lon": loc.get("HARDNESS"),                             # 원본 키 이름이 HARDNESS (경도)
                "image_path": info.get("IMAGE_PATH"),
            })
    return rows


def bearing_and_dist(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    brg = (np.degrees(np.arctan2(y, x)) + 360) % 360
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return brg, 6371000 * 2 * np.arcsin(np.sqrt(a))


def add_heading(df):
    """같은 drive · 카메라 안에서 다음 프레임 쪽 방위 (없으면 이전 프레임에서)."""
    df = df.sort_values(["drive", "camera", "file"]).reset_index(drop=True)
    g = df.groupby(["drive", "camera"])
    nlat, nlon = g["lat"].shift(-1), g["lon"].shift(-1)
    brg, dist = bearing_and_dist(df["lat"], df["lon"], nlat, nlon)
    plat, plon = g["lat"].shift(1), g["lon"].shift(1)
    pbrg, pdist = bearing_and_dist(plat, plon, df["lat"], df["lon"])
    df["heading"] = np.where(dist >= MIN_MOVE_M, brg, np.where(pdist >= MIN_MOVE_M, pbrg, np.nan))
    df["move_m"] = dist
    return df


def main(root):
    root = Path(root)
    rows = []
    for split in ("Training", "Validation"):
        for p in sorted((root / split).glob("*.zip")):
            rows += read_zip(p, split)
    df = pd.DataFrame(rows)
    df["lat"], df["lon"] = pd.to_numeric(df["lat"], errors="coerce"), pd.to_numeric(df["lon"], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], format="%Y%m%d_%H%M%S", errors="coerce")
    df = add_heading(df)

    ok = df["time"].notna() & df["lat"].notna() & df["lon"].notna()
    pos = sun_position(df.loc[ok, "time"], df.loc[ok, "lat"], df.loc[ok, "lon"])
    df.loc[ok, "sun_alt"], df.loc[ok, "sun_az"] = pos["altitude"].to_numpy(), pos["azimuth"].to_numpy()
    df["gap"] = heading_gap(df["sun_az"], df["heading"])
    df["front_low"] = is_front_low(df["sun_alt"], df["gap"]) & df["heading"].notna()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"{len(df):,} 장 → {OUT}  (읽지 못한 JSON {len(BAD)}개)")
    for b in BAD[:5]:
        print("  ", b)


if __name__ == "__main__":
    main(sys.argv[1])
