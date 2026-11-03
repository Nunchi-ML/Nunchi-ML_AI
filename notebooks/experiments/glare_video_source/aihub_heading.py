"""AI Hub 「특수환경 자율주행 3D 이미지」 주행(drive)별 진행 방향 복원 — 사진 · 라벨은 쓰지 않음.

라벨 JSON 의 GPS · 시각은 주행마다 하나뿐이지만, 같은 날 주행이 1분 간격으로 이어져 있어
(지역 이름이 seoul1 → seoul2 → seoul3 처럼 바뀌어도 한 경로) '앞 주행 → 뒤 주행' 이동 방향을 진행 방향으로 쓴다.

정리 규칙 (모든 주행에 똑같이):
    1. 같은 날 · 같은 도시(지역 이름 숫자 뺀 것) 주행을 시각 순으로 잇는다
    2. GPS 튐: 앞뒤 주행끼리는 MAX_STEP_M 안인데 자기만 양쪽과 JUMP_M 넘게 떨어지면 GPS 오류로 보고
       앞뒤 주행 가운데 지점으로 바꿈 (예: 서울 drive_1178 · 1189 · 1200 이 약 11분마다 같은 엉뚱한 좌표)
    3. 진행 방향 = 앞 주행 → 뒤 주행 방위 (양쪽 다 있으면 가운데 차분, 끝이면 한쪽). 1분 이동이
       MAX_STEP_M 보다 길면(끊긴 주행) 그쪽은 안 씀. 둘 다 못 쓰면 방향 없음
    4. 해 위치 · Δ 는 nunchi_ml.sun 으로 다시 계산

    python notebooks/experiments/glare_video_source/aihub_heading.py
결과: data/aihub_special/drive_heading.csv (주행별 위치 · 방향 · 해 높이 · Δ · 고친 표시)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from nunchi_ml.sun import heading_gap, sun_position

OUT = Path(__file__).resolve().parents[3] / "data" / "aihub_special"
MAX_STEP_M = 2500                 # 1분에 이보다 멀리 가면 이어진 주행이 아님 (150 km/h)
JUMP_M = 3000


def dist_bearing(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371000 * 2 * np.arcsin(np.sqrt(a)), (np.degrees(np.arctan2(y, x)) + 360) % 360


def chain(g):
    g = g.sort_values("time").copy()
    lat, lon = g.lat.to_numpy(float).copy(), g.lon.to_numpy(float).copy()
    fixed = np.zeros(len(g), bool)
    for i in range(1, len(g) - 1):
        d_nb, _ = dist_bearing(lat[i - 1], lon[i - 1], lat[i + 1], lon[i + 1])
        d_p, _ = dist_bearing(lat[i - 1], lon[i - 1], lat[i], lon[i])
        d_n, _ = dist_bearing(lat[i], lon[i], lat[i + 1], lon[i + 1])
        if d_nb <= 2 * MAX_STEP_M and d_p > JUMP_M and d_n > JUMP_M:
            lat[i], lon[i] = (lat[i - 1] + lat[i + 1]) / 2, (lon[i - 1] + lon[i + 1]) / 2
            fixed[i] = True
    head = np.full(len(g), np.nan)
    for i in range(len(g)):
        p = i - 1 if i > 0 and dist_bearing(lat[i - 1], lon[i - 1], lat[i], lon[i])[0] <= MAX_STEP_M else None
        n = i + 1 if i < len(g) - 1 and dist_bearing(lat[i], lon[i], lat[i + 1], lon[i + 1])[0] <= MAX_STEP_M else None
        a, b = (p if p is not None else i), (n if n is not None else i)
        if a != b:
            head[i] = dist_bearing(lat[a], lon[a], lat[b], lon[b])[1]
    return g.assign(lat=lat, lon=lon, gps_fixed=fixed, heading=head)


def main():
    d = pd.read_csv(OUT / "meta.csv", parse_dates=["time"])
    dr = d.dropna(subset=["time", "lat"]).groupby("drive").agg(
        split=("split", "first"), area=("area", "first"), time=("time", "first"),
        lat=("lat", "first"), lon=("lon", "first"), photos=("file", "size")).reset_index()
    dr["city"] = dr.area.str.replace(r"\d+$", "", regex=True)
    dr = pd.concat([chain(g) for _, g in dr.groupby([dr.city, dr.time.dt.date])], ignore_index=True)
    pos = sun_position(dr.time, dr.lat, dr.lon)
    dr["sun_alt"], dr["sun_az"] = pos.altitude.round(2).to_numpy(), pos.azimuth.round(2).to_numpy()
    dr["gap"] = heading_gap(dr.sun_az, dr.heading).round(1)
    dr.to_csv(OUT / "drive_heading.csv", index=False, encoding="utf-8-sig")
    print(f"주행 {len(dr)}개 · 방향 있음 {int(dr.heading.notna().sum())} · GPS 튐 고침 {int(dr.gps_fixed.sum())}")
    lab = pd.read_csv(OUT / "labels_merged.csv")
    lab["drive_name"] = lab.sequence.str.split("/").str[-1]
    dr["drive_name"] = dr.drive.str.split("/").str[-1]
    v = dr[dr.split.eq("Validation") & dr.area.isin(["seoul2", "daejeon", "changwon"])]
    print(v[["area", "drive_name", "time", "gps_fixed", "heading", "sun_alt", "gap"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
