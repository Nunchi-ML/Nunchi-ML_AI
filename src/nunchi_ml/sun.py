"""해 위치와 '해가 정면에 낮게 있나' 판정 — 모델 ① · ② · ③이 같이 쓰는 기준.

정면 저각도(눈부심 조건): 해 높이 0~25° 이고, 해 방위와 진행 방향 차이 Δ가 ±25° 이내
(노션 데이터 명세서 「공통 · 태양 위치 계산」과 같은 기준. 민감도 ±20° · ±30°)

    from nunchi_ml.sun import sun_position, heading_gap, is_front_low
    pos = sun_position(["2026-11-02 07:30"], [35.857], [128.63])   # 한국시간
    gap = heading_gap(pos["azimuth"], 100)                          # 진행 방위 100° (동남동)
"""
import numpy as np
import pandas as pd
import pvlib

KST = "Asia/Seoul"
ALT_MAX = 25.0                    # 해 높이 상한 (°)
GAP_MAX = 25.0                    # 해 방위 – 진행 방향 차이 상한 (°)


def sun_position(times, lat, lon, tz=KST):
    """시각 · 위도 · 경도 → 해 높이(altitude) · 방위(azimuth, 북=0 시계방향), 단위 °.

    times: 시간대 없는 값은 tz(기본 한국시간)로 봄. 같은 길이의 lat · lon 배열.
    """
    t = pd.DatetimeIndex(pd.to_datetime(times))
    t = t.tz_localize(tz) if t.tz is None else t
    lat, lon = np.broadcast_to(np.asarray(lat, float), len(t)), np.broadcast_to(np.asarray(lon, float), len(t))
    alt, az = np.empty(len(t)), np.empty(len(t))
    # pvlib 은 위치 하나씩 계산 → 같은 위치끼리 묶어 한 번에 (사진 수천 장도 빠르게)
    key = pd.Series(range(len(t))).groupby([np.round(lat, 4), np.round(lon, 4)]).groups
    for (la, lo), idx in key.items():
        idx = np.asarray(idx)
        sp = pvlib.solarposition.get_solarposition(t[idx], la, lo)
        alt[idx], az[idx] = sp["apparent_elevation"].to_numpy(), sp["azimuth"].to_numpy()
    return pd.DataFrame({"altitude": alt, "azimuth": az})


def heading_gap(sun_azimuth, heading):
    """해 방위와 진행(카메라) 방향의 차이 Δ, 0~180°. 0이면 해가 정면."""
    d = (np.asarray(sun_azimuth, float) - np.asarray(heading, float) + 180) % 360 - 180
    return np.abs(d)


def is_front_low(altitude, gap, alt_max=ALT_MAX, gap_max=GAP_MAX):
    """정면 저각도 눈부심 조건 (해가 떠 있고 · 낮고 · 정면)."""
    altitude, gap = np.asarray(altitude, float), np.asarray(gap, float)
    return (altitude > 0) & (altitude <= alt_max) & (gap <= gap_max)
