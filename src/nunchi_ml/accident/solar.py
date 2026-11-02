"""사고 일시(한국 시간)와 위경도 좌표를 기반으로
pvlib을 사용하여 태양 고도(Elevation)와 방위각(Azimuth)을 계산하고
주행 방향과의 역광 각도차를 도출하는 모듈.
"""
import numpy as np
import pandas as pd
import pvlib


def compute_solar_positions(df):
    """사고일자, 사고시간, X좌표(위도), Y좌표(경도)를 받아
    태양 고도 및 방위각을 계산.
    """
    # 1. datetime 파싱 (한국 표준시 KST)
    times_str = df["사고일자"].astype(str) + " " + df["사고시간"].astype(str)
    # 초 단위가 없는 경우 대비
    times_dt = pd.to_datetime(times_str, errors="coerce")
    times_kst = times_dt.dt.tz_localize("Asia/Seoul", ambiguous="NaT", nonexistent="shift_forward")

    lats = pd.to_numeric(df["X좌표"], errors="coerce").values
    lons = pd.to_numeric(df["Y좌표"], errors="coerce").values

    # pvlib은 배치 시계열 또는 단일 좌표에 특화되어 있으므로, N개 행에 대해 get_solarposition 벡터화 처리
    solar_df = pvlib.solarposition.get_solarposition(
        time=times_kst,
        latitude=lats,
        longitude=lons
    )

    solar_elevation = solar_df["apparent_elevation"].values
    solar_azimuth = solar_df["azimuth"].values

    return solar_elevation, solar_azimuth


def compute_relative_angle(travel_bearing, solar_azimuth):
    """주행 방향과 태양 방위각 간의 최소 각도차(0~180도) 계산.
    0도: 완전 정면 역광
    180도: 등 뒤 순광
    """
    diff = np.abs(travel_bearing - solar_azimuth) % 360
    rel_angle = np.minimum(diff, 360 - diff)
    return rel_angle


def classify_glare_condition(solar_elevation, rel_angle, elev_range=(0, 25), angle_thresh=25):
    """노션 공통 기준에 따른 눈부심 조건 판정:
    - 태양 고도 0~25도
    - 주행 방향과 방위각 차이 ±25도 이내
    """
    is_low_sun = (solar_elevation >= elev_range[0]) & (solar_elevation <= elev_range[1])
    is_front_sun = rel_angle <= angle_thresh
    is_back_sun = rel_angle >= (180 - angle_thresh)

    glare_status = np.full(len(solar_elevation), "해당없음", dtype=object)
    glare_status[is_low_sun & is_front_sun] = "해_정면"
    glare_status[is_low_sun & is_back_sun] = "해_등뒤"

    return glare_status
