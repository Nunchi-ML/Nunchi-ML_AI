"""고속도로 사고 데이터와 도로중심선 이정좌표를 결합하여
차량 주행 진행 방위각(Road Bearing)을 계산하는 모듈.
"""
import numpy as np
import pandas as pd

# 사고노선명 -> 도로중심선 도로명 매핑 사전
ROUTE_MAP = {
    "논산천안선,호남선": "논산천안선/호남선",
    "대구포항선": "새만금포항선(대구~포항)",
    "통영대전선,중부선": "통영대전선/중부선",
}


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """두 위경도 좌표 간의 구면 대원 거리(km) 계산."""
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def calculate_initial_compass_bearing(point_a, point_b):
    """두 위경도 좌표(lat, lon) 사이의 초기 방위각(0~360도) 계산.
    point_a: (lat1, lon1)
    point_b: (lat2, lon2)
    """
    lat1 = np.radians(point_a[0])
    lat2 = np.radians(point_b[0])
    diff_long = np.radians(point_b[1] - point_a[1])

    x = np.sin(diff_long) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(diff_long))
    initial_bearing = np.arctan2(x, y)

    # 0~360도로 정규화
    initial_bearing = np.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing


def get_route_direction_map(df_line, df_acc):
    """각 노선별 '방향' 텍스트(예: 서울, 부산)가 이정 증가 방향(상행/종점방향)인지
    이정 감소 방향(하행/기점방향)인지 판별하는 딕셔너리 생성.
    거리 계산에는 하버사인 구면 거리를 사용하여 위경도 제곱오차 왜곡을 방지.
    """
    direction_is_increasing = {}
    for acc_route, sub in df_acc.groupby("사고노선"):
        line_route = ROUTE_MAP.get(acc_route, acc_route)
        sub_line = df_line[df_line["도로명"] == line_route].sort_values("이정")
        if len(sub_line) < 2:
            continue

        p_start = sub_line.iloc[0]
        p_end = sub_line.iloc[-1]
        start_lat, start_lon = p_start["X좌표값"], p_start["Y좌표값"]
        end_lat, end_lon = p_end["X좌표값"], p_end["Y좌표값"]

        # 해당 노선의 주요 방향 레이블에 대해 평균 사고 좌표의 위치 확인
        for d, d_sub in sub.groupby("방향"):
            if len(d_sub) == 0:
                continue
            mean_lat = d_sub["X좌표"].mean()
            mean_lon = d_sub["Y좌표"].mean()

            # 하버사인 구면 대원 거리로 기점·종점 근접도 계산
            dist_to_start = haversine_distance_km(mean_lat, mean_lon, start_lat, start_lon)
            dist_to_end = haversine_distance_km(mean_lat, mean_lon, end_lat, end_lon)

            # 종점(end)에 더 가까운 방향이면 이정 증가 방향(True)
            direction_is_increasing[(acc_route, d)] = bool(dist_to_end < dist_to_start)

    return direction_is_increasing


def compute_road_bearing(df_acc, df_line):
    """사고 데이터의 각 행마다 도로중심선 이정좌표를 참조하여
    주행 진행 방위각(0~360도)을 산출.
    매핑 실패 시 기본값 True 대신 np.nan을 반환하여 잠재적 왜곡 방지.
    """
    dir_map = get_route_direction_map(df_line, df_acc)
    bearings = []
    unmapped_count = 0

    # 노선별 중심선 데이터 인덱싱
    line_by_route = {}
    for r, group in df_line.groupby("도로명"):
        line_by_route[r] = group.sort_values("이정").dropna(subset=["이정", "X좌표값", "Y좌표값"])

    for idx, row in df_acc.iterrows():
        acc_route = row["사고노선"]
        acc_mp = row["사고발생이정"]
        acc_dir = row["방향"]
        line_route = ROUTE_MAP.get(acc_route, acc_route)

        sub_line = line_by_route.get(line_route)
        if sub_line is None or len(sub_line) < 2:
            bearings.append(np.nan)
            continue

        # 방향 매핑 확인: 미확인 시 nan 처리
        is_inc = dir_map.get((acc_route, acc_dir), None)
        if is_inc is None:
            unmapped_count += 1
            bearings.append(np.nan)
            continue

        # 가장 가까운 이정 지점 탐색
        diffs = np.abs(sub_line["이정"] - acc_mp)
        closest_idx = diffs.idxmin()
        pos = sub_line.index.get_loc(closest_idx)

        # 전후 지점을 사용해 이정 증가 방향(기점 -> 종점) 벡터 산출
        if pos == 0:
            p1 = sub_line.iloc[0]
            p2 = sub_line.iloc[1]
        elif pos == len(sub_line) - 1:
            p1 = sub_line.iloc[-2]
            p2 = sub_line.iloc[-1]
        else:
            p1 = sub_line.iloc[pos - 1]
            p2 = sub_line.iloc[pos + 1]

        base_bearing = calculate_initial_compass_bearing(
            (p1["X좌표값"], p1["Y좌표값"]),
            (p2["X좌표값"], p2["Y좌표값"])
        )

        if is_inc:
            travel_bearing = base_bearing
        else:
            travel_bearing = (base_bearing + 180) % 360

        bearings.append(travel_bearing)

    if unmapped_count > 0:
        print(f"  [주의] 방향 매핑 실패 사고: {unmapped_count}건 ({unmapped_count / len(df_acc) * 100:.2f}%)")
    else:
        print(f"  [방향 판정 완료] 방향 매핑 성공률 100.0% (총 {len(df_acc)}건 중 실패 0건)")

    return np.array(bearings)
