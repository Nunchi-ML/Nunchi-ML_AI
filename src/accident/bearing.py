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
    """
    direction_is_increasing = {}
    for acc_route, sub in df_acc.groupby("사고노선"):
        line_route = ROUTE_MAP.get(acc_route, acc_route)
        sub_line = df_line[df_line["도로명"] == line_route].sort_values("이정")
        if len(sub_line) < 2:
            continue

        p_start = sub_line.iloc[0]
        p_end = sub_line.iloc[-1]
        start_pt = (p_start["X좌표값"], p_start["Y좌표값"])
        end_pt = (p_end["X좌표값"], p_end["Y좌표값"])

        # 해당 노선의 주요 방향 레이블에 대해 평균 사고 좌표의 위치 확인
        for d, d_sub in sub.groupby("방향"):
            if len(d_sub) == 0:
                continue
            mean_lat = d_sub["X좌표"].mean()
            mean_lon = d_sub["Y좌표"].mean()
            # 종점(end)에 가까운 방향이면, 해당 방향은 이정이 증가하는 방향으로 달리는 차량
            dist_to_start = (mean_lat - start_pt[0])**2 + (mean_lon - start_pt[1])**2
            dist_to_end = (mean_lat - end_pt[0])**2 + (mean_lon - end_pt[1])**2
            direction_is_increasing[(acc_route, d)] = (dist_to_end < dist_to_start)

    return direction_is_increasing


def compute_road_bearing(df_acc, df_line):
    """사고 데이터의 각 행마다 도로중심선 이정좌표를 참조하여
    주행 진행 방위각(0~360도)을 산출.
    """
    dir_map = get_route_direction_map(df_line, df_acc)
    bearings = []

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

        is_inc = dir_map.get((acc_route, acc_dir), True)
        if is_inc:
            travel_bearing = base_bearing
        else:
            travel_bearing = (base_bearing + 180) % 360

        bearings.append(travel_bearing)

    return np.array(bearings)
