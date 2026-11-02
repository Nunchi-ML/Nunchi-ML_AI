"""고속도로 중심선(58,846개 지점)을 1km 구간 단위로 집계하고,
구간별 주행 방위각, 도로 굴곡도(인접 지점 방위각 변화량), 터널 입·출구 인접성,
연간 잠재적 정면 저각도 태양 노출 시간(pvlib 기하학적 계산),
그리고 2023~2025년 사고 건수(타깃)를 결합하여 모델 ① 정식 데이터셋(①-v1)을 생성하는 스크립트.

실행:
    python -m src.accident.build_segments
"""
import os
import sys
import numpy as np
import pandas as pd
import pvlib

from pathlib import Path
from nunchi_ml.accident.bearing import ROUTE_MAP, calculate_initial_compass_bearing

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = str(Path(__file__).resolve().parents[3] / "data")
LINE_PATH = os.path.join(DATA_DIR, "expressway", "한국도로공사_도로중심선_이정좌표.csv")
ACC_PATH = os.path.join(DATA_DIR, "expressway", "accident_with_solar_features.csv")
TUNNEL_PATH = os.path.join(DATA_DIR, "tunnel", "국토교통부_전국도로터널정보표준데이터.csv")
OUTPUT_SEG_PATH = os.path.join(DATA_DIR, "expressway", "segment_1km_dataset_v1.csv")


def compute_annual_solar_grid(year=2024, freq="15min"):
    """대한민국 고속도로 대표 위경도(북위 36.5°, 동경 127.8°) 기준
    1년치 태양 고도·방위각 시계열을 계산하여 저각도(0~25도) 시간대만 반환.
    ※ 전국 대표 좌표 기반 기하학적 근사치 (잠재적 정면 저각도 태양 노출 시간)
    """
    times_kst = pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 23:45:00", freq=freq, tz="Asia/Seoul")
    times_utc = times_kst.tz_convert("UTC")
    solpos = pvlib.solarposition.get_solarposition(times_utc, latitude=36.5, longitude=127.8)
    elev = solpos["apparent_elevation"].values
    azim = solpos["azimuth"].values
    hours = times_kst.hour.values

    mask_low = (elev > 0.0) & (elev <= 25.0)
    return elev[mask_low], azim[mask_low], hours[mask_low], 0.25  # 15분 = 0.25시간


def calc_glare_exposure_hours(bearing_fwd, bearing_bwd, low_azim, low_hours, dt_hours, thresh=25.0):
    """구간의 양방향(순방향/역방향) 방위각에 대해 연간 정면 태양 노출 시간(h)을 벡터 연산으로 산출."""
    diff_fwd = np.abs(bearing_fwd[:, None] - low_azim[None, :]) % 360.0
    diff_fwd = np.where(diff_fwd > 180.0, 360.0 - diff_fwd, diff_fwd)

    diff_bwd = np.abs(bearing_bwd[:, None] - low_azim[None, :]) % 360.0
    diff_bwd = np.where(diff_bwd > 180.0, 360.0 - diff_bwd, diff_bwd)

    in_glare_fwd = diff_fwd <= thresh
    in_glare_bwd = diff_bwd <= thresh
    in_glare_any = in_glare_fwd | in_glare_bwd  # 양방향 합집합(중복 제거)

    total_hours_fwd = in_glare_fwd.sum(axis=1) * dt_hours
    total_hours_bwd = in_glare_bwd.sum(axis=1) * dt_hours
    total_hours_union = in_glare_any.sum(axis=1) * dt_hours
    total_hours_sum = (in_glare_fwd.sum(axis=1) + in_glare_bwd.sum(axis=1)) * dt_hours

    # 출퇴근 2시간 창: 07:00~08:59 (7~8시), 17:00~18:59 (17~18시) 엄밀한 2시간 구간
    commute_mask = ((low_hours >= 7) & (low_hours < 9)) | ((low_hours >= 17) & (low_hours < 19))
    commute_hours = in_glare_any[:, commute_mask].sum(axis=1) * dt_hours

    return total_hours_fwd, total_hours_bwd, total_hours_union, total_hours_sum, commute_hours


def build_segments():
    print("1. 도로중심선 이정좌표 로드 및 1km 구간 집계 중...")
    df_line = pd.read_csv(LINE_PATH, encoding="cp949")
    df_line = df_line.dropna(subset=["도로명", "이정", "X좌표값", "Y좌표값"]).copy()
    df_line["segment_km"] = np.floor(df_line["이정"]).astype(int)
    df_line = df_line.sort_values(["도로명", "이정"])

    records = []
    for (route, seg_km), grp in df_line.groupby(["도로명", "segment_km"], sort=False):
        if len(grp) < 2:
            continue
        lats = grp["X좌표값"].values
        lons = grp["Y좌표값"].values
        mps = grp["이정"].values

        # 실제 이정 차이 기반 구간 길이 산출
        mp_span = float(mps.max() - mps.min())
        # 단일 이정이거나 오차 보정: 100m 점 개수와 비교
        seg_len = round(max(mp_span, len(grp) * 0.1), 3)
        is_partial = int(seg_len < 0.8)

        # 구간 전체 순방향 방위각 (시작점 -> 끝점)
        b_fwd = calculate_initial_compass_bearing((lats[0], lons[0]), (lats[-1], lons[-1]))
        b_bwd = (b_fwd + 180.0) % 360.0

        # 100m 세부 구간별 방위각 변화량 (도로 굴곡도 지표)
        step_bearings = calculate_initial_compass_bearing(
            (lats[:-1], lons[:-1]), (lats[1:], lons[1:])
        )
        if len(step_bearings) > 1:
            diffs = np.abs(np.diff(step_bearings)) % 360.0
            diffs = np.where(diffs > 180.0, 360.0 - diffs, diffs)
            curvature_mean = float(np.mean(diffs))
            curvature_max = float(np.max(diffs))
        else:
            curvature_mean = 0.0
            curvature_max = 0.0

        # 동서 방향성 지수: |sin(방위각)| (1: 동서, 0: 남북)
        ew_alignment = float(np.abs(np.sin(np.radians(b_fwd))))

        records.append({
            "route": route,
            "segment_km": int(seg_km),
            "lat": float(np.mean(lats)),
            "lon": float(np.mean(lons)),
            "n_points": len(grp),
            "segment_length_km": seg_len,
            "is_partial_segment": is_partial,
            "start_mp": float(mps.min()),
            "end_mp": float(mps.max()),
            "bearing_fwd": float(b_fwd),
            "bearing_bwd": float(b_bwd),
            "ew_alignment": ew_alignment,
            "curvature_mean": curvature_mean,
            "curvature_max": curvature_max,
        })

    df_seg = pd.DataFrame(records)
    print(f"  - 생성된 1km 고속도로 구간 수: {len(df_seg):,}개 (노선 {df_seg['route'].nunique()}개)")
    print(f"  - 1km 미만 잔여(마지막) 구간 수: {df_seg['is_partial_segment'].sum()}개")

    print("\n2. 연간 태양 궤적 기반 구간별 잠재적 정면 저각도 태양 노출 시간(pvlib) 계산 중...")
    _, low_azim, low_hours, dt_hours = compute_annual_solar_grid(year=2024, freq="15min")
    fwd_h, bwd_h, union_h, sum_h, commute_h = calc_glare_exposure_hours(
        df_seg["bearing_fwd"].values,
        df_seg["bearing_bwd"].values,
        low_azim,
        low_hours,
        dt_hours,
        thresh=25.0,
    )
    df_seg["annual_glare_hours_fwd"] = fwd_h
    df_seg["annual_glare_hours_bwd"] = bwd_h
    df_seg["annual_glare_hours_total"] = union_h  # 합집합(중복 없는 시간)
    df_seg["annual_glare_hours_sum"] = sum_h
    df_seg["commute_glare_hours"] = commute_h

    # 민감도 분석용 (±20도, ±30도)
    _, _, union_20, _, _ = calc_glare_exposure_hours(
        df_seg["bearing_fwd"].values, df_seg["bearing_bwd"].values, low_azim, low_hours, dt_hours, thresh=20.0
    )
    _, _, union_30, _, _ = calc_glare_exposure_hours(
        df_seg["bearing_fwd"].values, df_seg["bearing_bwd"].values, low_azim, low_hours, dt_hours, thresh=30.0
    )
    df_seg["annual_glare_hours_20deg"] = union_20
    df_seg["annual_glare_hours_30deg"] = union_30

    print("\n3. 고속국도 터널 입·출구 인접성 피처 결합 중...")
    df_t = pd.read_csv(TUNNEL_PATH, encoding="utf-8")
    hw_t = df_t[df_t["도로종류"] == "고속국도"].copy()

    portal_lats = []
    portal_lons = []
    portal_lens = []
    for _, r in hw_t.iterrows():
        t_len = float(r["터널연장"]) if pd.notna(r["터널연장"]) else 0.0
        if pd.notna(r["터널시작점위도"]) and pd.notna(r["터널시작점경도"]):
            portal_lats.append(float(r["터널시작점위도"]))
            portal_lons.append(float(r["터널시작점경도"]))
            portal_lens.append(t_len)
        if pd.notna(r["터널종료점위도"]) and pd.notna(r["터널종료점경도"]):
            portal_lats.append(float(r["터널종료점위도"]))
            portal_lons.append(float(r["터널종료점경도"]))
            portal_lens.append(t_len)

    if len(portal_lats) > 0:
        p_lats = np.array(portal_lats)
        p_lons = np.array(portal_lons)
        p_lens = np.array(portal_lens)

        seg_lats = df_seg["lat"].values[:, None]
        seg_lons = df_seg["lon"].values[:, None]
        dlat_km = (seg_lats - p_lats[None, :]) * 111.0
        dlon_km = (seg_lons - p_lons[None, :]) * 88.8
        dist_matrix = np.sqrt(dlat_km**2 + dlon_km**2)

        min_tunnel_dist = dist_matrix.min(axis=1)
        near_mask = dist_matrix <= 1.0  # 반경 1km 이내 터널 입출구 좌표 존재 여부
        tunnel_portals_count = near_mask.sum(axis=1)
        max_tunnel_len = np.where(near_mask, p_lens[None, :], 0.0).max(axis=1)

        df_seg["min_tunnel_dist_km"] = np.round(min_tunnel_dist, 3)
        df_seg["has_tunnel_within_1km"] = (tunnel_portals_count > 0).astype(int)
        df_seg["tunnel_portals_1km"] = tunnel_portals_count
        df_seg["max_tunnel_length_1km"] = max_tunnel_len
    else:
        df_seg["min_tunnel_dist_km"] = 999.0
        df_seg["has_tunnel_within_1km"] = 0
        df_seg["tunnel_portals_1km"] = 0
        df_seg["max_tunnel_length_1km"] = 0.0

    print("\n4. 고속도로 사고 데이터(4,458건) 구간 매핑 및 타깃 집계 중...")
    df_acc = pd.read_csv(ACC_PATH)
    df_acc["route"] = df_acc["사고노선"].map(lambda x: ROUTE_MAP.get(x, x))
    df_acc["segment_km"] = np.floor(df_acc["사고발생이정"]).astype(int)

    acc_grp = df_acc.groupby(["route", "segment_km"]).agg(
        accident_count=("사고일자", "count"),
        glare_front_count=("glare_status", lambda s: (s == "해_정면").sum()),
        glare_back_count=("glare_status", lambda s: (s == "해_등뒤").sum()),
        clear_weather_count=("날씨", lambda s: (s == "맑음").sum()),
        severe_count=("사망", lambda s: (s >= 1).sum()),
        casualty_sum=("부상", "sum"),
    ).reset_index()

    df_seg = df_seg.merge(acc_grp, on=["route", "segment_km"], how="left")
    fill_zero_cols = [
        "accident_count",
        "glare_front_count",
        "glare_back_count",
        "clear_weather_count",
        "severe_count",
        "casualty_sum",
    ]
    for c in fill_zero_cols:
        df_seg[c] = df_seg[c].fillna(0).astype(int)

    # 저장
    df_seg.to_csv(OUTPUT_SEG_PATH, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 65)
    print(f"구간 데이터셋(①-v1) 생성 완료: {OUTPUT_SEG_PATH}")
    print("=" * 65)
    print(f"총 1km 구간 수: {len(df_seg):,}개")
    print(f"총 매핑된 사고 수: {df_seg['accident_count'].sum():,}건 / 원본 {len(df_acc):,}건")
    zero_ratio = (df_seg["accident_count"] == 0).mean() * 100
    print(f"무사고(0건) 구간 비율: {zero_ratio:.1f}% (평균 {df_seg['accident_count'].mean():.3f}건, 최대 {df_seg['accident_count'].max()}건)")

    q_high = df_seg["commute_glare_hours"].quantile(0.75)
    q_low = df_seg["commute_glare_hours"].quantile(0.25)
    high_grp = df_seg[df_seg["commute_glare_hours"] >= q_high]
    low_grp = df_seg[df_seg["commute_glare_hours"] <= q_low]
    print(f"\n[출퇴근 태양 정면 노출 수준별 사고 건수]")
    print(f"  - 출퇴근 노출 상위 25% 구간 (n={len(high_grp)}): 평균 전체사고 {high_grp['accident_count'].mean():.3f}건, 정면태양 사고 {high_grp['glare_front_count'].mean():.3f}건")
    print(f"  - 출퇴근 노출 하위 25% 구간 (n={len(low_grp)}): 평균 전체사고 {low_grp['accident_count'].mean():.3f}건, 정면태양 사고 {low_grp['glare_front_count'].mean():.3f}건")


if __name__ == "__main__":
    build_segments()
