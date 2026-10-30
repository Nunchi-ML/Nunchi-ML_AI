"""고속도로 사고 데이터(4,458건)에 도로 방위각과 태양 위치를 결합하여
Q1 연구 질문(해 정면 vs 등 뒤 사고 비교)을 검증하고 결과를 요약하는 스크립트.

실행:
    python -m src.accident.build_features
"""
import os
import numpy as np
import pandas as pd

from src.accident.bearing import compute_road_bearing
from src.accident.solar import compute_solar_positions, compute_relative_angle, classify_glare_condition

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
ACC_PATH = os.path.join(DATA_DIR, "expressway", "한국도로공사_사고이정별좌표및기상_2023-2025.csv")
LINE_PATH = os.path.join(DATA_DIR, "expressway", "한국도로공사_도로중심선_이정좌표.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "expressway", "accident_with_solar_features.csv")


def main():
    print("1. 데이터 로드 중...")
    df_acc = pd.read_csv(ACC_PATH, encoding="cp949")
    df_line = pd.read_csv(LINE_PATH, encoding="cp949")
    print(f"  - 사고 데이터: {len(df_acc)}건")
    print(f"  - 도로중심선 데이터: {len(df_line)}개 지점")

    print("\n2. 도로 주행 진행 방위각 계산 중...")
    road_bearings = compute_road_bearing(df_acc, df_line)
    df_acc["travel_bearing"] = road_bearings
    valid_bearings = (~np.isnan(road_bearings)).sum()
    print(f"  - 방위각 계산 성공률: {valid_bearings} / {len(df_acc)} ({valid_bearings/len(df_acc)*100:.1f}%)")

    print("\n3. pvlib 기반 태양 고도 및 방위각 계산 중...")
    elev, azim = compute_solar_positions(df_acc)
    df_acc["solar_elevation"] = elev
    df_acc["solar_azimuth"] = azim

    print("\n4. 상대 각도차 및 눈부심 조건(정면 vs 등 뒤) 판정 중...")
    rel_angles = compute_relative_angle(df_acc["travel_bearing"].values, df_acc["solar_azimuth"].values)
    df_acc["relative_sun_angle"] = rel_angles

    glare_status = classify_glare_condition(elev, rel_angles, elev_range=(0, 25), angle_thresh=25)
    df_acc["glare_status"] = glare_status

    # 결과 요약 출력
    print("\n" + "="*50)
    print("📊 [Q1 연구 질문] 해 정면 vs 등 뒤 사고 비교 요약")
    print("="*50)
    print(df_acc["glare_status"].value_counts(dropna=False))

    print("\n[맑은 날 기준 (날씨 == '맑음')]")
    clear_acc = df_acc[df_acc["날씨"] == "맑음"]
    print(clear_acc["glare_status"].value_counts(dropna=False))

    # 민감도 분석 (±20도, ±30도)
    print("\n[각도 임계값 민감도 분석]")
    for th in [20, 25, 30]:
        st = classify_glare_condition(elev, rel_angles, elev_range=(0, 25), angle_thresh=th)
        n_front = (st == "해_정면").sum()
        n_back = (st == "해_등뒤").sum()
        print(f"  - ±{th}° 기준: 정면 {n_front}건 (맑음 {(st[df_acc['날씨']=='맑음'] == '해_정면').sum()}건) | "
              f"등뒤 {n_back}건 (맑음 {(st[df_acc['날씨']=='맑음'] == '해_등뒤').sum()}건)")

    # 저장
    df_acc.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n✅ 특성 결합 데이터 저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
