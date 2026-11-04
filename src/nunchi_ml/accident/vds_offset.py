"""VDS Traffic Volume Mapping & Exposure Offset Preparation Pipeline.

Steps:
    1. Define VDS Zone schema (Route, Start MP, End MP, AADT, Direction).
    2. Interval Join: Map each 1km segment [start_mp, end_mp] to VDS zone.
    3. Calculate Exposure E_i = AADT * length_km * 365 * 3 (or sum over years).
    4. Quality check: mapping rate, coverage, unmapped segments, missing statistics.
"""
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "expressway"


def map_vds_to_segments(segments_df: pd.DataFrame, vds_df: pd.DataFrame) -> pd.DataFrame:
    """고속도로 1km 구간 데이터에 VDS 교통량(AADT) 매핑.

    Args:
        segments_df: 5,782개 1km 구간 (route, segment_km, start_mp, end_mp, segment_length_km 등)
        vds_df: VDS 존별 일평균 교통량 (route, start_mp, end_mp, aadt, direction 등)

    Returns:
        pd.DataFrame: aadt_daily, exposure_vkt_3yr, log_exposure_offset 컬럼이 추가된 DataFrame
    """
    df = segments_df.copy()
    mid_mp = (df["start_mp"] + df["end_mp"]) / 2.0
    df["mid_mp"] = mid_mp

    # 노선별 구간 인터벌 매핑
    aadt_list = []
    matched_vds_id = []

    for idx, row in df.iterrows():
        r_name = row["route"]
        mp = row["mid_mp"]

        # 동일 노선 VDS 필터링
        vds_sub = vds_df[vds_df["route"] == r_name]
        match = vds_sub[(vds_sub["start_mp"] <= mp) & (mp <= vds_sub["end_mp"])]

        if len(match) > 0:
            # 매칭된 VDS 존의 평균 AADT 취득
            avg_aadt = float(match["aadt"].mean())
            vds_id = str(match["vds_zone_id"].iloc[0])
        else:
            avg_aadt = np.nan
            vds_id = None

        aadt_list.append(avg_aadt)
        matched_vds_id.append(vds_id)

    df["aadt"] = aadt_list
    df["vds_zone_id"] = matched_vds_id

    # 교통 노출량 E_i = AADT * L_i * 365 * 3 (VKT, Vehicle Kilometers Traveled)
    # 결측치는 노선 평균 또는 중앙값으로 대치하는 정책 적용 가능
    df["exposure_vkt_3yr"] = df["aadt"] * df["segment_length_km"] * 365.0 * 3.0
    df["log_exposure"] = np.log(np.maximum(df["exposure_vkt_3yr"], 1.0))

    return df


def evaluate_mapping_quality(mapped_df: pd.DataFrame) -> dict:
    """VDS 매핑 품질 지표 산출."""
    total = len(mapped_df)
    matched = mapped_df["aadt"].notna().sum()
    match_rate = matched / total * 100.0

    return {
        "total_segments": total,
        "matched_segments": int(matched),
        "unmatched_segments": int(total - matched),
        "mapping_rate_percent": round(match_rate, 2),
        "aadt_min": float(mapped_df["aadt"].min()) if matched > 0 else 0.0,
        "aadt_median": float(mapped_df["aadt"].median()) if matched > 0 else 0.0,
        "aadt_max": float(mapped_df["aadt"].max()) if matched > 0 else 0.0,
    }
