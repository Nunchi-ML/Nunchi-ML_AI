"""모델 ①의 고속도로 5,782개 구간에 대한 최종 위험도 예측 결과를
프론트엔드(웹 시뮬레이터) 연동용 JSON 파일로 추출하는 스크립트.

출력 파일:
    data/expressway_segments_risk.json
"""
import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = str(Path(__file__).resolve().parents[3])
MODEL_PATH = os.path.join(BASE_DIR, "models", "model1-v1.joblib")
SEG_PATH = os.path.join(BASE_DIR, "data", "expressway", "segment_1km_dataset_v1.csv")
OUT_JSON_PATH = os.path.join(BASE_DIR, "data", "expressway_segments_risk.json")


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")
    if not os.path.exists(SEG_PATH):
        raise FileNotFoundError(f"구간 데이터셋이 없습니다: {SEG_PATH}")

    pkg = joblib.load(MODEL_PATH)
    lgb_model = pkg["lgb_model"]
    feature_names = pkg["feature_names"]

    df = pd.read_csv(SEG_PATH)
    if "tunnel_glare_interaction" not in df.columns:
        df["tunnel_glare_interaction"] = df["has_tunnel_within_1km"] * df["commute_glare_hours"]

    # 모델 추론
    preds = np.clip(lgb_model.predict(df[feature_names]), 1e-6, None)
    df["pred_risk"] = np.round(preds, 5)

    # 위험 등급 산정 (분위수 기준)
    df["risk_grade"] = np.where(
        df["pred_risk"] >= 0.035, "DANGER",
        np.where(df["pred_risk"] >= 0.025, "WARNING", "SAFE")
    )

    # 평균 위험 대비 배율 (FE road.ts ratio 대응)
    mean_risk = float(df["pred_risk"].mean())
    df["risk_ratio"] = np.round(df["pred_risk"] / mean_risk, 2)

    # 프론트엔드 경량 JSON 구조로 정제
    records = []
    for _, r in df.iterrows():
        records.append({
            "route": str(r["route"]),
            "segment_km": int(r["segment_km"]),
            "length_km": float(r["segment_length_km"]),
            "curvature_max": float(round(r["curvature_max"], 5)),
            "has_tunnel": int(r["has_tunnel_within_1km"]),
            "ew_alignment": float(round(r["ew_alignment"], 2)),
            "commute_glare_hours": float(round(r["commute_glare_hours"], 1)),
            "pred_risk": float(r["pred_risk"]),
            "risk_ratio": float(r["risk_ratio"]),
            "risk_grade": str(r["risk_grade"]),
        })

    summary = {
        "version": "1.0.0",
        "description": "한국도로공사 고속도로 1km 구간별 정면 태양 눈부심 사고 위험도 (모델 ① 최종)",
        "total_segments": len(records),
        "mean_risk": round(mean_risk, 5),
        "danger_count": int((df["risk_grade"] == "DANGER").sum()),
        "warning_count": int((df["risk_grade"] == "WARNING").sum()),
        "safe_count": int((df["risk_grade"] == "SAFE").sum()),
        "segments": records,
    }

    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    size_kb = os.path.getsize(OUT_JSON_PATH) / 1024
    print("=" * 70)
    print(f"✅ 프론트엔드 연동용 위험도 JSON 파일 생성 완료!")
    print(f"   * 저장 경로: {OUT_JSON_PATH} ({size_kb:.1f} KB)")
    print(f"   * 총 구간 수: {summary['total_segments']:,}개")
    print(f"   * 등급 분포: DANGER(고위험) {summary['danger_count']}개, WARNING(주의) {summary['warning_count']}개, SAFE(안전) {summary['safe_count']}개")
    print("=" * 70)


if __name__ == "__main__":
    main()
