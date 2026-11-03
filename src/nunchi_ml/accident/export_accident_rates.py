"""고속도로 5,782개 1km 구간에 대해 평소 사고율 및 모델 ① 정면 역광 위험도 예측치/위험 등급을 산출하여 최재혁 님(모델 ③ 터널 순응) 연동용
'구간 평소 사고율 표(segment_accident_rates_v1.csv)'를 생성하는 스크립트.
"""

from pathlib import Path
import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(r"C:\Users\campus4D052\Desktop\프로젝트\Nunchi-ML_AI")
sys.path.insert(0, str(BASE_DIR / "src"))

from nunchi_ml.accident.predict import AccidentRiskPredictor

DATA_PATH = BASE_DIR / "data" / "expressway" / "segment_1km_dataset_v1.csv"
MODEL_PATH = BASE_DIR / "models" / "model1-v1.joblib"
OUT_PATH = BASE_DIR / "data" / "expressway" / "segment_accident_rates_v1.csv"

print(f"데이터셋 로드: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"구간 수: {len(df):,}개 | 노선 수: {df['route'].nunique()}개")

print(f"모델 ① 로드: {MODEL_PATH}")
predictor = AccidentRiskPredictor(model_path=str(MODEL_PATH))

# 11개 기본 피처
INPUT_COLS = [
    "segment_length_km",
    "is_partial_segment",
    "curvature_mean",
    "curvature_max",
    "min_tunnel_dist_km",
    "has_tunnel_within_1km",
    "tunnel_portals_1km",
    "max_tunnel_length_1km",
    "ew_alignment",
    "annual_glare_hours_fwd",
    "commute_glare_hours",
]

print("전체 5,782개 구간 예측 수행 중...")
batch_res = predictor.predict_batch(df[INPUT_COLS])

# 결과 데이터프레임 구성
res = pd.DataFrame()
res["route"] = df["route"]
res["segment_km"] = df["segment_km"]
res["start_mp"] = df["start_mp"]
res["end_mp"] = df["end_mp"]
res["lat"] = df["lat"].round(6)
res["lon"] = df["lon"].round(6)
res["bearing_fwd"] = df["bearing_fwd"].round(2)
res["segment_length_km"] = df["segment_length_km"].round(3)

# 평소 사고율 (3개년 전체사고: 2023.01.01 ~ 2025.12.31, 3년 = 1,096일)
res["accident_count_3yr"] = df["accident_count"]
res["annual_crash_rate"] = (df["accident_count"] / 3.0).round(4)
res["daily_crash_rate"] = (df["accident_count"] / 1096.0).round(6)

# 정면 저각도 태양 역광 사고 실측 건수
res["glare_front_count_3yr"] = df["glare_front_count"]
res["glare_front_ratio"] = (
    df["glare_front_count"] / df["accident_count"].replace(0, 1)
).round(4)

# 주요 도로 기하 / 환경 피처
res["ew_alignment"] = df["ew_alignment"].round(4)
res["annual_glare_hours_fwd"] = df["annual_glare_hours_fwd"].round(2)
res["commute_glare_hours"] = df["commute_glare_hours"].round(2)
res["min_tunnel_dist_km"] = df["min_tunnel_dist_km"].round(3)
res["has_tunnel_within_1km"] = df["has_tunnel_within_1km"]

# 모델 ① 예측 위험도 및 위험 등급
res["predicted_glare_risk"] = batch_res["pred_accident_risk"]
res["risk_grade"] = batch_res["risk_grade"]

# 저장
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
res.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

file_size_kb = os.path.getsize(OUT_PATH) / 1024
print(f"✅ 구간 평소 사고율 표 저장 완료: {OUT_PATH} ({file_size_kb:.1f} KB)")
print("\n[위험 등급별 분포]")
print(res["risk_grade"].value_counts())
print("\n[경부선 샘플 5개 미리보기]")
gyeongbu = res[res["route"].str.contains("경부", na=False)]
print(
    gyeongbu[
        [
            "route",
            "segment_km",
            "accident_count_3yr",
            "annual_crash_rate",
            "glare_front_count_3yr",
            "predicted_glare_risk",
            "risk_grade",
        ]
    ].head(5)
)
