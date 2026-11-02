"""사고 발생 시점의 특성(심각도 및 주시태만)과 역광(태양 정면) 조건 간의
관계 및 통계적/분류 베이스라인을 검증하는 스크립트.
사후 변수 누수를 원천 차단한 이진 분류 모델 비교:
  - 타깃 A: 중대 사고 여부 (사망 >= 1 또는 사상자 >= 2) [PR-AUC 평가]
  - 타깃 B: 주시태만 사고 여부 [ROC-AUC, PR-AUC 평가]

실행:
    python -m src.accident.evaluate_baseline
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
INPUT_PATH = os.path.join(DATA_DIR, "expressway", "accident_with_solar_features.csv")


def run_experiment(df, target_col, feature_cols, group_col, model_fn):
    y = df[target_col].values
    groups = df[group_col].values
    preds = np.zeros(len(df))

    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(df, y, groups):
        X_tr = df.iloc[tr][feature_cols]
        y_tr = y[tr]
        X_te = df.iloc[te][feature_cols]

        m = model_fn()
        m.fit(X_tr, y_tr)
        preds[te] = m.predict_proba(X_te)[:, 1]

    roc = roc_auc_score(y, preds)
    pr = average_precision_score(y, preds)
    return roc, pr


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"로드 완료: 총 {len(df)}건")

    # 1. 타깃 생성
    # 타깃 A: 중대 사고 (사망 1명 이상 또는 사상자 2명 이상)
    df["target_severe"] = ((df["사망"] >= 1) | ((df["사망"] + df["부상"]) >= 2)).astype(int)
    # 타깃 B: 주시태만 사고
    df["target_negligence"] = (df["사고원인"] == "주시태만").astype(int)

    # 2. 전처리 및 특징 가공
    df["dt"] = pd.to_datetime(df["사고일자"] + " " + df["사고시간"], errors="coerce")
    df["month"] = df["dt"].dt.month
    df["dayofweek"] = df["dt"].dt.dayofweek
    df["hour"] = df["dt"].dt.hour + df["dt"].dt.minute / 60.0

    # 시간 주기성 인코딩
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    # 범주형 인코딩 (사고노선, 날씨)
    df["route_code"] = df["사고노선"].astype("category").cat.codes
    df["weather_code"] = df["날씨"].astype("category").cat.codes

    # 3. 입력 피처 셋 정의 (누수 엄격 차단!)
    # 기본 환경 피처
    base_features = ["사고발생이정", "month", "dayofweek", "hour_sin", "hour_cos", "route_code", "weather_code"]
    # 태양/역광 결합 피처
    solar_features = base_features + ["solar_elevation", "relative_sun_angle"]

    print("\n" + "="*60)
    print("🎯 [타깃 A: 중대 사고 여부 (양성 642건, 14.4%)] - 노선 단위 5-Fold GroupKFold")
    print("="*60)
    print("※ 사고원인은 누수이므로 입력 피처에서 완전 제외됨")
    
    # Random Forest
    rf_base_roc, rf_base_pr = run_experiment(df, "target_severe", base_features, "사고노선",
                                             lambda: RandomForestClassifier(n_estimators=100, max_depth=6, random_state=0))
    rf_sol_roc, rf_sol_pr = run_experiment(df, "target_severe", solar_features, "사고노선",
                                           lambda: RandomForestClassifier(n_estimators=100, max_depth=6, random_state=0))

    # LightGBM
    lgb_base_roc, lgb_base_pr = run_experiment(df, "target_severe", base_features, "사고노선",
                                               lambda: LGBMClassifier(n_estimators=100, num_leaves=15, random_state=0, verbose=-1))
    lgb_sol_roc, lgb_sol_pr = run_experiment(df, "target_severe", solar_features, "사고노선",
                                             lambda: LGBMClassifier(n_estimators=100, num_leaves=15, random_state=0, verbose=-1))

    no_skill_pr_a = df["target_severe"].mean()
    print(f"기준선 (No-Skill PR-AUC): {no_skill_pr_a:.3f}")
    print(f"  [RandomForest] 기본 피처 -> ROC-AUC: {rf_base_roc:.3f} | PR-AUC: {rf_base_pr:.3f}")
    print(f"  [RandomForest] +태양각도 -> ROC-AUC: {rf_sol_roc:.3f} | PR-AUC: {rf_sol_pr:.3f}")
    print(f"  [LightGBM]     기본 피처 -> ROC-AUC: {lgb_base_roc:.3f} | PR-AUC: {lgb_base_pr:.3f}")
    print(f"  [LightGBM]     +태양각도 -> ROC-AUC: {lgb_sol_roc:.3f} | PR-AUC: {lgb_sol_pr:.3f}")

    print("\n" + "="*60)
    print("🎯 [타깃 B: 주시태만 사고 여부 (양성 2,046건, 45.9%)] - 노선 단위 5-Fold GroupKFold")
    print("="*60)
    print("※ 사망/부상 건수는 누수이므로 입력 피처에서 완전 제외됨")

    lgb_b_base_roc, lgb_b_base_pr = run_experiment(df, "target_negligence", base_features, "사고노선",
                                                   lambda: LGBMClassifier(n_estimators=100, num_leaves=15, random_state=0, verbose=-1))
    lgb_b_sol_roc, lgb_b_sol_pr = run_experiment(df, "target_negligence", solar_features, "사고노선",
                                                 lambda: LGBMClassifier(n_estimators=100, num_leaves=15, random_state=0, verbose=-1))

    print(f"기준선 (No-Skill PR-AUC): {df['target_negligence'].mean():.3f}")
    print(f"  [LightGBM] 기본 피처 -> ROC-AUC: {lgb_b_base_roc:.3f} | PR-AUC: {lgb_b_base_pr:.3f}")
    print(f"  [LightGBM] +태양각도 -> ROC-AUC: {lgb_b_sol_roc:.3f} | PR-AUC: {lgb_b_sol_pr:.3f}")


if __name__ == "__main__":
    main()
