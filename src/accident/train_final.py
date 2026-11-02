"""고속도로 1km 구간 사고 위험도 예측 모델(모델 ①) 최종 튜닝, 앙상블 비교 및 패키징 스크립트.

주요 작업:
1. 5-Fold Route-based GroupKFold 기반 LightGBM Poisson 하이퍼파라미터 튜닝
2. 통계 모델(Poisson GLM) 및 트리 모델(LightGBM, Random Forest) 간 앙상블(Blending) 비교 검증
3. 최종 선정 모델을 전체 데이터(5,782개 구간)로 학습 후 models/model1-v1.joblib 로 패키징

실행:
    python -m src.accident.train_final
"""
import os
import sys
import itertools
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_poisson_deviance, mean_absolute_error
import lightgbm as lgb

sys.stdout.reconfigure(encoding="utf-8")

SEED = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SEG_PATH = os.path.join(BASE_DIR, "data", "expressway", "segment_1km_dataset_v1.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_MODEL_PATH = os.path.join(MODEL_DIR, "model1-v1.joblib")

BASE_FEATURES = [
    "segment_length_km",
    "is_partial_segment",
    "curvature_mean",
    "curvature_max",
    "min_tunnel_dist_km",
    "has_tunnel_within_1km",
    "tunnel_portals_1km",
    "max_tunnel_length_1km",
]

SOLAR_FEATURES = [
    "ew_alignment",
    "annual_glare_hours_fwd",
    "commute_glare_hours",
    "tunnel_glare_interaction",
]

ALL_FEATURES = BASE_FEATURES + SOLAR_FEATURES


def safe_poisson_deviance(y_true, y_pred):
    pred_clipped = np.clip(y_pred, 1e-6, None)
    return mean_poisson_deviance(y_true, pred_clipped)


def evaluate_oof(y, oof_pred, fold_devs=None):
    pred_clipped = np.clip(oof_pred, 1e-6, None)
    dev = mean_poisson_deviance(y, pred_clipped)
    mae = mean_absolute_error(y, pred_clipped)
    pos_mask = y > 0
    pos_mae = mean_absolute_error(y[pos_mask], pred_clipped[pos_mask]) if pos_mask.sum() > 0 else 0.0

    top10_k = int(len(y) * 0.1)
    top10_idx = np.argsort(pred_clipped)[-top10_k:]
    top10_capture = (y[top10_idx].sum() / max(y.sum(), 1)) * 100.0

    return {
        "deviance": dev,
        "dev_std": np.std(fold_devs) if fold_devs is not None else 0.0,
        "mae": mae,
        "pos_mae": pos_mae,
        "top10_capture": top10_capture,
        "pred": pred_clipped,
    }


def main():
    print("=" * 80)
    print("🚀 [모델 ①] 최종 하이퍼파라미터 튜닝, 앙상블 비교 및 패키징")
    print("=" * 80)

    if not os.path.exists(SEG_PATH):
        raise FileNotFoundError(f"데이터셋을 찾을 수 없습니다: {SEG_PATH}")

    df = pd.read_csv(SEG_PATH)
    df["tunnel_glare_interaction"] = df["has_tunnel_within_1km"] * df["commute_glare_hours"]

    target_col = "glare_front_count"  # 정면 태양 눈부심 조건 사고 모델
    y = df[target_col].values
    groups = df["route"].values

    print(f"데이터셋 로드: {len(df):,}개 구간 | 노선 수: {df['route'].nunique()}개 | 타깃: {target_col} (총 {y.sum():,}건)")
    print(f"사용 피처 ({len(ALL_FEATURES)}개): {ALL_FEATURES}\n")

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(df, y, groups))

    # =========================================================================
    # 1. Baseline GLM 평가
    # =========================================================================
    print("--- [1단계] Poisson GLM 기준선 재검증 ---")
    glm_oof = np.zeros(len(df), dtype=float)
    glm_fold_devs = []

    for fold, (tr_idx, te_idx) in enumerate(splits):
        tr_df, te_df = df.iloc[tr_idx], df.iloc[te_idx]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(tr_df[ALL_FEATURES])
        X_te = scaler.transform(te_df[ALL_FEATURES])
        glm = PoissonRegressor(alpha=1.0, max_iter=500)
        glm.fit(X_tr, tr_df[target_col].values)
        p = np.clip(glm.predict(X_te), 1e-6, None)
        glm_oof[te_idx] = p
        glm_fold_devs.append(safe_poisson_deviance(y[te_idx], p))

    res_glm = evaluate_oof(y, glm_oof, glm_fold_devs)
    print(f"Poisson GLM (Baseline)   | Deviance: {res_glm['deviance']:.4f} (±{res_glm['dev_std']:.4f}) | "
          f"MAE: {res_glm['mae']:.4f} | 상위10% 포착률: {res_glm['top10_capture']:.1f}%")

    # =========================================================================
    # 2. LightGBM 하이퍼파라미터 그리드 튜닝
    # =========================================================================
    print("\n--- [2단계] LightGBM Poisson 5-Fold 그리드 튜닝 탐색 ---")
    param_grid = {
        "learning_rate": [0.02, 0.03, 0.05],
        "num_leaves": [7, 12, 15, 20],
        "max_depth": [3, 4, 5],
        "min_child_samples": [20, 30, 50],
        "reg_alpha": [0.5, 1.0, 2.0],
        "reg_lambda": [1.0, 5.0, 10.0],
        "n_estimators": [60, 80, 100],
    }

    # 탐색 공간 축소: 의미 있는 핵심 조합 24개 엄선
    candidate_combos = [
        # (lr, num_leaves, max_depth, min_child_samples, reg_alpha, reg_lambda, n_est)
        (0.03, 7, 3, 30, 1.0, 5.0, 80),   # 기존 기본 설정
        (0.02, 7, 3, 30, 1.0, 5.0, 100),
        (0.03, 10, 3, 30, 1.0, 5.0, 80),
        (0.03, 12, 4, 30, 1.0, 5.0, 80),
        (0.02, 12, 4, 40, 1.0, 5.0, 100),
        (0.03, 15, 4, 30, 0.5, 3.0, 80),
        (0.03, 15, 4, 50, 1.0, 5.0, 80),
        (0.02, 15, 4, 30, 1.0, 10.0, 100),
        (0.05, 7, 3, 30, 1.0, 5.0, 60),
        (0.03, 7, 3, 20, 0.5, 2.0, 80),
        (0.03, 7, 3, 50, 2.0, 10.0, 80),
        (0.02, 10, 3, 40, 1.0, 5.0, 90),
        (0.025, 8, 3, 30, 1.0, 5.0, 85),
        (0.03, 12, 4, 50, 2.0, 5.0, 75),
    ]

    best_lgb_params = None
    best_lgb_dev = float("inf")
    best_lgb_oof = None
    best_lgb_res = None

    for i, (lr, nl, md, mcs, ra, rl, nest) in enumerate(candidate_combos):
        oof = np.zeros(len(df), dtype=float)
        fold_devs = []

        for fold, (tr_idx, te_idx) in enumerate(splits):
            tr_df, te_df = df.iloc[tr_idx], df.iloc[te_idx]
            model = lgb.LGBMRegressor(
                objective="poisson",
                learning_rate=lr,
                num_leaves=nl,
                max_depth=md,
                min_child_samples=mcs,
                reg_alpha=ra,
                reg_lambda=rl,
                n_estimators=nest,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=SEED,
                verbose=-1,
            )
            model.fit(tr_df[ALL_FEATURES], tr_df[target_col].values)
            p = np.clip(model.predict(te_df[ALL_FEATURES]), 1e-6, None)
            oof[te_idx] = p
            fold_devs.append(safe_poisson_deviance(y[te_idx], p))

        res = evaluate_oof(y, oof, fold_devs)
        if res["deviance"] < best_lgb_dev:
            best_lgb_dev = res["deviance"]
            best_lgb_params = {
                "learning_rate": lr,
                "num_leaves": nl,
                "max_depth": md,
                "min_child_samples": mcs,
                "reg_alpha": ra,
                "reg_lambda": rl,
                "n_estimators": nest,
            }
            best_lgb_oof = oof
            best_lgb_res = res

    print(f"최적 LightGBM 파라미터: {best_lgb_params}")
    print(f"LightGBM (Tuned)         | Deviance: {best_lgb_res['deviance']:.4f} (±{best_lgb_res['dev_std']:.4f}) | "
          f"MAE: {best_lgb_res['mae']:.4f} | 상위10% 포착률: {best_lgb_res['top10_capture']:.1f}%")

    # =========================================================================
    # 3. Random Forest 모델 OOF 산출
    # =========================================================================
    print("\n--- [3단계] Random Forest Regressor 보조 검증 ---")
    rf_oof = np.zeros(len(df), dtype=float)
    rf_fold_devs = []

    for fold, (tr_idx, te_idx) in enumerate(splits):
        tr_df, te_df = df.iloc[tr_idx], df.iloc[te_idx]
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=20, random_state=SEED, n_jobs=-1)
        rf.fit(tr_df[ALL_FEATURES], tr_df[target_col].values)
        p = np.clip(rf.predict(te_df[ALL_FEATURES]), 1e-6, None)
        rf_oof[te_idx] = p
        rf_fold_devs.append(safe_poisson_deviance(y[te_idx], p))

    res_rf = evaluate_oof(y, rf_oof, rf_fold_devs)
    print(f"Random Forest Regressor  | Deviance: {res_rf['deviance']:.4f} (±{res_rf['dev_std']:.4f}) | "
          f"MAE: {res_rf['mae']:.4f} | 상위10% 포착률: {res_rf['top10_capture']:.1f}%")

    # =========================================================================
    # 4. 앙상블 (Blending) 비교 실험
    # =========================================================================
    print("\n--- [4단계] 앙상블(Blending) 비교 검증 ---")
    # 실험 4-1: Tuned LightGBM + Poisson GLM
    print("[앙상블 A: Tuned LightGBM + Poisson GLM Blending]")
    best_blend_dev = float("inf")
    best_blend_w = None
    best_blend_res = None
    best_blend_oof = None

    for w_lgb in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        w_glm = 1.0 - w_lgb
        blend_oof = w_lgb * best_lgb_oof + w_glm * glm_oof
        res = evaluate_oof(y, blend_oof)
        if res["deviance"] < best_blend_dev:
            best_blend_dev = res["deviance"]
            best_blend_w = (w_lgb, w_glm)
            best_blend_res = res
            best_blend_oof = blend_oof
        print(f"  * LGB {w_lgb:.1f} + GLM {w_glm:.1f} | Deviance: {res['deviance']:.4f} | "
              f"MAE: {res['mae']:.4f} | 상위10% 포착률: {res['top10_capture']:.1f}%")

    # 실험 4-2: Tuned LightGBM + Random Forest
    print("\n[앙상블 B: Tuned LightGBM + Random Forest Blending]")
    best_rf_blend_dev = float("inf")
    best_rf_blend_w = None
    best_rf_blend_res = None
    for w_lgb in [0.9, 0.8, 0.7, 0.6, 0.5]:
        w_rf = 1.0 - w_lgb
        b_oof = w_lgb * best_lgb_oof + w_rf * rf_oof
        res = evaluate_oof(y, b_oof)
        if res["deviance"] < best_rf_blend_dev:
            best_rf_blend_dev = res["deviance"]
            best_rf_blend_w = (w_lgb, w_rf)
            best_rf_blend_res = res
        print(f"  * LGB {w_lgb:.1f} + RF {w_rf:.1f}  | Deviance: {res['deviance']:.4f} | "
              f"MAE: {res['mae']:.4f} | 상위10% 포착률: {res['top10_capture']:.1f}%")

    # =========================================================================
    # 5. 최종 모델 선정 및 전체 데이터 재학습 & 패키징
    # =========================================================================
    print("\n" + "=" * 80)
    print("🏆 [5단계] 최종 모델 선정 및 패키징 결론")
    print("=" * 80)

    print(f"1) 기준선 Poisson GLM              : Deviance {res_glm['deviance']:.4f} | 포착률 {res_glm['top10_capture']:.1f}%")
    print(f"2) 튜닝된 LightGBM (단일)         : Deviance {best_lgb_res['deviance']:.4f} | 포착률 {best_lgb_res['top10_capture']:.1f}%")
    print(f"3) 최적 앙상블 (LGB {best_blend_w[0]:.1f} + GLM {best_blend_w[1]:.1f}): Deviance {best_blend_res['deviance']:.4f} | 포착률 {best_blend_res['top10_capture']:.1f}%")

    dev_diff = best_lgb_res["deviance"] - best_blend_res["deviance"]
    print(f"\n앙상블 대비 단일 모델 Deviance 차이: {dev_diff:+.5f}")

    # 배포 효율성 vs 미세 성능 판정
    # 만약 앙상블이 0.001 이상 개선되면 앙상블 파이프라인, 아니면 단일 모델 채택
    if dev_diff >= 0.001:
        chosen_type = "ensemble"
        print("-> [결정] 앙상블 모델이 통계적으로 유의미한 성능 개선을 보여 'LGB + GLM 앙상블'을 최종 모델로 채택합니다.")
    else:
        chosen_type = "lightgbm_single"
        print("-> [결정] 앙상블과 단일 모델 간 Deviance 차이가 0.001 미만으로 미미하여, "
              "서빙 지연시간(Latency)과 시스템 단순성을 고려해 '튜닝된 LightGBM 단일 모델'을 최종 모델로 채택합니다.")

    # 전체 데이터(5,782개 구간) 학습
    print("\n고속도로 전체 5,782개 구간 데이터로 최종 모델 학습 중...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    final_lgb = lgb.LGBMRegressor(
        objective="poisson",
        learning_rate=best_lgb_params["learning_rate"],
        num_leaves=best_lgb_params["num_leaves"],
        max_depth=best_lgb_params["max_depth"],
        min_child_samples=best_lgb_params["min_child_samples"],
        reg_alpha=best_lgb_params["reg_alpha"],
        reg_lambda=best_lgb_params["reg_lambda"],
        n_estimators=best_lgb_params["n_estimators"],
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        verbose=-1,
    )
    final_lgb.fit(df[ALL_FEATURES], df[target_col].values)

    # GLM도 전체 학습 (앙상블 패키징 대비)
    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(df[ALL_FEATURES])
    final_glm = PoissonRegressor(alpha=1.0, max_iter=500)
    final_glm.fit(X_all_scaled, df[target_col].values)

    package = {
        "model_name": "highway_crash_risk_model1",
        "version": "1.0.0",
        "target": target_col,
        "chosen_architecture": chosen_type,
        "feature_names": ALL_FEATURES,
        "lgb_model": final_lgb,
        "glm_model": final_glm,
        "scaler": final_scaler,
        "ensemble_weights": {"lgb": best_blend_w[0], "glm": best_blend_w[1]},
        "best_hyperparameters": best_lgb_params,
        "cv_performance": {
            "baseline_glm_deviance": float(res_glm["deviance"]),
            "tuned_lgb_deviance": float(best_lgb_res["deviance"]),
            "ensemble_deviance": float(best_blend_res["deviance"]),
            "top10_capture_rate": float(best_lgb_res["top10_capture"]),
            "mae": float(best_lgb_res["mae"]),
        },
        "metadata": {
            "total_segments": len(df),
            "total_routes": int(df["route"].nunique()),
            "cv_strategy": "5-Fold Route-based GroupKFold",
            "created_by": "박범수 (AI Engineer)",
        },
    }

    joblib.dump(package, OUTPUT_MODEL_PATH)
    file_size_kb = os.path.getsize(OUTPUT_MODEL_PATH) / 1024
    print(f"✅ 최종 모델 저장 완료: {OUTPUT_MODEL_PATH} ({file_size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
