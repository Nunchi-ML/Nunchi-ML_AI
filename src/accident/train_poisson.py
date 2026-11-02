"""고속도로 1km 구간 데이터셋(①-v1)을 사용하여
노선 단위 GroupKFold(5겹) 교차검증 기반 포아송 회귀(GLM) 및 LightGBM(Poisson) 모델을 평가하고,
폴드별 성능 편차(mean ± std), 위험 구간(상위 10%) 포착률, 사고 발생 구간 MAE 등
심층 검증 지표를 산출하는 스크립트.

실행:
    python -m src.accident.train_poisson
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance, mean_absolute_error
import lightgbm as lgb

sys.stdout.reconfigure(encoding="utf-8")

SEED = 0
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
SEG_PATH = os.path.join(DATA_DIR, "expressway", "segment_1km_dataset_v1.csv")
FOLDS_PATH = os.path.join(DATA_DIR, "expressway", "folds_model1_v1.csv")

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


def safe_poisson_deviance(y_true, y_pred):
    """예측값이 0 이하가 되지 않도록 클리핑한 뒤 포아송 deviance 계산."""
    pred_clipped = np.clip(y_pred, 1e-6, None)
    return mean_poisson_deviance(y_true, pred_clipped)


def evaluate_cv_detailed(df, target_col, feature_cols, model_builder, n_splits=5):
    y = df[target_col].values
    groups = df["route"].values
    oof_pred = np.zeros(len(df), dtype=float)

    fold_devs = []
    fold_maes = []
    fold_pos_maes = []

    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(df, y, groups)):
        tr_df = df.iloc[tr_idx]
        te_df = df.iloc[te_idx]
        y_te = y[te_idx]

        if feature_cols is None:
            rate = tr_df[target_col].sum() / max(tr_df["segment_length_km"].sum(), 1e-6)
            pred = te_df["segment_length_km"].values * rate
        else:
            model = model_builder()
            model.fit(tr_df[feature_cols], tr_df[target_col].values)
            pred = model.predict(te_df[feature_cols])

        pred_clipped = np.clip(pred, 1e-6, None)
        oof_pred[te_idx] = pred_clipped

        # 폴드별 지표
        f_dev = safe_poisson_deviance(y_te, pred_clipped)
        f_mae = mean_absolute_error(y_te, pred_clipped)
        pos_mask = y_te > 0
        f_pos_mae = mean_absolute_error(y_te[pos_mask], pred_clipped[pos_mask]) if pos_mask.sum() > 0 else 0.0

        fold_devs.append(f_dev)
        fold_maes.append(f_mae)
        fold_pos_maes.append(f_pos_mae)

    overall_dev = safe_poisson_deviance(y, oof_pred)
    overall_mae = mean_absolute_error(y, oof_pred)
    pos_overall = y > 0
    overall_pos_mae = mean_absolute_error(y[pos_overall], oof_pred[pos_overall])

    # 상위 10% 위험 구간 실제 사고 포착률 (Capture Rate)
    top10_k = int(len(df) * 0.1)
    top10_idx = np.argsort(oof_pred)[-top10_k:]
    captured_accidents = y[top10_idx].sum()
    total_accidents = y.sum()
    top10_capture_rate = (captured_accidents / max(total_accidents, 1)) * 100.0

    return {
        "overall_dev": overall_dev,
        "overall_mae": overall_mae,
        "dev_mean": np.mean(fold_devs),
        "dev_std": np.std(fold_devs),
        "mae_mean": np.mean(fold_maes),
        "mae_std": np.std(fold_maes),
        "pos_mae": overall_pos_mae,
        "top10_capture": top10_capture_rate,
        "oof_pred": oof_pred,
        "fold_devs": fold_devs,
    }


def main():
    df = pd.read_csv(SEG_PATH)
    # 터널 인접 x 출퇴근 역광 시간 교호작용
    df["tunnel_glare_interaction"] = df["has_tunnel_within_1km"] * df["commute_glare_hours"]

    # 노선 단위 5-Fold 분할 인덱스 저장
    gkf = GroupKFold(n_splits=5)
    df["fold"] = -1
    for fold_idx, (_, te_idx) in enumerate(gkf.split(df, df["accident_count"], df["route"])):
        df.loc[df.index[te_idx], "fold"] = fold_idx

    df[["route", "segment_km", "fold", "accident_count", "glare_front_count", "commute_glare_hours"]].to_csv(
        FOLDS_PATH, index=False, encoding="utf-8-sig"
    )

    print("=" * 80)
    print("🛣️ 모델 ① 노선 단위 5-Fold 교차검증 상세 분석 (①-v1)")
    print("=" * 80)
    print(f"총 구간 수: {len(df):,}개 | 고속도로 노선 수: {df['route'].nunique()}개")
    print(f"분할 파일: {FOLDS_PATH}\n")

    print("[폴드별 노선 수 · 구간 수 · 사고 건수 분포]")
    for f in range(5):
        f_sub = df[df["fold"] == f]
        n_routes = f_sub["route"].nunique()
        n_segs = len(f_sub)
        n_accs = f_sub["accident_count"].sum()
        n_glare = f_sub["glare_front_count"].sum()
        print(f"  * Fold {f}: 노선 {n_routes:2d}개 | 구간 {n_segs:4d}개 ({n_segs/len(df)*100:4.1f}%) | "
              f"전체사고 {n_accs:4d}건 ({n_accs/df['accident_count'].sum()*100:4.1f}%) | 정면태양 사고 {n_glare:3d}건")

    all_features = BASE_FEATURES + SOLAR_FEATURES

    def glm_builder():
        return make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=500))

    def lgb_builder():
        return lgb.LGBMRegressor(
            objective="poisson",
            n_estimators=80,
            learning_rate=0.03,
            num_leaves=7,
            max_depth=3,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=SEED,
            verbose=-1,
        )

    print("\n" + "=" * 80)
    print("☀️ [실험 1] 정면 저각도 태양 조건 사고 건수 (glare_front_count, Q2 검증)")
    print("=" * 80)
    exp_glare = [
        ("기준선 0: 구간길이 비례 평균", None, None),
        ("기준선 1: GLM 기본 (도로·터널)", BASE_FEATURES, glm_builder),
        ("실험 A: GLM + 태양 변수", all_features, glm_builder),
        ("실험 B: LightGBM 기본 (도로·터널)", BASE_FEATURES, lgb_builder),
        ("실험 C: LightGBM + 태양 변수", all_features, lgb_builder),
    ]

    for name, feats, bld in exp_glare:
        res = evaluate_cv_detailed(df, "glare_front_count", feats, bld)
        print(f"- {name:<32} | Dev: {res['overall_dev']:.4f} (±{res['dev_std']:.4f}) | "
              f"MAE: {res['overall_mae']:.4f} (±{res['mae_std']:.4f}) | "
              f"사고구간 MAE: {res['pos_mae']:.4f} | 상위10% 포착률: {res['top10_capture']:.1f}%")

    print("\n" + "=" * 80)
    print("📊 [실험 2] 24시간 전체 사고 건수 (accident_count)")
    print("=" * 80)
    exp_total = [
        ("기준선 0: 구간길이 비례 평균", None, None),
        ("기준선 1: GLM 기본 (도로·터널)", BASE_FEATURES, glm_builder),
        ("실험 A: GLM + 태양 변수", all_features, glm_builder),
        ("실험 B: LightGBM 기본 (도로·터널)", BASE_FEATURES, lgb_builder),
        ("실험 C: LightGBM + 태양 변수", all_features, lgb_builder),
    ]

    for name, feats, bld in exp_total:
        res = evaluate_cv_detailed(df, "accident_count", feats, bld)
        print(f"- {name:<32} | Dev: {res['overall_dev']:.4f} (±{res['dev_std']:.4f}) | "
              f"MAE: {res['overall_mae']:.4f} (±{res['mae_std']:.4f}) | "
              f"사고구간 MAE: {res['pos_mae']:.4f} | 상위10% 포착률: {res['top10_capture']:.1f}%")

    # Feature importances
    m_glare = lgb_builder().fit(df[all_features], df["glare_front_count"])
    imp_glare = pd.Series(m_glare.booster_.feature_importance("gain"), index=all_features)
    imp_glare_norm = (imp_glare / imp_glare.sum()).sort_values(ascending=False).round(3)
    print("\n[정면 태양 사고 모델 LightGBM 변수 중요도 (Gain 비율)]")
    for k, v in imp_glare_norm.items():
        print(f"  * {k:<26}: {v*100:5.1f}%")


if __name__ == "__main__":
    main()
