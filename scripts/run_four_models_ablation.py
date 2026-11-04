import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_poisson_deviance
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/expressway/segment_vds_aadt_v1.csv')
# AADT 유효 구간 (통신 이상치 제외)
df = df[df['aadt'] > 0].copy().reset_index(drop=True)
df['log_aadt'] = np.log(df['aadt'])

ROAD_GEOM_FEATURES = [
    'segment_length_km', 'is_partial_segment', 'curvature_mean', 'curvature_max',
    'min_tunnel_dist_km', 'has_tunnel_within_1km', 'tunnel_portals_1km', 'max_tunnel_length_1km'
]

y_count = df['glare_front_count'].values
y_bin = (y_count > 0).astype(int)
groups = df['route'].values
n_routes = df['route'].nunique()

print("=" * 85)
print("🔬 [사용자 제안 4대 모델 비교 실험] 동일 표본 VDS 실측 구간 5-Fold GroupKFold")
print("=" * 85)
print(f"총 분석 구간: {len(df):,}개 | 고속도로 노선: {n_routes}개 | 정면 태양 사고: {y_count.sum()}건 (발생 구간: {y_bin.sum()}개, {y_bin.mean()*100:.2f}%)")

models_specs = {
    "Model A (노출시간 단독)": ['annual_glare_hours_fwd'],
    "Model B (노출시간 + AADT)": ['annual_glare_hours_fwd', 'log_aadt'],
    "Model C (노출시간 + 도로기하 8개)": ['annual_glare_hours_fwd'] + ROAD_GEOM_FEATURES,
    "Model D (노출시간 + AADT + 도로기하 8개)": ['annual_glare_hours_fwd', 'log_aadt'] + ROAD_GEOM_FEATURES,
}

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(df, y_count, groups))

# 1. LightGBM Poisson 결과 수집
lgb_results = {}
for name, feats in models_specs.items():
    oof_preds = np.zeros(len(df))
    train_devs, oof_devs = [], []
    train_rocs, oof_rocs = [], []
    
    for tr, te in splits:
        m = lgb.LGBMRegressor(
            objective='poisson', n_estimators=80, learning_rate=0.03, num_leaves=7, max_depth=3,
            min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=5.0, random_state=42, verbose=-1
        )
        m.fit(df.iloc[tr][feats], y_count[tr])
        p_te = np.clip(m.predict(df.iloc[te][feats]), 1e-6, None)
        p_tr = np.clip(m.predict(df.iloc[tr][feats]), 1e-6, None)
        oof_preds[te] = p_te
        
        train_devs.append(mean_poisson_deviance(y_count[tr], p_tr))
        oof_devs.append(mean_poisson_deviance(y_count[te], p_te))
        train_rocs.append(roc_auc_score(y_bin[tr], p_tr))
        oof_rocs.append(roc_auc_score(y_bin[te], p_te))
        
    k = int(len(df) * 0.1)
    top10_idx = np.argsort(oof_preds)[-k:]
    top10_cap = y_count[top10_idx].sum() / max(y_count.sum(), 1) * 100
    
    oof_dev = mean_poisson_deviance(y_count, np.clip(oof_preds, 1e-6, None))
    oof_roc = roc_auc_score(y_bin, oof_preds)
    oof_pr = average_precision_score(y_bin, oof_preds)
    
    lgb_results[name] = {
        "train_dev": np.mean(train_devs), "oof_dev": oof_dev,
        "train_roc": np.mean(train_rocs), "oof_roc": oof_roc,
        "oof_pr": oof_pr, "top10_cap": top10_cap, "oof_preds": oof_preds
    }

# 2. Poisson GLM (선형 회귀) 결과 수집 (비선형 트리 vs 선형 모델 비교)
glm_results = {}
for name, feats in models_specs.items():
    oof_preds = np.zeros(len(df))
    train_devs, oof_devs = [], []
    train_rocs, oof_rocs = [], []
    
    for tr, te in splits:
        pipe = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=500))
        pipe.fit(df.iloc[tr][feats], y_count[tr])
        p_te = np.clip(pipe.predict(df.iloc[te][feats]), 1e-6, None)
        p_tr = np.clip(pipe.predict(df.iloc[tr][feats]), 1e-6, None)
        oof_preds[te] = p_te
        
        train_devs.append(mean_poisson_deviance(y_count[tr], p_tr))
        oof_devs.append(mean_poisson_deviance(y_count[te], p_te))
        train_rocs.append(roc_auc_score(y_bin[tr], p_tr))
        oof_rocs.append(roc_auc_score(y_bin[te], p_te))
        
    k = int(len(df) * 0.1)
    top10_idx = np.argsort(oof_preds)[-k:]
    top10_cap = y_count[top10_idx].sum() / max(y_count.sum(), 1) * 100
    
    oof_dev = mean_poisson_deviance(y_count, np.clip(oof_preds, 1e-6, None))
    oof_roc = roc_auc_score(y_bin, oof_preds)
    oof_pr = average_precision_score(y_bin, oof_preds)
    
    glm_results[name] = {
        "train_dev": np.mean(train_devs), "oof_dev": oof_dev,
        "train_roc": np.mean(train_rocs), "oof_roc": oof_roc,
        "oof_pr": oof_pr, "top10_cap": top10_cap, "oof_preds": oof_preds
    }

print("\n" + "=" * 85)
print("1. [LightGBM Poisson] 피처 조합별 성능 및 과적합 격차")
print("=" * 85)
print(f"{'모델 구분':<30} | {'OOF Dev':<9} | {'OOF ROC':<9} | {'Train-OOF ROC Gap':<17} | {'OOF PR':<8} | {'Top10% 포착률':<11}")
print("-" * 85)
for name, r in lgb_results.items():
    roc_gap = r['train_roc'] - r['oof_roc']
    print(f"{name:<30} | {r['oof_dev']:.4f}    | {r['oof_roc']:.4f}    | {roc_gap:+.4f} (Tr:{r['train_roc']:.3f}) | {r['oof_pr']:.4f}   | {r['top10_cap']:5.1f}%")

print("\n" + "=" * 85)
print("2. [Poisson GLM (선형)] 피처 조합별 성능 및 과적합 격차")
print("=" * 85)
print(f"{'모델 구분':<30} | {'OOF Dev':<9} | {'OOF ROC':<9} | {'Train-OOF ROC Gap':<17} | {'OOF PR':<8} | {'Top10% 포착률':<11}")
print("-" * 85)
for name, r in glm_results.items():
    roc_gap = r['train_roc'] - r['oof_roc']
    print(f"{name:<30} | {r['oof_dev']:.4f}    | {r['oof_roc']:.4f}    | {roc_gap:+.4f} (Tr:{r['train_roc']:.3f}) | {r['oof_pr']:.4f}   | {r['top10_cap']:5.1f}%")

print("\n" + "=" * 85)
print("3. 핵심 발견 및 해석 요약")
print("=" * 85)
# Print Model A vs B comparison directly
dev_diff_lgb = lgb_results["Model B (노출시간 + AADT)"]['oof_dev'] - lgb_results["Model A (노출시간 단독)"]['oof_dev']
roc_diff_lgb = lgb_results["Model B (노출시간 + AADT)"]['oof_roc'] - lgb_results["Model A (노출시간 단독)"]['oof_roc']
print(f"• AADT 추가 효과 (LGBM Model B vs Model A): Deviance 변화={dev_diff_lgb:+.4f}, ROC-AUC 변화={roc_diff_lgb:+.4f}")

dev_diff_glm = glm_results["Model B (노출시간 + AADT)"]['oof_dev'] - glm_results["Model A (노출시간 단독)"]['oof_dev']
roc_diff_glm = glm_results["Model B (노출시간 + AADT)"]['oof_roc'] - glm_results["Model A (노출시간 단독)"]['oof_roc']
print(f"• AADT 추가 효과 (GLM Model B vs Model A):  Deviance 변화={dev_diff_glm:+.4f}, ROC-AUC 변화={roc_diff_glm:+.4f}")
