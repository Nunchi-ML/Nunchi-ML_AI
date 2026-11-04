import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_poisson_deviance, mean_absolute_error
from sklearn.linear_model import PoissonRegressor
import lightgbm as lgb

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/expressway/segment_1km_dataset_v1.csv')
df['tunnel_glare_interaction'] = df['has_tunnel_within_1km'] * df['commute_glare_hours']

BASE_FEATURES = [
    'segment_length_km', 'is_partial_segment', 'curvature_mean', 'curvature_max',
    'min_tunnel_dist_km', 'has_tunnel_within_1km', 'tunnel_portals_1km', 'max_tunnel_length_1km'
]
SOLAR_FEATURES = [
    'ew_alignment', 'annual_glare_hours_fwd', 'commute_glare_hours', 'tunnel_glare_interaction'
]
ALL_FEATURES = BASE_FEATURES + SOLAR_FEATURES

y = df['glare_front_count'].values
groups = df['route'].values
gkf = GroupKFold(n_splits=5)

oof_all_lgb = np.zeros(len(df))
oof_sol_lgb = np.zeros(len(df))
oof_rule_glm = np.zeros(len(df))
fold_ids = np.zeros(len(df), dtype=int)

for fold, (tr, te) in enumerate(gkf.split(df, y, groups)):
    # 1. LightGBM (All 12 features)
    m_all = lgb.LGBMRegressor(
        objective='poisson', n_estimators=80, learning_rate=0.03, num_leaves=7, max_depth=3,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=5.0, random_state=42, verbose=-1
    )
    m_all.fit(df.iloc[tr][ALL_FEATURES], y[tr])
    oof_all_lgb[te] = m_all.predict(df.iloc[te][ALL_FEATURES])
    
    # 2. LightGBM (Solar 4 features only)
    m_sol = lgb.LGBMRegressor(
        objective='poisson', n_estimators=80, learning_rate=0.03, num_leaves=7, max_depth=3,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=5.0, random_state=42, verbose=-1
    )
    m_sol.fit(df.iloc[tr][SOLAR_FEATURES], y[tr])
    oof_sol_lgb[te] = m_sol.predict(df.iloc[te][SOLAR_FEATURES])
    
    # 3. Rule GLM (OOF)
    glm = PoissonRegressor(alpha=1.0)
    glm.fit(df.iloc[tr][['annual_glare_hours_fwd']], y[tr])
    oof_rule_glm[te] = glm.predict(df.iloc[te][['annual_glare_hours_fwd']])
    
    fold_ids[te] = fold

df['oof_all'] = oof_all_lgb
df['oof_sol'] = oof_sol_lgb
df['oof_rule'] = oof_rule_glm
df['fold'] = fold_ids

# Normalized rank within fold
df['rank_norm_all'] = df.groupby('fold')['oof_all'].rank(pct=True)
df['rank_norm_sol'] = df.groupby('fold')['oof_sol'].rank(pct=True)
df['rank_norm_rule'] = df.groupby('fold')['annual_glare_hours_fwd'].rank(pct=True)

k = 578
print("=" * 70)
print("🎯 [검증 결과] LightGBM vs 단순 규칙 포착률 및 Deviance 비교")
print("=" * 70)
print(f"총 구간 수: {len(df):,}개 | 정면 태양 사고(y): {y.sum()}건 | 상위 10% k={k}개 구간\n")

print("--- [1. 글로벌 OOF 상위 10% 포착률 (Global Top 10% Pooling)] ---")
top_all = df.sort_values('oof_all', ascending=False).head(k)
cap_all = top_all['glare_front_count'].sum()
print(f"1. LightGBM (전체 12개 피처) Raw OOF:        {cap_all:2d}건 / 99건 ({cap_all/99*100:5.2f}%)")

top_norm_all = df.sort_values('rank_norm_all', ascending=False).head(k)
cap_norm_all = top_norm_all['glare_front_count'].sum()
print(f"2. LightGBM (전체 12개) Fold-Rank 정규화:    {cap_norm_all:2d}건 / 99건 ({cap_norm_all/99*100:5.2f}%)")

top_sol = df.sort_values('oof_sol', ascending=False).head(k)
cap_sol = top_sol['glare_front_count'].sum()
print(f"3. LightGBM (태양 4개 피처 전용) Raw OOF:     {cap_sol:2d}건 / 99건 ({cap_sol/99*100:5.2f}%)")

top_norm_sol = df.sort_values('rank_norm_sol', ascending=False).head(k)
cap_norm_sol = top_norm_sol['glare_front_count'].sum()
print(f"4. LightGBM (태양 4개 피처) Fold-Rank 정규화: {cap_norm_sol:2d}건 / 99건 ({cap_norm_sol/99*100:5.2f}%)")

top_rule = df.sort_values('annual_glare_hours_fwd', ascending=False).head(k)
cap_rule = top_rule['glare_front_count'].sum()
print(f"5. 단순 규칙 (annual_glare_hours_fwd 순위):   {cap_rule:2d}건 / 99건 ({cap_rule/99*100:5.2f}%)")

top_ew = df.sort_values('ew_alignment', ascending=False).head(k)
cap_ew = top_ew['glare_front_count'].sum()
print(f"6. 단순 규칙 (ew_alignment 동서축 순위):      {cap_ew:2d}건 / 99건 ({cap_ew/99*100:5.2f}%)")


print("\n--- [2. 폴드별(Within-Fold) 상위 10% 포착률 평균] ---")
fold_lgb_caps = []
fold_sol_caps = []
fold_rule_caps = []
for f in range(5):
    sub = df[df['fold'] == f]
    kf = int(len(sub) * 0.1)
    acc_total = sub['glare_front_count'].sum()
    
    t_all = sub.sort_values('oof_all', ascending=False).head(kf)['glare_front_count'].sum()
    t_sol = sub.sort_values('oof_sol', ascending=False).head(kf)['glare_front_count'].sum()
    t_rule = sub.sort_values('annual_glare_hours_fwd', ascending=False).head(kf)['glare_front_count'].sum()
    
    c_all = t_all / max(acc_total, 1) * 100
    c_sol = t_sol / max(acc_total, 1) * 100
    c_rule = t_rule / max(acc_total, 1) * 100
    
    fold_lgb_caps.append(c_all)
    fold_sol_caps.append(c_sol)
    fold_rule_caps.append(c_rule)
    print(f"Fold {f} (사고 {acc_total:2d}건): LGBM 12개={c_all:5.1f}% | LGBM 태양={c_sol:5.1f}% | 단순규칙={c_rule:5.1f}%")

print(f"\n>> 폴드별 평균 포착률: LGBM 12개={np.mean(fold_lgb_caps):.1f}% | LGBM 태양={np.mean(fold_sol_caps):.1f}% | 단순규칙={np.mean(fold_rule_caps):.1f}%")


print("\n--- [3. 포아송 Deviance 및 확률 캘리브레이션] ---")
def p_dev(y_true, y_pred):
    return mean_poisson_deviance(y_true, np.clip(y_pred, 1e-6, None))

print(f"LightGBM (전체 12개 피처) OOF Deviance: {p_dev(y, oof_all_lgb):.4f}")
print(f"LightGBM (태양 4개 피처)  OOF Deviance: {p_dev(y, oof_sol_lgb):.4f}")
print(f"단순 규칙 (Poisson GLM)   OOF Deviance: {p_dev(y, oof_rule_glm):.4f}")
