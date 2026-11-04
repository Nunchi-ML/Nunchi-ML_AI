import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_poisson_deviance
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import PoissonRegressor
import lightgbm as lgb

sys.stdout.reconfigure(encoding='utf-8')

# 데이터 로드
df = pd.read_csv('data/expressway/segment_vds_aadt_v1.csv')
df = df[df['aadt'] > 0].copy().reset_index(drop=True)
df['log_aadt'] = np.log(df['aadt'])

ROAD_GEOM_FEATURES = [
    'segment_length_km', 'is_partial_segment', 'curvature_mean', 'curvature_max',
    'min_tunnel_dist_km', 'has_tunnel_within_1km', 'tunnel_portals_1km', 'max_tunnel_length_1km'
]
MODEL_D_FEATURES = ['annual_glare_hours_fwd', 'log_aadt'] + ROAD_GEOM_FEATURES

y_count = df['glare_front_count'].values
y_bin = (y_count > 0).astype(int)
groups = df['route'].values
routes = np.unique(groups)
n_routes = len(routes)

print("=" * 85)
print("🔬 [사용자 지적 후속 검증] A-Rule vs A-ML 실질 비교, Bootstrap 유의성 및 구간별 사고율")
print("=" * 85)
print(f"표본 수: {len(df):,}개 | 노선 수: {n_routes}개 | 사고 건수: {y_count.sum()}건 (발생 구간: {y_bin.sum()}개, {y_bin.mean()*100:.2f}%)")

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(df, y_count, groups))

# =========================================================================
# 1. 모델 구축 및 OOF 예측
# =========================================================================
# 1-1. A-Rule: 연간 정면 노출시간 직접 정렬 (학습 없음)
raw_glare = df['annual_glare_hours_fwd'].values

# 1-2. A-Rule + 단변량 Poisson GLM (단순 노출시간을 확률/건수로 캘리브레이션)
oof_glm_a = np.zeros(len(df))
for tr, te in splits:
    glm = PoissonRegressor(alpha=1.0)
    glm.fit(df.iloc[tr][['annual_glare_hours_fwd']], y_count[tr])
    oof_glm_a[te] = glm.predict(df.iloc[te][['annual_glare_hours_fwd']])

# 1-3. A-ML: 단일 피처 LightGBM Poisson
oof_lgb_a = np.zeros(len(df))
train_lgb_a_rocs = []
for tr, te in splits:
    m = lgb.LGBMRegressor(
        objective='poisson', n_estimators=80, learning_rate=0.03, num_leaves=7, max_depth=3,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=5.0, random_state=42, verbose=-1
    )
    m.fit(df.iloc[tr][['annual_glare_hours_fwd']], y_count[tr])
    oof_lgb_a[te] = m.predict(df.iloc[te][['annual_glare_hours_fwd']])
    train_lgb_a_rocs.append(roc_auc_score(y_bin[tr], m.predict(df.iloc[tr][['annual_glare_hours_fwd']])))

# 1-4. Model D: 10개 피처 LightGBM Poisson
oof_lgb_d = np.zeros(len(df))
train_lgb_d_rocs = []
for tr, te in splits:
    m = lgb.LGBMRegressor(
        objective='poisson', n_estimators=80, learning_rate=0.03, num_leaves=7, max_depth=3,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=5.0, random_state=42, verbose=-1
    )
    m.fit(df.iloc[tr][MODEL_D_FEATURES], y_count[tr])
    oof_lgb_d[te] = m.predict(df.iloc[te][MODEL_D_FEATURES])
    train_lgb_d_rocs.append(roc_auc_score(y_bin[tr], m.predict(df.iloc[tr][MODEL_D_FEATURES])))

k = int(len(df) * 0.1)

def calc_metrics(preds, is_calibrated=True):
    p_clip = np.clip(preds, 1e-6, None)
    dev = mean_poisson_deviance(y_count, p_clip) if is_calibrated else None
    roc = roc_auc_score(y_bin, preds)
    pr = average_precision_score(y_bin, preds)
    top10_idx = np.argsort(preds)[-k:]
    cap = y_count[top10_idx].sum() / max(y_count.sum(), 1) * 100
    return dev, roc, pr, cap

dev_rule, roc_rule, pr_rule, cap_rule = calc_metrics(raw_glare, is_calibrated=False)
dev_glm_a, roc_glm_a, pr_glm_a, cap_glm_a = calc_metrics(oof_glm_a, is_calibrated=True)
dev_lgb_a, roc_lgb_a, pr_lgb_a, cap_lgb_a = calc_metrics(oof_lgb_a, is_calibrated=True)
dev_lgb_d, roc_lgb_d, pr_lgb_d, cap_lgb_d = calc_metrics(oof_lgb_d, is_calibrated=True)

print("\n" + "=" * 85)
print("1. [A-Rule vs A-ML vs Model D 실질 비교]")
print("=" * 85)
print(f"{'모델 구분':<32} | {'OOF Dev':<9} | {'OOF ROC':<9} | {'Train ROC':<9} | {'ROC Gap':<8} | {'Top10% 포착률':<11}")
print("-" * 85)
print(f"{'A-Rule (노출시간 직접 정렬)':<32} | {'(N/A)':<9} | {roc_rule:.4f}    | {roc_rule:.4f}    | {'+0.0000':<8} | {cap_rule:5.1f}%")
print(f"{'A-GLM  (노출시간 + Poisson GLM)':<32} | {dev_glm_a:.4f}    | {roc_glm_a:.4f}    | {'-':<9} | {'-':<8} | {cap_glm_a:5.1f}%")
print(f"{'A-ML   (노출시간 단일 LightGBM)':<32} | {dev_lgb_a:.4f}    | {roc_lgb_a:.4f}    | {np.mean(train_lgb_a_rocs):.4f}    | {np.mean(train_lgb_a_rocs)-roc_lgb_a:+.4f}  | {cap_lgb_a:5.1f}%")
print(f"{'Model D (10개 피처 LightGBM)':<32} | {dev_lgb_d:.4f}    | {roc_lgb_d:.4f}    | {np.mean(train_lgb_d_rocs):.4f}    | {np.mean(train_lgb_d_rocs)-roc_lgb_d:+.4f}  | {cap_lgb_d:5.1f}%")

# =========================================================================
# 2. A-Rule vs Model D Paired Bootstrap (2,000회 노선 군집)
# =========================================================================
print("\n" + "=" * 85)
print("2. [Paired Bootstrap 2,000회 검정] A-Rule vs Model D (복잡 모델 성능 저하 유의성)")
print("=" * 85)

np.random.seed(42)
n_bootstraps = 2000
cluster_indices = {r: np.where(groups == r)[0] for r in routes}

diff_rocs = []
diff_caps = []

for _ in range(n_bootstraps):
    sampled_routes = np.random.choice(routes, size=n_routes, replace=True)
    sample_idx = np.concatenate([cluster_indices[r] for r in sampled_routes])
    
    y_s = y_bin[sample_idx]
    yc_s = y_count[sample_idx]
    if y_s.sum() == 0 or y_s.sum() == len(y_s) or yc_s.sum() == 0:
        continue
        
    # ROC 차이
    r_roc = roc_auc_score(y_s, raw_glare[sample_idx])
    d_roc = roc_auc_score(y_s, oof_lgb_d[sample_idx])
    diff_rocs.append(r_roc - d_roc)
    
    # 상위 10% 포착률 차이
    k_s = int(len(sample_idx) * 0.1)
    top_r = np.argsort(raw_glare[sample_idx])[-k_s:]
    top_d = np.argsort(oof_lgb_d[sample_idx])[-k_s:]
    cap_r = yc_s[top_r].sum() / yc_s.sum() * 100
    cap_d = yc_s[top_d].sum() / yc_s.sum() * 100
    diff_caps.append(cap_r - cap_d)

diff_rocs = np.array(diff_rocs)
diff_caps = np.array(diff_caps)

ci_roc = np.percentile(diff_rocs, [2.5, 97.5])
p_roc = np.mean(diff_rocs <= 0)

ci_cap = np.percentile(diff_caps, [2.5, 97.5])
p_cap = np.mean(diff_caps <= 0)

print(f"• ROC-AUC 차이 (A-Rule - Model D):")
print(f"  평균 차이: {np.mean(diff_rocs):+.4f} | 95% 신뢰구간: [{ci_roc[0]:+.4f}, {ci_roc[1]:+.4f}] | p-value: {p_roc:.4f}")
print(f"• 상위 10% 사고 포착률 차이 (A-Rule - Model D):")
print(f"  평균 차이: {np.mean(diff_caps):+.2f}%p | 95% 신뢰구간: [{ci_cap[0]:+.2f}%p, {ci_cap[1]:+.2f}%p] | p-value: {p_cap:.4f}")

# =========================================================================
# 3. 연간 정면 노출시간 구간별 실제 사고 발생률 분석 (Binned Analysis)
# =========================================================================
print("\n" + "=" * 85)
print("3. [노출시간 구간별 실제 사고율 분석] 태양 노출시간 증가에 따른 실제 사고율 추이")
print("=" * 85)

df['glare_bin'] = pd.qcut(df['annual_glare_hours_fwd'], q=5, duplicates='drop')
binned = df.groupby('glare_bin', observed=True).agg(
    구간수=('annual_glare_hours_fwd', 'count'),
    평균노출시간=('annual_glare_hours_fwd', 'mean'),
    최소노출=('annual_glare_hours_fwd', 'min'),
    최대노출=('annual_glare_hours_fwd', 'max'),
    총사고건수=('glare_front_count', 'sum'),
    사고발생구간수=('glare_front_count', lambda x: (x > 0).sum()),
).reset_index()

binned['구간당평균사고건수'] = binned['총사고건수'] / binned['구간수']
binned['구간양성률(%)'] = (binned['사고발생구간수'] / binned['구간수']) * 100

print(f"{'노출시간 분위(5분위)':<25} | {'구간수':<6} | {'평균노출(h)':<10} | {'사고건수':<8} | {'구간당 평균사고':<14} | {'양성 구간 비율(%)':<14}")
print("-" * 85)
for _, r in binned.iterrows():
    print(f"{str(r['glare_bin']):<25} | {r['구간수']:<6} | {r['평균노출시간']:8.1f}h  | {r['총사고건수']:4d}건   | {r['구간당평균사고건수']:.5f}건/km    | {r['구간양성률(%)']:6.2f}%")

# 상위 10% 초고노출 구간 별도 확인
q90 = df['annual_glare_hours_fwd'].quantile(0.9)
top10_sub = df[df['annual_glare_hours_fwd'] >= q90]
bot90_sub = df[df['annual_glare_hours_fwd'] < q90]

rate_top = top10_sub['glare_front_count'].sum() / len(top10_sub)
rate_bot = bot90_sub['glare_front_count'].sum() / len(bot90_sub)

print("\n[상위 10% 초고노출 구간 vs 하위 90% 비교]")
print(f"• 상위 10% 구간 (노출시간 >= {q90:.1f}h, {len(top10_sub)}개): 총 사고 {top10_sub['glare_front_count'].sum()}건 (구간당 {rate_top:.4f}건)")
print(f"• 하위 90% 구간 (노출시간 <  {q90:.1f}h, {len(bot90_sub)}개): 총 사고 {bot90_sub['glare_front_count'].sum()}건 (구간당 {rate_bot:.4f}건)")
print(f"• 사고율 배수: 상위 10% 초고노출 구간이 하위 90% 대비 약 {rate_top / max(rate_bot, 1e-6):.2f}배 사고율 높음")
