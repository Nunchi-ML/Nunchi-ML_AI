import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_poisson_deviance
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import PoissonRegressor
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.stdout.reconfigure(encoding='utf-8')

# 데이터 로드
df = pd.read_csv('data/expressway/segment_vds_aadt_v1.csv')
df = df[df['aadt'] > 0].copy().reset_index(drop=True)
df['log_aadt'] = np.log(df['aadt'])
df['vkt_100m_3yr'] = (df['aadt'] * df['segment_length_km'] * 365 * 3) / 1e8  # 1억 대·km 단위

y_count = df['glare_front_count'].values
y_bin = (y_count > 0).astype(int)
groups = df['route'].values
routes = np.unique(groups)
n_routes = len(routes)

print("=" * 85)
print("🔬 [엄밀 통계 검증] A-Rule 신뢰구간, A-GLM 캘리브레이션 및 노출시간 추세 검정")
print("=" * 85)
print(f"분석 대상: {len(df):,}개 구간 | 노선 수: {n_routes}개 | 총 정면 태양 사고: {y_count.sum()}건\n")

# =========================================================================
# 1. A-Rule 노선 군집 Bootstrap 2,000회 신뢰구간 산출
# =========================================================================
print("--- [1. A-Rule 노선 군집 Bootstrap 2,000회 신뢰구간] ---")
np.random.seed(42)
n_bootstraps = 2000
cluster_indices = {r: np.where(groups == r)[0] for r in routes}

boot_rocs = []
boot_recalls = []

raw_glare = df['annual_glare_hours_fwd'].values

for _ in range(n_bootstraps):
    sampled_routes = np.random.choice(routes, size=n_routes, replace=True)
    sample_idx = np.concatenate([cluster_indices[r] for r in sampled_routes])
    
    y_s = y_bin[sample_idx]
    yc_s = y_count[sample_idx]
    if y_s.sum() == 0 or y_s.sum() == len(y_s) or yc_s.sum() == 0:
        continue
        
    roc = roc_auc_score(y_s, raw_glare[sample_idx])
    boot_rocs.append(roc)
    
    k_s = int(len(sample_idx) * 0.1)
    top_idx = np.argsort(raw_glare[sample_idx])[-k_s:]
    rec = yc_s[top_idx].sum() / yc_s.sum() * 100
    boot_recalls.append(rec)

boot_rocs = np.array(boot_rocs)
boot_recalls = np.array(boot_recalls)

ci_roc = np.percentile(boot_rocs, [2.5, 97.5])
ci_rec = np.percentile(boot_recalls, [2.5, 97.5])

point_roc = roc_auc_score(y_bin, raw_glare)
k_all = int(len(df) * 0.1)
point_rec = y_count[np.argsort(raw_glare)[-k_all:]].sum() / y_count.sum() * 100

print(f"• A-Rule ROC-AUC:")
print(f"  점추정치: {point_roc:.4f} | 부트스트랩 평균: {np.mean(boot_rocs):.4f} | 95% CI: [{ci_roc[0]:.4f}, {ci_roc[1]:.4f}]")
print(f"• A-Rule 상위 10% 사고 포착률 (Top 10% Recall):")
print(f"  점추정치: {point_rec:.1f}% | 부트스트랩 평균: {np.mean(boot_recalls):.1f}% | 95% CI: [{ci_rec[0]:.1f}%, {ci_rec[1]:.1f}%]\n")

# =========================================================================
# 2. A-GLM 5-Fold OOF 예측값의 Calibration Plot (분위수별 관측치 vs 예측치)
# =========================================================================
print("--- [2. A-GLM 5-Fold OOF 캘리브레이션 분석 (예측 사고건수 vs 실제 사고건수)] ---")
gkf = GroupKFold(n_splits=5)
oof_glm = np.zeros(len(df))

for tr, te in gkf.split(df, y_count, groups):
    glm = PoissonRegressor(alpha=1.0)
    glm.fit(df.iloc[tr][['annual_glare_hours_fwd']], y_count[tr])
    oof_glm[te] = glm.predict(df.iloc[te][['annual_glare_hours_fwd']])

df['oof_pred_glm'] = oof_glm

# 예측값 기준 5개 구간(Quintiles) 분할
df['pred_bin'] = pd.qcut(df['oof_pred_glm'], q=5, duplicates='drop')
calib = df.groupby('pred_bin', observed=True).agg(
    구간수=('oof_pred_glm', 'count'),
    평균예측건수=('oof_pred_glm', 'mean'),
    실제평균건수=('glare_front_count', 'mean'),
    총실제건수=('glare_front_count', 'sum')
).reset_index()

calib['오차(실제-예측)'] = calib['실제평균건수'] - calib['평균예측건수']
calib['비율(실제/예측)'] = calib['실제평균건수'] / np.maximum(calib['평균예측건수'], 1e-6)

print(f"{'A-GLM 예측 구간 (5분위)':<25} | {'구간수':<6} | {'평균 예측(건)':<12} | {'실제 관측(건)':<12} | {'오차':<10} | {'실제/예측 비율':<10}")
print("-" * 85)
for _, r in calib.iterrows():
    print(f"{str(r['pred_bin']):<25} | {r['구간수']:<6} | {r['평균예측건수']:.5f}      | {r['실제평균건수']:.5f}      | {r['오차(실제-예측)']:+.5f}  | {r['비율(실제/예측)']:5.2f}배")

# 회귀 기반 Calibration Slope (실제 ~ log(예측))
calib_model = smf.poisson('glare_front_count ~ np.log(oof_pred_glm)', data=df).fit(disp=0)
slope = calib_model.params['np.log(oof_pred_glm)']
slope_ci = calib_model.conf_int().loc['np.log(oof_pred_glm)']
print(f"\n• Calibration Slope (이상값 1.0): {slope:.3f} (95% CI: [{slope_ci[0]:.3f}, {slope_ci[1]:.3f}])\n")

# =========================================================================
# 3. 연간 정면 노출시간 분위수별 실제 사고율 추세 검정 (포아송 추세 회귀)
# =========================================================================
print("--- [3. 연간 정면 노출시간 분위수별 사고율 추세 검정 (구간당 건수 및 VKT당 사고율)] ---")
df['glare_quintile'] = pd.qcut(df['annual_glare_hours_fwd'], q=5, duplicates='drop')

trend_table = df.groupby('glare_quintile', observed=True).agg(
    구간수=('annual_glare_hours_fwd', 'count'),
    평균노출시간=('annual_glare_hours_fwd', 'mean'),
    총연장_km=('segment_length_km', 'sum'),
    총사고건수=('glare_front_count', 'sum'),
    사고발생구간수=('glare_front_count', lambda x: (x > 0).sum()),
    총VKT_억대km=('vkt_100m_3yr', 'sum')
).reset_index()

trend_table['구간당사고건수'] = trend_table['총사고건수'] / trend_table['구간수']
trend_table['100만VKT당사고율'] = (trend_table['총사고건수'] / (trend_table['총VKT_억대km'] * 100))  # 100만대km당
trend_table['양성구간비율(%)'] = (trend_table['사고발생구간수'] / trend_table['구간수']) * 100

print(f"{'노출시간 5분위':<22} | {'평균노출':<8} | {'사고수':<6} | {'구간당사고':<12} | {'100만VKT당사고율':<16} | {'사고구간비율':<10}")
print("-" * 85)
for _, r in trend_table.iterrows():
    print(f"{str(r['glare_quintile']):<22} | {r['평균노출시간']:6.1f}h | {r['총사고건수']:4d}건 | {r['구간당사고건수']:.5f}건/km | {r['100만VKT당사고율']:.5f}건/100만VKT | {r['양성구간비율(%)']:5.2f}%")

# 통계적 추세 검정 (Poisson Trend Regression: 노출시간 100시간 증가당 사고 발생률비 Rate Ratio)
df['glare_hours_100h'] = df['annual_glare_hours_fwd'] / 100.0

# VKT 노출량 오프셋 반영 포아송 추세 검정
trend_fit = smf.poisson('glare_front_count ~ glare_hours_100h', data=df, offset=np.log(df['vkt_100m_3yr'])).fit(disp=0)
rr = np.exp(trend_fit.params['glare_hours_100h'])
rr_ci = np.exp(trend_fit.conf_int().loc['glare_hours_100h'])
p_trend = trend_fit.pvalues['glare_hours_100h']

print(f"\n• 교통 노출량(VKT) 보정 포아송 추세 검정 결과:")
print(f"  연간 정면 노출시간 100시간 증가당 사고율비 (Rate Ratio): {rr:.3f}배 (95% CI: [{rr_ci[0]:.3f}, {rr_ci[1]:.3f}]) | p-value: {p_trend:.4e}")
