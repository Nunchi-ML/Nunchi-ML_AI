import sys
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/expressway/segment_vds_aadt_v1.csv')
df = df[df['aadt'] > 0].copy().reset_index(drop=True)
y_count = df['glare_front_count'].values
groups = df['route'].values

gkf = GroupKFold(n_splits=5)
oof = np.zeros(len(df))
for tr, te in gkf.split(df, y_count, groups):
    glm = PoissonRegressor(alpha=1.0)
    glm.fit(df.iloc[tr][['annual_glare_hours_fwd']], y_count[tr])
    oof[te] = glm.predict(df.iloc[te][['annual_glare_hours_fwd']])

df['oof'] = oof
df['pred_bin'] = pd.qcut(df['oof'], q=5, duplicates='drop')

print("=" * 90)
print("1. A-GLM 5분위별 관측치 vs 예측치 및 포아송 신뢰구간 (Exact Poisson Count 95% CI)")
print("=" * 90)
for name, group in df.groupby('pred_bin', observed=True):
    n_seg = len(group)
    pred_sum = group['oof'].sum()
    obs_sum = int(group['glare_front_count'].sum())
    pred_mean = group['oof'].mean()
    obs_mean = group['glare_front_count'].mean()
    
    ci_low = 0.5 * stats.chi2.ppf(0.025, 2 * obs_sum) if obs_sum > 0 else 0.0
    ci_high = 0.5 * stats.chi2.ppf(0.975, 2 * (obs_sum + 1))
    ratio = obs_sum / pred_sum if pred_sum > 0 else 0.0
    ratio_low = ci_low / pred_sum if pred_sum > 0 else 0.0
    ratio_high = ci_high / pred_sum if pred_sum > 0 else 0.0
    
    print(f"{str(name):<28} | N={n_seg:4d} | 예측합: {pred_sum:5.1f}건 | 관측: {obs_sum:2d}건 (95% CI: [{ci_low:4.1f}, {ci_high:4.1f}]) | 관측/예측: {ratio:4.2f}배 [{ratio_low:4.2f}, {ratio_high:4.2f}]")

print("\n" + "=" * 90)
print("2. Calibration 회귀분석 (절편 Intercept + 기울기 Slope)")
print("=" * 90)
calib = smf.poisson('glare_front_count ~ np.log(oof)', data=df).fit(disp=0)
print(calib.summary().tables[1])

print("• 절편 (Intercept, 이상치 0.0):", f"{calib.params['Intercept']:.4f}", f"(95% CI: [{calib.conf_int().loc['Intercept', 0]:.4f}, {calib.conf_int().loc['Intercept', 1]:.4f}])")
print("• 기울기 (Slope, 이상치 1.0):    ", f"{calib.params['np.log(oof)']:.4f}", f"(95% CI: [{calib.conf_int().loc['np.log(oof)', 0]:.4f}, {calib.conf_int().loc['np.log(oof)', 1]:.4f}])")
print(f"• 전체 합계: 관측 {y_count.sum()}건 vs 예측 {oof.sum():.2f}건 (전체 비율: {y_count.sum()/oof.sum():.2f}배)")

print("\n" + "=" * 90)
print("3. 주요 노선별 A-Rule ROC-AUC 및 사고 점유율 (특정 노선 쏠림 여부 검증)")
print("=" * 90)
route_stats = []
for r, grp in df.groupby('route'):
    c = grp['glare_front_count'].sum()
    pos = (grp['glare_front_count'] > 0).sum()
    n = len(grp)
    if pos > 0 and pos < n:
        roc = roc_auc_score((grp['glare_front_count'] > 0).astype(int), grp['annual_glare_hours_fwd'])
    else:
        roc = np.nan
    route_stats.append({'route': r, 'segments': n, 'crashes': c, 'crash_segs': pos, 'roc': roc})

r_df = pd.DataFrame(route_stats).sort_values(by='crashes', ascending=False)
for _, row in r_df.head(10).iterrows():
    roc_str = f"{row['roc']:.4f}" if not np.isnan(row['roc']) else "N/A"
    pct_crash = row['crashes'] / y_count.sum() * 100
    print(f"• {row['route']:<18} | 구간수: {row['segments']:4d} | 사고: {int(row['crashes']):2d}건 ({pct_crash:4.1f}%) | 사고구간: {int(row['crash_segs']):2d}개 | 노선 내 ROC-AUC: {roc_str}")
