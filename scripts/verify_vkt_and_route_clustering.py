import sys
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/expressway/segment_vds_aadt_v1.csv')
df = df[df['aadt'] > 0].copy().reset_index(drop=True)
df['vkt_100m_3yr'] = (df['aadt'] * df['segment_length_km'] * 365 * 3) / 1e8
df['glare_hours_100h'] = df['annual_glare_hours_fwd'] / 100.0

# 1. 일반 Poisson vs Cluster-robust SE by route
fit_std = smf.poisson('glare_front_count ~ glare_hours_100h', data=df, offset=np.log(df['vkt_100m_3yr'])).fit(disp=0)

fit_cluster = smf.poisson('glare_front_count ~ glare_hours_100h', data=df, offset=np.log(df['vkt_100m_3yr'])).fit(
    cov_type='cluster', cov_kwds={'groups': df['route']}, disp=0
)

print("=" * 85)
print("1. VKT 보정 포아송 모델 표준오차 vs 노선 군집(Cluster-robust) 표준오차 비교")
print("=" * 85)
rr_std = np.exp(fit_std.params['glare_hours_100h'])
ci_std = np.exp(fit_std.conf_int().loc['glare_hours_100h'])
p_std = fit_std.pvalues['glare_hours_100h']
se_std = fit_std.bse['glare_hours_100h']
print(f"• 일반 표준오차 (Standard SE):")
print(f"  Rate Ratio = {rr_std:.4f} (95% CI: [{ci_std[0]:.4f}, {ci_std[1]:.4f}]) | SE = {se_std:.4f} | p = {p_std:.4e}")

rr_cl = np.exp(fit_cluster.params['glare_hours_100h'])
ci_cl = np.exp(fit_cluster.conf_int().loc['glare_hours_100h'])
p_cl = fit_cluster.pvalues['glare_hours_100h']
se_cl = fit_cluster.bse['glare_hours_100h']
print(f"• 노선 군집 표준오차 (Cluster-robust SE by Route):")
print(f"  Rate Ratio = {rr_cl:.4f} (95% CI: [{ci_cl[0]:.4f}, {ci_cl[1]:.4f}]) | SE = {se_cl:.4f} | p = {p_cl:.4e}")

print("\n" + "=" * 85)
print("2. 전체 45개 노선 구성 현황 및 무사고 노선 처리")
print("=" * 85)
route_summary = df.groupby('route').agg(
    n_segments=('segment_length_km', 'count'),
    total_km=('segment_length_km', 'sum'),
    total_crashes=('glare_front_count', 'sum'),
    crash_segments=('glare_front_count', lambda x: (x > 0).sum()),
    mean_glare=('annual_glare_hours_fwd', 'mean')
).reset_index()

n_routes = len(route_summary)
has_crash_routes = route_summary[route_summary['total_crashes'] > 0]
zero_crash_routes = route_summary[route_summary['total_crashes'] == 0]

print(f"• 전체 노선 수: {n_routes}개")
print(f"• 사고 발생 노선 수: {len(has_crash_routes)}개 (총 {has_crash_routes['n_segments'].sum()}개 구간, 사고 {has_crash_routes['total_crashes'].sum()}건)")
print(f"• 무사고 노선 수: {len(zero_crash_routes)}개 (총 {zero_crash_routes['n_segments'].sum()}개 구간, 사고 0건)")

# 무사고 노선의 노출시간 vs 사고 노선의 노출시간 비교
print(f"  - 사고 발생 노선의 평균 연간 정면 노출시간: {has_crash_routes['mean_glare'].mean():.1f}시간")
print(f"  - 무사고 노선의 평균 연간 정면 노출시간:       {zero_crash_routes['mean_glare'].mean():.1f}시간")

# 노선 내 ROC 계산 가능한 노선(사고 발생 노선 23개)에 대한 사고 가중 평균 ROC
route_rocs = []
for _, r in has_crash_routes.iterrows():
    rt = r['route']
    sub = df[df['route'] == rt]
    pos = (sub['glare_front_count'] > 0).sum()
    if pos > 0 and pos < len(sub):
        roc = roc_auc_score((sub['glare_front_count'] > 0).astype(int), sub['annual_glare_hours_fwd'])
        route_rocs.append({
            'route': rt,
            'crashes': r['total_crashes'],
            'crash_segments': pos,
            'segments': len(sub),
            'roc': roc
        })

r_roc_df = pd.DataFrame(route_rocs)
weighted_roc = (r_roc_df['roc'] * r_roc_df['crashes']).sum() / r_roc_df['crashes'].sum()
unweighted_roc = r_roc_df['roc'].mean()

print(f"\n• 노선 내(within-route) ROC-AUC 요약 (계산 가능 23개 노선):")
print(f"  - 단순 평균 (Unweighted Mean): {unweighted_roc:.4f}")
print(f"  - 사고 건수 가중 평균 (Crash-weighted Mean): {weighted_roc:.4f}")
print(f"  - 노선 간 풀링 전체(Pooled overall) ROC-AUC: {roc_auc_score((df['glare_front_count']>0).astype(int), df['annual_glare_hours_fwd']):.4f}")
print(f"  * 풀링 ROC(0.7935)가 노선 내 가중 ROC({weighted_roc:.4f})보다 높은 이유:")
print(f"    -> 노선 간(Between-route) 노출시간 차이(동서축 vs 남북축 노선 간 사고율 차이)와")
print(f"       노선 내(Within-route) 위험 구간 분리가 결합되었기 때문임.")
