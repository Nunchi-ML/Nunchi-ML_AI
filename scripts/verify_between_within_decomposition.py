import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/expressway/segment_vds_aadt_v1.csv')
df = df[df['aadt'] > 0].copy().reset_index(drop=True)
y_bin = (df['glare_front_count'] > 0).astype(int)

# 1. 28개 사고 노선별 ROC 계산 가능 여부 전수 점검
route_summary = df.groupby('route').agg(
    n_segments=('segment_length_km', 'count'),
    total_crashes=('glare_front_count', 'sum'),
    pos_segments=('glare_front_count', lambda x: (x > 0).sum())
).reset_index()

crash_routes = route_summary[route_summary['total_crashes'] > 0].reset_index(drop=True)
print("=== 1. 사고 발생 노선 28개 중 노선 내 ROC 계산 가능 여부 점검 ===")
valid_rocs = []
excluded_routes = []

for _, r in crash_routes.iterrows():
    rt = r['route']
    sub = df[df['route'] == rt]
    pos = (sub['glare_front_count'] > 0).sum()
    n = len(sub)
    y_sub = (sub['glare_front_count'] > 0).astype(int).values
    x_sub = sub['annual_glare_hours_fwd'].values
    
    if pos == 0 or pos == n:
        excluded_routes.append((rt, n, pos, "양성 또는 음성만 존재 (ROC 정의 불가)"))
    elif len(np.unique(x_sub)) < 2:
        excluded_routes.append((rt, n, pos, "노출시간 값이 단일값(변동 없음)"))
    else:
        try:
            roc = roc_auc_score(y_sub, x_sub)
            valid_rocs.append({
                'route': rt,
                'n_segments': n,
                'crashes': r['total_crashes'],
                'pos_segments': pos,
                'roc': roc
            })
        except Exception as e:
            excluded_routes.append((rt, n, pos, str(e)))

print(f"• 계산 성공 노선 수: {len(valid_rocs)}개")
print(f"• 제외된 노선 수: {len(excluded_routes)}개")
for ex in excluded_routes:
    print(f"  - 제외: [{ex[0]}] 구간수={ex[1]}, 사고구간={ex[2]} | 사유: {ex[3]}")

# 2. Between vs Within 변별력 엄밀 분해
route_mean_glare = df.groupby('route')['annual_glare_hours_fwd'].transform('mean')
within_glare = df['annual_glare_hours_fwd'] - route_mean_glare

roc_pooled = roc_auc_score(y_bin, df['annual_glare_hours_fwd'])
roc_between = roc_auc_score(y_bin, route_mean_glare)
roc_within = roc_auc_score(y_bin, within_glare)

v_df = pd.DataFrame(valid_rocs)
unweighted_mean = v_df['roc'].mean()
weighted_mean = (v_df['roc'] * v_df['crashes']).sum() / v_df['crashes'].sum()

print("\n=== 2. A-Rule Between vs Within 변별력 엄밀 분해 결과 ===")
print(f"1) Raw A-Rule (전체 연간 정면 노출시간):         ROC-AUC = {roc_pooled:.4f}")
print(f"2) Between-Route Only (노선별 평균 노출시간만):   ROC-AUC = {roc_between:.4f}")
print(f"3) Within-Route Only (노선 내 중심화 노출시간만): ROC-AUC = {roc_within:.4f}")
print(f"4) 개별 노선 내 ROC 단순 평균:                  ROC-AUC = {unweighted_mean:.4f}")
print(f"5) 개별 노선 내 ROC 사고건수 가중 평균:         ROC-AUC = {weighted_mean:.4f}")
