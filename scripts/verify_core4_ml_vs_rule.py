import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, mean_poisson_deviance
from sklearn.linear_model import PoissonRegressor

sys.stdout.reconfigure(encoding='utf-8')

# 1. 동일한 데이터셋 및 코호트 정의 (4,984개 구간, 정면 태양 사고 90건)
df = pd.read_csv('data/expressway/segment_vds_aadt_v1.csv')
df = df[df['aadt'] > 0].copy().reset_index(drop=True)

df['glare_100h'] = df['annual_glare_hours_fwd'] / 100.0
df['log_aadt'] = np.log(df['aadt'])
df['curv_log'] = np.log1p(df['curvature_mean'] * 100.0)

CORE_4_FEATURES = ['glare_100h', 'log_aadt', 'curv_log', 'has_tunnel_within_1km']

y_count = df['glare_front_count'].values
y_bin = (y_count > 0).astype(int)
groups = df['route'].values
rule_vals = df['annual_glare_hours_fwd'].values

gkf = GroupKFold(n_splits=5)
k_global = int(len(df) * 0.1)  # 498개 구간

oof_ml_raw = np.zeros(len(df))
oof_ml_prior_adj = np.zeros(len(df))
oof_ml_rank = np.zeros(len(df))
oof_rule_rank = np.zeros(len(df))

fold_rows = []

for fold, (tr, te) in enumerate(gkf.split(df, y_count, groups)):
    m = PoissonRegressor(alpha=1.0)
    m.fit(df.iloc[tr][CORE_4_FEATURES], y_count[tr])
    
    pred_te = m.predict(df.iloc[te][CORE_4_FEATURES])
    pred_tr_mean = np.mean(m.predict(df.iloc[tr][CORE_4_FEATURES]))
    
    oof_ml_raw[te] = pred_te
    oof_ml_prior_adj[te] = pred_te / pred_tr_mean * np.mean(y_count)
    oof_ml_rank[te] = pd.Series(pred_te).rank(pct=True).values
    oof_rule_rank[te] = pd.Series(rule_vals[te]).rank(pct=True).values
    
    kf = int(len(te) * 0.1)
    crashes_fold = y_count[te].sum()
    
    rule_cap = y_count[te][np.argsort(rule_vals[te])[-kf:]].sum() / crashes_fold * 100.0
    ml_cap = y_count[te][np.argsort(pred_te)[-kf:]].sum() / crashes_fold * 100.0
    
    rule_roc = roc_auc_score(y_bin[te], rule_vals[te])
    ml_roc = roc_auc_score(y_bin[te], pred_te)
    
    fold_rows.append({
        'fold': fold,
        'n_seg': len(te),
        'crashes': crashes_fold,
        'train_rate': y_count[tr].mean(),
        'test_rate': y_count[te].mean(),
        'rule_top10': rule_cap,
        'ml_top10': ml_cap,
        'rule_roc': rule_roc,
        'ml_roc': ml_roc,
    })

fdf = pd.DataFrame(fold_rows)

print("=" * 90)
print("🔬 [동일 조건 비교] A-Rule (단일 노출시간) vs Core-4 ML (노출+교통량+곡률+터널)")
print("=" * 90)
print(f"데이터 코호트: {len(df):,}개 구간 | 노선: {df['route'].nunique()}개 | 정면 태양 사고: {y_count.sum()}건 (양성 구간 {y_bin.sum()}개)")
print(f"교차검증 분할: 5-Fold Route GroupKFold | Global Top 10% k = {k_global}개 구간\n")

print("--- [1. Fold별 상세 비교 (Fold 내부 Top 10% 포착률 및 ROC-AUC)] ---")
for _, r in fdf.iterrows():
    print(
        f"Fold {int(r['fold'])} (구간={int(r['n_seg']):4d}, 사고={int(r['crashes']):2d}건, "
        f"Train기저율={r['train_rate']:.4f}, Test기저율={r['test_rate']:.4f}) | "
        f"Top10: Rule={r['rule_top10']:5.1f}% vs ML={r['ml_top10']:5.1f}% | "
        f"ROC: Rule={r['rule_roc']:.4f} vs ML={r['ml_roc']:.4f}"
    )

# 3가지 평가 기준 통합 집계
rule_global_raw_top10 = y_count[np.argsort(rule_vals)[-k_global:]].sum() / y_count.sum() * 100.0
ml_global_raw_top10 = y_count[np.argsort(oof_ml_raw)[-k_global:]].sum() / y_count.sum() * 100.0
ml_global_adj_top10 = y_count[np.argsort(oof_ml_prior_adj)[-k_global:]].sum() / y_count.sum() * 100.0

rule_rank_top10 = y_count[np.argsort(oof_rule_rank)[-k_global:]].sum() / y_count.sum() * 100.0
ml_rank_top10 = y_count[np.argsort(oof_ml_rank)[-k_global:]].sum() / y_count.sum() * 100.0

print("\n--- [2. 평가 기준별 Top 10% 사고 포착률 및 ROC-AUC 요약] ---")
print(f"기준 ① Fold 내부 평균 (Within-Fold Mean)       : Rule = {fdf['rule_top10'].mean():5.1f}% (ROC {fdf['rule_roc'].mean():.4f}) | Core-4 ML = {fdf['ml_top10'].mean():5.1f}% (ROC {fdf['ml_roc'].mean():.4f})")
print(f"기준 ② Fold 순위 정규화 합산 (Rank-Normalized) : Rule = {rule_rank_top10:5.1f}% (ROC {roc_auc_score(y_bin, oof_rule_rank):.4f}) | Core-4 ML = {ml_rank_top10:5.1f}% (ROC {roc_auc_score(y_bin, oof_ml_rank):.4f})")
print(f"기준 ③ Global Raw OOF 합산 (비정규화 원시 합산): Rule = {rule_global_raw_top10:5.1f}% (ROC {roc_auc_score(y_bin, rule_vals):.4f}) | Core-4 ML = {ml_global_raw_top10:5.1f}% (ROC {roc_auc_score(y_bin, oof_ml_raw):.4f})")
print(f"기준 ④ 학습 기저율 보정 OOF (Prior-Adjusted)   : Rule = {rule_global_raw_top10:5.1f}% (ROC {roc_auc_score(y_bin, rule_vals):.4f}) | Core-4 ML = {ml_global_adj_top10:5.1f}% (ROC {roc_auc_score(y_bin, oof_ml_prior_adj):.4f})")
