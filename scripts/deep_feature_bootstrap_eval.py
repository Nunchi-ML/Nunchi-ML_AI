import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_poisson_deviance
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

sys.stdout.reconfigure(encoding='utf-8')

# 데이터 로드
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

y_count = df['glare_front_count'].values
y_bin = (y_count > 0).astype(int)
groups = df['route'].values
routes = df['route'].unique()

print("=" * 80)
print("🔬 모델 ① 피처 기여도 분석, 학습-검증 격차(과적합) 및 Paired Bootstrap 검증")
print("=" * 80)
print(f"전체 구간 수: {len(df):,}개 | 노선 수: {len(routes)}개 | 사고 구간(양성): {y_bin.sum()}개 ({y_bin.sum()/len(df)*100:.2f}%)")

gkf = GroupKFold(n_splits=5)

# 실험할 모델 구성
models_config = {
    "LGBM_Binary_All12": (ALL_FEATURES, "binary"),
    "LGBM_Binary_Solar4": (SOLAR_FEATURES, "binary"),
    "LGBM_Poisson_All12": (ALL_FEATURES, "poisson"),
    "LGBM_Poisson_Solar4": (SOLAR_FEATURES, "poisson"),
}

results = {}

for name, (feats, mtype) in models_config.items():
    oof_preds = np.zeros(len(df))
    train_preds_list = []
    train_y_list = []
    
    for tr, te in gkf.split(df, y_bin, groups):
        if mtype == "binary":
            m = lgb.LGBMClassifier(
                n_estimators=80, learning_rate=0.03, num_leaves=7, max_depth=3,
                min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=5.0, random_state=42, verbose=-1
            )
            m.fit(df.iloc[tr][feats], y_bin[tr])
            oof_preds[te] = m.predict_proba(df.iloc[te][feats])[:, 1]
            train_preds_list.append(m.predict_proba(df.iloc[tr][feats])[:, 1])
            train_y_list.append(y_bin[tr])
        else:
            m = lgb.LGBMRegressor(
                objective='poisson', n_estimators=80, learning_rate=0.03, num_leaves=7, max_depth=3,
                min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=5.0, random_state=42, verbose=-1
            )
            m.fit(df.iloc[tr][feats], y_count[tr])
            oof_preds[te] = m.predict(df.iloc[te][feats])
            train_preds_list.append(m.predict(df.iloc[tr][feats]))
            train_y_list.append(y_bin[tr])
            
    # 전체 Train 점수 (평균)
    train_rocs = [roc_auc_score(ty, tp) for ty, tp in zip(train_y_list, train_preds_list)]
    train_prs = [average_precision_score(ty, tp) for ty, tp in zip(train_y_list, train_preds_list)]
    
    # OOF 점수
    oof_roc = roc_auc_score(y_bin, oof_preds)
    oof_pr = average_precision_score(y_bin, oof_preds)
    
    results[name] = {
        "oof_preds": oof_preds,
        "train_roc": np.mean(train_rocs),
        "oof_roc": oof_roc,
        "train_pr": np.mean(train_prs),
        "oof_pr": oof_pr,
        "mtype": mtype,
    }

# 단순 규칙
rule_preds = df['annual_glare_hours_fwd'].values
results["Rule_annual_glare"] = {
    "oof_preds": rule_preds,
    "train_roc": roc_auc_score(y_bin, rule_preds),
    "oof_roc": roc_auc_score(y_bin, rule_preds),
    "train_pr": average_precision_score(y_bin, rule_preds),
    "oof_pr": average_precision_score(y_bin, rule_preds),
    "mtype": "rule",
}

print("\n" + "=" * 80)
print("1. 모델별 학습(Train) vs OOF(일반화) 성능 비교 (과적합 검증)")
print("=" * 80)
print(f"{'모델명':<24} | {'Train ROC':<10} | {'OOF ROC':<10} | {'ROC Gap':<8} | {'Train PR':<10} | {'OOF PR':<10} | {'PR Gap':<8}")
print("-" * 80)

for name in ["Rule_annual_glare", "LGBM_Binary_Solar4", "LGBM_Binary_All12", "LGBM_Poisson_Solar4", "LGBM_Poisson_All12"]:
    r = results[name]
    roc_gap = r['train_roc'] - r['oof_roc']
    pr_gap = r['train_pr'] - r['oof_pr']
    print(f"{name:<24} | {r['train_roc']:.4f}     | {r['oof_roc']:.4f}   | {roc_gap:+.4f}  | {r['train_pr']:.4f}     | {r['oof_pr']:.4f}   | {pr_gap:+.4f}")

print("\n" + "=" * 80)
print("2. Paired Bootstrap 불확실성 검증 (노선 군집 2,000회 부트스트랩)")
print("=" * 80)
print("단순 규칙 대비 차이 (Rule - Model) : 양수면 단순 규칙 우위, 95% CI가 0을 미포함 시 유의미")

np.random.seed(42)
n_bootstraps = 2000
unique_routes = np.unique(groups)
n_routes = len(unique_routes)

# 노선 단위 클러스터 부트스트랩 인덱스 사전 생성
cluster_indices = {r: np.where(groups == r)[0] for r in unique_routes}

comparisons = [
    ("LGBM_Binary_All12", "단순 규칙 vs Binary 12개 피처"),
    ("LGBM_Binary_Solar4", "단순 규칙 vs Binary 태양 4개 피처"),
    ("LGBM_Poisson_All12", "단순 규칙 vs Poisson 12개 피처"),
    ("LGBM_Poisson_Solar4", "단순 규칙 vs Poisson 태양 4개 피처"),
]

for model_key, label in comparisons:
    diff_rocs = []
    diff_prs = []
    
    pred_rule = results["Rule_annual_glare"]["oof_preds"]
    pred_m = results[model_key]["oof_preds"]
    
    for _ in range(n_bootstraps):
        sampled_routes = np.random.choice(unique_routes, size=n_routes, replace=True)
        sample_idx = np.concatenate([cluster_indices[r] for r in sampled_routes])
        
        y_samp = y_bin[sample_idx]
        if y_samp.sum() == 0 or y_samp.sum() == len(y_samp):
            continue
            
        r_roc = roc_auc_score(y_samp, pred_rule[sample_idx])
        m_roc = roc_auc_score(y_samp, pred_m[sample_idx])
        diff_rocs.append(r_roc - m_roc)
        
        r_pr = average_precision_score(y_samp, pred_rule[sample_idx])
        m_pr = average_precision_score(y_samp, pred_m[sample_idx])
        diff_prs.append(r_pr - m_pr)
        
    diff_rocs = np.array(diff_rocs)
    diff_prs = np.array(diff_prs)
    
    ci_roc = np.percentile(diff_rocs, [2.5, 97.5])
    p_val_roc = np.mean(diff_rocs <= 0) if np.mean(diff_rocs) > 0 else np.mean(diff_rocs >= 0)
    
    ci_pr = np.percentile(diff_prs, [2.5, 97.5])
    p_val_pr = np.mean(diff_prs <= 0) if np.mean(diff_prs) > 0 else np.mean(diff_prs >= 0)
    
    print(f"\n[{label}]")
    print(f"  * ROC-AUC 차이 (Rule - M): 평균 {np.mean(diff_rocs):+.4f} | 95% CI: [{ci_roc[0]:+.4f}, {ci_roc[1]:+.4f}] | p-value: {p_val_roc:.4f}")
    print(f"  * PR-AUC 차이  (Rule - M): 평균 {np.mean(diff_prs):+.4f} | 95% CI: [{ci_pr[0]:+.4f}, {ci_pr[1]:+.4f}] | p-value: {p_val_pr:.4f}")
