"""Fair Comparison Experiment: Base Model vs Poisson Offset Model.

Using the exact same identical cohort (3,304 mapped segments) and 5-Fold GroupKFold.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from sklearn.metrics import mean_poisson_deviance, mean_absolute_error

# 1. Load Segments and VDS AADT
REPO_ROOT = Path(__file__).resolve().parents[3]
seg_path = REPO_ROOT / "data" / "expressway" / "segment_1km_dataset_v1.csv"
aadt_path = REPO_ROOT / "data" / "expressway" / "vds_aadt_2023_2025.csv"

seg_df = pd.read_csv(seg_path)
vds_df = pd.read_csv(aadt_path)
vids = vds_df["vds_id"].astype(str)
vds_df["code"] = vids.str[:4]
vds_df["vds_mp"] = vids.str[7:].astype(float) / 100.0


# 2. VDS Mapping with origin normalization
def get_vds_matches(route_name, seg_km, s_mp, e_mp):
    cands = []
    if route_name == "경부선":
        cands.append(("0010", s_mp, e_mp))
    elif route_name == "서해안선":
        cands.append(("0150", s_mp, e_mp))
    elif route_name == "중부내륙선":
        cands.append(("0450", s_mp, e_mp))
    elif route_name == "영동선":
        cands.append(("0500", s_mp, e_mp))
    elif route_name == "중앙선":
        cands.append(("0550", s_mp, e_mp))
    elif route_name == "호남선":
        cands.append(("0251", s_mp, e_mp))
    elif route_name == "통영대전선":
        cands.append(("0351", s_mp, e_mp))
    elif route_name == "중부선":
        cands.append(("0352", s_mp - 215.0, e_mp - 215.0))
    elif route_name == "통영대전선/중부선":
        if seg_km < 215:
            cands.append(("0351", s_mp, e_mp))
        else:
            cands.append(("0352", s_mp - 215.0, e_mp - 215.0))
    elif route_name == "논산천안선/호남선":
        cands.append(("0251", s_mp, e_mp))
    elif route_name == "논산천안선":
        cands.append(("0252", s_mp, e_mp))
    elif "남해선" in route_name:
        cands.append(("0100", s_mp, e_mp))
        cands.append(("0101", s_mp, e_mp))
    elif route_name == "광주대구선":
        cands.append(("0120", s_mp, e_mp))
    elif route_name == "평택제천선":
        cands.append(("0175", s_mp, e_mp))
    elif "서산영덕" in route_name:
        cands.append(("0300", s_mp, e_mp))
        cands.append(("0301", s_mp, e_mp))
    elif route_name == "서울양양선":
        cands.append(("0600", s_mp, e_mp))
    elif "수도권제1순환선" in route_name:
        cands.append(("1000", s_mp, e_mp))
    elif route_name == "순천완주선":
        cands.append(("0270", s_mp, e_mp))
    elif "동해선" in route_name:
        cands.append(("0650", s_mp, e_mp))
        cands.append(("0653", s_mp, e_mp))
    return cands


mapped_aadt = []
mapped_mask = []

for _, s in seg_df.iterrows():
    cands = get_vds_matches(s["route"], s["segment_km"], s["start_mp"], s["end_mp"])
    vals = []
    for code, smp, emp in cands:
        sub = vds_df[vds_df["code"] == code]
        pts = sub[(sub["vds_mp"] >= smp - 0.2) & (sub["vds_mp"] <= emp + 0.2)]
        if len(pts) > 0:
            vals.append(pts["aadt"].mean())
    if vals:
        mapped_aadt.append(float(np.mean(vals)))
        mapped_mask.append(True)
    else:
        mapped_aadt.append(np.nan)
        mapped_mask.append(False)

seg_df["aadt"] = mapped_aadt
seg_df["tunnel_glare_interaction"] = seg_df["has_tunnel_within_1km"] * seg_df["commute_glare_hours"]

# Filter identical cohort
df = seg_df[mapped_mask].copy().reset_index(drop=True)
print(f"Identical Cohort Size: {len(df)} segments | Routes: {df['route'].nunique()}")
print(f"Total Target Accidents (glare_front_count): {df['glare_front_count'].sum():.0f} 건")

# Calculate Exposure E_i = AADT * length * 365 * 3
df["exposure_vkt"] = df["aadt"] * df["segment_length_km"] * 365.0 * 3.0
# Log offset
df["log_exposure"] = np.log(np.maximum(df["exposure_vkt"], 1000.0))

features = [
    "segment_length_km",
    "is_partial_segment",
    "curvature_mean",
    "curvature_max",
    "min_tunnel_dist_km",
    "has_tunnel_within_1km",
    "tunnel_portals_1km",
    "max_tunnel_length_1km",
    "ew_alignment",
    "annual_glare_hours_fwd",
    "commute_glare_hours",
    "tunnel_glare_interaction",
]

X = df[features]
y = df["glare_front_count"].to_numpy()
groups = df["route"].to_numpy()
offset = df["log_exposure"].to_numpy()

# 5-Fold GroupKFold
gkf = GroupKFold(n_splits=5)

base_dev_list, base_mae_list, base_rec_list = [], [], []
offset_dev_list, offset_mae_list, offset_rec_list = [], [], []

base_oof_preds = np.zeros(len(df))
offset_oof_preds = np.zeros(len(df))

params = {
    "objective": "poisson",
    "learning_rate": 0.03,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 20,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "n_estimators": 80,
    "verbose": -1,
    "random_state": 42,
}

for fold, (trn_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
    X_trn, y_trn = X.iloc[trn_idx], y[trn_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    # --- 1. Base Model (Without Offset) ---
    m_base = lgb.LGBMRegressor(**params)
    m_base.fit(X_trn, y_trn)
    pred_base = np.clip(m_base.predict(X_val), 1e-6, None)
    base_oof_preds[val_idx] = pred_base

    dev_base = mean_poisson_deviance(y_val, pred_base)
    mae_base = mean_absolute_error(y_val, pred_base)

    # Top 10% recall
    k = max(1, int(len(val_idx) * 0.1))
    top_k_base = np.argsort(pred_base)[-k:]
    rec_base = y_val[top_k_base].sum() / max(1.0, y_val.sum())

    base_dev_list.append(dev_base)
    base_mae_list.append(mae_base)
    base_rec_list.append(rec_base)

    # --- 2. Offset Model (With log_exposure Offset) ---
    # In LightGBM, offset is passed via init_score
    init_score_trn = offset[trn_idx]
    init_score_val = offset[val_idx]

    m_offset = lgb.LGBMRegressor(**params)
    m_offset.fit(X_trn, y_trn, init_score=init_score_trn)
    # predict with init_score
    pred_offset = np.clip(m_offset.predict(X_val, init_score=init_score_val), 1e-6, None)
    offset_oof_preds[val_idx] = pred_offset

    dev_offset = mean_poisson_deviance(y_val, pred_offset)
    mae_offset = mean_absolute_error(y_val, pred_offset)

    top_k_offset = np.argsort(pred_offset)[-k:]
    rec_offset = y_val[top_k_offset].sum() / max(1.0, y_val.sum())

    offset_dev_list.append(dev_offset)
    offset_mae_list.append(mae_offset)
    offset_rec_list.append(rec_offset)

# Overall results
k_all = int(len(df) * 0.1)
top_k_base_all = np.argsort(base_oof_preds)[-k_all:]
top_k_offset_all = np.argsort(offset_oof_preds)[-k_all:]

overall_rec_base = y[top_k_base_all].sum() / y.sum()
overall_rec_offset = y[top_k_offset_all].sum() / y.sum()

# Rank Shuffle (교체율)
overlap_count = len(set(top_k_base_all).intersection(set(top_k_offset_all)))
shuffle_rate = (k_all - overlap_count) / k_all * 100.0

print("\n" + "=" * 60)
print("🏆 [동일 표본 3,304개 구간 대상 5-Fold GroupKFold 비교 결과]")
print("=" * 60)
print(f"지표                   | 기존 Model 1 (건수) | Offset Model (사고율)")
print("-" * 60)
print(f"Poisson Deviance (↓)   | {np.mean(base_dev_list):.4f} (±{np.std(base_dev_list):.4f}) | {np.mean(offset_dev_list):.4f} (±{np.std(offset_dev_list):.4f})")
print(f"MAE (↓)                | {np.mean(base_mae_list):.4f}             | {np.mean(offset_mae_list):.4f}")
print(f"상위 10% 사고 포착률(↑) | {overall_rec_base*100:.1f}%              | {overall_rec_offset*100:.1f}%")
print(f"고위험 순위 교체율     | 기준                 | {shuffle_rate:.1f}% 재편")
print("=" * 60)
