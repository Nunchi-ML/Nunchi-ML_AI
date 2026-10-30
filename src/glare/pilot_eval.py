"""시범 라벨로 '밝기 규칙' 기준선과 LightGBM을 영상 단위 교차검증으로 비교한다.

사용:
    python -m src.glare.pilot_eval data/glare_features.csv data/glare_pilot_labels.csv

라벨 파일 칸: vid, file, label (0 정상 / 1 눈부심 / 2 심한 눈부심)
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupKFold

from .features import FEATURES

SEED = 0


def load(features_csv, labels_csv):
    f = pd.read_csv(features_csv)
    lab = pd.read_csv(labels_csv)
    keys = ["vid", "file"]
    if lab.duplicated(keys).any() or f.duplicated(keys).any():
        raise ValueError("Duplicate image keys in labels or features")
    if not lab.label.isin([0, 1, 2]).all() or set(lab.label) != {0, 1, 2}:
        raise ValueError("Labels must contain all three classes: 0, 1, 2")
    d = lab.merge(f, on=keys, how="left", validate="one_to_one")
    if not np.isfinite(d[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Missing or non-finite image features")
    if d.vid.nunique() < 5:
        raise ValueError("At least five videos are required for GroupKFold")
    return d


def cross_val(d, fit_predict, n_splits=5):
    """같은 영상이 학습과 시험에 섞이지 않게 영상(vid) 단위로 나눈다."""
    y = d.label.values
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=n_splits).split(d, y, d.vid):
        pred[te] = fit_predict(d.iloc[tr], d.iloc[te])
    return pred


def brightness_rule(feature):
    """숫자 하나에 임계값 두 개를 두는 규칙. 임계값은 학습 폴드에서 macro F1이 가장 높은 값."""
    def fit_predict(tr, te):
        v, y = tr[feature].values, tr.label.values
        qs = np.quantile(v, np.linspace(0.05, 0.95, 19))
        best = max(((f1_score(y, np.digitize(v, [a, b]), average="macro"), (a, b))
                    for a in qs for b in qs if b > a), key=lambda t: t[0])
        return np.digitize(te[feature].values, best[1])
    return fit_predict


def lightgbm_model():
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=5,
                              class_weight="balanced", random_state=SEED, n_jobs=1, verbose=-1)


def lightgbm(tr, te):
    m = lightgbm_model().fit(tr[FEATURES], tr.label)
    return m.predict(te[FEATURES])


def main(features_csv, labels_csv):
    d = load(features_csv, labels_csv)
    y = d.label.values
    print(f"라벨 {len(d)}장 (정상 {(y == 0).sum()} / 눈부심 {(y == 1).sum()} / 심한 {(y == 2).sum()}), 영상 {d.vid.nunique()}개")
    for name, fn in [("규칙: 전체 밝기", brightness_rule("mean_all")),
                     ("규칙: 하얀 픽셀 비율", brightness_rule("white_ratio")),
                     ("LightGBM", lightgbm)]:
        p = cross_val(d, fn)
        print(f"{name}: 정확도 {accuracy_score(y, p):.2f} · macro F1 {f1_score(y, p, average='macro'):.2f}"
              f" · 심한 눈부심 재현율 {recall_score(y, p, labels=[2], average='macro'):.2f}")
        if name == "LightGBM":
            print("혼동행렬 (행 = 정답 정상/눈부심/심한):\n", confusion_matrix(y, p))
    m = lightgbm_model().fit(d[FEATURES], y)
    imp = pd.Series(m.booster_.feature_importance("gain"), index=FEATURES)
    print("중요 변수 (gain):", (imp / imp.sum()).sort_values(ascending=False).head(5).round(2).to_dict())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
