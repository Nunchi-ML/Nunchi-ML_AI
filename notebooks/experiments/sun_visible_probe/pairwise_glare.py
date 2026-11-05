"""Fixed-label pairwise ranking and leakage-free ordinal score thresholds.

Ranking queries use only training photos. Percentile scores are fitted on a
ranker's own training predictions, not probabilities or population fractions.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

MODES = ["global", "capture"]
GRID = np.r_[-1e-9, np.arange(0., 1.001, .05), 1.+1e-9]


def percentile(reference, values):
    reference, values = np.asarray(reference, float), np.asarray(values, float)
    if reference.ndim != 1 or values.ndim != 1 or not len(reference) or not np.isfinite(reference).all() or not np.isfinite(values).all():
        raise ValueError("Finite training reference and evaluation score vectors required")
    ordered = np.sort(reference)
    return (np.searchsorted(ordered, values, side="left") + np.searchsorted(ordered, values, side="right")) / (2*len(ordered))


def grade(scores, thresholds):
    scores = np.asarray(scores, float)
    if scores.ndim != 1 or not np.isfinite(scores).all() or len(thresholds) != 2 or not np.isfinite(thresholds).all() or thresholds[0] > thresholds[1]:
        raise ValueError("Finite scores and ordered cut points required")
    return np.searchsorted(np.asarray(thresholds), scores, side="right").astype(int)


def thresholds_from_oof(y, scores):
    y, scores = np.asarray(y), np.asarray(scores, float)
    if y.ndim != 1 or len(y) != len(scores) or not len(y) or not np.isin(y, [0, 1, 2]).all() or not np.isfinite(scores).all():
        raise ValueError("Matched finite inner-OOF labels/scores required")
    candidates = np.array([(a, b) for a in GRID for b in GRID if a <= b])
    pred = (scores[None, :] >= candidates[:, :1]).astype(int) + (scores[None, :] >= candidates[:, 1:]).astype(int)
    counts = np.zeros((len(candidates), 3, 3), int)
    for a in range(3):
        for b in range(3):
            counts[:, a, b] = ((y[None, :] == a) & (pred == b)).sum(axis=1)
    diagonal = counts.diagonal(axis1=1, axis2=2)
    denominator = counts.sum(axis=1)+counts.sum(axis=2)
    f1 = np.divide(2*diagonal, denominator, out=np.zeros(diagonal.shape, float), where=denominator != 0).mean(axis=1)
    severe_recall = counts[:, 2, 2] / max(int((y == 2).sum()), 1)
    mae = np.abs(pred-y[None, :]).mean(axis=1)
    selected = max(range(len(candidates)), key=lambda k: (f1[k], severe_recall[k], -mae[k],
                    -abs(candidates[k, 0]-1/3)-abs(candidates[k, 1]-2/3), -k))
    return tuple(candidates[selected]), dict(inner_macro_f1=float(f1[selected]), inner_severe_recall=float(severe_recall[selected]), candidates=len(candidates))


def query_layout(frame, mode):
    if mode not in MODES or not frame.id.is_unique or frame.group.isna().any() or not frame.label.isin([0, 1, 2]).all():
        raise ValueError("Unique training photos, capture groups and valid grades required")
    qid = np.zeros(len(frame), int) if mode == "global" else pd.factorize(frame.group, sort=True)[0]
    order = np.argsort(qid, kind="stable")
    effective, active = 0, 0
    for group in np.unique(qid):
        counts = np.bincount(frame.label.to_numpy(int)[qid == group], minlength=3)
        pairs = int((counts.sum()**2 - (counts**2).sum()) // 2)
        effective += pairs
        active += pairs > 0
    return order, qid[order], dict(queries=int(len(np.unique(qid))), active_queries=int(active), possible_unequal_pairs=effective)


class RankScoreModel:
    def __init__(self, mode):
        self.mode = mode

    def fit(self, frame, columns):
        from xgboost import XGBRanker
        order, qid, self.diagnostics_ = query_layout(frame, self.mode)
        self.columns_ = list(columns)
        x = frame[self.columns_].to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite training features")
        self.model_ = None
        if self.diagnostics_["possible_unequal_pairs"]:
            self.model_ = XGBRanker(objective="rank:pairwise", n_estimators=200, max_depth=3,
                learning_rate=.05, min_child_weight=1, reg_lambda=1., tree_method="hist", n_jobs=1,
                random_state=0, lambdarank_pair_method="mean", lambdarank_num_pair_per_sample=8)
            self.model_.fit(x[order], frame.label.to_numpy(int)[order], qid=qid)
        self.reference_ = self.raw_score(frame)
        return self

    def raw_score(self, frame):
        x = frame[self.columns_].to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite prediction features")
        return np.zeros(len(frame)) if self.model_ is None else self.model_.predict(x)

    def score(self, frame):
        return percentile(self.reference_, self.raw_score(frame))


class ClassifierScoreModel:
    """Matching multiclass objective, with the same percentile/cut-point pipeline."""
    def __init__(self, mode):
        pass

    def fit(self, frame, columns):
        from ordinal_compare import make_estimator
        self.columns_ = list(columns)
        self.classes_, encoded = np.unique(frame.label.to_numpy(int), return_inverse=True)
        self.model_ = None if len(self.classes_) == 1 else make_estimator("xgboost", len(self.classes_))
        if self.model_ is not None:
            self.model_.fit(frame[self.columns_].to_numpy(float), encoded)
        self.reference_ = self.raw_score(frame)
        self.diagnostics_ = dict(queries=0, active_queries=0, possible_unequal_pairs=0)
        return self

    def probabilities(self, frame):
        if self.model_ is None:
            return np.ones((len(frame), 1))
        return self.model_.predict_proba(frame[self.columns_].to_numpy(float))

    def raw_score(self, frame):
        return self.probabilities(frame) @ self.classes_

    def score(self, frame):
        return percentile(self.reference_, self.raw_score(frame))


def nested_rank(train, test, columns, mode):
    if set(train.group) & set(test.group):
        raise ValueError("Outer capture leakage")
    n_groups = train.group.nunique()
    if n_groups < 3:
        raise ValueError("Three training capture groups needed")
    splits = list(GroupKFold(3, shuffle=True, random_state=0).split(train, groups=train.group))
    scores, seen = np.zeros(len(train)), np.zeros(len(train), int)
    factory = ClassifierScoreModel if mode == "multiclass" else RankScoreModel
    for tr, va in splits:
        a, b = train.iloc[tr], train.iloc[va]
        if set(a.group) & set(b.group):
            raise ValueError("Inner capture leakage")
        fitted = factory(mode).fit(a, columns)
        scores[va] = fitted.score(b)
        seen[va] += 1
    if not np.all(seen == 1):
        raise ValueError("Inner OOF coverage mismatch")
    thresholds, diagnostic = thresholds_from_oof(train.label.to_numpy(), scores)
    fitted = factory(mode).fit(train, columns)
    test_scores = fitted.score(test)
    information = dict(threshold_1=float(thresholds[0]), threshold_2=float(thresholds[1]), **diagnostic, **fitted.diagnostics_)
    if mode == "multiclass":
        information["argmax_predictions"] = fitted.classes_[fitted.probabilities(test).argmax(axis=1)]
    return grade(test_scores, thresholds), test_scores, information, scores


def pair_accuracy(y, scores):
    y, scores = np.asarray(y), np.asarray(scores)
    eligible = y[:, None] < y[None, :]
    n = int(eligible.sum())
    if not n:
        return None
    correct = (scores[:, None] < scores[None, :]) + .5*(scores[:, None] == scores[None, :])
    return float(correct[eligible].mean())
