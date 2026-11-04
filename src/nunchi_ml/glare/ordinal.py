"""Three ordered image-glare grades using two independent binary estimators.

This experimental estimator does not change the deployed glare predictor.
Class weighting is computed separately for y>=1 and y>=2 on training rows.
Optional sigmoid calibration uses group-held-out scores within training only.
Returned probabilities are model estimates, not sensor measurements.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_is_fitted, validate_data


def fit_with_sample_weight(model, X, y, weights):
    """Support both plain estimators and existing sklearn preprocessing pipelines."""
    key = f"{model.steps[-1][0]}__sample_weight" if isinstance(model, Pipeline) else "sample_weight"
    return model.fit(X, y, **{key: weights})


class FoldWeightedClassifier(ClassifierMixin, BaseEstimator):
    """Recompute balancing inside EACH calibration training fold.

    Passing weights computed from the whole outer training set into a CV
    calibrator would let inner validation labels influence training weights.
    Calibration itself receives only caller weights, never balanced weights.
    """
    def __init__(self, estimator, class_weight=None):
        self.estimator = estimator
        self.class_weight = class_weight

    def fit(self, X, y, sample_weight=None):
        X, y = validate_data(self, X, y, dtype=float)
        weights = compute_sample_weight("balanced", y) if self.class_weight else np.ones(len(y))
        if sample_weight is not None:
            weights = weights * sample_weight
        self.estimator_ = (DummyClassifier(strategy="constant", constant=y[0])
                           if len(np.unique(y)) == 1 else clone(self.estimator))
        fit_with_sample_weight(self.estimator_, X, y, weights)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(validate_data(self, X, reset=False, dtype=float))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def project_cumulative_probabilities(q):
    """Euclidean/isotonic projection onto 1 >= q1 >= q2 >= 0.

    For two thresholds, pooling an inverted pair to its arithmetic mean is
    the exact least-squares projection. Invalid input is rejected, not hidden.
    """
    q = np.array(q, dtype=float, copy=True)
    if q.ndim != 2 or q.shape[1] != 2:
        raise ValueError("Expected cumulative probabilities with shape (n, 2)")
    if not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
        raise ValueError("Cumulative probabilities must be finite and in [0, 1]")
    inverted = q[:, 0] < q[:, 1]
    q[inverted] = q[inverted].mean(axis=1, keepdims=True)
    return q


def cumulative_to_class_probabilities(q):
    q = project_cumulative_probabilities(q)
    p = np.column_stack((1 - q[:, 0], q[:, 0] - q[:, 1], q[:, 1]))
    if (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-12):
        raise AssertionError("Ordinal class probabilities must be a simplex")
    return p


class CumulativeOrdinalClassifier(ClassifierMixin, BaseEstimator):
    """Cloneable sklearn classifier; predict uses argmax of P(0), P(1), P(2).

    estimator_ge2 optionally supplies different parameters for the second
    threshold. Fit/predict use sklearn's feature name/order/finite checks.
    A threshold with one training class uses a constant predictor.
    """

    def __init__(self, estimator, estimator_ge2=None, class_weight=None,
                 calibration=None, calibration_splits=3):
        self.estimator = estimator
        self.estimator_ge2 = estimator_ge2
        self.class_weight = class_weight
        self.calibration = calibration
        self.calibration_splits = calibration_splits

    def fit(self, X, y, sample_weight=None, groups=None):
        X, y = validate_data(self, X, y, dtype=float)
        if not np.isin(y, [0, 1, 2]).all():
            raise ValueError("Ordinal labels must be 0, 1 or 2")
        if self.class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be None or balanced")
        if self.calibration not in (None, "sigmoid"):
            raise ValueError("calibration must be None or sigmoid")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if (sample_weight.shape != (len(y),) or not np.isfinite(sample_weight).all()
                    or (sample_weight < 0).any() or sample_weight.sum() <= 0):
                raise ValueError("Invalid sample_weight")
        self.classes_ = np.arange(3)
        self.estimators_ = []
        self.calibration_folds_ = []
        if self.calibration:
            if groups is None:
                raise ValueError("Calibration requires capture groups")
            groups = np.asarray(groups)
            if (groups.shape != (len(y),) or any(v is None or str(v).strip().lower() in ("", "nan", "nat") for v in groups)):
                raise ValueError("Invalid calibration groups")
            if not isinstance(self.calibration_splits, (int, np.integer)) or self.calibration_splits < 2:
                raise ValueError("calibration_splits must be at least 2")
            count = min(self.calibration_splits, len(np.unique(groups)))
            if count < 2:
                raise ValueError("Calibration requires at least two capture groups")
            self.calibration_folds_ = list(GroupKFold(count).split(X, y, groups))
        for threshold, base in ((1, self.estimator),
                                (2, self.estimator_ge2 if self.estimator_ge2 is not None else self.estimator)):
            binary = (y >= threshold).astype(int)
            if self.calibration and len(np.unique(binary)) > 1:
                model = CalibratedClassifierCV(
                    FoldWeightedClassifier(base, self.class_weight), method="sigmoid",
                    cv=self.calibration_folds_, ensemble=False, n_jobs=1)
                model.fit(X, binary, sample_weight=sample_weight)
                self.estimators_.append(model)
                continue
            weights = compute_sample_weight("balanced", binary) if self.class_weight else np.ones(len(y))
            if sample_weight is not None:
                weights = weights * sample_weight
            model = (DummyClassifier(strategy="constant", constant=int(binary[0]))
                     if np.unique(binary).size == 1 else clone(base))
            fit_with_sample_weight(model, X, binary, weights)
            self.estimators_.append(model)
        return self

    def predict_cumulative_proba(self, X, *, corrected=True):
        check_is_fitted(self, "estimators_")
        X = validate_data(self, X, reset=False, dtype=float)
        columns = []
        for model in self.estimators_:
            positive = np.flatnonzero(model.classes_ == 1)
            columns.append(model.predict_proba(X)[:, positive[0]] if len(positive) else np.zeros(len(X)))
        q = np.column_stack(columns)
        # Validate even when returning the original estimates for diagnostics.
        projected = project_cumulative_probabilities(q)
        return projected if corrected else q

    def predict_proba(self, X):
        return cumulative_to_class_probabilities(self.predict_cumulative_proba(X))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
