"""Non-neural, source-to-unlabelled-calibration covariance alignment.

Caller fits any scaler on source training data only. Evaluation data is never
accepted by fit. Mean alignment is an explicit extension/control to CORAL.
"""
import numpy as np


def matrix(x, columns=None):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("Finite two-dimensional features required")
    if columns is not None and x.shape[1] != columns:
        raise ValueError("Feature width changed")
    return x


def symmetric_power(covariance, exponent):
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    if not np.isfinite(values).all() or values.min() <= 0:
        raise ValueError("Positive definite regularized covariance required")
    return (vectors * (values ** exponent)) @ vectors.T


class CoralAlignment:
    def __init__(self, ridge=1.):
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("Positive fixed ridge required")
        self.ridge = float(ridge)

    def fit(self, source, calibration):
        source = matrix(source)
        calibration = matrix(calibration, source.shape[1])
        if min(len(source), len(calibration)) < 2:
            raise ValueError("At least two source and calibration rows required")
        self.source_mean_ = source.mean(axis=0)
        self.target_mean_ = calibration.mean(axis=0)
        identity = np.eye(source.shape[1]) * self.ridge
        self.source_covariance_ = np.atleast_2d(np.cov(source, rowvar=False)) + identity
        self.target_covariance_ = np.atleast_2d(np.cov(calibration, rowvar=False)) + identity
        self.mapping_ = symmetric_power(self.source_covariance_, -.5) @ symmetric_power(self.target_covariance_, .5)
        return self

    def transform_source(self, source):
        if not hasattr(self, "mapping_"):
            raise ValueError("Alignment is not fitted")
        source = matrix(source, len(self.source_mean_))
        return (source - self.source_mean_) @ self.mapping_ + self.target_mean_


def separated_folds(frame, group_column):
    if (not frame.id.is_unique or frame.image_sha256.duplicated().any()
            or frame[group_column].isna().any()):
        raise ValueError("Unique photos and nonmissing adaptation groups required")
    groups = sorted(frame[group_column].unique())
    if len(groups) < 2:
        raise ValueError("At least two capture groups required for adaptation")
    result = []
    for group in groups:
        evaluation = np.flatnonzero(frame[group_column].eq(group))
        calibration = np.flatnonzero(frame[group_column].ne(group))
        if len(calibration) < 2:
            raise ValueError("Insufficient disjoint calibration photos")
        result.append((str(group), calibration, evaluation))
    return result


def verify_disjoint(source, calibration, evaluation, group_column):
    sets = [set(f.image_sha256) for f in [source, calibration, evaluation]]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Source/calibration/evaluation photo overlap")
    if set(calibration[group_column]) & set(evaluation[group_column]):
        raise ValueError("Calibration/evaluation capture overlap")
