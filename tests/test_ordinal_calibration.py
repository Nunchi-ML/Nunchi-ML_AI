import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from nunchi_ml.glare.ordinal import CumulativeOrdinalClassifier


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    fits = []
    held_predictions = []

    def fit(self, X, y, sample_weight=None):
        self.ids_ = set(X[:, 0])
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        type(self).fits.append((np.array(X), np.array(y), np.array(sample_weight)))
        return self

    def predict_proba(self, X):
        type(self).held_predictions.append((self.ids_, set(X[:, 0])))
        if len(self.classes_) == 1:
            return np.ones((len(X), 1))
        p = 1 / (1 + np.exp(-np.clip(X[:, 1], -20, 20)))
        return np.column_stack([1-p, p])


def test_calibration_predictions_are_group_held_out_and_balancing_is_fold_local():
    groups = np.repeat(np.arange(9), 4)
    X = np.column_stack([np.arange(36), np.sin(np.arange(36))])
    y = np.tile([0, 0, 1, 2], 9)
    RecordingClassifier.fits = []
    RecordingClassifier.held_predictions = []
    model = CumulativeOrdinalClassifier(RecordingClassifier(), class_weight="balanced", calibration="sigmoid")
    model.fit(X, y, groups=groups)
    assert len(RecordingClassifier.fits) == 8  # 3 inner + 1 full fit, for each threshold
    assert len(RecordingClassifier.held_predictions) == 6
    for trained, predicted in RecordingClassifier.held_predictions:
        assert not trained & predicted
        assert not set(groups[list(map(int, trained))]) & set(groups[list(map(int, predicted))])
    for features, target, weights in RecordingClassifier.fits:
        np.testing.assert_allclose(weights, compute_sample_weight("balanced", target))
    for train, validation in model.calibration_folds_:
        assert not set(groups[train]) & set(groups[validation])


def test_rare_severity_missing_from_inner_training_fold_still_has_valid_output():
    groups = np.repeat(np.arange(6), 4)
    X = np.column_stack([np.arange(24), np.sin(np.arange(24))])
    y = np.tile([0, 0, 1, 1], 6)
    y[-4:] = 2  # All severity labels in one capture group, like the real data.
    m = CumulativeOrdinalClassifier(LogisticRegression(max_iter=500), class_weight="balanced", calibration="sigmoid")
    # sklearn warns when an inner fold cannot learn a missing class; never silently add a label.
    with pytest.warns(RuntimeWarning, match="Number of classes"):
        m.fit(X, y, groups=groups)
    p = m.predict_proba(X)
    assert p.shape == (24, 3)
    assert np.isfinite(p).all() and (p >= 0).all()
    np.testing.assert_allclose(p.sum(axis=1), 1)


def test_calibrated_scaler_feature_contract_serialization_and_constant_threshold(tmp_path):
    X = pd.DataFrame({"pixel": np.arange(24), "sun": np.cos(np.arange(24))})
    y = np.tile([0, 1, 1, 1], 6)  # No class2: B stays a constant zero.
    groups = np.repeat(np.arange(6), 4)
    base = make_pipeline(StandardScaler(), LogisticRegression())
    m = clone(CumulativeOrdinalClassifier(base, class_weight="balanced", calibration="sigmoid")).fit(X, y, groups=groups)
    np.testing.assert_array_equal(m.predict_proba(X)[:, 2], 0)
    with pytest.raises(ValueError, match="feature names"):
        m.predict(X[["sun", "pixel"]])
    path = tmp_path / "calibrated.joblib"
    joblib.dump(m, path)
    np.testing.assert_allclose(joblib.load(path).predict_proba(X), m.predict_proba(X))


@pytest.mark.parametrize("groups", [None, np.zeros(12), np.arange(10), [None]*12, ["nan"]*12])
def test_calibration_rejects_missing_or_ambiguous_group_plan(groups):
    with pytest.raises(ValueError, match="groups"):
        CumulativeOrdinalClassifier(LogisticRegression(), calibration="sigmoid").fit(
            np.arange(12)[:, None], np.tile([0, 1, 2], 4), groups=groups)


def test_uncalibrated_existing_predictions_are_unchanged_when_groups_are_supplied():
    X = np.arange(24).reshape(12, 2)
    y = np.repeat([0, 1, 2], 4)
    a = CumulativeOrdinalClassifier(LogisticRegression(), class_weight="balanced").fit(X, y)
    b = CumulativeOrdinalClassifier(LogisticRegression(), class_weight="balanced").fit(X, y, groups=np.arange(12))
    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X))
