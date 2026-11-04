import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nunchi_ml.glare.ordinal import (
    CumulativeOrdinalClassifier, cumulative_to_class_probabilities,
    project_cumulative_probabilities,
)


def test_projection_pools_only_inverted_pairs_without_mutating_input():
    q = np.array([[.8, .2], [.2, .8], [.5, .5], [0, 0], [1, 1]])
    original = q.copy()
    np.testing.assert_allclose(project_cumulative_probabilities(q),
                               [[.8, .2], [.5, .5], [.5, .5], [0, 0], [1, 1]])
    np.testing.assert_array_equal(q, original)
    np.testing.assert_allclose(cumulative_to_class_probabilities(q),
                               [[.2, .6, .2], [.5, 0, .5], [.5, 0, .5], [1, 0, 0], [0, 0, 1]])


def test_random_probabilities_have_order_simplex_and_idempotent_projection():
    q = np.random.default_rng(0).random((10000, 2))
    ordered = project_cumulative_probabilities(q)
    assert (ordered[:, 0] >= ordered[:, 1]).all()
    np.testing.assert_array_equal(project_cumulative_probabilities(ordered), ordered)
    p = cumulative_to_class_probabilities(q)
    assert (p >= 0).all()
    np.testing.assert_allclose(p.sum(axis=1), 1)
    np.testing.assert_allclose(p[:, 1:].sum(axis=1), ordered[:, 0])
    np.testing.assert_allclose(p[:, 2], ordered[:, 1])


@pytest.mark.parametrize("q", [[[1.1, .5]], [[.5, -.1]], [[np.nan, .2]],
                             [[.5, np.inf]], [.5, .2], [[.1, .2, .3]]])
def test_invalid_probabilities_are_rejected(q):
    with pytest.raises(ValueError):
        cumulative_to_class_probabilities(q)


@pytest.mark.parametrize("label", [0, 1, 2])
def test_single_class_thresholds_and_fixed_output_order(label):
    m = CumulativeOrdinalClassifier(LogisticRegression()).fit(np.arange(5)[:, None], np.full(5, label))
    p = m.predict_proba([[2], [3]])
    np.testing.assert_array_equal(p, np.eye(3)[[label, label]])


def test_cloning_feature_contract_serialization_and_separate_threshold_training(tmp_path):
    X = pd.DataFrame({"brightness": np.arange(18), "angle": np.sin(np.arange(18))})
    y = np.repeat([0, 1, 2], 6)
    m = clone(CumulativeOrdinalClassifier(LogisticRegression(), class_weight="balanced")).fit(X, y)
    assert m.estimators_[0] is not m.estimators_[1]
    assert not np.allclose(m.estimators_[0].coef_, m.estimators_[1].coef_)
    q = m.predict_cumulative_proba(X, corrected=False)
    np.testing.assert_allclose(m.predict_proba(X), cumulative_to_class_probabilities(q))
    np.testing.assert_array_equal(m.predict(X), m.predict_proba(X).argmax(axis=1))
    with pytest.raises(ValueError, match="feature names"):
        m.predict(X[["angle", "brightness"]])
    with pytest.raises(ValueError, match="feature names"):
        m.predict(X.assign(extra=1))
    path = tmp_path / "ordinal.joblib"
    joblib.dump(m, path)
    np.testing.assert_allclose(joblib.load(path).predict_proba(X), m.predict_proba(X))


@pytest.mark.parametrize("labels", [[0, 1, 3], [0, .5, 2], [0, -1, 2]])
def test_invalid_ordinal_labels_rejected(labels):
    with pytest.raises(ValueError, match="labels"):
        CumulativeOrdinalClassifier(LogisticRegression()).fit([[0], [1], [2]], labels)


def test_existing_scaler_pipeline_is_fitted_on_training_rows_only():
    X = np.arange(24).reshape(12, 2)
    base = make_pipeline(StandardScaler(), LogisticRegression())
    m = CumulativeOrdinalClassifier(base, class_weight="balanced").fit(X, np.repeat([0, 1, 2], 4))
    assert not hasattr(base[0], "mean_")
    before = m.estimators_[0][0].mean_.copy()
    m.predict_proba([[10000, 20000]])
    np.testing.assert_array_equal(before, X.mean(axis=0))
    np.testing.assert_array_equal(m.estimators_[0][0].mean_, before)
