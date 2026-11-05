import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.class_weight import compute_sample_weight


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("photo_weight_probe")


@pytest.fixture
def training():
    return pd.DataFrame(dict(label=[0, 0, 1, 1, 2, 2, 0, 1, 2],
                             group=["a"] * 6 + ["b"] * 3, camera=["x"] * 6 + ["y"] * 3,
                             feature=np.arange(9)))


def test_weight_policies_balance_class_mass_and_preserve_input(module, training):
    original = training.copy(deep=True)
    np.testing.assert_array_equal(module.training_weights(training, "class_only"), compute_sample_weight("balanced", training.label))
    for policy in module.POLICIES:
        weights = module.training_weights(training, policy)
        assert np.isfinite(weights).all() and (weights > 0).all()
        assert weights.mean() == pytest.approx(1)
        for label in [0, 1, 2]:
            assert weights[training.label.eq(label)].sum() == pytest.approx(len(training) / 3)
    pd.testing.assert_frame_equal(training, original)


def test_inverse_capture_reduces_duplicate_capture_dominance(module, training):
    enlarged = pd.concat([training[training.group.eq("a")]] * 5 + [training[training.group.eq("b")]], ignore_index=True)
    for frame in [training, enlarged]:
        weights = module.training_weights(frame, "capture_inverse")
        assert weights[frame.group.eq("a")].sum() == pytest.approx(weights[frame.group.eq("b")].sum())
    weights = module.training_weights(enlarged, "class_only")
    assert weights[enlarged.group.eq("a")].sum() > 5 * weights[enlarged.group.eq("b")].sum()


def test_invalid_training_metadata_and_missing_classes_are_rejected(module, training):
    broken = training.copy()
    broken.loc[0, "group"] = None
    for frame, policy in [(broken, "capture_inverse"), (training[training.label.ne(2)], "class_only"), (training, "unknown")]:
        with pytest.raises(ValueError, match="Training rows"):
            module.training_weights(frame, policy)


def test_fit_uses_single_class_correction_and_ignores_evaluation_labels(module, training, monkeypatch):
    calls = []

    class FakeEstimator:
        classes_ = np.array([0, 1, 2])

        def set_params(self, **kwargs):
            assert kwargs == {"class_weight": None}
            return self

        def fit(self, x, y, sample_weight):
            calls.append((x.copy(), y.copy(), sample_weight.copy()))

        def predict_proba(self, x):
            return np.tile([.2, .3, .5], (len(x), 1))

    monkeypatch.setattr(module.P, "model", lambda p: FakeEstimator())
    first = module.fit_predict(training, training.assign(label=0), ["feature"], {}, "capture_inverse")
    second = module.fit_predict(training, training.assign(label=2), ["feature"], {}, "capture_inverse")
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(calls[0][2], calls[1][2])
    pd.testing.assert_frame_equal(calls[0][0], training[["feature"]])
    np.testing.assert_array_equal(calls[0][2], module.training_weights(training, "capture_inverse"))


def test_training_selection_ties_keep_existing_class_weighting(module):
    records = [dict(policy=p, macro_f1=.4, recall_2=.2) for p in module.POLICIES]
    assert module.choose(records)["policy"] == "class_only"
    records[1]["macro_f1"] = .41
    assert module.choose(records)["policy"] == "capture_inverse"
