import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("ordinal_calibration_probe")


def test_balanced_prior_restore_known_odds_and_boundary_cases(probe):
    result = probe.restore_balanced_prior([[.5, .5], [0, 1]], [.8, .2], [True, True])
    np.testing.assert_allclose(result, [[.8, .2], [0, 1]])
    np.testing.assert_allclose(probe.restore_balanced_prior([[.6, .7]], [0, 1], [True, True]), [[0, 1]])
    np.testing.assert_allclose(probe.restore_balanced_prior([[.6, .7]], [.8, .2], [False, False]), [[.6, .7]])


def test_threshold_search_includes_default_and_uses_default_for_ties(probe):
    p = np.tile([.1, .8, .1], (9, 1))
    a, b, score = probe.select_thresholds(np.tile([0, 1, 2], 3), p)
    assert a == b == .5
    np.testing.assert_array_equal(probe.threshold_decision(p, a, b), 1)


def test_reference_tuning_scores_are_predicted_outside_their_capture_group(probe, monkeypatch):
    frame = pd.DataFrame({"id": np.arange(30), "group": np.repeat(np.arange(10), 3), "label": np.tile([0, 1, 2], 10)})
    seen = []
    def factory(columns):
        def predict(train, validation):
            assert not set(train.group) & set(validation.group)
            assert not set(train.id) & set(validation.id)
            seen.extend(validation.id)
            return np.tile([.2, .6, .2], (len(validation), 1))
        return predict
    monkeypatch.setattr(probe.O.P, "two_stage", factory)
    result = probe.inner_reference_predictions(frame)
    assert sorted(seen) == list(range(30))
    assert result.shape == (30, 3)


def test_calibration_spec_preserves_existing_severity_feature_subset(probe):
    base, balanced = probe.model_for("logistic_severity_g_A_none", "sigmoid")
    assert base.calibration == "sigmoid"
    assert balanced == [False, True]
    assert base.estimator_ge2[-1].class_weight == "balanced"
    assert base.estimator_ge2[0].transformers[0][2] == list(range(6))


def test_augmentation_adds_training_only_and_never_scores_the_training_aihub(probe, monkeypatch, tmp_path):
    rows = []
    for capture in range(8):
        ai = capture >= 6
        for label in range(3):
            identity = capture*3 + label
            rows.append(dict(id=str(identity), camera="AI Hub" if ai else "base", group=str(capture), label=label,
                             **{c: float(identity) for c in probe.O.SETS["g_rel"]}))
    data = pd.DataFrame(rows)
    augmentation_ids = set(data.loc[data.camera.eq("AI Hub"), "sun_alt"])
    fits = []
    class FakeModel:
        def fit(self, X, y, groups=None):
            self.train = set(X.sun_alt)
            assert augmentation_ids <= self.train
            fits.append(self.train)
            return self
        def predict_cumulative_proba(self, X, corrected=False):
            assert not self.train & set(X.sun_alt)
            assert not augmentation_ids & set(X.sun_alt)
            return np.tile([.7, .3], (len(X), 1))
    monkeypatch.setattr(probe, "model_for", lambda spec, calibration: (FakeModel(), [True, True]))
    def factory(columns):
        def prediction(train, validation):
            assert not set(train.id) & set(validation.id)
            assert not set(train.group) & set(validation.group)
            return np.tile([.3, .4, .3], (len(validation), 1))
        return prediction
    monkeypatch.setattr(probe.O.P, "two_stage", factory)
    args = SimpleNamespace(repeats=1, only_cv=False, augment_aihub=True, specs=["logistic"], workers=1, out=tmp_path)
    scores, _, _ = probe.evaluate(data, args)
    assert fits
    assert set(scores.test) == {"group_cv", "held_camera:base"}
    predictions = pd.read_csv(tmp_path / "predictions.csv", dtype={"id": str})
    assert not set(predictions.id) & set(data.loc[data.camera.eq("AI Hub"), "id"])
    controls = pd.read_csv(tmp_path / "thresholds.csv")
    assert controls.added_aihub_rows.eq(6).all()
