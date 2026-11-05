import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("coral_alignment"), importlib.import_module("coral_photo_probe")


def test_mapping_aligns_regularized_covariance_and_mean(modules):
    c, _ = modules
    rng = np.random.default_rng(0)
    source = rng.normal(size=(120, 4)) * [1, 2, 3, 4]
    calibration = rng.normal(size=(70, 4)) @ rng.normal(size=(4, 4)) + [5, -4, 3, 2]
    before = source.copy()
    model = c.CoralAlignment().fit(source, calibration)
    actual = model.mapping_.T @ model.source_covariance_ @ model.mapping_
    np.testing.assert_allclose(actual, model.target_covariance_, atol=1e-10)
    np.testing.assert_allclose(model.transform_source(source).mean(axis=0), calibration.mean(axis=0), atol=1e-12)
    np.testing.assert_array_equal(source, before)


def test_rank_deficiency_and_one_feature_are_finite(modules):
    c, _ = modules
    for source, target in [(np.ones((3, 5)), np.zeros((2, 5))), (np.arange(8).reshape(-1, 1), np.arange(4).reshape(-1, 1))]:
        aligned = c.CoralAlignment().fit(source, target).transform_source(source)
        assert aligned.shape == source.shape and np.isfinite(aligned).all()
    with pytest.raises(ValueError, match="ridge"):
        c.CoralAlignment(0)
    with pytest.raises(ValueError, match="two source"):
        c.CoralAlignment().fit(np.zeros((1, 2)), np.zeros((4, 2)))
    with pytest.raises(ValueError, match="width"):
        c.CoralAlignment().fit(np.zeros((4, 2)), np.zeros((4, 3)))
    with pytest.raises(ValueError, match="Finite"):
        c.CoralAlignment().fit(np.full((4, 2), np.nan), np.zeros((4, 2)))


def test_entire_videos_are_held_out_with_no_photo_reuse(modules):
    c, _ = modules
    frame = pd.DataFrame(dict(id=list("abcdef"), image_sha256=list("123456"), video=["v1", "v1", "v2", "v2", "v3", "v3"]))
    counts = np.zeros(len(frame), int)
    for _, calibration, evaluation in c.separated_folds(frame, "video"):
        assert not set(frame.iloc[calibration].video) & set(frame.iloc[evaluation].video)
        counts[evaluation] += 1
    np.testing.assert_array_equal(counts, np.ones(6))
    with pytest.raises(ValueError, match="two capture"):
        c.separated_folds(frame.assign(video="one"), "video")
    with pytest.raises(ValueError, match="Unique"):
        c.separated_folds(frame.assign(image_sha256="same"), "video")


def test_image_and_capture_overlap_are_rejected(modules):
    c, _ = modules
    source = pd.DataFrame(dict(image_sha256=["source"], group=["s"]))
    cal = pd.DataFrame(dict(image_sha256=["cal"], group=["c"]))
    test = pd.DataFrame(dict(image_sha256=["test"], group=["e"]))
    c.verify_disjoint(source, cal, test, "group")
    with pytest.raises(ValueError, match="photo overlap"):
        c.verify_disjoint(source, cal, test.assign(image_sha256="cal"), "group")
    with pytest.raises(ValueError, match="capture overlap"):
        c.verify_disjoint(source, cal, test.assign(group="c"), "group")


def test_evaluation_inputs_do_not_affect_fitted_transform_or_training(modules, monkeypatch):
    _, p = modules
    captured = []

    class Fake:
        classes_ = np.array([0, 1, 2])

        def fit(self, x, y):
            captured.append(np.asarray(x).copy())
            return self

        def predict_proba(self, x):
            return np.tile([.2, .3, .5], (len(x), 1))

    monkeypatch.setattr(p, "estimator", lambda family: Fake())
    source = pd.DataFrame(dict(a=[0., 1., 2., 3., 4., 5.], b=[1., 3., 2., 5., 4., 6.], label=[0, 1, 2, 0, 1, 2]))
    calibration = np.array([[4., 3.], [5., 9.], [6., 4.]])
    first, _ = p.fit_predictions(source, calibration, np.zeros((2, 2)), ["a", "b"], "lightgbm")
    p.fit_predictions(source, calibration, np.full((2, 2), 1e9), ["a", "b"], "lightgbm")
    for before, after in zip(captured[:3], captured[3:]):
        np.testing.assert_array_equal(before, after)
    assert set(first) == set(p.METHODS)
