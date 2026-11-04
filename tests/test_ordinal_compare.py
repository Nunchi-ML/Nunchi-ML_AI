import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def experiment(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "notebooks/experiments/sun_visible_probe"
    monkeypatch.syspath_prepend(str(folder))
    return importlib.import_module("ordinal_compare")


def test_grade_metrics_keep_extreme_errors_and_binary_task_distinct(experiment):
    y, pred = np.array([0, 1, 2]), np.array([2, 1, 0])
    grade = experiment.metrics(y, pred)
    assert grade["count_0_to_2"] == grade["count_2_to_0"] == 1
    assert grade["f1_1"] == 1
    assert grade["mae"] == pytest.approx(4 / 3)
    assert grade["qwk"] == pytest.approx(-1)
    binary = experiment.binary_metrics(y, pred >= 1)
    assert binary["alert_recall"] == .5
    assert binary["false_alert_rate"] == 1
    # A 1<->2 mistake is wrong for grading but correct for alert detection.
    same_alert = np.array([0, 2, 1])
    assert experiment.binary_metrics(y, same_alert >= 1)["binary_f1"] == 1
    assert experiment.metrics(y, same_alert)["macro_f1"] < 1


def test_nested_folds_never_share_capture_groups(experiment):
    groups = np.repeat(np.arange(8), 3)
    validation = []
    for train, test in experiment.inner_splits(groups):
        assert not set(groups[train]) & set(groups[test])
        validation.extend(test)
    assert sorted(validation) == list(range(len(groups)))
    with pytest.raises(ValueError, match="groups"):
        experiment.inner_splits(np.zeros(10))


def test_missing_training_grade_keeps_fixed_output_columns(experiment):
    X = np.arange(24).reshape(12, 2)
    y = np.repeat([0, 2], 6)
    p = experiment.task_probabilities("logistic", X, y, X, True, 3)
    assert p.shape == (12, 3)
    np.testing.assert_array_equal(p[:, 1], 0)
    np.testing.assert_allclose(p.sum(axis=1), 1)
    q = experiment.task_probabilities("logistic", X, np.ones(12, int), X, True, 2)
    np.testing.assert_array_equal(q, np.tile([0, 1], (12, 1)))


@pytest.mark.parametrize("value,bin_index", [(0, 0), (255, 7)])
def test_pixel_statistics_are_finite_for_uniform_images(experiment, value, bin_index):
    features = importlib.import_module("ordinal_image_features")
    f = features.extract_array(np.full((50, 80, 3), value, np.uint8))
    assert list(f) == features.EXTRA
    assert np.isfinite(list(f.values())).all()
    assert f["px_std"] == 0
    assert f["px_local_contrast"] == 0
    assert f["px_source_contrast"] == 0
    assert f[f"px_hist_{bin_index}"] == 1
    assert sum(f[f"px_hist_{i}"] for i in range(8)) == 1


def test_bright_region_centroid_is_normalized_pixel_location(experiment):
    features = importlib.import_module("ordinal_image_features")
    image = np.zeros((368, 640, 3), np.uint8)
    image[80:100, 400:420] = 255
    f = features.extract_array(image)
    assert f["px_bright_area"] == pytest.approx(400 / (368 * 640))
    assert f["px_bright_x"] == pytest.approx(409.5 / 640)
    assert f["px_bright_y"] == pytest.approx(89.5 / 368)
