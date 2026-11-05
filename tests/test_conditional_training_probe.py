import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("conditional_training_probe")


def test_conditional_scores_reconstruct_three_valid_classes(module):
    p = module.compose([0, 1, .8], [1, 0, .25])
    np.testing.assert_allclose(p, [[1, 0, 0], [0, 1, 0], [.2, .6, .2]])
    np.testing.assert_allclose(p.sum(1), 1)


@pytest.mark.parametrize("a,b", [([.5], [.4, .5]), ([np.nan], [.5]), ([.5], [1.1])])
def test_invalid_binary_scores_rejected(module, a, b):
    with pytest.raises(ValueError):
        module.compose(a, b)


def test_severity_training_excludes_normal_and_does_not_use_validation_labels(module, monkeypatch):
    train = pd.DataFrame(dict(label=[0, 1, 2, 0]))
    validation = pd.DataFrame(dict(label=[2, 2]))
    calls = []
    monkeypatch.setattr(module.R.C, "probabilities", lambda a, b, cols: np.tile([.5, .3, .2], (len(b), 1)))

    def binary(a, b, cols, target):
        calls.append((a.index.tolist(), np.asarray(target).tolist()))
        return np.full(len(b), .6)

    monkeypatch.setattr(module, "binary_probability", binary)
    first = module.fold_predictions(train, validation)
    assert calls == [([0, 1, 2, 3], [0, 1, 1, 0]), ([1, 2], [0, 1])]
    second = module.fold_predictions(train, validation.assign(label=[0, 0]))
    for method in first:
        np.testing.assert_allclose(first[method], second[method])


def test_single_class_training_has_explicit_constant_prediction(module):
    np.testing.assert_array_equal(module.binary_probability(pd.DataFrame({"x": [1, 2]}), pd.DataFrame({"x": [3]}), ["x"], [1, 1]), [1])
