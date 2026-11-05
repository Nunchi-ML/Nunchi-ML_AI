import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("forward_detail_probe")


def test_detail_ablation_keeps_training_targets_and_stage_feature_scope(module, monkeypatch):
    tr = pd.DataFrame(dict(label=[0, 1, 2, 0]))
    va = pd.DataFrame(dict(label=[2, 2]))
    calls = []
    monkeypatch.setattr(module.R.C, "probabilities", lambda a, b, cols: np.tile([.2, .5, .3], (len(b), 1)))

    def binary(a, b, cols, target):
        calls.append((a.index.tolist(), list(cols), list(target)))
        return np.full(len(b), .6 if set(module.F.FEATURES).issubset(cols) else .4)

    monkeypatch.setattr(module.M, "binary_probability", binary)
    result = module.predict_fold(tr, va)
    assert list(result) == module.NAMES
    assert calls[0][0] == calls[1][0] == [0, 1, 2, 3]
    assert calls[2][0] == calls[3][0] == [1, 2]
    assert calls[2][2] == calls[3][2] == [False, True]
    assert not set(module.F.FEATURES) & set(calls[2][1])
    assert set(module.F.FEATURES).issubset(calls[3][1])
    for p in result.values():
        np.testing.assert_allclose(p.sum(1), 1)


def test_validation_labels_are_not_used_in_new_detail_models(module, monkeypatch):
    tr = pd.DataFrame(dict(label=[0, 1, 2]))
    monkeypatch.setattr(module.R.C, "probabilities", lambda a, b, cols: np.tile([.4, .4, .2], (len(b), 1)))
    monkeypatch.setattr(module.M, "binary_probability", lambda a, b, cols, target: np.full(len(b), .6))
    first = module.predict_fold(tr, pd.DataFrame(dict(label=[0, 0])))
    second = module.predict_fold(tr, pd.DataFrame(dict(label=[2, 2])))
    for method in first:
        np.testing.assert_allclose(first[method], second[method])
