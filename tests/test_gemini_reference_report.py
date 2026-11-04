import importlib
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("gemini_reference_report")


def test_abstentions_do_not_become_class_zero_or_change_pair_denominator(module):
    ids = pd.Index(["a", "b", "c", "d"])
    center = module.numeric_grades(pd.Series(["0", "abstain", "2", "1"], index=ids))
    temporal = module.numeric_grades(pd.Series(["1", "2", "abstain", "1"], index=ids))
    common = module.paired_ids(dict(center=center, temporal=temporal), ids)
    assert center.index.tolist() == ["a", "c", "d"]
    assert common.tolist() == ["a", "d"]


def test_unknown_and_duplicate_auxiliary_labels_fail(module):
    with pytest.raises(ValueError):
        module.numeric_grades(pd.Series(["0", "3"], index=["a", "b"]))
    with pytest.raises(ValueError):
        module.numeric_grades(pd.Series(["0", "1"], index=["a", "a"]))
    with pytest.raises(ValueError):
        module.paired_ids({"extra": pd.Series([0], index=["unseen"])}, pd.Index(["a"]))


def test_undefined_statistics_are_null_not_invented_zero(module):
    assert module.clean({"qwk": float("nan"), "scores": [float("inf"), 0.]}) == {"qwk": None, "scores": [None, 0.]}
