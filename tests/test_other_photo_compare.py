import importlib
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("other_photo_compare")


def test_outside_partition_excludes_human_and_preserves_abstention(module):
    outside, accepted, pending = module.partition_ids(list("ABCDE"), ["A"], list("ACD"))
    assert outside.tolist() == list("BCDE")
    assert accepted.tolist() == list("CD") and pending.tolist() == list("BE")


@pytest.mark.parametrize("accepted", [["C", "C"], ["UNKNOWN"]])
def test_invalid_reference_ids_cannot_enter_evaluation(module, accepted):
    with pytest.raises(ValueError):
        module.partition_ids(list("ABCDE"), ["A"], accepted)


@pytest.mark.parametrize("fault", ["none", "changed", "missing", "duplicate"])
def test_exact_baseline_reproduction_gate(module, fault):
    original = pd.DataFrame(dict(id=["a", "b"], method=["fixed", "fixed"], pred=[0, 2]))
    actual = original.iloc[::-1].copy()
    if fault == "changed":
        actual.iloc[0, 2] = 1
    elif fault == "missing":
        actual = actual.iloc[:1]
    elif fault == "duplicate":
        actual = pd.concat([actual, actual.iloc[:1]])
    if fault == "none":
        module.verify_reproduction(actual, original)
    else:
        with pytest.raises(ValueError):
            module.verify_reproduction(actual, original)
