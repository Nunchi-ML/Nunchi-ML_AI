import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("photo_regularization_probe"), importlib.import_module("photo_regularization_analysis")


def test_captures_remain_whole_and_every_sample_is_held_out_once(modules):
    probe, _ = modules
    frame = pd.DataFrame(dict(id=[str(i) for i in range(60)], group=np.repeat(np.arange(10), 6), label=np.tile([0, 1, 2], 20)))
    coverage = np.zeros(60, int)
    for tr, va in probe.grouped_splits(frame):
        assert not set(frame.iloc[tr].group) & set(frame.iloc[va].group)
        coverage[va] += 1
    np.testing.assert_array_equal(coverage, np.ones(60))
    frame.loc[1, "id"] = frame.loc[0, "id"]
    with pytest.raises(ValueError, match="Unique IDs"):
        probe.grouped_splits(frame)


def test_selection_prefers_training_macro_then_severe_then_simpler_model(modules):
    probe, _ = modules
    base = dict(macro_f1=.4, recall_2=.2, num_leaves=7, min_child_samples=10, reg_lambda=1.)
    rows = [base, dict(base, macro_f1=.3, recall_2=1.), dict(base, num_leaves=3),
            dict(base, num_leaves=3, min_child_samples=30),
            dict(base, num_leaves=3, min_child_samples=30, reg_lambda=10.)]
    assert probe.select(rows) is rows[-1]
    improved = dict(base, macro_f1=.41)
    assert probe.select(rows + [improved]) is improved
    severe = dict(base, recall_2=.3)
    assert probe.select(rows + [severe]) is severe


def test_grid_budget_and_factory_keep_other_baseline_parameters(modules):
    probe, _ = modules
    assert len(probe.GRID) == 9
    assert len({p["id"] for p in probe.GRID}) == 9
    baseline = probe.R.lightgbm_model().get_params()
    for parameters in probe.GRID:
        result = probe.model(parameters).get_params()
        for name in baseline:
            if name not in ["num_leaves", "min_child_samples", "reg_lambda"]:
                assert result[name] == baseline[name]


def test_three_class_macro_and_paired_group_identity(modules):
    _, analysis = modules
    frame = pd.DataFrame(dict(video=["a", "a", "b", "b"], label=[0, 1, 0, 2], p=[0, 0, 0, 2]))
    counts = np.zeros((3, 3), int)
    np.add.at(counts, (frame.label, frame.p), 1)
    assert analysis.macro_from_counts(counts) == pytest.approx(f1_score(frame.label, frame.p, labels=[0, 1, 2], average="macro", zero_division=0))
    result = analysis.paired_interval(frame, "p", "p", draws=50)
    assert result["groups"] == 2
    assert result["delta"] == result["low"] == result["high"] == 0
