import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("conditional_capture_weight_probe")


def positive_frame(m):
    frame = pd.DataFrame(dict(id=list("abcdef"), group=["a", "a", "a", "b", "c", "c"], label=[1, 1, 2, 2, 1, 2]))
    rng = np.random.default_rng(7)
    for column in m.V.M.R.G + m.V.EXTRA:
        frame[column] = rng.normal(size=len(frame))
    return frame


def test_capture_weighting_keeps_class_mass_and_total_loss_scale(module):
    frame = positive_frame(module)
    for power in module.POWERS.values():
        weights = module.balanced_capture_weights(frame, power)
        assert np.isfinite(weights).all() and (weights > 0).all()
        assert weights.sum() == pytest.approx(len(frame))
        for label in [1, 2]:
            assert weights[frame.label.eq(label)].sum() == pytest.approx(len(frame)/2)
    np.testing.assert_array_equal(module.balanced_capture_weights(frame, 0), np.ones(6))
    with pytest.raises(ValueError, match="positive grades"):
        module.balanced_capture_weights(frame.assign(label=0), .5)


def test_power_zero_reproduces_reference_balanced_LR_and_scaler(module):
    train = positive_frame(module)
    test = train.iloc[:3].copy().assign(group="test")
    actual, _ = module.weighted_B(train, test, 0)
    expected = module.V.binary(train, test, module.V.M.R.G+module.V.EXTRA, train.label.eq(2), "geometry_lr")
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    changed, _ = module.weighted_B(train, test.assign(label=[0, 2, 1]), 0)
    np.testing.assert_array_equal(actual, changed)


def test_B_uses_constant_for_single_class_and_rejects_capture_overlap(module):
    train = positive_frame(module).assign(label=2)
    test = train.iloc[:2].copy().assign(group="test")
    probability, _ = module.weighted_B(train, test, .5)
    np.testing.assert_array_equal(probability, [1., 1.])
    with pytest.raises(ValueError, match="capture leakage"):
        module.weighted_B(train, train.iloc[:1], .5)


def test_weighted_rule_baseline_matches_existing_grid_and_ignores_test_labels(module):
    train = pd.DataFrame(dict(id=list("abcdefgh"), group=list("aaaabbcc"), label=[0, 1, 2, 1, 0, 2, 1, 0],
                              sun_alt=[-2, 10, 12, 20, 60, 8, 15, -1], gap=[4, 30, 5, 25, 5, 9, 35, 30], cloud=[1, 3, 0, 7, 0, 2, 5, 0]))
    train = module.V.geometry(train)
    test = train.copy().assign(label=2)
    outputs, choice = module.rule_candidates(train, test)
    np.testing.assert_array_equal(outputs["class"], module.V.rule_predictions(train, test)["rule_tuned"])
    changed, changed_choice = module.rule_candidates(train, test.assign(label=0))
    assert choice == changed_choice
    for policy in outputs:
        np.testing.assert_array_equal(outputs[policy], changed[policy])


def gate_scores(m):
    rows = []
    for regime, repeat in [("group_cv320", r) for r in m.REPEATS]+[("held_camera:"+str(c), -1) for c in range(3)]:
        for method in m.NAMES:
            rows.append(dict(test=regime, repeat=repeat, method=method,
                             macro_f1=.62 if method == "aug_B_sqrt" else .6,
                             recall_2=.7, count_0_to_2=1, count_2_to_0=1))
    return pd.DataFrame(rows)


def test_predeclared_gate_stops_on_severe_loss_without_switching_to_inverse(module):
    frame = gate_scores(module)
    assert module.expansion_gate(frame)["expand_primary"]
    frame.loc[frame.method.eq("aug_B_sqrt"), "recall_2"] = .5
    frame.loc[frame.method.eq("aug_B_inverse"), "macro_f1"] = .99
    result = module.expansion_gate(frame)
    assert not result["expand_primary"]
    assert not result["checks"]["mean_severe_recall_loss_le_01"]
    assert result["inverse_is_diagnostic_not_a_test_selected_replacement"]
    with pytest.raises(ValueError, match="three predeclared"):
        module.expansion_gate(frame[~frame["repeat"].eq(2)])


def test_existing_output_is_preserved(module, tmp_path):
    with pytest.raises(ValueError, match="new private"):
        module.run(tmp_path, tmp_path)
