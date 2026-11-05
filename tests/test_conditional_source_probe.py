import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("conditional_source_probe")


def frames():
    train = pd.DataFrame(dict(id=list("abcdef"), group=["a", "a", "b", "b", "c", "c"],
                              image_sha256=list("abcdef"), label=[0, 1, 1, 2, 0, 2]))
    auxiliary = pd.DataFrame(dict(id=["x", "y", "z"], group=["x", "y", "z"],
                                  image_sha256=["x", "y", "z"], label=[0, 1, 2]))
    test = pd.DataFrame(dict(id=["t", "u"], group=["t", "t"], image_sha256=["t", "u"], label=[0, 2]))
    return train, auxiliary, test


def spies(m, monkeypatch):
    calls, rules = [], []
    def binary(train, test, columns, target, kind):
        assert not set(train.group) & set(test.group)
        calls.append((train.copy(), columns, list(np.asarray(target, int))))
        return np.full(len(test), .6 if kind == "conditional_lr" else .4 + .02*len(train))
    def rule(train, test):
        assert not set(train.group) & set(test.group)
        rules.append(train.copy())
        return {"rule_tuned": np.full(len(test), 1)}
    monkeypatch.setattr(m.V, "binary", binary)
    monkeypatch.setattr(m.V, "rule_predictions", rule)
    return calls, rules


def test_fixed_A_positive_B_and_equal_data_rule_pools(module, monkeypatch):
    m = module
    train, auxiliary, test = frames()
    calls, rules = spies(m, monkeypatch)
    a, outputs = m.fit_candidates(train, auxiliary, test, ["x", "y", "z"])
    assert set(calls[0][0].id) == set(train.id)
    assert calls[0][2] == [0, 1, 1, 1, 0, 1]
    assert len(calls) == 5  # one A, four distinct positive-only B pools
    for pool, columns, target in calls[1:]:
        assert pool.label.ge(1).all()
        assert columns == m.V.M.R.G + m.V.EXTRA
        assert target == list(pool.label.eq(2).astype(int))
    for mask, pool in zip(m.MASKS, rules):
        expected = set(train.id) | set(m.subset(auxiliary, mask, ["x", "y", "z"]).id)
        assert set(pool.id) == expected
    np.testing.assert_array_equal(outputs["000"]["b"], outputs["100"]["b"])
    np.testing.assert_array_equal(outputs["011"]["b"], outputs["111"]["b"])
    np.testing.assert_array_equal(a, [.6, .6])


def test_nested_choice_uses_only_training_OOF_and_excludes_whole_captures(module, monkeypatch):
    m = module
    train, auxiliary, test = frames()
    spies(m, monkeypatch)
    first, choice, trace = m.predict(train, auxiliary, test, ["x", "y", "z"])
    second, changed, _ = m.predict(train, auxiliary, test.assign(label=[2, 0]), ["x", "y", "z"])
    assert choice == changed
    assert len(trace) == 8*len(train)
    t = pd.DataFrame(trace)
    assert set(t.id) == set(train.id) and not set(t.id) & set(test.id)
    assert t.groupby("group").inner_fold.nunique().eq(1).all()
    for name in first:
        np.testing.assert_array_equal(first[name][0], second[name][0])
    assert first["B_inner_subset"][3] == choice["selected"]["B"]


def test_subset_selection_uses_inner_score_then_declared_parsimony(module):
    m = module
    outputs = {mask: dict(prediction=np.array([0, 1, 2]), rule=np.array([0, 1, 2]),
                         auxiliary_positive=int(mask, 2), auxiliary_rows=int(mask, 2)) for mask in m.MASKS}
    assert m.choose_subset(np.array([0, 1, 2]), outputs, "B")[0] == "000"
    outputs["000"]["prediction"] = np.array([0, 0, 0])
    assert m.choose_subset(np.array([0, 1, 2]), outputs, "B")[0] == "001"
    assert m.choose_subset(np.array([0, 1, 2]), outputs, "rule")[0] == "000"


def test_duplicate_auxiliary_image_or_outer_capture_leakage_rejected(module, monkeypatch):
    m = module
    train, auxiliary, test = frames()
    spies(m, monkeypatch)
    for bad in [auxiliary.assign(image_sha256=["a", "y", "z"]), auxiliary.assign(group=["t", "y", "z"])]:
        with pytest.raises(ValueError, match="leakage"):
            m.fit_candidates(train, bad, test, sorted(bad.group.unique()))


def test_undeclared_subset_and_existing_output_are_rejected(module, tmp_path):
    _, auxiliary, _ = frames()
    with pytest.raises(ValueError, match="three declared"):
        module.subset(auxiliary, "1100", ["x", "y", "z"])
    with pytest.raises(ValueError, match="new private"):
        module.run(tmp_path, tmp_path)
