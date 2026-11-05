import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("pairwise_glare")


def test_training_percentiles_preserve_ties_and_ignore_test_distribution(module):
    reference = np.array([1., 1., 3., 5.])
    np.testing.assert_allclose(module.percentile(reference, [-10., 1., 2., 5., 99.]), [0., .25, .5, .875, 1.])
    assert module.percentile(reference, [2.])[0] == module.percentile(reference, [2., 1e10])[0]
    np.testing.assert_array_equal(reference, [1., 1., 3., 5.])
    with pytest.raises(ValueError, match="reference"):
        module.percentile([], [1.])


def test_query_groups_limit_comparisons_and_exclude_equal_grades(module):
    frame = pd.DataFrame(dict(id=list("abcd"), group=["g2", "g1", "g2", "g1"], label=[0, 0, 1, 2]))
    order, qid, stats = module.query_layout(frame, "capture")
    assert np.all(qid[:-1] <= qid[1:])
    assert stats == dict(queries=2, active_queries=2, possible_unequal_pairs=2)
    _, _, global_stats = module.query_layout(frame, "global")
    assert global_stats["possible_unequal_pairs"] == 5
    assert frame.iloc[order].group.tolist() == ["g1", "g1", "g2", "g2"]


def test_cut_points_fit_inner_labels_and_handle_constant_scores(module):
    y, scores = np.array([0, 0, 1, 1, 2, 2]), np.array([.1, .15, .45, .5, .8, .9])
    cuts, info = module.thresholds_from_oof(y, scores)
    np.testing.assert_array_equal(module.grade(scores, cuts), y)
    assert info["inner_macro_f1"] == pytest.approx(1.)
    cuts, _ = module.thresholds_from_oof(y, np.full(6, .5))
    assert cuts[0] <= cuts[1] and np.isfinite(cuts).all()
    with pytest.raises(ValueError, match="Matched"):
        module.thresholds_from_oof(y, [np.nan]*6)
    with pytest.raises(ValueError, match="ordered"):
        module.grade(scores, [np.nan, .5])


def test_outer_labels_cannot_change_inner_cuts_or_predictions(module, monkeypatch):
    fits = []

    class Spy:
        def __init__(self, mode):
            pass

        def fit(self, frame, columns):
            self.groups = set(frame.group)
            self.diagnostics_ = dict(queries=1, active_queries=1, possible_unequal_pairs=1)
            fits.append(frame.id.tolist())
            return self

        def score(self, frame):
            assert not self.groups & set(frame.group)
            return frame.feature.to_numpy(float)

    monkeypatch.setattr(module, "RankScoreModel", Spy)
    train = pd.DataFrame(dict(id=[str(i) for i in range(18)], group=np.repeat(np.arange(6), 3),
                              label=np.tile([0, 1, 2], 6), feature=np.tile([.1, .5, .9], 6)))
    test = pd.DataFrame(dict(id=["a", "b"], group=[100, 101], label=[0, 0], feature=[.2, .8]))
    first = module.nested_rank(train, test, ["feature"], "global")
    second = module.nested_rank(train, test.assign(label=2), ["feature"], "global")
    np.testing.assert_array_equal(first[0], second[0])
    assert first[2] == second[2]
    assert len(fits) == 8
    assert set(fits[3]) == set(train.id)


def test_pair_metric_gives_half_credit_for_ties(module):
    assert module.pair_accuracy([0, 1, 2], [0., 1., 2.]) == 1.
    assert module.pair_accuracy([0, 1, 2], [2., 1., 0.]) == 0.
    assert module.pair_accuracy([0, 1, 2], [1., 1., 1.]) == .5
    assert module.pair_accuracy([1, 1], [0., 1.]) is None


def test_invalid_queries_and_outer_capture_overlap_are_rejected(module):
    frame = pd.DataFrame(dict(id=["a", "a", "c"], group=["g1", "g2", "g3"], label=[0, 1, 2], feature=[0., .5, 1.]))
    with pytest.raises(ValueError, match="Unique"):
        module.query_layout(frame, "capture")
    frame["id"] = ["a", "b", "c"]
    with pytest.raises(ValueError, match="Outer capture"):
        module.nested_rank(frame, frame.iloc[:1], ["feature"], "global")
