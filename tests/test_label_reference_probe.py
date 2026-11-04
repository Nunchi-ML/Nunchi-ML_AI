import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def probe(monkeypatch):
    here = Path(__file__).resolve().parents[1] / "notebooks/experiments"
    monkeypatch.syspath_prepend(str(here / "sun_visible_probe"))
    monkeypatch.syspath_prepend(str(here / "glare_video_source"))
    return importlib.import_module("label_reference_probe")


def test_same_ids_compare_label_definition_not_sample_composition(probe):
    pred = pd.DataFrame({"fixed": [0, 1, 2, 0]}, index=list("abcd"))
    labels = {"human": pd.Series([0, 1, 2, 0], index=list("abcd")),
              "claude": pd.Series([0, 2, 0], index=list("abc")),
              "codex": pd.Series([0, 1, 2], index=list("abd"))}
    scores, agreement, common = probe.evaluate(labels, pred)
    assert common.tolist() == list("ab")
    paired = scores[scores.regime.eq("common_all_available")]
    assert paired.rows.eq(2).all()
    assert set(paired.reference) == set(labels)
    assert len(agreement) == 3
    pd.testing.assert_frame_equal(pred, pd.DataFrame({"fixed": [0, 1, 2, 0]}, index=list("abcd")))


def test_disjoint_reference_ids_are_not_silently_scored(probe):
    with pytest.raises(ValueError, match="No shared"):
        probe.shared_ids({"a": pd.Series([0], index=["a"]), "b": pd.Series([1], index=["b"])}, ["a", "b"])


def test_unknown_reference_id_is_rejected(probe):
    with pytest.raises(ValueError, match="Invalid reference"):
        probe.evaluate({"a": pd.Series([0], index=["unknown"])}, pd.DataFrame({"m": [0]}, index=["known"]))


def test_missing_or_duplicate_frozen_predictions_rejected(probe, tmp_path):
    path = tmp_path / "pred.csv"
    data = pd.DataFrame(dict(id=["a", "a"], method=["m", "m"], test=["human50_photo_only"]*2, pred=[0, 1]))
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        probe.checked_predictions(path, ["a"])
    data.iloc[:1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="Every method"):
        probe.checked_predictions(path, ["a", "b"])
