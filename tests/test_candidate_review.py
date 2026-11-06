import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("prepare_candidate_review")


def saved(module):
    originals = pd.DataFrame(dict(id=["a", "b", "c"], label=[0, 1, 2],
                                 group=["g1", "g2", "g3"], camera=["one"] * 3))
    rows = []
    for method in module.METHODS:
        for row in originals.itertuples():
            rows.append(dict(test="group_cv320", repeat=0, method=method, id=row.id,
                             group=row.group, label=row.label, pred=row.label))
    probabilities = pd.DataFrame(dict(test=["group_cv320"] * 3, repeat=[0] * 3,
                                     id=originals.id, source_a=[.2, .7, .8],
                                     source_b=[.1, .2, .7], geometry_b=[.1, .3, .6]))
    return pd.DataFrame(rows), probabilities, originals


def test_saved_comparisons_keep_rules_without_fabricated_probabilities(module):
    predictions, probabilities, originals = saved(module)
    result = module.score_saved(predictions, probabilities, originals)
    assert len(result) == 4 and result.macro_f1.eq(1.).all()
    rules = result[result.method.str.contains("rule")]
    assert rules.probability_status.eq("not_saved").all()
    assert rules.multiclass_log_loss.isna().all()
    ml = result[~result.method.str.contains("rule")]
    assert np.isfinite(ml.multiclass_log_loss).all()


@pytest.mark.parametrize("change", ["missing", "duplicate_score", "changed_grade", "wrong_score"])
def test_corrupt_or_incomplete_preserved_results_are_rejected(module, change):
    predictions, probabilities, originals = saved(module)
    if change == "missing":
        predictions = predictions.iloc[1:]
    elif change == "duplicate_score":
        probabilities = pd.concat([probabilities, probabilities.iloc[:1]])
    elif change == "changed_grade":
        predictions.loc[0, "label"] = 2
    else:
        probabilities.loc[0, "source_a"] = .9
        probabilities.loc[0, "geometry_b"] = .9
    with pytest.raises(ValueError):
        module.score_saved(predictions, probabilities, originals)


def test_output_cannot_overwrite_previous_review(module, tmp_path):
    with pytest.raises(ValueError, match="new private output"):
        module.run(tmp_path, tmp_path)


def test_review_cannot_silently_pick_a_new_model(module, tmp_path):
    with pytest.raises(ValueError, match="Explicitly confirmed"):
        module.run(tmp_path, tmp_path / "new", selected_prototype="new_after_seeing_test")


def test_untrusted_photo_metadata_cannot_close_the_json_script(module):
    text = module.page({"caption": "</script><script>alert(1)</script>"})
    assert "<script>alert(1)" not in text
    assert "\\u003c/script>" in text
