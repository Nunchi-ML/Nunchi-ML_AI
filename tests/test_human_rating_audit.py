import importlib
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("human_rating_audit")


def test_disagreement_and_abstention_keep_valid_denominators(module):
    cat = pd.DataFrame(dict(id=["a", "b", "c"]))
    votes = pd.DataFrame(dict(id=["a", "a", "b", "b", "c", "c"],
                             rater=["r1", "r2"] * 3, label=["0", "2", "1", "x", "", "x"],
                             unsure=["0", "0", "1", "0", "0", "0"]))
    out = module.aggregate_votes(cat, votes).set_index("id")
    assert out.loc["a", "n"] == 2
    assert out.loc["a", "legacy_severe_share"] == .5
    assert out.loc["a", "legacy_any_glare_share"] == .5
    assert out.loc["b", "n"] == 1
    assert out.loc["b", "legacy_any_glare_share"] == 1
    assert out.loc["b", "legacy_severe_share"] == 0
    assert out.loc["b", "abstain_n"] == out.loc["b", "uncertain_n"] == 1
    assert out.loc["c", "n"] == 0
    assert pd.isna(out.loc["c", "legacy_severe_share"])
    assert out.loc["c", "missing_n"] == 1


def test_same_person_repeats_cannot_inflate_panel_size(module):
    votes = pd.DataFrame(dict(id=["a", "a"], rater=["r1", "r1"], label=["1", "2"], unsure=["0", "0"]))
    with pytest.raises(ValueError, match="Repeated person"):
        module.aggregate_votes(pd.DataFrame(dict(id=["a"])), votes)


def test_raw_identity_image_and_label_checks(module):
    cat = pd.DataFrame(dict(id=["a"], file=["images/a.jpg"], sequence=["seq"]))
    raw = pd.DataFrame(dict(id=["a"], file=["images/a.jpg"], sequence=["seq"], label=["2"],
                            labeler=["human"], unsure=["0"], note=[""]))
    original = raw.copy(deep=True)
    assert module.read_votes(cat, raw, "human").label.tolist() == ["2"]
    pd.testing.assert_frame_equal(raw, original)
    with pytest.raises(ValueError, match="identity"):
        module.read_votes(cat, raw, "another human")
    with pytest.raises(ValueError, match="reference mismatch"):
        module.read_votes(cat, raw.assign(file="images/other.jpg"), "human")
    with pytest.raises(ValueError, match="Unknown raw label"):
        module.read_votes(cat, raw.assign(label="0.7"), "human")
    with pytest.raises(ValueError, match="Duplicate/unknown"):
        module.read_votes(cat, pd.concat([raw, raw]), "human")


def test_incomplete_photo_is_not_given_a_normal_label(module):
    cat = pd.DataFrame(dict(id=["a", "b"]))
    votes = pd.DataFrame(dict(id=["a"], rater=["r1"], label=["1"], unsure=["0"]))
    result = module.aggregate_votes(cat, votes).set_index("id")
    assert result.loc["b", "n"] == result.loc["b", "recorded_people"] == 0
    assert pd.isna(result.loc["b", "legacy_any_glare_share"])


def test_missing_or_duplicate_ids_are_rejected(module, tmp_path):
    path = tmp_path / "votes.csv"
    pd.DataFrame(dict(id=["a", "a"], label=[0, 1])).to_csv(path, index=False)
    with pytest.raises(ValueError, match="photo ID"):
        module.read_table(path)


def test_audit_refuses_to_overwrite_existing_output(module, tmp_path):
    existing = tmp_path / "audit"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("preserved")
    with pytest.raises(ValueError, match="new private"):
        module.run(tmp_path, existing)
    assert sentinel.read_text() == "preserved"
