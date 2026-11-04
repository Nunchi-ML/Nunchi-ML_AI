import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("new_camera_audit")


def label_file(tmp_path, filename, who, labels, ids=None):
    path = tmp_path / filename
    pd.DataFrame(dict(id=ids or [str(i) for i in range(len(labels))], label=labels, labeler=who)).to_csv(path, index=False)
    return path


def test_two_raters_cannot_turn_missing_or_excluded_votes_into_consensus(audit, tmp_path):
    a = label_file(tmp_path, "a.csv", "Alice", [0, 1, 2, 0])
    b = label_file(tmp_path, "b.csv", "Bob", [0, 2, "x", ""])
    labels, review, info = audit.load_labels([a, b], ["0", "1", "2", "3"], "human_consensus")
    assert labels.to_dict() == {"0": 0}
    assert review.accepted.tolist() == [True, False, False, False]
    assert info["pairwise"][0]["rows"] == 2
    assert info["pairwise"][0]["agreement"] == .5


def test_three_raters_need_strict_majority_not_plurality(audit, tmp_path):
    paths = [label_file(tmp_path, f"{i}.csv", str(i), values)
             for i, values in enumerate(([1, 0, 2], [1, 1, "x"], ["x", 2, 2]))]
    labels, _, _ = audit.load_labels(paths, ["0", "1", "2"], "human_consensus")
    assert labels.to_dict() == {"0": 1, "2": 2}


def test_identical_file_cannot_be_two_raters(audit, tmp_path):
    a = label_file(tmp_path, "a.csv", "Alice", [0])
    with pytest.raises(ValueError, match="same label file"):
        audit.load_labels([a, a], ["0"], "human_consensus")


def test_copied_file_with_same_identity_cannot_be_two_raters(audit, tmp_path):
    a = label_file(tmp_path, "a.csv", "Alice", [0])
    b = label_file(tmp_path, "b.csv", "alice", [0])
    with pytest.raises(ValueError, match="distinct"):
        audit.load_labels([a, b], ["0"], "human_consensus")


@pytest.mark.parametrize("values,ids,message", [([3], ["0"], "Invalid label"),
    ([0], ["unknown"], "Unknown frame"), ([0, 1], ["0", "0"], "duplicate")])
def test_bad_labels_rejected_before_evaluation(audit, tmp_path, values, ids, message):
    a = label_file(tmp_path, "a.csv", "Alice", values, ids)
    with pytest.raises(ValueError, match=message):
        audit.load_labels([a], ["0"], "human_single")


def test_ai_labels_cannot_be_reported_as_human(audit, tmp_path):
    a = label_file(tmp_path, "a.csv", "Claude(AI 임시 라벨)", [0])
    with pytest.raises(ValueError, match="AI-labelled"):
        audit.load_labels([a], ["0"], "human_single")
    labels, _, info = audit.load_labels([a], ["0"], "ai_provisional")
    assert labels.iloc[0] == 0 and info["kind"] == "ai_provisional"


def test_one_human_cannot_be_consensus(audit, tmp_path):
    a = label_file(tmp_path, "a.csv", "Alice", [0])
    with pytest.raises(ValueError, match="at least two"):
        audit.load_labels([a], ["0"], "human_consensus")


def test_constant_baseline_exposes_prevalence_accuracy(audit):
    score = audit.score([0] * 9 + [2], [0] * 10)
    assert score["alert_accuracy"] == .9
    assert score["alert_balanced_accuracy"] == .5
    assert score["alert_recall"] == score["severe_recall"] == 0
    assert score["alert_precision"] is None


def test_one_class_video_does_not_invent_balanced_accuracy(audit):
    assert audit.score([0, 0], [0, 1])["alert_balanced_accuracy"] is None


def test_reference_abstention_is_not_a_human_vote_or_normal_label(audit):
    human = pd.Series([0, 1, 2], index=["a", "b", "c"], name="label")
    reference = pd.Series([0, 1], index=["a", "c"])
    paired, info = audit.compare_reference(human, reference)
    assert len(paired) == 3 and pd.isna(paired.loc["b", "reference"])
    assert human.tolist() == [0, 1, 2]
    assert info["jointly_labelled"] == 2 and info["reference_excluded_or_missing"] == 1
    assert info["exact_agreement"] == .5
    assert info["reference_as_prediction"]["severe_recall"] == 0


def test_paired_video_resampling_preserves_identical_predictions(audit):
    y = np.array([0, 1, 2, 0, 0])
    pred = np.array([0, 1, 1, 1, 0])
    _, differences = audit.bootstrap_accuracy(y, {"ML1": pred, "rule": pred.copy()}, np.array(["a", "a", "b", "b", "b"]), 100)
    assert differences == [dict(method="ML1", reference="rule", metric="alert_accuracy", difference=0., lower=0., upper=0.)]


def test_blind_sample_preserves_ids_and_excludes_answers(audit, tmp_path):
    import json
    import re
    import zipfile
    prepare = importlib.import_module("prepare_human_review").prepare
    root = tmp_path / "private"
    pack = root / "label_pack"
    (pack / "images/s01").mkdir(parents=True)
    records = []
    for i in range(4):
        file = f"images/s01/{i}.jpg"
        (pack / file).write_bytes(f"preserve-image-{i}".encode())
        records.append(dict(no=i+1, id=str(i), file=file, sequence="s01", label="", note=""))
    sheet = pack / "label_sheet_blind.csv"
    pd.DataFrame(records).to_csv(sheet, index=False)
    before = sheet.read_bytes()
    out = root / "human_sample"
    prepare(root, out, count=2)
    assert sheet.read_bytes() == before
    sample = pd.read_csv(out / "label_sheet_blind.csv", dtype={"id": str})
    assert len(sample) == 2 and sample.id.isin(["0", "1", "2", "3"]).all()
    assert sample.label.isna().all()
    for row in sample.itertuples():
        assert (out / row.file).read_bytes() == (pack / row.file).read_bytes()
    html = (out / "label_tool.html").read_text(encoding="utf-8")
    embedded = json.loads(re.search(r'<script id="embeddedRows" type="application/json">(.*?)</script>', html).group(1))
    assert len(embedded) == 2 and all(set(r) == {"no", "id", "file", "sequence"} for r in embedded)
    with zipfile.ZipFile(out.with_suffix(".zip")) as z:
        assert len([p for p in z.namelist() if p.endswith(".jpg")]) == 2
    with pytest.raises(FileExistsError):
        prepare(root, out, count=2)
