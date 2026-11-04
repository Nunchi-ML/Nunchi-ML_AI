import csv
import importlib
import json
from pathlib import Path
import re

import pytest


@pytest.fixture
def audit_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("temporal_priority_audit")


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def example(audit_module, tmp_path):
    m = audit_module
    root = tmp_path / "temporal"
    private = root / "private_audit"
    private.mkdir(parents=True)
    observation = private / "observations.csv"
    row = dict(zip(m.FIELDS, ["T001", "identifiable", "not_confirmed", "not_confirmed",
                             "bright_scene_only", "no", "no", "no", "yes", "Visible road <test>", "No causal reference"]))
    write_csv(observation, m.FIELDS, [row])
    rubric = private / "rubric_before_review.json"
    rubric.write_text('{"version":"example"}', encoding="utf-8")
    provenance = dict(observations_sha256=m.sha(observation), source_rubric_sha256=m.sha(rubric),
                      blind=False, class_labels_assigned=False)
    (private / "observation_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    images = root / "review/T001"
    images.mkdir(parents=True)
    frames = []
    for offset in m.OFFSETS:
        image = images / ("center.jpg" if offset == 0 else f"neighbor_{offset:+.0f}.jpg")
        image.write_bytes(f"mock image {offset}".encode())
        frames.append(dict(case_id="T001", offset=offset, output_sha256=m.sha(image)))
    write_csv(root / "frame_audit.csv", ["case_id", "offset", "output_sha256"], frames)
    pfields = ["case_id", "video", "original_sha256", "middle_missed", "severe_to_ai_middle"]
    write_csv(root / "private_priority.csv", pfields, [
        dict(zip(pfields, ["T001", "v1", m.sha(images / "center.jpg"), True, False])),
        dict(zip(pfields, ["T002", "v2", "unselected", False, False]))])
    source = root / "original_labels.csv"
    source.write_bytes(b"ORIGINAL LABELS")
    (root / "method.json").write_text(json.dumps(dict(status="complete", input_sha256={str(source): m.sha(source)})), encoding="utf-8")
    return root, observation, private / "output", source, row


def test_private_report_coverage_unknown_targets_and_no_relabel(audit_module, example):
    root, observations, out, source, row = example
    before = source.read_bytes()
    result = audit_module.audit(root, observations, out)
    assert (result["cases"], result["not_observed"], result["frames"], result["video_groups"]) == (1, 1, 5, 1)
    assert result["counts"]["vehicle"] == {"not_confirmed": 1}
    assert source.read_bytes() == before and "macro_f1" not in result
    assert "class-grade conversion" in " ".join(audit_module.NEXT_PROTOCOL["guards"])
    page = (out / "report.html").read_text(encoding="utf-8")
    assert "&lt;test&gt;" in page
    links = re.findall(r'href="([^"]+)"', page)
    assert len(links) == 5 and all((out / p).resolve().is_file() for p in links)
    with pytest.raises(ValueError, match="new output"):
        audit_module.audit(root, observations, out)


@pytest.mark.parametrize("corrupt", ["original", "image", "observations"])
def test_changed_evidence_fails_before_output(audit_module, example, corrupt):
    root, observations, out, source, row = example
    target = {"original": source, "image": root / "review/T001/center.jpg", "observations": observations}[corrupt]
    with target.open("ab") as f:
        f.write(b" changed")
    with pytest.raises(ValueError, match="hash|changed"):
        audit_module.audit(root, observations, out)
    assert not out.exists()


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown_state", "label_column"])
def test_incomplete_or_unsafe_observation_schema_is_rejected(audit_module, example, change):
    root, observation, out, source, row = example
    _, priorities = audit_module.read_rows(root / "private_priority.csv")
    fields, rows = list(audit_module.FIELDS), [row.copy()]
    if change == "missing":
        rows = []
    elif change == "duplicate":
        rows.append(row.copy())
    elif change == "unknown_state":
        rows[0]["vehicle"] = "unidentifiable"  # unknown existence cannot be recast as a failed task in v0.1
    else:
        fields.append("human_label")
    with pytest.raises(ValueError):
        audit_module.validate_observations(fields, rows, audit_module.keyed(priorities))
