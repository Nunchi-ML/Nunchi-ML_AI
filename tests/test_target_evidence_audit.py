import copy
import csv
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("target_evidence_audit")


def example(module, cid="T001"):
    return dict(case_id=cid, scene_context="forward_or_aligned_road", scene_evidence="road aligned",
                observation="Road <evidence>", temporal_valid_offsets=[-2, -1, 0, 1, 2], temporal_exclusion_reason=None,
                same_target_recovery="not_established", solar_evidence="uncertain", targets=[
                    dict(target_id=cid + "_" + k, task=task, near_or_distant="image_near", center_normalized_box_or_corridor=[.1, .2, .8, .9],
                         center_visibility="identifiable", existence_evidence="visible_in_center", impairment_region=None,
                         uncertainty_reason="coarse region; no causal reference", alternative_causes=["motion"], solar_evidence="uncertain")
                    for k, task in module.TASKS.items()])


def test_unknown_target_is_explicitly_unevaluable(module):
    row = example(module)
    row["targets"][2].update(near_or_distant="not_applicable", existence_evidence="not_established",
                             center_visibility="not_evaluable", center_normalized_box_or_corridor=None)
    module.validate([row], {"T001"})


@pytest.mark.parametrize("error", ["missing_task", "unknown_failed", "invalid_box", "duplicate", "recovery", "missing_case", "new_label"])
def test_invalid_evidence_does_not_become_labels(module, error):
    row = example(module)
    rows, expected = [row], {"T001"}
    if error == "missing_task":
        row["targets"].pop()
    elif error == "unknown_failed":
        row["targets"][0].update(existence_evidence="not_established", center_visibility="unidentifiable")
    elif error == "invalid_box":
        row["targets"][0]["center_normalized_box_or_corridor"] = [.8, .2, .1, float("nan")]
    elif error == "duplicate":
        rows.append(copy.deepcopy(row))
    elif error == "recovery":
        row["same_target_recovery"] = "yes"
    elif error == "missing_case":
        expected.add("T002")
    else:
        row["new_label"] = 2
    with pytest.raises(ValueError):
        module.validate(rows, expected)


def test_scope_score_keeps_three_class_definition(module):
    result = module.score([0, 0], [0, 0])
    assert result["macro_f1"] == pytest.approx(1 / 3)
    assert result["middle_f1"] == 0 and result["counts"] == [2, 0, 0]


def test_full_audit_preserves_inputs_and_scores_fixed_scope(module, tmp_path):
    root = tmp_path / "temporal"
    private = root / "observations"
    private.mkdir(parents=True)
    rows = [example(module), example(module, "T002")]
    rows[1]["scene_context"] = "rear_marked_road"
    observations = private / "frozen.json"
    observations.write_text(json.dumps(rows), encoding="utf-8")
    protocol = private / "protocol_before_review.json"
    protocol.write_text('{}', encoding="utf-8")
    (private / "provenance.json").write_text(json.dumps(dict(blind=False, class_labels_assigned=False,
        observations_sha256=module.sha(observations), protocol_sha256=module.sha(protocol))), encoding="utf-8")
    priorities, frames = [], []
    for cid, y, pred, scene in [("T001", 2, 0, "forward"), ("T002", 0, 1, "rear")]:
        images = root / "review" / cid
        images.mkdir(parents=True)
        for offset in [-2, -1, 0, 1, 2]:
            image = images / ("center.jpg" if offset == 0 else f"neighbor_{offset:+.0f}.jpg")
            image.write_bytes(f"{cid} {offset}".encode())
            frames.append(dict(case_id=cid, offset=offset, output_sha256=module.sha(image)))
        priorities.append(dict(case_id=cid, human=y, rel_road_logistic=pred, video=scene, original_sha256=module.sha(images / "center.jpg")))
    for filename, values in [("private_priority.csv", priorities), ("frame_audit.csv", frames)]:
        with (root / filename).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(values[0])); w.writeheader(); w.writerows(values)
    source = root / "source_labels.csv"
    source.write_bytes(b"original labels stay fixed")
    (root / "method.json").write_text(json.dumps(dict(status="complete", input_sha256={str(source): module.sha(source)})), encoding="utf-8")
    out = private / "result"
    result = module.audit(root, observations, out)
    assert (result["cases"], result["frames"], result["task_records"]) == (2, 10, 6)
    assert result["scores"]["all_prepared"]["fixed_relative_road_logistic"]["n"] == 2
    assert result["scores"]["forward_or_aligned_only"]["fixed_relative_road_logistic"]["n"] == 1
    assert "&lt;evidence&gt;" in (out / "report.html").read_text(encoding="utf-8")
    assert source.read_bytes() == b"original labels stay fixed"
    with pytest.raises(ValueError, match="new output"):
        module.audit(root, observations, out)
    (root / "review/T001/center.jpg").write_bytes(b"modified")
    with pytest.raises(ValueError, match="image changed"):
        module.audit(root, observations, private / "must_not_exist")
    assert not (private / "must_not_exist").exists()
