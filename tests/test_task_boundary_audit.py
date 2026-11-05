import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("task_boundary_audit")


@pytest.fixture
def records():
    originals = {cid: dict(targets=[dict(target_id=cid + "_" + task,
                                       center_normalized_box_or_corridor=[.1, .1, .2, .2])
                                  for task in ["lane", "vehicle", "sign"]]) for cid in ["T001", "T022"]}
    notes = [dict(case_id=cid, reviewed_offsets=[-2, -1, 0, 1, 2], distant_information="not_established",
                  same_target_recovery="not_established", solar_grounding="uncertain", scope="forward_or_aligned_road",
                  visual_evidence="visible representative vehicle; distant area unmeasured",
                  temporal_confound="camera and objects move", boundary_issue="near target does not clear distant information",
                  checked_target_ids=[t["target_id"] for t in originals[cid]["targets"]]) for cid in originals]
    rois = [dict(target_id=cid + "_sign", old_box=[.1, .1, .2, .2], region_result="panel_located",
                 kind="guide_panel", audited_box=[.1, .1, .2, .2], evidence="panel within old region", global_loss_inference="none") for cid in originals]
    return originals, notes, rois


def test_draft_does_not_make_unknown_targets_normal_or_create_whole_image_grades(module):
    draft = module.boundary_draft()
    assert draft["status"] == "proposal_not_applied" and draft["labels_created"] is False
    assert "never fill with0" in draft["abstention"]
    assert "UNRESOLVED" in draft["aggregation"] and "whole-frame" in draft["aggregation"]


def test_complete_posthoc_notes_and_region_inventory(module, records):
    originals, notes, rois = records
    module.validate_notes(notes, dict(priority_cases=list(originals)), originals)
    module.validate_rois(rois, originals)


@pytest.mark.parametrize("key", ["label", "new_label", "proposed_grade", "global_grade"])
def test_rejects_grade_assignment_in_evidence_audit(module, records, key):
    originals, notes, _ = records
    notes[0][key] = 0
    with pytest.raises(ValueError, match="grades"):
        module.validate_notes(notes, dict(priority_cases=list(originals)), originals)


@pytest.mark.parametrize("fault", ["missing_case", "duplicate_case", "missing_frame", "wrong_target", "distant_clear", "causal_recovery", "empty_evidence", "invalid_scope"])
def test_rejects_missing_evidence_or_global_overclaims(module, records, fault):
    originals, notes, _ = copy.deepcopy(records)
    if fault == "missing_case":
        notes.pop()
    elif fault == "duplicate_case":
        notes[1] = notes[0]
    elif fault == "missing_frame":
        notes[0]["reviewed_offsets"] = [0]
    elif fault == "wrong_target":
        notes[0]["checked_target_ids"][0] = "another_vehicle"
    elif fault == "distant_clear":
        notes[0]["distant_information"] = "intact"
    elif fault == "causal_recovery":
        notes[0]["same_target_recovery"] = "confirmed"
    elif fault == "empty_evidence":
        notes[0]["temporal_confound"] = ""
    else:
        notes[0]["scope"] = "driver_vision_validated"
    with pytest.raises(ValueError):
        module.validate_notes(notes, dict(priority_cases=list(originals)), originals)


def test_wrong_object_and_location_correction_remain_separate_from_grade(module, records):
    originals, _, rois = records
    rois[0].update(region_result="wrong_object", kind="non_sign_structure", audited_box=None)
    rois[1].update(region_result="mislocalized_corrected", kind="panel_type_uncertain", audited_box=[.3, .3, .4, .4])
    module.validate_rois(rois, originals)
    assert originals["T022"]["targets"][-1]["center_normalized_box_or_corridor"] == [.1, .1, .2, .2]


@pytest.mark.parametrize("fault", ["missing_region", "modified_original", "invalid_box", "fake_correction", "wrong_object_kept", "global_inference"])
def test_rejects_unsupported_region_corrections(module, records, fault):
    originals, _, rois = copy.deepcopy(records)
    if fault == "missing_region":
        rois.pop()
    elif fault == "modified_original":
        rois[0]["old_box"] = [.2, .2, .3, .3]
    elif fault == "invalid_box":
        rois[0].update(region_result="mislocalized_corrected", audited_box=[.5, .5, .3, .3])
    elif fault == "fake_correction":
        rois[0]["region_result"] = "mislocalized_corrected"
    elif fault == "wrong_object_kept":
        rois[0].update(region_result="wrong_object", kind="non_sign_structure")
    else:
        rois[0]["global_loss_inference"] = "all_clear"
    with pytest.raises(ValueError):
        module.validate_rois(rois, originals)
