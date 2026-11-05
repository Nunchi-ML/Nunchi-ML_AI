import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("auxiliary_label_audit")


def accepted_record():
    return dict(image_sha256="train", group="capture1", source_kind="adjudicated_annotation",
                quality_accepted=True, task_version="lane_presence_v1", value=0)


def test_missing_is_not_normal_but_explicit_zero_is_valid(module):
    record = accepted_record()
    assert module.admission_reasons(record, {"train"}, set(), set()) == []
    record["value"] = None
    assert "missing_value" in module.admission_reasons(record, {"train"}, set(), set())


@pytest.mark.parametrize("reserved_hashes,reserved_groups", [({"train"}, set()), (set(), {"capture1"})])
def test_reserved_photo_or_capture_group_cannot_enter_training(module, reserved_hashes, reserved_groups):
    assert "reserved_evaluation_photo_or_group" in module.admission_reasons(
        accepted_record(), {"train"}, reserved_hashes, reserved_groups)


def test_proxies_and_unreviewed_ai_are_not_accepted_annotations(module):
    for kind in ("generated_feature", "ai_reference", "blank_template"):
        record = {**accepted_record(), "source_kind": kind, "quality_accepted": False}
        reasons = module.admission_reasons(record, {"train"}, set(), set())
        assert {"not_adjudicated_annotation", "quality_not_accepted"} <= set(reasons)


def test_mapping_requires_same_training_image_bytes_and_exact_scope(module):
    rows = [{"review_id": "R1", **dict.fromkeys(module.TASKS, "")}]
    mapping = [dict(review_id="R1", id="a", image_sha256="same")]
    assert set(module.template_counts(rows, mapping, {"a": {"image_sha256": "same"}}).values()) == {0}
    with pytest.raises(ValueError, match="hash mismatch"):
        module.template_counts(rows, mapping, {"a": {"image_sha256": "other"}})
    with pytest.raises(ValueError, match="identity mismatch"):
        module.template_counts(rows, mapping * 2, {"a": {"image_sha256": "same"}})


def test_repeated_id_and_existing_output_fail_before_any_mutation(module, tmp_path):
    source = tmp_path / "bad.csv"
    source.write_text("id,value\na,0\na,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        module.table(source, "id")
    with pytest.raises(ValueError, match="new private output"):
        module.audit(tmp_path, tmp_path)
