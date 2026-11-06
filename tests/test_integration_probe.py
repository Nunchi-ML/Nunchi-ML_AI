"""Synthetic contract cases, not evidence of model3 accuracy or paired data."""
from copy import deepcopy
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("integration_probe")


def query():
    return dict(observation_id="synthetic-photo", segment_id="synthetic-entry", image_sha256="a" * 64,
                lat=35.8, lon=128.6, heading=1., captured_at="2026-11-06T08:00:00+09:00",
                as_of="2026-11-06T08:00:05+09:00", eta="2026-11-06T08:00:05+09:00", zone="entry")


def component(model, level, a=None, b=None):
    a = (.2 if level == 0 else .8) if a is None else a
    b = (.8 if level == 2 else .2) if b is None else b
    context = {k: v for k, v in query().items() if k not in ("as_of", "eta", "zone")}
    context.update(received_at="2026-11-06T08:00:01+09:00", heading=359.)
    return dict(context=context, source="recorded_photo", status="candidate", artifact_sha256="b" * 64,
                model_version="photo-g-candidate-v1" if model == 2 else "synthetic-model3-not-a-trained-artifact",
                variant="B_geometry_aug" if model == 2 else "ridge_estimated_luminance_two_stage",
                target="photo_glare_0_1_2" if model == 2 else "entrance_contrast_low_mid_high",
                zone="entry", estimated_luminance_cd_m2=1000., level=level,
                alert_score=a, conditional_high_score=b, joint_scores=[1-a, a*(1-b), a*b])


def compose(module, a=0, b=0, q=None):
    return module.compose_guidance(query() if q is None else q,
                                   None if a is None else component(2, a),
                                   None if b is None else component(3, b), module.Policy(10, 5, 5))


@pytest.mark.parametrize("a,b", [(a, b) for a in range(3) for b in range(3)])
def test_all_joint_grade_combinations_preserve_originals_and_causes(module, a, b):
    result = compose(module, a, b)
    assert result["current_observation_level"] == result["requested_level"] == max(a, b)
    assert result["completeness"] == "complete"
    assert result["components"]["model2"]["level"] == a
    assert result["components"]["model3"]["level"] == b
    assert result["causes"] == [name for name, value in [("model2", a), ("model3", b)] if value > 0]
    assert result["accident_probability"] is None
    assert not result["policy_validated"] and not result["adaptation_time_validated"]
    assert not result["artifact_bytes_verified_by_probe"] and not result["inference_provenance_verified_by_probe"]


@pytest.mark.parametrize("a,b,level,completeness", [(None, None, None, "unknown"),
    (0, None, None, "partial"), (None, 0, None, "partial"),
    (2, None, 2, "partial"), (None, 1, 1, "partial")])
def test_missing_component_never_becomes_full_normal(module, a, b, level, completeness):
    result = compose(module, a, b)
    assert result["current_observation_level"] == level
    assert result["completeness"] == completeness
    assert result["current_level_is_lower_bound"] == (level is not None)
    if level is None:
        assert result["requested_state"] == "unknown"


@pytest.mark.parametrize("field,value,reason", [
    ("observation_id", "other", "observation_mismatch"), ("segment_id", "other", "segment_mismatch"),
    ("image_sha256", "c" * 64, "photo_mismatch"), ("captured_at", "2026-11-06T08:00:02+09:00", "invalid_component"),
    ("lat", 35.9, "position_mismatch"), ("heading", 180., "direction_mismatch"),
    ("captured_at", "2026-11-06T07:59:00+09:00", "stale_photo"),
    ("received_at", "2026-11-06T08:00:06+09:00", "not_available_at_query"),
])
def test_misaligned_model_cannot_clear_or_warn_for_another_photo(module, field, value, reason):
    m3 = component(3, 2)
    m3["context"][field] = value
    result = module.compose_guidance(query(), component(2, 0), m3, module.Policy(10, 5, 5))
    assert result["requested_level"] is None
    assert result["components"]["model3"]["state"] == "unknown"
    assert reason in result["components"]["model3"]["reasons"]
    assert result["causes"] == []


@pytest.mark.parametrize("field,value", [("level", True), ("level", 0), ("alert_score", float("nan")),
    ("estimated_luminance_cd_m2", -1), ("estimated_luminance_cd_m2", float("inf")),
    ("artifact_sha256", ""), ("joint_scores", [.1, .1, .1]),
    ("source", "simulator-example"), ("status", "rule"), ("target", "pupil_diameter"),
    ("variant", "legacy_brightness_formula")])
def test_invalid_or_legacy_model3_stays_unknown(module, field, value):
    m3 = component(3, 2)
    m3[field] = value
    result = module.compose_guidance(query(), component(2, 0), m3, module.Policy(10, 5, 5))
    assert result["requested_level"] is None and result["missing_components"] == ["model3"]


@pytest.mark.parametrize("zone", ["road", "exit", "unknown"])
def test_entrance_only_model_cannot_validate_exit_or_other_zones(module, zone):
    result = compose(module, 0, 2, q={**query(), "zone": zone})
    assert result["requested_level"] is None
    assert result["components"]["model3"]["reasons"] == ["unsupported_zone"]


def test_future_eta_cannot_turn_photo_into_future_state(module):
    result = compose(module, 2, 1, q={**query(), "eta": "2026-11-06T08:01:00+09:00"})
    assert result["current_observation_level"] == 2 and result["requested_level"] is None
    assert result["requested_state"] == "unknown" and result["reasons"] == ["future_state_unverified"]
    result = compose(module, 0, 0, q={**query(), "eta": "2026-11-06T08:00:04+09:00"})
    assert result["requested_level"] is None and result["reasons"] == ["past_eta"]


def test_exact_binary_ties_follow_each_original_model(module):
    m2, m3 = component(2, 2, .5, .5), component(3, 0, .5, .8)
    result = module.compose_guidance(query(), m2, m3, module.Policy(10, 5, 5))
    assert result["components"]["model2"]["level"] == 2
    assert result["components"]["model3"]["level"] == 0
    m3 = component(3, 1, .8, .5)
    result = module.compose_guidance(query(), component(2, 0), m3, module.Policy(10, 5, 5))
    assert result["components"]["model3"]["level"] == 1


def test_query_and_trial_policy_must_be_explicit_and_valid(module):
    for field, value in [("heading", True), ("heading", 360), ("as_of", "2026-11-06T08:00:05"),
                         ("lat", float("nan")), ("image_sha256", ""), ("zone", "assumed-entry")]:
        with pytest.raises(ValueError):
            compose(module, q={**query(), field: value})
    for args in [(float("nan"), 5, 5), (10, -1, 5), (10, 5, 91), (True, 5, 5)]:
        with pytest.raises(ValueError):
            module.Policy(*args)


def test_batch_has_distinct_ids_no_input_mutation_and_no_faked_accuracy(module):
    case = dict(case_id="synthetic-contract", evidence="synthetic_contract_not_ml_accuracy", query=query(),
                model2=component(2, 0), model3=None,
                policy=dict(max_age_seconds=10, max_distance_m=5, heading_tolerance_degrees=5))
    before = deepcopy(case)
    result = module.run_cases([case])[0]
    assert case == before
    assert result["evidence"] == "synthetic_contract_not_ml_accuracy"
    assert result["completeness"] == "partial" and result["requested_level"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        module.run_cases([case, case])
    with pytest.raises(ValueError, match="Nonempty"):
        module.run_cases([])


def test_cli_labels_synthetic_counts_and_refuses_overwriting_results(module, tmp_path, monkeypatch):
    case = dict(case_id="contract-fixture", evidence="synthetic_contract", query=query(),
                model2=component(2, 0), model3=component(3, 0),
                policy=dict(max_age_seconds=10, max_distance_m=5, heading_tolerance_degrees=5))
    source, out = tmp_path / "input.json", tmp_path / "out"
    source.write_text(json.dumps([case]), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["integration_probe", "--input", str(source), "--out", str(out)])
    module.main()
    report = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert report["evidence_counts"] == {"synthetic_contract": 1}
    assert report["contract_readiness_counts"]["complete"] == 1
    assert report["counts_are_contract_checks_not_real_inference_counts"]
    assert not report["real_joint_accuracy_evaluated"]
    before = (out / "results.json").read_bytes()
    with pytest.raises(FileExistsError):
        module.main()
    assert (out / "results.json").read_bytes() == before
