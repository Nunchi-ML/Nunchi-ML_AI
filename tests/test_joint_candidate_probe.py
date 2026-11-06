"""Same-photo orchestration transport tests use explicit synthetic fixtures."""
from copy import deepcopy
import importlib
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("joint_candidate_probe")


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / "fixture.jpg"
    assert cv2.imwrite(str(path), np.full((24, 32, 3), 100, np.uint8))
    metadata = dict(lat=35.8, lon=128.6, heading=90., focal=.55, width=32, height=24,
                    captured_at="2026-11-06T08:00:00+09:00", cloud=0., sunshine=.5,
                    weather_observed_at="2026-11-06T08:00:00+09:00", weather_source="synthetic fixture",
                    geometry_source="synthetic fixture", projection="legacy_level_assumed")
    return path, metadata


class FixturePredictor:
    """Contract double, never claimed as actual trained inference."""
    meta = {"parameters_sha256": "a" * 64}

    def __init__(self, name):
        self.name, self.calls = name, []

    def predict_image(self, image, metadata, variant=None, *, zone=None):
        self.calls.append((image, deepcopy(metadata), variant, zone))
        result = dict(model_version="photo-g-candidate-v1" if self.name == 2 else "model3-ridge-two-stage-candidate-v1",
                      variant="B_geometry_aug" if self.name == 2 else "ridge_estimated_luminance_two_stage",
                      status="candidate", level=np.array([2]), alert_score=np.array([.8]),
                      joint_scores=np.array([[.2, .16, .64]]))
        if self.name == 2:
            result["conditional_severe_score"] = np.array([.8])
        else:
            result.update(conditional_high_score=np.array([.8]), estimated_luminance_cd_m2=np.array([5000.]))
        return result


def run(module, sample, evidence="recorded_photo", date="2026-11-06T08:00:05+09:00", **changes):
    path, metadata = sample
    m2, m3 = FixturePredictor(2), FixturePredictor(3)
    result = module.run_photo(path, metadata, model2=m2, model3=m3,
        observation_id="synthetic-fixture", segment_id="synthetic-fixture", zone="entry",
        zone_source="synthetic-contract-only", policy=module.Policy(10, 5, 5), evidence=evidence, now=lambda: date, **changes)
    return result, m2, m3


def test_same_bytes_metadata_and_separate_preprocessors(module, sample):
    result, m2, m3 = run(module, sample)
    assert result["same_photo_bytes_checked"]
    assert result["model2"]["context"] == result["model3"]["context"]
    assert m2.calls[0][0].shape == (24, 32, 3) and m2.calls[0][2] == "B_geometry_aug"
    assert "u" in m2.calls[0][1] and "sun_alt" in m2.calls[0][1]
    assert m3.calls[0][0] == sample[0] and m3.calls[0][1] == sample[1]
    assert m3.calls[0][3] == "entry"
    assert result["guidance"]["requested_level"] == 2
    assert not result["scene_verified_by_probe"] and not result["route_coverage_verified"]


def test_synthetic_source_never_becomes_current_model_guidance(module, sample):
    result, _, _ = run(module, sample, evidence="synthetic_contract")
    assert result["raw_model2"]["level"] == [2] and result["raw_model3"]["level"] == [2]
    assert result["guidance"]["requested_level"] is None
    assert result["guidance"]["completeness"] == "unknown"


def test_historical_photo_does_not_become_fresh_by_running_models_today(module, sample):
    result, _, _ = run(module, sample, date="2026-11-07T08:00:05+09:00")
    assert result["raw_model2"]["level"] == [2]
    assert result["guidance"]["requested_level"] is None
    for item in result["guidance"]["components"].values():
        assert "stale_photo" in item["reasons"]


def test_changing_photo_between_models_is_detected(module, sample):
    m3 = FixturePredictor(3)
    original = m3.predict_image
    def replace(path, metadata, *, zone):
        result = original(path, metadata, zone=zone)
        path.write_bytes(b"changed")
        return result
    m3.predict_image = replace
    with pytest.raises(ValueError, match="changed"):
        module.run_photo(sample[0], sample[1], model2=FixturePredictor(2), model3=m3,
            observation_id="synthetic", segment_id="synthetic", zone="entry", zone_source="synthetic",
            policy=module.Policy(10, 5, 5), now=lambda: "2026-11-06T08:00:05+09:00")


@pytest.mark.parametrize("field,value", [("focal", None), ("width", 31), ("heading", 360),
                                      ("cloud", None), ("captured_at", "2026-11-06T08:00:00")])
def test_bad_photo_context_never_reaches_either_model(module, sample, field, value):
    m2, m3 = FixturePredictor(2), FixturePredictor(3)
    with pytest.raises(ValueError):
        module.run_photo(sample[0], {**sample[1], field: value}, model2=m2, model3=m3,
            observation_id="synthetic", segment_id="synthetic", zone="entry", zone_source="synthetic",
            policy=module.Policy(10, 5, 5), now=lambda: "2026-11-06T08:00:05+09:00")
    assert not m2.calls and not m3.calls


def test_unidentified_scene_is_not_assumed_to_be_an_entrance(module, sample):
    m2, m3 = FixturePredictor(2), FixturePredictor(3)
    with pytest.raises(ValueError, match="entrance"):
        module.run_photo(sample[0], sample[1], model2=m2, model3=m3,
            observation_id="synthetic", segment_id="synthetic", zone="unknown", zone_source="",
            policy=module.Policy(10, 5, 5))
    assert not m2.calls and not m3.calls
