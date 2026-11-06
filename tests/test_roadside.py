import numpy as np
import pytest

from nunchi_ml.glare.roadside import associate_observation, photo_g


def observation(**overrides):
    return dict(camera_id="synthetic", segment_id="trial", heading=359.,
                observed_at="2026-11-06T08:00:00+09:00", received_at="2026-11-06T08:00:01+09:00",
                image_sha256="a" * 64, model_version="synthetic-test", level=2,
                coverage_verified=True, coverage_source="synthetic-test-fixture", **overrides)


def associate(obs, **segment):
    return associate_observation(obs, dict(segment_id="trial", heading=1., eta="2026-11-06T08:00:05+09:00", **segment),
                                 as_of="2026-11-06T08:00:05+09:00", max_age_seconds=10,
                                 heading_tolerance_degrees=5)


def test_matching_current_observation_wraps_north_heading():
    result = associate(observation())
    assert result["status"] == "candidate_observation" and result["level"] == 2


def test_current_photo_is_not_a_future_prediction():
    result = associate_observation(observation(), dict(segment_id="trial", heading=1., eta="2026-11-06T08:01:00+09:00"),
                                   as_of="2026-11-06T08:00:05+09:00", max_age_seconds=10,
                                   heading_tolerance_degrees=5)
    assert result["level"] is None and result["current_observation_level"] == 2
    assert result["reasons"] == ["future_state_unverified"]


@pytest.mark.parametrize("changes,reason", [
    ({"heading": 180.}, "direction_mismatch"),
    ({"coverage_verified": False}, "coverage_unverified"),
    ({"segment_id": "another"}, "outside_verified_segment"),
    ({"observed_at": "2026-11-06T07:59:00+09:00"}, "stale_photo"),
    ({"received_at": "2026-11-06T08:00:10+09:00"}, "not_available_at_query")])
def test_unusable_observations_keep_grade_unknown(changes, reason):
    obs = observation()
    obs.update(changes)
    result = associate(obs)
    assert result["level"] is None and result["current_observation_level"] is None
    assert reason in result["reasons"]


def test_no_camera_and_naive_timestamp_are_not_normal():
    assert associate(None)["reasons"] == ["not_observed"]
    obs = observation()
    obs["observed_at"] = "2026-11-06 08:00:00"
    with pytest.raises(ValueError, match="timezone"):
        associate(obs)


def metadata():
    return dict(lat=35.86, lon=128.59, heading=100., focal=.8, width=640, height=368,
                captured_at="2026-11-06T08:30:00+09:00", cloud=2., sunshine=.8,
                weather_observed_at="2026-11-06T08:00:00+09:00", weather_source="synthetic-weather",
                geometry_source="synthetic-camera", projection="legacy_level_assumed")


def test_photo_geometry_requires_provenance_and_exposes_pose_assumption():
    m = metadata()
    result = photo_g(m, (368, 640, 3))
    assert np.isfinite(list(result["features"].values())).all()
    assert result["camera_pose_validated"] is False
    assert result["weather_availability_verified"] is False
    for key in ("heading", "focal", "geometry_source", "captured_at"):
        invalid = dict(m)
        invalid.pop(key)
        with pytest.raises(ValueError, match="Missing"):
            photo_g(invalid, (368, 640, 3))
    with pytest.raises(ValueError, match="dimensions"):
        photo_g(m, (320, 640, 3))


def test_future_weather_and_missing_weather_are_rejected():
    for key, value in [("weather_observed_at", "2026-11-06T09:00:00+09:00"),
                       ("weather_available_at", "2026-11-06T08:35:00+09:00"),
                       ("cloud", None)]:
        m = metadata()
        m[key] = value
        with pytest.raises(ValueError):
            photo_g(m, (368, 640, 3))
