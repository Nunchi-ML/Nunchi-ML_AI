"""Weather cannot silently use future or unpublished observations."""
import importlib
from pathlib import Path

import pandas as pd
import pytest

from nunchi_ml.weather import WEATHER_COLUMNS, align_hourly_weather, observation_hours


def observations(times, **extra):
    return pd.DataFrame({"tm": times, **{c: list(range(1, len(times) + 1)) for c in WEATHER_COLUMNS}, **extra})


def test_preceding_hour_preserves_order_duplicate_indices_and_exact_hour():
    t = pd.Series(["2026-11-04 15:45", "2026-11-04 15:00"], index=[7, 7])
    result = align_hourly_weather(t, observations(["2026-11-04 15:00", "2026-11-04 16:00"]))
    assert result.index.tolist() == [7, 7]
    assert result.cloud.tolist() == [1, 1]
    assert result.weather_age_minutes.tolist() == [45, 0]
    assert not result.weather_availability_verified.any()


def test_missing_hour_does_not_take_future_or_stale_weather():
    result = align_hourly_weather(["2026-11-04 15:45"], observations(["2026-11-04 14:00", "2026-11-04 16:00"]))
    assert result[list(WEATHER_COLUMNS)].isna().all().all()
    assert result.weather_observed_at.isna().all()


def test_timezone_aware_utc_capture_matches_korean_hour():
    result = align_hourly_weather(["2026-11-04T06:45:00Z"], observations(["2026-11-04 15:00"]))
    assert result.cloud.iloc[0] == 1
    assert result.weather_age_minutes.iloc[0] == 45


def test_year_boundary_legacy_nearest_and_previous_are_explicit():
    t = ["2026-12-31 23:45"]
    assert observation_hours(t).iloc[0] == pd.Timestamp("2026-12-31 23:00")
    assert observation_hours(t, "nearest").iloc[0] == pd.Timestamp("2027-01-01 00:00")


def test_publication_after_capture_or_unknown_is_withheld():
    a = observations(["2026-11-04 15:00"], received=["2026-11-04 15:10"])
    result = align_hourly_weather(["2026-11-04 15:05", "2026-11-04 15:20"], a,
                                  available_at_col="received", require_available=True)
    assert pd.isna(result.cloud.iloc[0])
    assert result.cloud.iloc[1] == 1
    assert result.weather_availability_verified.tolist() == [False, True]
    a["received"] = None
    assert align_hourly_weather(["2026-11-04 15:20"], a, available_at_col="received").cloud.isna().all()


@pytest.mark.parametrize("kwargs", [{"require_available": True},
                                    {"require_available": True, "available_at_col": "received", "alignment": "nearest"}])
def test_live_checks_cannot_claim_historical_cache_is_available(kwargs):
    with pytest.raises(ValueError, match="Availability checks"):
        align_hourly_weather(["2026-11-04 15:45"], observations(["2026-11-04 15:00"]), **kwargs)


def test_unknown_weather_is_not_imputed_to_clear_sky():
    a = observations(["2026-11-04 15:00"], cloud=[None], sunshine=[None])
    result = align_hourly_weather(["2026-11-04 15:45"], a)
    assert result[["cloud", "sunshine"]].isna().all().all()


@pytest.mark.parametrize("times", [[None], ["invalid timestamp"]])
def test_invalid_capture_time_is_rejected(times):
    with pytest.raises(ValueError):
        align_hourly_weather(times, observations(["2026-11-04 15:00"]))


def test_ambiguous_observations_and_impossible_publication_are_rejected():
    with pytest.raises(ValueError, match="unique timestamps"):
        align_hourly_weather(["2026-11-04 15:45"], observations(["2026-11-04 15:00"] * 2))
    with pytest.raises(ValueError, match="before its observation"):
        align_hourly_weather(["2026-11-04 15:45"], observations(["2026-11-04 15:00"], received=["2026-11-04 14:00"]),
                             available_at_col="received")


def test_fetch_wrapper_loads_selected_day_and_preserves_station(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "notebooks/experiments/glare_video_source"
    monkeypatch.syspath_prepend(str(scripts))
    fetch = importlib.import_module("fetch_asos")
    requests = []

    def cached(days, stn):
        requests.append((list(days), stn))
        return observations(["2027-01-01 00:00"])

    monkeypatch.setattr(fetch, "ensure", cached)
    result = fetch.attach(pd.DataFrame({"captured_at_kst": ["2026-12-31 23:45"]}), stn=108, alignment="nearest")
    assert requests == [(["20270101"], 108)]
    assert result.cloud.iloc[0] == 1
    assert result.weather_age_minutes.iloc[0] == -15
    assert not result.weather_availability_verified.iloc[0]


def test_fetch_wrapper_defaults_to_previous_without_mutating_cache(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "notebooks/experiments/glare_video_source"
    monkeypatch.syspath_prepend(str(scripts))
    fetch = importlib.import_module("fetch_asos")
    weather = observations(["2026-11-04 15:00", "2026-11-04 16:00"])
    original = weather.copy(deep=True)
    monkeypatch.setattr(fetch, "ensure", lambda days, stn: weather)
    result = fetch.attach(pd.DataFrame({"captured_at_kst": ["2026-11-04 15:45"]}))
    assert result.cloud.iloc[0] == 1
    assert result.weather_time_policy.iloc[0] == "previous"
    assert not result.weather_availability_verified.iloc[0]
    pd.testing.assert_frame_equal(weather, original)


def test_offline_replay_recomputes_serialized_metadata(monkeypatch, tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "notebooks/experiments/glare_video_source"
    monkeypatch.syspath_prepend(str(scripts))
    probe = importlib.import_module("g_weather_time_probe")
    observations(["2026-11-04 15:00", "2026-11-04 16:00"]).to_csv(tmp_path / "daegu_143.csv", index=False)
    snapshot = pd.DataFrame({"camera": ["도로 광각"], "area": [None],
                             "captured_at_kst": ["2026-11-04 15:45"],
                             "weather_observed_at": ["2026-11-04 16:00"],
                             "weather_availability_verified": [True]})
    result = probe.replay(snapshot, tmp_path, "previous")
    assert result.cloud.iloc[0] == 1
    assert result.weather_observed_at.iloc[0] == pd.Timestamp("2026-11-04 15:00")
    assert not result.weather_availability_verified.iloc[0]
    assert snapshot.weather_availability_verified.iloc[0]
