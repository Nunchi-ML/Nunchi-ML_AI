"""Hourly weather alignment; observation time does not prove publication time."""
from collections.abc import Sequence
from typing import Literal

import pandas as pd

Alignment = Literal["previous", "nearest"]
WEATHER_COLUMNS = ("cloud", "sunshine", "radiation", "visibility", "rain")


def _kst(values) -> pd.Series:
    """Naive timestamps mean KST; aware timestamps are converted to KST."""
    times = pd.to_datetime(pd.Series(values), errors="raise")
    if times.isna().any():
        raise ValueError("Weather alignment requires non-missing timestamps")
    if times.dt.tz is not None:
        times = times.dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    return times


def observation_hours(times, alignment: Alignment = "previous") -> pd.Series:
    """Select the preceding hour, or reproduce the legacy nearest-hour study."""
    t = _kst(times)
    if alignment == "previous":
        return t.dt.floor("h")
    if alignment == "nearest":
        return t.dt.round("h")
    raise ValueError("alignment must be 'previous' or 'nearest'")


def align_hourly_weather(
    times,
    observations: pd.DataFrame,
    *,
    alignment: Alignment = "previous",
    columns: Sequence[str] = WEATHER_COLUMNS,
    available_at_col: str | None = None,
    require_available: bool = False,
) -> pd.DataFrame:
    """Join the exact selected hour; missing rows stay missing without fallback.

    ``previous`` prevents future observations, but historical ASOS rows have no
    proof of availability at capture time. A live/replay evaluation must supply
    publication/receipt timestamps and set ``require_available=True``. Provided
    availability timestamps are always enforced; unavailable values stay NaN.
    No publication delay is guessed. ``nearest`` is historical reproduction only.
    """
    t = _kst(times)
    hours = observation_hours(t, alignment)
    if require_available and (available_at_col is None or alignment != "previous"):
        raise ValueError("Availability checks require previous alignment and an availability column")
    a = observations.copy()
    a["tm"] = _kst(a["tm"]).to_numpy()
    if a.tm.duplicated().any():
        raise ValueError("Hourly observations must have unique timestamps for one station")
    a[list(columns)] = a[list(columns)].apply(pd.to_numeric, errors="coerce")
    a["weather_observed_at"] = a.tm
    if available_at_col is not None:
        # Unknown publication times must remain unknown, not default to tm.
        raw = a[available_at_col]
        known = raw.notna()
        available = pd.Series(pd.NaT, index=a.index, dtype="datetime64[ns]")
        available.loc[known] = _kst(raw[known]).to_numpy()
        if (available[known] < a.loc[known, "tm"]).any():
            raise ValueError("Weather cannot be available before its observation time")
        a["weather_available_at"] = available
    else:
        a["weather_available_at"] = pd.NaT
    result = a.set_index("tm").reindex(hours.to_numpy())[
        [*columns, "weather_observed_at", "weather_available_at"]
    ].copy()
    result.index = t.index
    result["weather_requested_hour"] = hours.to_numpy()
    result["weather_age_minutes"] = (t - result.weather_observed_at).dt.total_seconds() / 60
    result["weather_time_policy"] = alignment
    verified = (result.weather_available_at.notna()
                & result.weather_available_at.le(t)
                & result.weather_observed_at.le(t))
    result["weather_availability_verified"] = verified
    if available_at_col is not None:
        result.loc[~verified, list(columns)] = float("nan")
    return result
