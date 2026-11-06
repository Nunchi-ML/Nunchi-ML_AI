"""Offline photo-time geometry and conservative route association for candidates.

No network calls, inferred camera bearings, or conversion of unknown to normal.
Policy limits are caller-supplied trial settings, not validated warning limits.
"""
import re

import numpy as np
import pandas as pd

from nunchi_ml.sun import sun_position
from .hybrid import geometry


def timestamp(value):
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError):
        raise ValueError("An explicit timestamp with timezone is required") from None
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("An explicit timestamp with timezone is required")
    return result


def number(value, low, high, name):
    if isinstance(value, (bool, str)) or not np.isscalar(value):
        raise ValueError("Numeric input required: " + name)
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError("Numeric input required: " + name) from None
    if not np.isfinite(value) or not low <= value <= high:
        raise ValueError("Input out of range: " + name)
    return value


def photo_g(metadata, image_shape):
    """Legacy projection at observation time; explicit level-pose assumption.

The training table rounded sun angles to .01 degrees and heading gap to .1.
The legacy projection clips side angle to +/-89, including rear rays, solely
to reproduce the existing candidate. It is not camera-pose validation.
"""
    required = {"lat", "lon", "heading", "focal", "width", "height", "captured_at",
                "cloud", "sunshine", "weather_observed_at", "weather_source",
                "geometry_source", "projection"}
    if not required <= set(metadata):
        raise ValueError("Missing photo/G metadata: " + ", ".join(sorted(required - set(metadata))))
    if metadata["projection"] != "legacy_level_assumed":
        raise ValueError("Explicit legacy_level_assumed projection required")
    if any(not isinstance(metadata[k], str) or not metadata[k].strip()
           for k in ("geometry_source", "weather_source")):
        raise ValueError("Geometry/weather provenance is required")
    lat = number(metadata["lat"], -90, 90, "lat")
    lon = number(metadata["lon"], -180, 180, "lon")
    heading = number(metadata["heading"], 0, np.nextafter(360., 0), "heading")
    focal = number(metadata["focal"], 1e-6, 100, "focal")
    width = number(metadata["width"], 1, 100000, "width")
    height = number(metadata["height"], 1, 100000, "height")
    if len(image_shape) != 3 or tuple(image_shape[:2]) != (height, width):
        raise ValueError("Camera metadata and raw photo dimensions differ")
    captured, weather_at = timestamp(metadata["captured_at"]), timestamp(metadata["weather_observed_at"])
    weather_age = (captured - weather_at).total_seconds()
    if not 0 <= weather_age <= 3600:
        raise ValueError("Previous/identical-hour weather within one hour required")
    available = metadata.get("weather_available_at")
    if available is not None and timestamp(available) > captured:
        raise ValueError("Weather became available after the photo")
    cloud = number(metadata["cloud"], 0, 10, "cloud")
    sunshine = number(metadata["sunshine"], 0, 1, "sunshine")
    position = sun_position([captured], [lat], [lon]).iloc[0]
    alt, az = np.round([position.altitude, position.azimuth], 2)
    side = (az - heading + 180) % 360 - 180
    angle = np.radians(np.clip(side, -89, 89))
    fpx = focal * max(width, height)
    u = np.tan(angle) * 2 * fpx / width
    v = np.tan(np.radians(np.clip(alt, -89, 89))) * 2 * fpx / height / np.cos(angle)
    row = dict(sun_alt=alt, u=u, v=v,
               in_fov=float(abs(u) <= 1 and 0 < v <= 1 and abs(side) < 90),
               gap=np.round(abs(side), 1), cloud=cloud, sunshine=sunshine)
    return dict(features=geometry(pd.DataFrame([row])).iloc[0].to_dict(),
                captured_at=captured.isoformat(), weather_age_seconds=weather_age,
                weather_availability_verified=available is not None,
                projection="legacy_level_assumed", camera_pose_validated=False)


def associate_observation(observation, segment, *, as_of, max_age_seconds,
                          heading_tolerance_degrees):
    """Attach a recorded current condition to one verified segment/direction.

For future ETA expose current observation separately and keep future level
unknown. This candidate has no validated photo-to-future transfer model.
"""
    now, eta = timestamp(as_of), timestamp(segment["eta"])
    age_limit = number(max_age_seconds, 0, 86400, "max_age_seconds")
    tolerance = number(heading_tolerance_degrees, 0, 90, "heading_tolerance_degrees")
    route_heading = number(segment["heading"], 0, np.nextafter(360., 0), "route heading")
    result = dict(segment_id=segment["segment_id"], eta=eta.isoformat(),
                  level=None, current_observation_level=None, status="unknown", reasons=[])
    if observation is None:
        result["reasons"] = ["not_observed"]
        return result
    observed, received = timestamp(observation["observed_at"]), timestamp(observation["received_at"])
    heading = number(observation["heading"], 0, np.nextafter(360., 0), "camera heading")
    if observation["level"] not in (0, 1, 2) or isinstance(observation["level"], bool):
        raise ValueError("Original grade 0/1/2 required")
    if not re.fullmatch(r"[a-f0-9]{64}", observation["image_sha256"]):
        raise ValueError("Actual photo SHA256 is required")
    if received < observed:
        raise ValueError("Receipt cannot precede observation")
    reasons = result["reasons"]
    if observation["segment_id"] != segment["segment_id"]:
        reasons.append("outside_verified_segment")
    if observation.get("coverage_verified") is not True or not str(observation.get("coverage_source", "")).strip():
        reasons.append("coverage_unverified")
    if observed > now or received > now:
        reasons.append("not_available_at_query")
    if (now - observed).total_seconds() > age_limit:
        reasons.append("stale_photo")
    if abs((heading - route_heading + 180) % 360 - 180) > tolerance:
        reasons.append("direction_mismatch")
    if eta < now:
        reasons.append("past_eta")
    result.update(observed_at=observed.isoformat(), age_seconds=(now - observed).total_seconds(),
                  model_version=observation["model_version"], image_sha256=observation["image_sha256"])
    if not reasons:
        result["current_observation_level"] = observation["level"]
        if eta > now:
            reasons.append("future_state_unverified")
        else:
            result.update(level=observation["level"], status="candidate_observation")
    return result
