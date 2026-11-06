"""Offline model-1 guidance contract probe, not a deployed or validated policy.

Preserve model-2 image grades and model-3 entrance contrast grades separately.
No fitting, physiological inference, accident probability, or simulated fallback.
"""
import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

VERSION = "guidance-contract-probe-v1"


def number(value, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Finite numeric value required")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError("Value out of range")
    return float(value)


def clock(value):
    if not isinstance(value, str):
        raise ValueError("Timestamp string required")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Explicit timezone required")
    return result


def text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Nonblank identifier required")
    return value


def sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("Photo/artifact SHA256 required")
    return value


@dataclass(frozen=True)
class Policy:
    """Caller-selected trial limits, never claimed as validated warning limits."""
    max_age_seconds: float
    max_distance_m: float
    heading_tolerance_degrees: float

    def __post_init__(self):
        number(self.max_age_seconds, 0, 86400)
        number(self.max_distance_m, 0, 10000)
        number(self.heading_tolerance_degrees, 0, 90)


def distance_m(a, b):
    la, lo = (math.radians(number(a[k], *bounds)) for k, bounds in
              [("lat", (-90, 90)), ("lon", (-180, 180))])
    lb, lob = (math.radians(number(b[k], *bounds)) for k, bounds in
               [("lat", (-90, 90)), ("lon", (-180, 180))])
    h = math.sin((la - lb) / 2) ** 2 + math.cos(la) * math.cos(lb) * math.sin((lo - lob) / 2) ** 2
    return 6371000 * 2 * math.asin(math.sqrt(min(1, max(0, h))))


def validate_query(query):
    for key in ("observation_id", "segment_id"):
        text(query[key])
    sha(query["image_sha256"])
    number(query["heading"], 0, math.nextafter(360, 0))
    distance_m(query, query)
    for key in ("captured_at", "as_of", "eta"):
        clock(query[key])
    if query["zone"] not in ("entry", "exit", "road", "unknown"):
        raise ValueError("Explicit scene zone required")


def context_reasons(context, query, policy):
    for key in ("observation_id", "segment_id"):
        text(context[key])
    sha(context["image_sha256"])
    heading = number(context["heading"], 0, math.nextafter(360, 0))
    captured, received, now = (clock(context["captured_at"]), clock(context["received_at"]), clock(query["as_of"]))
    if received < captured:
        raise ValueError("Receipt before photo")
    reasons = []
    if context["observation_id"] != query["observation_id"]:
        reasons.append("observation_mismatch")
    if context["segment_id"] != query["segment_id"]:
        reasons.append("segment_mismatch")
    if context["image_sha256"] != query["image_sha256"]:
        reasons.append("photo_mismatch")
    if captured != clock(query["captured_at"]):
        reasons.append("capture_time_mismatch")
    if captured > now or received > now:
        reasons.append("not_available_at_query")
    elif (now - captured).total_seconds() > policy.max_age_seconds:
        reasons.append("stale_photo")
    if distance_m(context, query) > policy.max_distance_m:
        reasons.append("position_mismatch")
    if abs((heading - query["heading"] + 180) % 360 - 180) > policy.heading_tolerance_degrees:
        reasons.append("direction_mismatch")
    return reasons


def check_component(item, model, query, policy):
    if item is None:
        return dict(state="unknown", level=None, reasons=["model_missing"], model_version=None)
    try:
        version = text(item["model_version"])
        sha(item["artifact_sha256"])
        reasons = context_reasons(item["context"], query, policy)
        if item["source"] != "recorded_photo":
            reasons.append("unsupported_source")
        if item["status"] not in ("candidate", "prototype"):
            reasons.append("unsupported_status")
        if model == "model2":
            if version != "photo-g-candidate-v1" or item["variant"] != "B_geometry_aug":
                reasons.append("wrong_model_variant")
            if item["target"] != "photo_glare_0_1_2":
                reasons.append("wrong_target")
        else:
            if item["variant"] != "ridge_estimated_luminance_two_stage" or item["target"] != "entrance_contrast_low_mid_high":
                reasons.append("wrong_model_variant_or_target")
            if item["zone"] != "entry" or query["zone"] != "entry":
                reasons.append("unsupported_zone")
            number(item["estimated_luminance_cd_m2"], math.nextafter(0, 1), 1e12)
        level = item["level"]
        if type(level) is not int or level not in (0, 1, 2):
            raise ValueError("Original ordinal grade required")
        a, b = (number(item[k], 0, 1) for k in ("alert_score", "conditional_high_score"))
        joint = item["joint_scores"]
        if not isinstance(joint, list) or len(joint) != 3:
            raise ValueError("Three joint scores required")
        expected = [1 - a, a * (1 - b), a * b]
        if any(not math.isclose(number(x, 0, 1), y, rel_tol=0, abs_tol=1e-9) for x, y in zip(joint, expected)):
            raise ValueError("Inconsistent conditional scores")
        # Model2 explicitly uses >=.5. Notebook model3 sklearn.predict chooses
        # the first (False) class on an exact .5 tie, so it uses strict >.5.
        alert, high = (a >= .5, b >= .5) if model == "model2" else (a > .5, b > .5)
        if level != (0 if not alert else 2 if high else 1):
            raise ValueError("Conditional decision mismatch")
        return dict(state="unknown" if reasons else "ready", level=None if reasons else level,
                    reasons=reasons, model_version=version, target=item["target"],
                    artifact_sha256=item["artifact_sha256"],
                    original_level=level, joint_scores=joint)
    except (ValueError, TypeError, KeyError, OverflowError):
        return dict(state="unknown", level=None, reasons=["invalid_component"], model_version=None)


def compose_guidance(query, model2, model3, policy):
    """Trial ordinal-max display; originals and unknown components remain visible.

Both original grades use 0/1/2 but measure different targets. This mapping is
a proposed display policy, not a learned joint target or an efficacy result.
"""
    validate_query(query)
    components = {name: check_component(item, name, query, policy)
                  for name, item in [("model2", model2), ("model3", model3)]}
    missing = [name for name, item in components.items() if item["state"] != "ready"]
    levels = [item["level"] for item in components.values() if item["state"] == "ready"]
    maximum = max(levels) if levels else None
    # A known warning survives partial data. A known normal cannot clear an
    # unknown other model. Never return full normal from one missing component.
    current = maximum if not missing or maximum in (1, 2) else None
    causes = [name for name, item in components.items() if item["level"] in (1, 2)]
    eta, now = clock(query["eta"]), clock(query["as_of"])
    reasons = ["future_state_unverified"] if eta > now else ["past_eta"] if eta < now else []
    return dict(version=VERSION, status="prototype", policy_validated=False,
                artifact_bytes_verified_by_probe=False, inference_provenance_verified_by_probe=False,
                observation_id=query["observation_id"], segment_id=query["segment_id"],
                captured_at=query["captured_at"], eta=query["eta"],
                completeness="complete" if not missing else "partial" if levels else "unknown",
                current_observation_level=current, current_level_is_lower_bound=bool(missing and current is not None),
                requested_level=None if reasons else current, requested_state="unknown" if reasons or current is None else "attention" if current else "observed_normal",
                reasons=reasons, missing_components=missing, causes=causes, components=components,
                accident_probability=None, driver_visibility_validated=False,
                adaptation_time_validated=False, future_state_validated=False)


def run_cases(cases):
    if not isinstance(cases, list) or not cases:
        raise ValueError("Nonempty case list required")
    ids = [text(case["case_id"]) for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate case identifiers")
    return [dict(case_id=case["case_id"], evidence=text(case["evidence"]),
                 **compose_guidance(case["query"], case.get("model2"), case.get("model3"), Policy(**case["policy"])))
            for case in cases]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = run_cases(json.loads(args.input.read_text(encoding="utf-8-sig")))
    args.out.mkdir(parents=True, exist_ok=False)
    evidence_counts = {kind: sum(r["evidence"] == kind for r in results)
                       for kind in sorted({r["evidence"] for r in results})}
    report = dict(version=VERSION, input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  cases=len(results), policy_validated=False, real_joint_accuracy_evaluated=False,
                  evidence_counts=evidence_counts, counts_are_contract_checks_not_real_inference_counts=True,
                  contract_readiness_counts=dict(
                      model2_ready=sum(r["components"]["model2"]["state"] == "ready" for r in results),
                      model3_ready=sum(r["components"]["model3"]["state"] == "ready" for r in results),
                      complete=sum(r["completeness"] == "complete" for r in results)))
    for name, value in [("results.json", results), ("summary.json", report)]:
        (args.out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
