"""Run fixed model2 and model3 on one photo, then probe guidance alignment.

Local/offline only. Caller must identify an entrance and supply actual photo/G
metadata. Historical photos stay stale for a current query. No route validation,
future-state transfer, or real-world efficacy is inferred from a software run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from integration_probe import Policy, clock, compose_guidance, text
from model3_export_probe import Model3Candidate

VERSION = "same-photo-joint-probe-v1"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    return value


def run_photo(path, metadata, *, model2, model3, observation_id, segment_id,
              zone, zone_source, policy, evidence="recorded_photo", now=utc_now):
    if zone != "entry":
        raise ValueError("A caller-identified entrance is required; no automatic scene assumption")
    text(zone_source)
    text(observation_id)
    text(segment_id)
    if evidence not in ("recorded_photo", "synthetic_contract"):
        raise ValueError("Explicit evidence type required")
    path = Path(path)
    received = now()
    clock(received)
    started = perf_counter()
    raw = path.read_bytes()
    if not 0 < len(raw) <= 1048576 or not raw.startswith(b"\xff\xd8"):
        raise ValueError("JPEG photo at most 1MiB required")
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[0] * image.shape[1] > 8_000_000:
        raise ValueError("Readable JPEG at most 8 million pixels required")
    image_hash = hashlib.sha256(raw).hexdigest()
    from nunchi_ml.glare.roadside import photo_g
    prepared = photo_g(metadata, image.shape)
    m2 = model2.predict_image(image, prepared["features"], "B_geometry_aug")
    m3 = model3.predict_image(path, metadata, zone=zone)
    if hashlib.sha256(path.read_bytes()).hexdigest() != image_hash:
        raise ValueError("Photo changed between the two model executions")
    as_of = now()
    if clock(as_of) < clock(received):
        raise ValueError("Receipt cannot follow query")
    context = dict(observation_id=observation_id, segment_id=segment_id,
                   image_sha256=image_hash, lat=metadata["lat"], lon=metadata["lon"],
                   heading=metadata["heading"], captured_at=prepared["captured_at"], received_at=received)
    envelopes = []
    for name, model, prediction in [("model2", model2, m2), ("model3", model3, m3)]:
        expected_version = "photo-g-candidate-v1" if name == "model2" else "model3-ridge-two-stage-candidate-v1"
        if prediction["model_version"] != expected_version or prediction["status"] != "candidate":
            raise ValueError("Unexpected model runtime contract")
        envelope = dict(context=context, source=evidence, status=prediction["status"],
            model_version=prediction["model_version"], variant=prediction["variant"],
            artifact_sha256=model.meta["parameters_sha256"],
            target="photo_glare_0_1_2" if name == "model2" else "entrance_contrast_low_mid_high",
            level=int(prediction["level"][0]), alert_score=float(prediction["alert_score"][0]),
            conditional_high_score=float(prediction["conditional_severe_score" if name == "model2" else "conditional_high_score"][0]),
            joint_scores=prediction["joint_scores"][0].tolist())
        if name == "model3":
            envelope.update(zone="entry", estimated_luminance_cd_m2=float(prediction["estimated_luminance_cd_m2"][0]))
        envelopes.append(envelope)
    query = {k: v for k, v in context.items() if k != "received_at"}
    query.update(as_of=as_of, eta=as_of, zone=zone)
    result = dict(version=VERSION, evidence=evidence, image_sha256=image_hash,
                  zone=zone, zone_source=zone_source, scene_verified_by_probe=False,
                  route_coverage_verified=False, same_photo_bytes_checked=True,
                  geometry_provenance=prepared, raw_model2=serial(m2), raw_model3=serial(m3),
                  model2=envelopes[0], model3=envelopes[1],
                  guidance=compose_guidance(query, envelopes[0], envelopes[1], policy),
                  local_processing_ms=(perf_counter() - started) * 1000)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model2-bundle", type=Path, required=True)
    parser.add_argument("--model3-bundle", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--observation-id", required=True)
    parser.add_argument("--segment-id", required=True)
    parser.add_argument("--zone", choices=["entry"], required=True)
    parser.add_argument("--zone-source", required=True)
    parser.add_argument("--evidence", choices=["recorded_photo", "synthetic_contract"], required=True)
    parser.add_argument("--max-age-seconds", type=float, required=True)
    parser.add_argument("--max-distance-m", type=float, required=True)
    parser.add_argument("--heading-tolerance-degrees", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not args.out.resolve().is_relative_to((Path(__file__).resolve().parents[3] / "data").resolve()):
        parser.error("Use a new private output file under AI/data")
    from nunchi_ml.glare.hybrid import HybridCandidate
    policy = Policy(args.max_age_seconds, args.max_distance_m, args.heading_tolerance_degrees)
    result = run_photo(args.image, json.loads(args.context.read_text(encoding="utf-8-sig")),
                       model2=HybridCandidate(args.model2_bundle), model3=Model3Candidate(args.model3_bundle),
                       observation_id=args.observation_id, segment_id=args.segment_id,
                       zone=args.zone, zone_source=args.zone_source, policy=policy, evidence=args.evidence)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(version=VERSION, evidence=result["evidence"],
                         same_photo_bytes_checked=True, guidance=result["guidance"]["requested_state"],
                         real_joint_accuracy_evaluated=False)))


if __name__ == "__main__":
    main()
