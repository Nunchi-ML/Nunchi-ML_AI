"""Inventory existing auxiliary evidence; never generate labels or train a model.

The admission check is a conservative prerequisite, not a declaration of label
accuracy. Missing values remain missing and development photos stay reserved.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys


TASKS = ("sun_disc", "sun_occluder", "bloom", "lane_visibility",
         "vehicle_visibility", "signal_visibility", "degradation_cause",
         "sun_x", "sun_y")
PROXIES = {
    "u,v,in_fov": "geometric sun projection; not observed visible sun location",
    "peak_x,peak_y,brightest_x,brightest_y,r_peak_x,r_peak_y,px_bright_x,px_bright_y":
        "computed bright-region coordinates; not annotated sun positions",
    "peak_area,bright_blob,px_bright_area,road_clipped_fraction":
        "computed pixel statistics; not annotated flare segmentation",
    "visibility": "weather visibility; not road-target identifiability",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(path, key):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [r[key] for r in rows]
    if any(not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("Missing or duplicate row identity")
    return rows


def admission_reasons(record, training_hashes, reserved_hashes, reserved_groups):
    """Require explicit review acceptance and disjoint training-photo identity.

    An accepted record still needs a task-specific rubric, adequate coverage,
    grouped validation and a genuinely shared non-neural model before fitting.
    Neither a feature proxy nor an unreviewed AI reference passes this gate.
    """
    reasons = []
    digest = record.get("image_sha256")
    if digest not in training_hashes:
        reasons.append("not_training_photo")
    if digest in reserved_hashes or record.get("group") in reserved_groups:
        reasons.append("reserved_evaluation_photo_or_group")
    if record.get("source_kind") != "adjudicated_annotation":
        reasons.append("not_adjudicated_annotation")
    if record.get("quality_accepted") is not True:
        reasons.append("quality_not_accepted")
    if not record.get("task_version"):
        reasons.append("task_version_missing")
    if not record.get("group"):
        reasons.append("group_missing")
    if record.get("value") is None or record.get("value") == "":
        reasons.append("missing_value")
    return reasons


def template_counts(rows, mapping, training):
    """Match both ID and hash; count blank fields without imputing normal/zero."""
    mapped = {r["review_id"]: r for r in mapping}
    if (len(mapped) != len(mapping) or len(rows) != len(mapped)
            or {r["review_id"] for r in rows} != set(mapped)):
        raise ValueError("Annotation/mapping identity mismatch")
    for row in rows:
        item = mapped[row["review_id"]]
        if item["id"] not in training or training[item["id"]]["image_sha256"] != item["image_sha256"]:
            raise ValueError("Training photo hash mismatch")
    return {name: sum(bool(r[name].strip()) for r in rows) for name in TASKS}


def audit(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out.exists() or out.parent != root / "data":
        raise ValueError("Use a new private output directly under data")
    source_sha256 = sha(__file__)
    inputs = {}

    def checked(relative):
        path = root / relative
        inputs[relative] = sha(path)
        return path

    feature_path = checked("data/road_visibility_20261104/run30/training_features.csv")
    training_rows = table(feature_path, "id")
    training = {r["id"]: r for r in training_rows}
    training_hashes = {r["image_sha256"] for r in training_rows}
    if len(training_hashes) != len(training_rows):
        raise ValueError("Duplicate training image bytes")
    base = "data/analysis_20261104/step3_visibility"
    ready = base + "/annotation_ready"
    mapping = table(checked(ready + "/private_review_mapping.csv"), "review_id")
    blank = table(checked(ready + "/review_bundle/blank_annotations.csv"), "review_id")
    counts = template_counts(blank, mapping, training)
    if any(counts.values()):
        raise ValueError("Template now has answers; audit submitted provenance before admission")
    # These files are a known template, not a submitted annotation source. A later
    # submission needs a separate rubric/provenance audit, never silent admission.
    method = json.loads(checked(ready + "/review_method.json").read_text(encoding="utf-8"))
    if method["rows"] != len(blank) or method["sample_uses_labels"] is not False:
        raise ValueError("Prepared sample protocol mismatch")
    for name in ("private_review_mapping.csv", "review_bundle/blank_annotations.csv"):
        if sha(checked(base + "/" + name)) != sha(checked(ready + "/" + name)):
            raise ValueError("Template copies diverged; inspect before counting")
    image_digest = hashlib.sha256()
    for row in blank:
        asset = (root / ready / "review_bundle" / row["image"]).resolve()
        if not asset.is_relative_to(root / ready / "review_bundle"):
            raise ValueError("Review asset escapes bundle")
        relative = asset.relative_to(root).as_posix()
        checked(relative)
        # Bundle images were deliberately resized/re-encoded without EXIF. The
        # mapping hash describes the original, not the anonymous JPEG bytes.
        image_digest.update(row["review_id"].encode())
        image_digest.update(asset.read_bytes())
    html = checked(ready + "/review_bundle/review.html").read_text(encoding="utf-8")
    payload = re.search(r"const payload=(\{[^\n]*\});", html)
    if not payload:
        raise ValueError("Bundle payload missing")
    manifest = hashlib.sha256(payload[1].encode() + image_digest.digest()).hexdigest()
    if manifest != method["manifest"] or f'const manifest="{manifest}";' not in html:
        raise ValueError("Anonymous review bundle manifest mismatch")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "glare_video_source"))
    import pandas as pd
    from g_visibility_probe import image_paths
    checked("notebooks/experiments/glare_video_source/prepare_visibility_review.py")
    for row, path in zip(training_rows, image_paths(pd.DataFrame(training_rows), root / "data")):
        if sha(checked(path.relative_to(root).as_posix())) != row["image_sha256"]:
            raise ValueError("Original training image bytes mismatch")

    temporal = "data/temporal_visibility_20261104"
    priorities = table(checked(temporal + "/private_priority.csv"), "case_id")
    priority = {r["case_id"]: r for r in priorities}
    observation_file = checked(temporal + "/full50_evidence_v02/observations_frozen.json")
    observations = json.loads(observation_file.read_text(encoding="utf-8"))
    provenance = json.loads(checked(temporal + "/full50_evidence_v02/provenance.json").read_text(encoding="utf-8"))
    if provenance["observations_sha256"] != sha(observation_file) or provenance["blind"] is not False:
        raise ValueError("Observation provenance mismatch")
    protocol = checked(temporal + "/full50_evidence_v02/protocol_before_review.json")
    if provenance["protocol_sha256"] != sha(protocol):
        raise ValueError("Observation protocol mismatch")
    from target_evidence_audit import validate
    validate(observations, priority)
    for row in priorities:
        path = checked(temporal + "/review/" + row["case_id"] + "/center.jpg")
        if sha(path) != row["original_sha256"]:
            raise ValueError("Reserved center image mismatch")
    reserved_hashes = {r["original_sha256"] for r in priorities}
    reserved_groups = {r["video"] for r in priorities}
    if reserved_hashes & training_hashes:
        raise ValueError("Development evidence overlaps training bytes")

    candidates = []
    for row in observations:
        item = priority[row["case_id"]]
        for target in row["targets"]:
            record = dict(image_sha256=item["original_sha256"], group=item["video"],
                          source_kind="ai_reference", quality_accepted=False,
                          task_version="target_visibility_evidence_v0.2",
                          value=None if target["center_visibility"] == "not_evaluable" else target["center_visibility"])
            candidates.append(dict(case_id=row["case_id"], target_id=target["target_id"],
                                   state=target["center_visibility"],
                                   exclusion_reasons=admission_reasons(record, training_hashes, reserved_hashes, reserved_groups)))
    boundary = json.loads(checked("data/target_boundary_20261104/result_v2/report.json").read_text(encoding="utf-8"))
    checked("data/target_boundary_20261104/result_v2/sign_region_audit.csv")
    gemini = json.loads(checked("data/gemini_reference_20261104_v3/report_v2/annotations.json").read_text(encoding="utf-8"))
    for record in gemini:
        p = priority[record["temporal_case_id"]]
        if record["id"] != p["id"]:
            raise ValueError("AI reference photo identity mismatch")
    # Bound this inventory to the locally persisted AIHub folder. Absence here
    # says nothing about annotations available from the upstream public dataset.
    aihub_files = sorted(p for p in (root / "data/aihub_special").rglob("*") if p.is_file())
    structured = [p for p in aihub_files if p.suffix.lower() in {".json", ".xml"}]
    for path in structured:
        checked(path.relative_to(root).as_posix())
    report = dict(
        status="audit_complete_training_deferred", new_training=False, performance=None,
        training_photos=len(training_rows), training_groups=len({r["group"] for r in training_rows}),
        original_photos=sum(r["camera"] != "AI Hub" for r in training_rows),
        template=dict(unique_photos=len(blank), groups=len({r["group"] for r in mapping}),
                      nonempty_by_field=counts, duplicated_copy_count=2, copies_identical=True,
                      anonymous_asset_manifest=manifest, original_and_thumbnail_hashes_distinguished=True),
        observation=dict(photos=len(observations), video_groups=len(reserved_groups),
                         task_records=len(candidates), training_hash_matches=len(reserved_hashes & training_hashes),
                         states={k: dict(Counter(t["center_visibility"] for r in observations for t in r["targets"]
                                                if t["target_id"].endswith("_" + k))) for k in ("lane", "vehicle", "sign")},
                         limitation="project-aware posthoc development evidence; coarse boxes, not detection/segmentation truth"),
        boundary=dict(sign_roi_audit=boundary["sign_roi_audit"], confirmed_solar_recovery_pairs=boundary["confirmed_solar_recovery_pairs"]),
        gemini=dict(condition_records=len(gemini), unique_photos=len({r["id"] for r in gemini}),
                    condition_counts=dict(Counter(r["condition"] for r in gemini)),
                    limitation="same development photos, two conditions, no independent adjudicated auxiliary truth"),
        aihub_local=dict(structured_json_xml_files=len(structured),
                         limitation="bounded local inventory only; upstream annotation availability not assessed"),
        accepted_auxiliary_training_records=0, feature_proxies=PROXIES,
        decision="No shared multi-task fit: no accepted matching training-photo auxiliary annotations in audited sources.",
        next_steps=["Reuse existing 96-photo bundle; freeze task rubric and provenance before annotation.",
                    "Keep unknown presence/not_evaluable separate; audit regions and image hashes before admission.",
                    "Use capture-group outer/inner splits; never train on development50 or its video groups.",
                    "Only then compare a shared non-neural learner with original320 grouped CV; independent output models are not shared learning.",
                    "Human-response target definition/data/calibration and sensor requirements/paired collection remain separate pending stages."],
        input_sha256=inputs)
    # Verify every read input again before committing a new immutable artifact.
    for relative, digest in inputs.items():
        if sha(root / relative) != digest:
            raise ValueError("Input changed during audit")
    if sha(__file__) != source_sha256:
        raise ValueError("Audit source changed during run")
    out.mkdir()
    (out / "source_at_audit.py").write_bytes(Path(__file__).read_bytes())
    report["source_sha256"] = source_sha256
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "excluded_development_targets.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "target_id", "state", "exclusion_reasons"])
        writer.writeheader()
        writer.writerows({**r, "exclusion_reasons": ";".join(r["exclusion_reasons"])} for r in candidates)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.out)
    print(json.dumps({k: result[k] for k in ("status", "training_photos", "accepted_auxiliary_training_records", "new_training")}, ensure_ascii=False))
