"""Validate and summarize project-aware observations, never generate class labels.

Consumes private artifacts from temporal_visibility_review.py. Independent of
that module's code; no image/model downloads or training. Counts are descriptive
of selected cases, not prevalence, causal estimates, or classifier performance.
"""
import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import shutil

FIELDS = ["case_id", "lane", "vehicle", "sign", "sun_evidence",
          "same_target_recovery", "localized_detail_concern", "fog_like_band",
          "motion_or_viewpoint", "observation", "limitation"]
STATES = {"identifiable", "partly_degraded", "not_confirmed"}
SUN = {"direct_sun_or_flare", "bright_scene_only", "uncertain"}
BOOL_FIELDS = ["same_target_recovery", "localized_detail_concern", "fog_like_band", "motion_or_viewpoint"]
OFFSETS = [-2, -1, 0, 1, 2]
NEXT_PROTOCOL = {
    "version": "target_visibility_evidence_v0.2",
    "status": "frozen_followup_observation_schema_not_canonical_class_labels",
    "designed_after": "17 selected project-aware temporal cases; not preregistered validation",
    "task": {"lane": "near travel corridor/boundary, not missing painted-line detection",
             "vehicle": "presence/outline/position; record near vs distant separately",
             "sign": "existence/shape; text reading requires a separate specified task"},
    "required_per_target": ["target_id", "task", "near_or_distant", "center_normalized_box_or_corridor",
                            "existence_evidence", "center_visibility", "impairment_region",
                            "solar_evidence", "alternative_causes", "uncertainty_reason"],
    "existence_evidence": ["visible_in_center", "same_target_in_neighbor", "not_established"],
    "visibility": ["identifiable", "partly_degraded", "unidentifiable", "not_evaluable"],
    "comparison_required_for_recovery": ["reference_offset", "same_target_correspondence",
                                         "reference_target_region", "occlusion_motion_scale_checks"],
    "guards": ["Unknown existence is not unidentifiable", "Absent target is not a failed task",
               "Small text failure is not vehicle-presence failure",
               "One visible target does not rule out severe loss elsewhere",
               "Solar evidence alone does not determine severity",
               "No physical luminance, in-person vision, adaptation time, or accident-risk claim",
               "No class-grade conversion, training-label overwrite, or feature input from future frames"],
    "next_scope": "remaining 33 cases before prevalence or full50 comparisons; existing17 need extra region fields",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def keyed(rows):
    result = {}
    for row in rows:
        key = row["case_id"]
        if not re.fullmatch(r"T\d{3}", key) or key in result:
            raise ValueError("Invalid or duplicate case ID")
        result[key] = row
    return result


def is_true(value):
    if value not in {"True", "False"}:
        raise ValueError("Priority flags must be explicit True/False")
    return value == "True"


def validate_observations(fields, rows, priority):
    if fields != FIELDS:
        raise ValueError("Observation schema differs; no class labels or scores allowed")
    observations = keyed(rows)
    flags = {k: (is_true(v["middle_missed"]), is_true(v["severe_to_ai_middle"])) for k, v in priority.items()}
    selected = {k for k, v in flags.items() if any(v)}
    if not selected or observations.keys() != selected:
        raise ValueError("Observation coverage must equal the entire priority set")
    for row in rows:
        if any(row[k] not in STATES for k in ["lane", "vehicle", "sign"]) or row["sun_evidence"] not in SUN:
            raise ValueError("Unknown observation state")
        if any(row[k] not in {"yes", "no", "uncertain"} for k in BOOL_FIELDS):
            raise ValueError("Use explicit yes/no/uncertain")
        if not row["observation"].strip() or not row["limitation"].strip():
            raise ValueError("Every observation requires evidence and limitations")
    return observations, flags


def audit(root, observation_file, out):
    root, observation_file, out = (Path(p).resolve() for p in [root, observation_file, out])
    if (out.exists() or out.parent != observation_file.parent or observation_file.parent.parent != root
            or out.is_relative_to(root / "review")):
        raise ValueError("Use a new output child beside private observations")
    provenance_file = observation_file.parent / "observation_provenance.json"
    rubric_file = observation_file.parent / "rubric_before_review.json"
    provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
    if provenance["observations_sha256"] != sha(observation_file) or provenance["source_rubric_sha256"] != sha(rubric_file):
        raise ValueError("Frozen observation/rubric hash mismatch")
    if provenance["blind"] is not False or provenance["class_labels_assigned"] is not False:
        raise ValueError("This audit requires explicit nonblind, no-class-label provenance")
    fields, rows = read_rows(observation_file)
    _, priority_rows = read_rows(root / "private_priority.csv")
    priority = keyed(priority_rows)
    observations, flags = validate_observations(fields, rows, priority)
    method = json.loads((root / "method.json").read_text(encoding="utf-8"))
    if method["status"] != "complete":
        raise ValueError("Temporal frame preparation is incomplete")
    for filename, expected in method["input_sha256"].items():
        if sha(filename) != expected:
            raise ValueError("Original source input hash changed")
    _, audit_rows = read_rows(root / "frame_audit.csv")
    frame_rows = [r for r in audit_rows if r["case_id"] in observations]
    frames = {}
    for row in frame_rows:
        cid, offset = row["case_id"], float(row["offset"])
        if offset not in OFFSETS or (cid, offset) in frames:
            raise ValueError("Invalid/duplicate frame offset")
        name = "center.jpg" if offset == 0 else f"neighbor_{offset:+.0f}.jpg"
        image = root / "review" / cid / name
        if sha(image) != row["output_sha256"]:
            raise ValueError("Review image changed")
        if offset == 0 and sha(image) != priority[cid]["original_sha256"]:
            raise ValueError("Center differs from original labelled JPEG")
        frames[cid, offset] = row["output_sha256"]
    if set(frames) != {(cid, float(offset)) for cid in observations for offset in OFFSETS}:
        raise ValueError("Every observation needs all five audited frames")
    report = dict(scope="selected project-aware descriptive audit, no class labels or performance",
                  cases=len(rows), total_prepared=len(priority), not_observed=len(priority)-len(rows),
                  frames=len(frames), video_groups=len({priority[c]["video"] for c in observations}),
                  priority_middle=sum(flags[c][0] for c in observations),
                  priority_severe_ai_difference=sum(flags[c][1] for c in observations),
                  counts={k: dict(sorted(Counter(r[k] for r in rows).items())) for k in FIELDS[1:-2]},
                  provenance=provenance, original_inputs_verified=len(method["input_sha256"]),
                  hashes={str(p): sha(p) for p in [observation_file, provenance_file, rubric_file,
                                                  root / "private_priority.csv", root / "frame_audit.csv", root / "method.json"]},
                  limitations=["Selected cases and multiple cases per video are not independent",
                               "Not a blind rater, relabeling, causal or in-person vision validation",
                               "No recovery reference is not proof of no impairment",
                               "Visible near targets do not establish that all forward information is intact"])
    out.mkdir()
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "next_protocol.json").write_text(json.dumps(NEXT_PROTOCOL, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(__file__, out / Path(__file__).name)
    (out / "source_sha256.txt").write_text(sha(__file__), encoding="ascii")
    cards = []
    for row in rows:
        cid = row["case_id"]
        images = "".join(f'<a href="../../review/{cid}/{name}"><img width="250" src="../../review/{cid}/{name}" alt="{cid} {tag}"></a>'
                         for name, tag in [("neighbor_-2.jpg", "-2s"), ("neighbor_-1.jpg", "-1s"),
                                           ("center.jpg", "center"), ("neighbor_+1.jpg", "+1s"), ("neighbor_+2.jpg", "+2s")])
        # The private observation directory is an immediate child of root.
        cards.append(f'<section><h2>{cid}</h2><div>{images}</div><p>{html.escape(row["observation"])}</p><p>한계: {html.escape(row["limitation"])}</p></section>')
    page = '<!doctype html><html lang="ko"><meta charset="utf-8"><title>전후 관찰 기록</title><style>body{font:16px system-ui;margin:24px}img{max-width:45%;height:auto}section{border-top:1px solid #aaa;padding:12px 0}</style><h1>선택된 사례의 전후 관찰 기록</h1><p>프로젝트를 아는 AI의 탐색 관찰입니다. 정답 교체·모델 성능·실제 사람 시야 검증이 아닙니다.</p>' + "".join(cards) + '</html>'
    (out / "report.html").write_text(page, encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/temporal_visibility_20261104"))
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.observations, args.out)
    print(json.dumps({k: result[k] for k in ["cases", "not_observed", "frames", "video_groups", "counts"]}, ensure_ascii=False))
