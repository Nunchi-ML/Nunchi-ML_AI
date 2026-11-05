"""Audit target scope/locations before interpreting ordinal label disagreement.

No grading, label replacement, model fitting or new accuracy measurement.
Manual notes are explicitly posthoc and project-aware; Gemini has no shared
object IDs/regions, so case-level attributes are not object-level agreement.
"""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import shutil

import pandas as pd

from target_evidence_audit import box_valid, validate as validate_original

OFFSETS = [-2, -1, 0, 1, 2]
KINDS = {"guide_panel", "regulatory_panel", "parking_panel", "panel_type_uncertain",
         "non_sign_structure", "traffic_signal", "commercial_or_facility_panel"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def boundary_draft():
    return dict(version="task_limited_boundary_draft_v0.1", status="proposal_not_applied",
                axes={"solar_evidence": "visible solar/flare, uncertain, or no clear evidence; not physical luminance",
                      "target_information": "specified object/region/task and image-relative distance",
                      "attribution": "compatible solar evidence, alternative cause, or unknown; co-occurrence is not isolated causality"},
                scope="Forward/aligned road tasks only; scene eligibility precedes grading",
                task_level_ordinal_proposal={
                    "0": "Prespecified, confirmed-present target is identifiable for the named task; says nothing about unexamined targets",
                    "1": "Confirmed target is partly degraded but named task remains answerable",
                    "2": "Confirmed-present target cannot be identified for the named task; supply region, existence and correspondence evidence where applicable"},
                abstention="Unknown presence, wrong scene, missing target, unmatched task/region, uncertain visibility or attribution remain separate; never fill with0",
                aggregation="UNRESOLVED: representative near targets cannot establish whole-frame absence of loss; no max/mean/vote creates a global grade",
                prerequisites=["Freeze target kinds, task, distance/size handling, scope and representative-region selection before new evaluation",
                               "Separate traffic-sign shape from signal presence/color, advertising/facility panels and text reading",
                               "For temporal evidence, link the same object with frame regions; record motion, occlusion and cuts",
                               "Freeze a separate proposed label version before fitting; retain original labels and development-set status"],
                labels_created=False, thresholds_tuned=False, deployment_changed=False)


def validate_notes(notes, protocol, originals):
    expected = set(protocol["priority_cases"])
    if len(notes) != len(expected) or {r["case_id"] for r in notes} != expected:
        raise ValueError("Exact priority-case coverage required")
    for row in notes:
        cid = row["case_id"]
        if any(k in row for k in ["label", "new_label", "proposed_grade", "global_grade"]):
            raise ValueError("Audit notes cannot assign or replace grades")
        if row["reviewed_offsets"] != OFFSETS or row["distant_information"] != "not_established":
            raise ValueError("All priority frames required; near evidence cannot clear distant information")
        if row["same_target_recovery"] != "not_established":
            raise ValueError("Recovery needs a different schema with validated correspondences")
        if row["solar_grounding"] not in {"visible_solar_or_flare", "uncertain", "not_applicable"}:
            raise ValueError("Invalid photographic solar grounding")
        if row["scope"] not in {"forward_or_aligned_road", "side_view_road"}:
            raise ValueError("Invalid priority scene scope")
        for field in ["visual_evidence", "temporal_confound", "boundary_issue"]:
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError("Concrete evidence/confounds required")
        targets = row["checked_target_ids"]
        if set(targets) != {t["target_id"] for t in originals[cid]["targets"]} or len(targets) != 3:
            raise ValueError("Target identity/task coverage mismatch")


def validate_rois(rows, originals):
    expected = {t["target_id"]: t for row in originals.values() for t in row["targets"]
                if t["target_id"].endswith("_sign") and t["center_normalized_box_or_corridor"] is not None}
    if len(rows) != len(expected) or {r["target_id"] for r in rows} != set(expected):
        raise ValueError("Every prior sign region needs exactly one audit")
    for row in rows:
        old = expected[row["target_id"]]["center_normalized_box_or_corridor"]
        if row["old_box"] != old or row["kind"] not in KINDS or not row["evidence"].strip():
            raise ValueError("Original region/kind/evidence mismatch")
        status, box = row["region_result"], row["audited_box"]
        if status == "panel_located":
            if box != old or row["kind"] == "non_sign_structure":
                raise ValueError("Located panel must retain its actual original box")
        elif status == "mislocalized_corrected":
            if not box_valid(box) or box == old or row["kind"] == "non_sign_structure":
                raise ValueError("Correction requires a distinct valid panel region")
        elif status == "wrong_object":
            if box is not None or row["kind"] != "non_sign_structure":
                raise ValueError("Wrong-object region cannot become a traffic target")
        else:
            raise ValueError("Invalid region audit status")
        if row.get("global_loss_inference") != "none":
            raise ValueError("A localized panel cannot establish whole-image visibility")


def audit(root, data, out):
    root, data, out = (Path(p).resolve() for p in [root, data, out])
    if out.exists() or out.parent != root:
        raise ValueError("Use new output inside the private audit folder")
    protocol_file = root / "protocol_before_audit.json"
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    source_hashes = dict(protocol["source_sha256"])
    if any(sha(path) != value for path, value in source_hashes.items()):
        raise ValueError("Frozen source labels/predictions/observations changed")
    temporal = data / "temporal_visibility_20261104"
    mapping = pd.read_csv(temporal / "private_id_map.csv", dtype={"id": str})
    if len(mapping) != 50 or not mapping.id.is_unique or not mapping.case_id.is_unique:
        raise ValueError("Expected50 unique original cases")
    human = pd.read_csv(data / "new_cameras/human_review_50/labels_성진.csv", dtype={"id": str}).set_index("id").label
    if len(human) != 50 or not human.index.is_unique or set(human.index) != set(mapping.id) or not human.isin([0, 1, 2]).all():
        raise ValueError("Expected50 matched unique original human labels")
    original_file = temporal / "full50_evidence_v02/observations_frozen.json"
    original_list = json.loads(original_file.read_text(encoding="utf-8"))
    validate_original(original_list, set(mapping.case_id))
    originals = {row["case_id"]: row for row in original_list}
    gemini_dir = data / "gemini_reference_20261104_v3/report_v2"
    annotations = pd.read_json(gemini_dir / "annotations.json")
    annotations["id"] = annotations.id.astype(str)
    if len(annotations) != 100 or annotations.duplicated(["condition", "id"]).any():
        raise ValueError("Expected100 unique actual Gemini judgments")
    wide = annotations.pivot(index="id", columns="condition", values="reference_grade")
    severe = set(mapping.loc[mapping.id.isin(human[human.eq(2)].index), "case_id"])
    changed = set(mapping.loc[mapping.id.isin(wide[wide.center.ne(wide.temporal)].index), "case_id"])
    if severe != set(protocol["human_severe"]) or changed != set(protocol["condition_changes"]) or set(protocol["priority_cases"]) != severe | changed | set(protocol["scene_scope_dispute"]):
        raise ValueError("Frozen priority selection differs from preserved actual inputs")
    frozen = root / "audit_frozen.json"
    provenance_file = root / "provenance.json"
    provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
    if (provenance["blind"] is not False or provenance["labels_created"] is not False
            or provenance["audit_sha256"] != sha(frozen) or provenance["protocol_sha256"] != sha(protocol_file)):
        raise ValueError("Explicit posthoc provenance/hash required")
    for path, digest in provenance["qa_sha256"].items():
        if not Path(path).resolve().is_relative_to(root / "qa") or sha(path) != digest:
            raise ValueError("Reviewed diagnostic image changed/outside audit")
        source_hashes[path] = digest
    notes = json.loads(frozen.read_text(encoding="utf-8"))
    validate_notes(notes["cases"], protocol, originals)
    validate_rois(notes["sign_regions"], originals)
    reviewed = {(cid, offset) for cid in mapping.case_id for offset in ([0] if cid not in protocol["priority_cases"] else OFFSETS)}
    frame_table = pd.read_csv(temporal / "frame_audit.csv")
    for cid, offset in sorted(reviewed):
        path = temporal / "review" / cid / ("center.jpg" if offset == 0 else f"neighbor_{offset:+d}.jpg")
        rows = frame_table[frame_table.case_id.eq(cid) & frame_table.offset.eq(offset)]
        if len(rows) != 1 or sha(path) != rows.iloc[0].output_sha256:
            raise ValueError("Reviewed frame changed/missing from original audit")
        source_hashes[str(path)] = sha(path)
    source_hashes.update({str(p): sha(p) for p in [protocol_file, frozen, provenance_file]})
    flags = []
    for _, case in mapping.iterrows():
        original = originals[case.case_id]
        for condition in ["center", "temporal"]:
            item = annotations[annotations.condition.eq(condition) & annotations.id.eq(case.id)]
            if len(item) != 1 or item.iloc[0].temporal_case_id != case.case_id:
                raise ValueError("Gemini/source sample mismatch")
            item = item.iloc[0]
            for target in original["targets"]:
                task = target["target_id"].rsplit("_", 1)[1]
                flags.append(dict(case_id=case.case_id, condition=condition, task=task,
                                  prior_state=target["center_visibility"], gemini_state=item[task],
                                  shared_target_correspondence="not_recorded", object_accuracy_evaluable=False,
                                  state_diff=target["center_visibility"] != item[task],
                                  unlocated_positive=item[task] != "not_evaluable" and target["existence_evidence"] == "not_established"))
    flags = pd.DataFrame(flags)
    scope = [cid for cid, row in originals.items() if (row["scene_context"] == "forward_or_aligned_road") != bool(annotations[annotations.temporal_case_id.eq(cid)].scene.eq("forward_or_aligned_road").all())]
    severe_targets = [t for cid in severe for t in originals[cid]["targets"]]
    summary = dict(status="complete_posthoc_target_audit", priority_cases=len(notes["cases"]), priority_frames=len(notes["cases"])*5,
                   all_center_inventory=50, unique_reviewed_frames=len(reviewed), human_severe_cases=len(severe),
                   human_severe_video_groups=int(mapping[mapping.case_id.isin(severe)].video.nunique()),
                   severe_samples_per_video=sorted(mapping[mapping.case_id.isin(severe)].video.value_counts().tolist(), reverse=True),
                   severe_task_states={task: dict(Counter(t["center_visibility"] for t in severe_targets if t["target_id"].endswith("_"+task))) for task in ["lane", "vehicle", "sign"]},
                   sign_roi_audit=dict(Counter(r["region_result"] for r in notes["sign_regions"])),
                   case_attribute_screening={condition: {task: dict(state_differences=int(part.state_diff.sum()),
                                                                 unlocated_positives=int(part.unlocated_positive.sum()),
                                                                 object_accuracy_evaluable=False)
                                                        for task in ["lane", "vehicle", "sign"]
                                                        for part in [flags[flags.condition.eq(condition) & flags.task.eq(task)]]}
                                            for condition in ["center", "temporal"]},
                   scene_scope_disagreements=scope, confirmed_solar_recovery_pairs=0, labels_created=False, new_training=False,
                   grade_boundary=boundary_draft(), hashes=source_hashes, source_sha256=sha(__file__),
                   limitations=["Case attribute differences are a queue, not object agreement or error rates",
                                "Visible representative targets do not clear unexamined distant information",
                                "No isolated solar causal contrast or actual driver-vision validation",
                                "Posthoc selected development cases, project-aware reviewer and coarse regions",
                                "No reassignment of severe labels, new F1, or improvement over .6229"])
    out.mkdir()
    dump(out / "report.json", summary)
    dump(out / "boundary_draft.json", boundary_draft())
    flags.to_csv(out / "attribute_screening.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(notes["sign_regions"]).to_csv(out / "sign_region_audit.csv", index=False, encoding="utf-8-sig")
    shutil.copyfile(__file__, out / Path(__file__).name)
    cards = []
    for row in notes["cases"]:
        cid = row["case_id"]
        pictures = ''.join('<a href="' + (temporal / "review" / cid / ("center.jpg" if off == 0 else f"neighbor_{off:+d}.jpg")).as_uri() + '"><img loading="lazy" width="360" src="' + (temporal / "review" / cid / ("center.jpg" if off == 0 else f"neighbor_{off:+d}.jpg")).as_uri() + '" alt="' + cid + f' {off:+d}s"></a>' for off in OFFSETS)
        cards.append('<section><h2>' + cid + '</h2>' + pictures + '<pre>' + html.escape(json.dumps(row, ensure_ascii=False, indent=2)) + '</pre></section>')
    region_cards = []
    for row in notes["sign_regions"]:
        cid = row["target_id"].rsplit("_", 1)[0]
        source = (temporal / "review" / cid / "center.jpg").as_uri()
        overlays = []
        for name, color, box in [("old", "red", row["old_box"]), ("audited", "lime", row["audited_box"])]:
            if box is not None:
                x0, y0, x1, y1 = box
                overlays.append(f'<div title="{name}" style="position:absolute;left:{x0*100}%;top:{y0*100}%;width:{(x1-x0)*100}%;height:{(y1-y0)*100}%;border:2px solid {color};box-sizing:border-box"></div>')
        region_cards.append('<section><h2>' + cid + ': ' + html.escape(row["region_result"]) + '</h2><p>Red=old region; green=audited region (if retained/corrected). No new grade.</p><div style="position:relative;max-width:900px"><img style="display:block;width:100%" src="' + source + '" alt="' + cid + ' region audit">' + ''.join(overlays) + '</div><p>' + html.escape(row["evidence"]) + '</p></section>')
    inventory = ''.join('<a href="' + p.as_uri() + '"><img loading="lazy" width="500" src="' + p.as_uri() + '" alt="center inventory"></a>' for p in sorted((root / "qa").glob("centers_*.jpg")))
    page = '<!doctype html><meta charset="utf-8"><title>Target boundary audit</title><style>body{font:16px system-ui;margin:24px}pre{white-space:pre-wrap}section{border-top:1px solid #aaa;padding:18px 0}img{max-width:100%;height:auto}</style><h1>Posthoc target audit — no new labels</h1><p>Original human/AI labels unchanged. Attribute mismatch is not an object error rate. No new F1 or final model.</p><pre>' + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre><h1>Prior sign region audits</h1>' + ''.join(region_cards) + '<h1>Priority temporal review</h1>' + ''.join(cards) + '<h1>All50 center inventory — resized diagnostics</h1>' + inventory
    (out / "report.html").write_text(page, encoding="utf-8")
    if any(sha(p) != digest for p, digest in source_hashes.items()):
        raise ValueError("Source changed during audit")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/target_boundary_20261104"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.data, args.out)
    print(json.dumps({k: v for k, v in result.items() if k not in {"hashes", "grade_boundary"}}, ensure_ascii=False))
