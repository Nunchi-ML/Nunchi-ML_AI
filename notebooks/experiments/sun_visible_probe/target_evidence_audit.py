"""Audit full50 target observations and fixed-prediction scope sensitivity.

No relabeling/training. Scope choices follow exploratory, project-aware review;
subset scores are development sensitivity, never improvement over original CV.
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

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

TASKS = {"lane": "near_road_corridor", "vehicle": "presence_outline_position", "sign": "presence_shape"}
SCENES = {"forward_or_aligned_road", "rear_marked_road", "walkway", "side_view_road"}
STATES = {"identifiable", "partly_degraded", "unidentifiable", "not_evaluable"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def box_valid(box):
    return (isinstance(box, list) and len(box) == 4
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) for v in box)
            and 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)


def validate(rows, expected):
    seen = set()
    for row in rows:
        cid = row["case_id"]
        if not re.fullmatch(r"T\d{3}", cid) or cid not in expected or cid in seen:
            raise ValueError("Unknown or duplicate case")
        seen.add(cid)
        if any(k in row for k in ["label", "new_label", "human_label", "proposed_grade"]):
            raise ValueError("This observation schema does not assign class labels")
        if row["scene_context"] not in SCENES or not row["scene_evidence"].strip() or not row["observation"].strip():
            raise ValueError("Scene and observation evidence required")
        offsets = row["temporal_valid_offsets"]
        if (not isinstance(offsets, list) or 0 not in offsets or len(set(offsets)) != len(offsets)
                or any(isinstance(x, bool) or x not in [-2, -1, 0, 1, 2] for x in offsets)):
            raise ValueError("Invalid temporal references")
        if len(offsets) < 5 and not row["temporal_exclusion_reason"]:
            raise ValueError("Temporal exclusions need a reason")
        if row["same_target_recovery"] != "not_established":
            raise ValueError("Recovery claims need a new schema with target correspondences")
        targets = row["targets"]
        if len(targets) != 3 or {t["target_id"] for t in targets} != {cid + "_" + k for k in TASKS}:
            raise ValueError("Exactly one explicit task record per target kind required")
        for target in targets:
            kind = target["target_id"].rsplit("_", 1)[1]
            box = target["center_normalized_box_or_corridor"]
            visibility = target["center_visibility"]
            if target["task"] != TASKS[kind] or visibility not in STATES:
                raise ValueError("Invalid target task/state")
            if target["near_or_distant"] not in {"image_near", "image_distant", "not_applicable"}:
                raise ValueError("Use image-relative distance only")
            if target["existence_evidence"] == "not_established":
                if box is not None or visibility != "not_evaluable" or target["near_or_distant"] != "not_applicable":
                    raise ValueError("Unknown existence cannot be a failed visibility task")
            elif target["existence_evidence"] == "visible_in_center":
                if not box_valid(box) or visibility == "not_evaluable" or target["near_or_distant"] == "not_applicable":
                    raise ValueError("Visible target needs a valid representative region")
            else:
                raise ValueError("Neighbor-only existence requires correspondence records")
            if target["impairment_region"] is not None and not box_valid(target["impairment_region"]):
                raise ValueError("Invalid impairment region")
            if not target["uncertainty_reason"].strip() or not target["alternative_causes"]:
                raise ValueError("Uncertainty and alternative explanations required")
    if seen != set(expected):
        raise ValueError("Full prepared-case coverage required")


def score(y, pred):
    return dict(n=len(y), counts=np.bincount(y, minlength=3).tolist(),
                macro_f1=float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
                middle_f1=float(f1_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)[1]),
                accuracy=float(accuracy_score(y, pred)), confusion=confusion_matrix(y, pred, labels=[0, 1, 2]).tolist())


def audit(root, observations, out):
    root, observations, out = (Path(p).resolve() for p in [root, observations, out])
    if out.exists() or observations.parent.parent != root or out.parent != observations.parent or out.is_relative_to(root / "review"):
        raise ValueError("Use a new output beside private observations")
    provenance_file = observations.parent / "provenance.json"
    provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
    if (provenance["blind"] is not False or provenance["class_labels_assigned"] is not False
            or provenance["observations_sha256"] != sha(observations)
            or provenance["protocol_sha256"] != sha(observations.parent / "protocol_before_review.json")):
        raise ValueError("Frozen provenance/hash mismatch")
    with (root / "private_priority.csv").open(encoding="utf-8-sig", newline="") as f:
        priority_rows = list(csv.DictReader(f))
    priority = {r["case_id"]: r for r in priority_rows}
    if len(priority) != len(priority_rows):
        raise ValueError("Duplicate private case mapping")
    rows = json.loads(observations.read_text(encoding="utf-8"))
    validate(rows, priority)
    method_file = root / "method.json"
    method = json.loads(method_file.read_text(encoding="utf-8"))
    if method["status"] != "complete":
        raise ValueError("Incomplete temporal preparation")
    for filename, expected in method["input_sha256"].items():
        if sha(filename) != expected:
            raise ValueError("Original labels/predictions/mapping changed")
    with (root / "frame_audit.csv").open(encoding="utf-8-sig", newline="") as f:
        frame_rows = list(csv.DictReader(f))
    checked = set()
    for row in frame_rows:
        cid, offset = row["case_id"], float(row["offset"])
        if cid not in priority or offset not in [-2, -1, 0, 1, 2] or (cid, offset) in checked:
            raise ValueError("Invalid frame coverage")
        name = "center.jpg" if offset == 0 else f"neighbor_{offset:+.0f}.jpg"
        actual = sha(root / "review" / cid / name)
        if actual != row["output_sha256"] or (offset == 0 and actual != priority[cid]["original_sha256"]):
            raise ValueError("Source review image changed")
        checked.add((cid, offset))
    if checked != {(cid, offset) for cid in priority for offset in [-2, -1, 0, 1, 2]}:
        raise ValueError("Every case needs five original audited frames")
    subsets = {"all_prepared": rows, "forward_or_aligned_only": [r for r in rows if r["scene_context"] == "forward_or_aligned_road"]}
    scores = {}
    for name, subset in subsets.items():
        if not subset:
            raise ValueError("Cannot score an empty scope")
        originals = [priority[r["case_id"]] for r in subset]
        y = np.array([int(r["human"]) for r in originals])
        pred = np.array([int(r["rel_road_logistic"]) for r in originals])
        if not np.isin(y, [0, 1, 2]).all() or not np.isin(pred, [0, 1, 2]).all():
            raise ValueError("Invalid original class labels/predictions")
        scores[name] = {"fixed_relative_road_logistic": score(y, pred), "always_normal": score(y, np.zeros_like(y))}
    report = dict(scope="posthoc project-aware observation and fixed-prediction development sensitivity",
                  cases=len(rows), video_groups=len({r["video"] for r in priority_rows}), task_records=sum(len(r["targets"]) for r in rows), frames=len(checked),
                  scene_counts=dict(Counter(r["scene_context"] for r in rows)), solar_counts=dict(Counter(r["solar_evidence"] for r in rows)),
                  task_counts={k: dict(Counter(t["center_visibility"] for r in rows for t in r["targets"] if t["target_id"].endswith("_" + k))) for k in TASKS},
                  temporal_exclusions={r["case_id"]: {"kept": r["temporal_valid_offsets"], "reason": r["temporal_exclusion_reason"]} for r in rows if len(r["temporal_valid_offsets"]) < 5},
                  scores=scores, provenance=provenance,
                  hashes={str(p): sha(p) for p in [observations, provenance_file, method_file, root / "private_priority.csv", root / "frame_audit.csv"]},
                  limitations=["One representative coarse region per task, not exhaustive boxes or physical distance",
                               "Scene scope designed after exploratory review, not an independent test",
                               "Preserved source human labels; no new ordinal target or model training",
                               "Subset scores cannot be compared as improvement over original320 CV .6229",
                               "No actual driver vision or causal solar impairment validation"])
    out.mkdir()
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(__file__, out / Path(__file__).name)
    (out / "source_sha256.txt").write_text(sha(__file__), encoding="ascii")
    cards = []
    for row in rows:
        cid = row["case_id"]
        images = "".join(f'<a href="../../review/{cid}/{name}"><img width="250" src="../../review/{cid}/{name}" alt="{cid} {offset}s"></a>'
                         for offset in [-2, -1, 0, 1, 2] for name in ["center.jpg" if offset == 0 else f"neighbor_{offset:+.0f}.jpg"])
        cards.append(f'<section><h2>{cid}: {row["scene_context"]}</h2>{images}<p>{html.escape(row["observation"])}</p><pre>{html.escape(json.dumps(row["targets"], ensure_ascii=False, indent=2))}</pre></section>')
    (out / "report.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>50사례 대상별 관찰</title><style>body{font:15px system-ui;margin:24px}pre{white-space:pre-wrap}section{border-top:1px solid #aaa;padding:15px 0}img{max-width:45%;height:auto}</style><h1>50사례 대상별 관찰</h1><p>프로젝트를 아는 AI의 탐색 관찰. 원본 라벨 유지, 새 학습·정답·현장 사람 시야 검증 아님. 참조에서 제외한 화면도 원본 보존을 위해 표시함. 제외 근거는 JSON 참조.</p>' + "".join(cards) + '</html>', encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/temporal_visibility_20261104"))
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.observations, args.out)
    print(json.dumps({k: result[k] for k in ["cases", "frames", "task_records", "scene_counts", "task_counts", "scores"]}, ensure_ascii=False))
