"""Paired same-image comparison of frozen predictions and real Gemini outputs.

Gemini grades are auxiliary rubric opinions, never measured driver ground truth.
The two conditions share one model and are not two independent raters.
"""
import argparse
import hashlib
from html import escape
import itertools
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from gemini_reference_probe import validate_packet as validate_read_packet
from label_reference_probe import checked_predictions
from new_camera_audit import load_labels
from ordinal_compare import metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric_grades(series):
    if not series.index.is_unique or not series.isin(["0", "1", "2", "abstain"]).all():
        raise ValueError("Invalid auxiliary grades/duplicate IDs")
    return series[series.ne("abstain")].astype(int)


def paired_ids(labels, expected):
    result = pd.Index(expected)
    for values in labels.values():
        if not values.index.is_unique or len(values.index.difference(expected)) or not values.isin([0, 1, 2]).all():
            raise ValueError("Invalid labels/IDs")
        result = result.intersection(values.index, sort=False)
    return result


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def report(experiment, data, out):
    experiment, data, out = (Path(p).resolve() for p in [experiment, data, out])
    if out.exists() or not out.is_relative_to(experiment):
        raise ValueError("Use a new private report output")
    manifest_file, progress_file = experiment / "manifest.json", experiment / "progress.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    validator = validate_read_packet
    if manifest.get("transport") == "acp_images":
        from gemini_acp_transport import validate_packet as validator
    progress = json.loads(progress_file.read_text(encoding="utf-8"))
    if progress["status"] != "complete" or len(progress["packets"]) != 20:
        raise ValueError("Cannot report planned100 judgments as complete without20 valid batches")
    for filename, expected in manifest["source_sha256"].items():
        if sha(filename) != expected:
            raise ValueError("Original source changed")
    fingerprints = {str(p): sha(p) for p in [manifest_file, progress_file, experiment / "private_pairs.json"]}
    for pattern in ["transport_provenance.json", "transport_amendment_*/provenance.json", "transport_amendment_*/*.py", "progress_before_resume_*.json", "source_at_prepare.py", "gemini_acp_transport.py"]:
        fingerprints.update({str(p): sha(p) for p in experiment.glob(pattern)})
    pairs = pd.DataFrame(json.loads((experiment / "private_pairs.json").read_text(encoding="utf-8")))
    if len(pairs) != 100 or pairs.duplicated(["condition", "case_id"]).any() or pairs.duplicated(["condition", "temporal_case_id"]).any():
        raise ValueError("Expected100 one-to-one blinded-condition mappings")
    records, conversations, usages = [], [], []
    for packet in manifest["packets"]:
        stdout = experiment / "runs" / packet["name"] / "stdout.jsonl"
        events = [json.loads(line) for line in stdout.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        validated = validator(packet, events)
        fingerprints[str(stdout)] = sha(stdout)
        conversations.append(validated["conversation_id"])
        usages.append(validated["usage"])
        for item in validated["items"]:
            records.append(dict(condition=packet["condition"], **item))
    if len(set(conversations)) != 20:
        raise ValueError("Conditions/batches must use fresh conversations")
    annotations = pd.DataFrame(records).merge(pairs, on=["condition", "case_id"], how="left", validate="one_to_one")
    temporal = data / "temporal_visibility_20261104"
    mapping_file = temporal / "private_id_map.csv"
    mapping = pd.read_csv(mapping_file, dtype={"id": str}).rename(columns={"case_id": "temporal_case_id"})
    if len(mapping) != 50 or not mapping.id.is_unique or not mapping.temporal_case_id.is_unique:
        raise ValueError("Invalid original mapping")
    annotations = annotations.merge(mapping[["temporal_case_id", "id"]], on="temporal_case_id", how="left", validate="many_to_one")
    if annotations.id.isna().any() or annotations.duplicated(["condition", "id"]).any():
        raise ValueError("Unmatched/duplicate source images")
    image_order = mapping.id
    files = {"human": data / "new_cameras/human_review_50/labels_성진.csv", "claude": data / "new_cameras/labels_claude.csv", "codex": data / "ai_label_review_20261104/labels_codex.csv"}
    sheet_file = data / "new_cameras/label_pack/label_sheet_blind.csv"
    expected = pd.read_csv(sheet_file, dtype={"id": str}).id
    labels = {}
    for name, filename in files.items():
        values, _, _ = load_labels([filename], expected, "human_single" if name == "human" else "ai_provisional")
        labels[name] = values.reindex(image_order).dropna().astype(int)
    for condition in ["center", "temporal"]:
        values = annotations[annotations.condition.eq(condition)].set_index("id").reference_grade
        if len(values) != 50:
            raise ValueError("Every condition requires50 actual responses")
        labels["gemini_" + condition] = numeric_grades(values)
    photo_file = data / "road_visibility_20261104/run30/predictions.csv"
    predictions = checked_predictions(photo_file, image_order)
    fingerprints.update({str(p): sha(p) for p in [mapping_file, sheet_file, photo_file, *files.values()]})
    coverage = [dict(reference=name, total=50, accepted=len(values), abstain=50-len(values), **{f"class{k}": int(values.eq(k).sum()) for k in range(3)}) for name, values in labels.items()]
    regimes = {"paired_gemini_conditions_human": {k: labels[k] for k in ["human", "gemini_center", "gemini_temporal"]}, "paired_all_five_references": labels}
    rows, common = [], {}
    for regime, references in regimes.items():
        ids = paired_ids(references, predictions.index)
        common[regime] = ids.tolist()
        for reference, values in references.items():
            if len(ids):
                for method in predictions:
                    rows.append(dict(regime=regime, reference=reference, method=method, rows=len(ids), **metrics(values.loc[ids].to_numpy(), predictions.loc[ids, method].to_numpy())))
    for reference, values in labels.items():
        if len(values):
            for method in predictions:
                rows.append(dict(regime="own_coverage_not_paired", reference=reference, method=method, rows=len(values), **metrics(values.to_numpy(), predictions.loc[values.index, method].to_numpy())))
    agreement = []
    for (a, va), (b, vb) in itertools.combinations(labels.items(), 2):
        ids = paired_ids({a: va, b: vb}, image_order)
        if len(ids):
            ya, yb = va.loc[ids].to_numpy(), vb.loc[ids].to_numpy()
            agreement.append(dict(a=a, b=b, rows=len(ids), agreement=float((ya == yb).mean()), kappa=float(cohen_kappa_score(ya, yb, labels=[0, 1, 2])), qwk=float(cohen_kappa_score(ya, yb, labels=[0, 1, 2], weights="quadratic")), severe_disagreements=int((np.abs(ya-yb) == 2).sum()), confusion=confusion_matrix(ya, yb, labels=[0, 1, 2]).tolist()))
    wide = annotations.pivot(index="id", columns="condition", values="reference_grade").reindex(image_order)
    transition = pd.crosstab(wide.center, wide.temporal).reindex(index=["0", "1", "2", "abstain"], columns=["0", "1", "2", "abstain"], fill_value=0)
    changed = wide[wide.center.ne(wide.temporal)].copy()
    state_counts = {condition: {field: annotations.loc[annotations.condition.eq(condition), field].value_counts().to_dict()
                               for field in ["scene", "sun_evidence", "lane", "vehicle", "sign", "reference_comparability"]}
                    for condition in ["center", "temporal"]}
    task_changes = {}
    for field in ["scene", "sun_evidence", "lane", "vehicle", "sign"]:
        pivot = annotations.pivot(index="id", columns="condition", values=field).reindex(image_order)
        task_changes[field] = dict(changed=int(pivot.center.ne(pivot.temporal).sum()),
                                   transitions=pd.crosstab(pivot.center, pivot.temporal).to_dict())
    result = dict(status="complete_real_gemini_outputs", model=manifest["model"], transport=manifest.get("transport", "read_file"),
                  protocol=manifest["protocol"], cli_version=manifest["cli_version"], seed=manifest["seed"],
                  coverage=coverage, common_counts={k: len(v) for k,v in common.items()},
                  agreement=agreement, condition_changes=len(changed), transition_order=["0", "1", "2", "abstain"], transition=transition.to_numpy().tolist(),
                  state_counts=state_counts, task_changes=task_changes,
                  total_usage={k: sum(u.get(k, 0) for u in usages) for k in ["input_tokens", "output_tokens", "thinking_tokens", "total_tokens"]},
                  files=fingerprints, source_sha256=sha(__file__),
                  limitations=["Same model two conditions, not independent raters; fresh conversation per five cases",
                               "Original in-image watermarks remain; supplied attachments/read traces are audited, model viewing claims do not prove attention",
                               "Auxiliary legacy-rubric grades, no new canonical information-loss label boundaries",
                               "Reference scores reflect label definition, not objective improvement or human error",
                               "Reused development50 without G inputs; no training or original320 CV .6229 improvement",
                               "Temporal offsets may be reference-only; undocumented center0 requirement removed after first batch, raw judgments unchanged and amendment preserved",
                               "Own coverage scores have different sample composition; use paired tables"])
    out.mkdir()
    pd.DataFrame(rows).to_csv(out / "scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(agreement).to_csv(out / "agreement.csv", index=False, encoding="utf-8-sig")
    annotations.to_json(out / "annotations.json", orient="records", force_ascii=False, indent=2)
    changed.to_csv(out / "condition_changed_cases.csv", encoding="utf-8-sig")
    for condition in ["center", "temporal"]:
        lab = labels["gemini_" + condition].rename("label").reindex(image_order)
        lab.to_csv(out / f"labels_gemini_{condition}.csv", encoding="utf-8-sig")
    (out / "report.json").write_text(json.dumps(clean(result), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    shutil.copyfile(__file__, out / Path(__file__).name)
    page = '<!doctype html><meta charset="utf-8"><title>Gemini reference comparison</title><h1>Real Gemini outputs, fixed predictions</h1><p>Auxiliary rubric opinions. No training, canonical relabeling, objective driver ground truth or final model selection. Same-image paired scores only.</p>'
    page += '<h2>Coverage</h2>' + pd.DataFrame(coverage).to_html(index=False) + '<h2>Agreement</h2>' + pd.DataFrame(agreement).round(4).to_html(index=False)
    page += '<h2>Center→temporal, including abstain</h2>' + transition.to_html()
    for regime in regimes:
        frame = pd.DataFrame(rows)
        page += '<h2>' + regime + '</h2>' + (frame[frame.regime.eq(regime)].round(4).to_html(index=False) if len(frame) else '<p>No shared accepted labels</p>')
    (out / "report.html").write_text(page, encoding="utf-8")
    review = ['<!doctype html><meta charset="utf-8"><title>Analyst reference review</title>',
              '<style>body{font:16px sans-serif;max-width:1200px;margin:24px auto}img{max-width:100%}td{vertical-align:top;width:50%;padding:12px}details{margin:20px 0}summary{cursor:pointer}</style>',
              '<h1>Analyst review — original labels remain unchanged</h1><p>Contains answers. These are auxiliary opinions and posthoc review flags, not ground truth. Priority: human severe, condition change or abstention. All50 remain available below.</p>']
    for identifier in image_order:
        case = annotations[annotations.id.eq(identifier)].set_index("condition")
        cid = case.iloc[0].temporal_case_id
        human_grade = int(labels["human"].loc[identifier])
        flag = human_grade == 2 or identifier in changed.index or case.reference_grade.eq("abstain").any()
        photo = temporal / "review" / cid / "center.jpg"
        review.append('<details' + (' open' if flag else '') + '><summary>' + escape(f'{cid}: human={human_grade}, center={case.loc["center", "reference_grade"]}, temporal={case.loc["temporal", "reference_grade"]}') + '</summary>')
        review.append('<img loading="lazy" src="' + escape(photo.as_uri(), quote=True) + '" alt="center photograph">')
        review.append('<p>' + ' · '.join('<a href="' + escape((temporal / "review" / cid / f"neighbor_{offset:+d}.jpg").as_uri(), quote=True) + '">' + f'{offset:+d}s reference' + '</a>' for offset in [-2, -1, 1, 2]) + '</p><table><tr>')
        for condition in ["center", "temporal"]:
            item = case.loc[condition]
            review.append('<td><b>' + condition + '</b><p>' + escape(str(item.evidence)) + '</p><p>' + escape(str(item.uncertainty)) + '</p><p>' + escape(f'lane={item.lane}; vehicle={item.vehicle}; sign={item.sign}; scene={item.scene}') + '</p></td>')
        review.append('</tr></table></details>')
    (out / "disagreement_review.html").write_text("\n".join(review), encoding="utf-8")
    if any(sha(p) != v for p, v in fingerprints.items()):
        raise ValueError("Inputs changed during reporting")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.experiment, args.data, args.out)
    print(json.dumps(clean({k: result[k] for k in ["coverage", "common_counts", "condition_changes", "agreement"]}), ensure_ascii=False))
