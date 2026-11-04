"""Rescore FROZEN photo predictions against separate human/AI label versions.

No model fitting, relabeling, reference-dependent selection, G imputation or
claim of objective driver vision. Compare labels on identical image IDs;
also expose coverage and the effect of abstentions on sample composition.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "glare_video_source"))
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from new_camera_audit import load_labels
from ordinal_compare import metrics


def checked_predictions(path, expected_ids):
    raw = pd.read_csv(path, dtype={"id": str})
    data = raw[raw.test.eq("human50_photo_only")].copy()
    if data.duplicated(["id", "method"]).any():
        raise ValueError("Duplicate frozen prediction ID/method")
    matrix = data.pivot(index="id", columns="method", values="pred").reindex(expected_ids)
    if matrix.isna().any().any() or not matrix.isin([0, 1, 2]).all().all():
        raise ValueError("Every method needs valid predictions for every frozen ID")
    return matrix.astype(int)


def shared_ids(labels, expected_ids):
    result = pd.Index(expected_ids)
    for label in labels.values():
        result = result.intersection(label.index, sort=False)
    if result.empty:
        raise ValueError("No shared labelled image IDs")
    return result


def evaluate(labels, predictions):
    for name, reference in labels.items():
        if (not reference.index.is_unique or reference.index.hasnans
                or not reference.isin([0, 1, 2]).all()
                or len(reference.index.difference(predictions.index))):
            raise ValueError(f"Invalid reference labels: {name}")
    common = shared_ids(labels, predictions.index)
    regimes = [("common_all_available", name, common) for name in labels]
    regimes += [("own_coverage_not_paired", name, labels[name].index) for name in labels]
    if "human" in labels and "claude" in labels:
        pair = shared_ids({k: labels[k] for k in ["human", "claude"]}, predictions.index)
        regimes += [("common_human_claude", name, pair) for name in ["human", "claude"]]
    rows = []
    for regime, name, ids in regimes:
        y = labels[name].loc[ids].to_numpy(int)
        for method in predictions.columns:
            pred = predictions.loc[ids, method].to_numpy(int)
            rows.append(dict(regime=regime, reference=name, method=method, rows=len(ids),
                             **{f"class{k}_count": int((y == k).sum()) for k in range(3)},
                             alert_accuracy=float(((y >= 1) == (pred >= 1)).mean()),
                             **metrics(y, pred)))
    agreement = []
    for a, b in itertools.combinations(labels, 2):
        ids = shared_ids({k: labels[k] for k in [a, b]}, predictions.index)
        ya, yb = labels[a].loc[ids].to_numpy(int), labels[b].loc[ids].to_numpy(int)
        agreement.append(dict(a=a, b=b, rows=len(ids), agreement=float((ya == yb).mean()),
                              kappa=float(cohen_kappa_score(ya, yb, labels=[0, 1, 2])),
                              qwk=float(cohen_kappa_score(ya, yb, labels=[0, 1, 2], weights="quadratic")),
                              severe_disagreement=int((np.abs(ya-yb) == 2).sum()),
                              confusion_matrix=json.dumps(confusion_matrix(ya, yb, labels=[0, 1, 2]).tolist())))
    return pd.DataFrame(rows), pd.DataFrame(agreement), common


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("data/ai_label_review_20261104/evaluation"))
    ap.add_argument("--gemini-labels", type=Path, help="Optional actual Gemini CSV; never fabricate missing labels")
    a = ap.parse_args()
    if not a.out.resolve().is_relative_to(a.data_root.resolve()):
        ap.error("Output must remain in private data-root")
    a.out.mkdir(parents=True, exist_ok=True)
    root = a.data_root / "ai_label_review_20261104"
    human_file = a.data_root / "new_cameras/human_review_50/labels_성진.csv"
    claude_file = a.data_root / "new_cameras/labels_claude.csv"
    codex_file = root / "labels_codex.csv"
    photo_file = a.data_root / "road_visibility_20261104/run30/predictions.csv"
    old_file = a.data_root / "new_cameras/human_review_50_evaluation/predictions.csv"
    human_ids = pd.read_csv(human_file, dtype={"id": str}).id
    expected = pd.read_csv(a.data_root / "new_cameras/label_pack/label_sheet_blind.csv", dtype={"id": str}).id
    label_files = {"human": human_file, "claude": claude_file, "codex": codex_file}
    if a.gemini_labels:
        label_files["gemini"] = a.gemini_labels
    sources = [photo_file, old_file, root / "codex_annotations_frozen.json", root / "rubric.txt", *label_files.values()]
    fingerprints = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    labels, evidence = {}, {}
    for name, path in label_files.items():
        series, _, info = load_labels([path], expected, "human_single" if name == "human" else "ai_provisional")
        labels[name] = series.reindex(human_ids).dropna().astype(int)
        evidence[name] = dict(info, accepted_in_human50=len(labels[name]))
    pred = checked_predictions(photo_file, human_ids)
    old = pd.read_csv(old_file, dtype={"id": str}).set_index("id").reindex(human_ids)
    for current, previous in [("screen_lgb", "ML1 LightGBM(화면 12, 기존 모델 ② 형태)"),
                              ("rel_logistic", "ML2 로지스틱(상대 12)")]:
        if not np.array_equal(pred[current], old[previous]):
            raise ValueError("Frozen predictions differ from original photo baseline")
    for col in old:
        if col.startswith("R") or col == "B1 always alert":
            pred[col] = old[col].to_numpy()
    train = pd.read_csv(a.data_root / "road_visibility_20261104/run30/human_features.csv", dtype={"id": str}).set_index("id").reindex(human_ids)
    if not np.array_equal(old.image_sha256, train.image_sha256):
        raise ValueError("Photo identity changed between evaluations")
    scores, agreement, common = evaluate(labels, pred)
    scores.to_csv(a.out / "scores.csv", index=False, encoding="utf-8-sig")
    agreement.to_csv(a.out / "label_agreement.csv", index=False, encoding="utf-8-sig")
    joined = pred.copy()
    for name, lab in labels.items():
        joined["reference_"+name] = lab.reindex(joined.index)
    joined["common_all_available"] = joined.index.isin(common)
    joined.to_csv(a.out / "frozen_predictions_and_references.csv", encoding="utf-8-sig")
    report = dict(status="complete_available_references", common_rows=len(common), total_rows=len(pred),
                  evidence=evidence, files=fingerprints,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  codex_provenance=json.loads((root / "provenance.json").read_text(encoding="utf-8")),
                  limitations=["No new fitting or objective driver-vision ground truth", "Reference-specific F1 measures agreement with that rater", "Own-coverage tables are not paired comparisons", "Same data reused for development", "Project-aware Codex judgment is not fully independent", "Gemini absent unless a real CSV is supplied"])
    if fingerprints != {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}:
        raise AssertionError("Frozen inputs changed during scoring")
    (a.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    html = '<!doctype html><meta charset="utf-8"><title>Label reference sensitivity</title><h1>Frozen predictions, separate label references</h1><p>AI labels are reference judgments, not measured driver vision. Same-image paired comparisons only. No retraining, label replacement or final selection. Gemini annotations are absent unless a real CSV is provided.</p>'
    html += '<h2>Coverage</h2>' + pd.DataFrame([dict(reference=k, accepted=len(v), abstained_or_missing=len(pred)-len(v), **{f"class{j}": int((v==j).sum()) for j in range(3)}) for k,v in labels.items()]).to_html(index=False)
    html += '<h2>Pairwise rater agreement</h2>' + agreement.round(4).to_html(index=False)
    for regime in scores.regime.unique():
        html += '<h2>'+regime+'</h2>' + scores[scores.regime.eq(regime)].round(4).to_html(index=False)
    (a.out / "report.html").write_text(html, encoding="utf-8")
    print(agreement.round(4).to_string(index=False))
    print('COMMON SAME-IMAGE COMPARISON',len(common))
    print(scores[scores.regime.eq("common_all_available")][["reference","method","rows","macro_f1","f1_1","recall_2","accuracy","alert_accuracy"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
