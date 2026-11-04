"""Label validation and reports for new-video evaluation; no model selection."""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score


def load_labels(paths, expected_ids, kind):
    """Require two votes and a strict majority for declared human consensus."""
    resolved = [Path(p).resolve() for p in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("The same label file cannot count as two raters")
    if kind == "human_consensus" and len(paths) < 2:
        raise ValueError("human_consensus requires at least two independent raters")
    if kind != "human_consensus" and len(paths) != 1:
        raise ValueError("Single-rater/AI evaluation requires exactly one label file")
    expected = pd.Index(expected_ids, name="id")
    if expected.hasnans or not expected.is_unique:
        raise ValueError("Blind sheet IDs must be present and unique")
    votes, identities, uncertain = [], [], []
    for p in paths:
        frame = pd.read_csv(p, dtype=str, keep_default_na=False)
        if not {"id", "label", "labeler"}.issubset(frame.columns):
            raise ValueError(f"Missing id/label/labeler in {Path(p).name}")
        frame["id"] = frame.id.str.strip()
        if frame.id.eq("").any() or not frame.id.is_unique:
            raise ValueError(f"Missing/duplicate label ID in {Path(p).name}")
        if len(pd.Index(frame.id).difference(expected)):
            raise ValueError(f"Unknown frame ID in {Path(p).name}")
        identity = frame.labeler.str.strip().unique()
        if len(identity) != 1 or not identity[0]:
            raise ValueError("Each label file needs one non-empty rater identity")
        identities.append(identity[0].casefold())
        if kind.startswith("human") and any(s in identities[-1] for s in ("claude", "chatgpt", "ai 임시")):
            raise ValueError("AI-labelled files cannot be declared human labels")
        value = frame.label.str.strip().str.lower()
        if not value.isin(["", "x", "0", "1", "2"]).all():
            raise ValueError(f"Invalid label in {Path(p).name}; use 0/1/2/x/blank")
        value.index = frame.id
        votes.append(pd.to_numeric(value.replace({"x": ""}), errors="coerce").reindex(expected))
        unsure = frame.get("unsure", pd.Series("", index=frame.index)).str.strip().str.lower()
        uncertain.append(int(unsure.isin(["1", "true"]).sum()))
    if len(set(identities)) != len(identities):
        raise ValueError("Rater identities must be distinct")
    lab = pd.concat(votes, axis=1)
    lab.columns = [f"rater_{i + 1}" for i in range(len(votes))]
    counts = pd.DataFrame({c: lab.eq(c).sum(axis=1) for c in (0, 1, 2)})
    needed = max(2, len(votes) // 2 + 1) if kind == "human_consensus" else 1
    accepted = counts.max(axis=1).ge(needed)
    labels = counts.idxmax(axis=1).where(accepted).dropna().astype(int).rename("label")
    review = lab.copy()
    review["valid_votes"] = lab.notna().sum(axis=1)
    review["label"] = counts.idxmax(axis=1).where(accepted)
    review["accepted"] = accepted
    pairs = []
    for i, j in combinations(range(len(votes)), 2):
        pair = lab.iloc[:, [i, j]].dropna()
        kappa = None
        if len(pair) and len(np.unique(pair.to_numpy())) > 1:
            kappa = float(cohen_kappa_score(pair.iloc[:, 0], pair.iloc[:, 1]))
        pairs.append(dict(raters=[identities[i], identities[j]], rows=len(pair),
                          agreement=float(pair.iloc[:, 0].eq(pair.iloc[:, 1]).mean()) if len(pair) else None,
                          kappa=kappa))
    info = dict(kind=kind, provenance="Declared by caller; independence requires the collection protocol",
                raters=identities, rows=len(expected), accepted=len(labels),
                rejected_or_pending=int((~accepted).sum()), unsure_counts=uncertain,
                minimum_matching_votes=needed, pairwise=pairs)
    if labels.empty:
        raise ValueError("No accepted labels; collect independent votes or resolve disagreements")
    return labels, review, info


def score(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    truth, alert = y >= 1, prediction >= 1
    normal = ~truth
    recall = float(alert[truth].mean()) if truth.any() else None
    specificity = float((~alert[normal]).mean()) if normal.any() else None
    return dict(rows=len(y), normal_count=int(normal.sum()), alert_count=int(truth.sum()),
                alert_accuracy=float((truth == alert).mean()),
                alert_balanced_accuracy=(recall + specificity) / 2 if recall is not None and specificity is not None else None,
                macro_f1=float(f1_score(y, prediction, labels=[0, 1, 2], average="macro", zero_division=0)),
                false_alert_rate=1 - specificity if specificity is not None else None,
                alert_recall=recall,
                alert_precision=float(truth[alert].mean()) if alert.any() else None,
                severe_recall=float((prediction[y == 2] == 2).mean()) if (y == 2).any() else None,
                confusion_matrix=confusion_matrix(y, prediction, labels=[0, 1, 2]).tolist())


def compare_reference(human, reference):
    """Compare an AI/reference label after human review, without adding a vote."""
    paired = human.to_frame("human").join(reference.rename("reference"), how="left")
    valid = paired.dropna().astype(int)
    if valid.empty:
        raise ValueError("No jointly labelled frames for agreement")
    a, b = valid.human.to_numpy(), valid.reference.to_numpy()
    kappa = {}
    for weights in (None, "linear", "quadratic"):
        value = cohen_kappa_score(a, b, labels=[0, 1, 2], weights=weights) if len(np.unique([a, b])) > 1 else None
        kappa[weights or "unweighted"] = float(value) if value is not None and np.isfinite(value) else None
    paired["same_grade"] = paired.human.eq(paired.reference).where(paired.reference.notna())
    info = dict(human_rows=len(human), jointly_labelled=len(valid),
                reference_excluded_or_missing=int(paired.reference.isna().sum()),
                exact_agreement=float((a == b).mean()), kappa=kappa,
                extreme_disagreements=int((np.abs(a-b) == 2).sum()),
                reference_as_prediction=score(a, b),
                limitation="Agreement with one human is not two-person consensus. Reference abstentions remain excluded/missing.")
    return paired, info


def bootstrap_accuracy(y, predictions, groups, repeats=3000):
    """Resample entire videos, paired across all methods, with a fixed seed."""
    rng = np.random.default_rng(0)
    unique = np.unique(groups)
    index = {c: np.flatnonzero(groups == c) for c in unique}
    draws = {name: [] for name in predictions}
    for _ in range(repeats):
        ii = np.concatenate([index[c] for c in rng.choice(unique, len(unique))])
        for name, pred in predictions.items():
            draws[name].append(float(np.mean((y[ii] >= 1) == (pred[ii] >= 1))))
    intervals, differences = [], []
    for name, values in draws.items():
        lo, hi = np.percentile(values, [2.5, 97.5])
        intervals.append(dict(method=name, metric="alert_accuracy", lower=float(lo), upper=float(hi)))
        for ml in (k for k in predictions if k.startswith("ML")):
            if ml == name:
                continue
            delta = np.asarray(draws[ml]) - values
            lo, hi = np.percentile(delta, [2.5, 97.5])
            observed = np.mean((y >= 1) == (predictions[ml] >= 1)) - np.mean((y >= 1) == (predictions[name] >= 1))
            differences.append(dict(method=ml, reference=name, metric="alert_accuracy",
                                    difference=float(observed), lower=float(lo), upper=float(hi)))
    return intervals, differences
