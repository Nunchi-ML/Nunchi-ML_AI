"""Change training targets, keeping original320 labels, folds and .5/.5 rules.

One-factor controls: binary alert A, positive-only severe B, and their fixed
combination. Reproduce every legacy prediction before reporting any candidate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import road_visibility_probe as R
from ordinal_compare import metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compose(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 1 or a.shape != b.shape or not np.isfinite([a, b]).all() or ((a < 0) | (a > 1) | (b < 0) | (b > 1)).any():
        raise ValueError("Expected paired valid binary scores")
    return np.column_stack([1-a, a*(1-b), a*b])


def binary_probability(train, test, cols, target):
    target = np.asarray(target, int)
    if len(train) != len(target) or not len(train) or not np.isin(target, [0, 1]).all():
        raise ValueError("Nonempty matched binary training target required")
    if len(np.unique(target)) == 1:
        return np.full(len(test), float(target[0]))
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    model.fit(train[cols], target)
    return model.predict_proba(test[cols])[:, list(model.classes_).index(1)]


def fold_predictions(train, test):
    cols = R.G + R.P.REL_R
    pf, pg = R.C.probabilities(train, test, cols), R.C.probabilities(train, test, R.G)
    old_a = pf[:, 1:].sum(1)
    old_b = pg[:, 2] / np.maximum(pg[:, 1:].sum(1), 1e-9)
    new_a = binary_probability(train, test, cols, train.label.ge(1).astype(int))
    positive = train[train.label.ge(1)]
    new_b = binary_probability(positive, test, R.G, positive.label.eq(2).astype(int))
    return {"legacy": compose(old_a, old_b), "binary_A_only": compose(new_a, old_b),
            "positive_only_B": compose(old_a, new_b), "binary_A_positive_B": compose(new_a, new_b)}


def run(data, out):
    data, out = Path(data).resolve(), Path(out).resolve()
    if out.exists() or not out.is_relative_to(data):
        raise ValueError("Use new private output folder")
    base = data / "road_visibility_20261104/run30"
    files = [base / "training_features.csv", base / "predictions.csv", base / "splits.csv", Path(__file__), Path(R.__file__), Path(R.C.__file__)]
    hashes = {str(p): sha(p) for p in files}
    full = pd.read_csv(files[0], dtype={"id": str})
    train = full[full.camera.ne("AI Hub")].reset_index(drop=True)
    if len(train) != 320 or train.group.nunique() != 24 or not train.id.is_unique:
        raise ValueError("Expected original320/24groups")
    splits = pd.read_csv(files[2], dtype={"id": str})
    reference = pd.read_csv(files[1], dtype={"id": str})
    reference = reference[reference.test.eq("group_cv320") & reference.method.eq("legacy")]
    if len(reference) != 30*320 or reference.duplicated(["repeat", "id"]).any():
        raise ValueError("Expected9600 frozen baseline predictions")
    out.mkdir()
    method = dict(status="running", rows=320, groups=24, repeats=30, folds="Preserved run30 splits, no reselection",
                  class_labels="Original0/1/2 unchanged", alert_threshold=.5, severe_threshold=.5,
                  candidates=["binary_A_only", "positive_only_B", "binary_A_positive_B"],
                  training="A=binary0vs1+ on outer-train; B=binary1vs2 on outer-train positives only; balanced logistic C1; scaler fitted only on training",
                  hashes=hashes, selection="Fixed ablations, no tuning/calibration or label replacement")
    (out / "method_before_run.json").write_text(json.dumps(method, indent=2), encoding="utf-8")
    shutil.copyfile(__file__, out / Path(__file__).name)
    scores, predictions, support = [], [], []
    for repeat in range(30):
        part = splits[splits["repeat"].eq(repeat)]
        if len(part) != 320 or not part.id.is_unique or set(part.id) != set(train.id):
            raise ValueError("Invalid frozen split coverage")
        all_p, seen = {}, np.zeros(320, int)
        for fold in range(5):
            valid_ids = set(part.loc[part.fold.eq(fold), "id"])
            mask = train.id.isin(valid_ids).to_numpy()
            a, b = train[~mask], train[mask]
            if not len(b) or set(a.group) & set(b.group):
                raise ValueError("Capture-group leakage")
            for name, p in fold_predictions(a, b).items():
                R.validate_probabilities(p)
                all_p.setdefault(name, np.zeros((320, 3)))[mask] = p
            seen[mask] += 1
            support.append(dict(repeat=repeat, fold=fold, positive_rows=int(a.label.ge(1).sum()), severe_rows=int(a.label.eq(2).sum())))
        if not np.all(seen == 1):
            raise ValueError("Every row needs one OOF prediction")
        for name, p in all_p.items():
            pred = R.C.decision(p)
            if name == "legacy":
                old = reference[reference["repeat"].eq(repeat)].set_index("id").loc[train.id, "pred"].to_numpy()
                if not np.array_equal(pred, old):
                    raise ValueError("Legacy baseline changed")
            scores.append(dict(repeat=repeat, method=name, **metrics(train.label.to_numpy(), pred)))
            predictions.extend(dict(repeat=repeat, method=name, id=cid, pred=int(value)) for cid, value in zip(train.id, pred))
        pd.DataFrame(scores).to_csv(out / "scores.csv", index=False, encoding="utf-8-sig")
        print("conditional training repeat", repeat+1, "/30", flush=True)
    frame = pd.DataFrame(scores)
    numeric = [c for c in frame.select_dtypes("number") if c != "repeat"]
    summary = frame.groupby("method")[numeric].mean()
    paired = frame.pivot(index="repeat", columns="method", values="macro_f1")
    summary["delta_vs_legacy"] = summary.macro_f1-summary.loc["legacy", "macro_f1"]
    summary["improved_repeats"] = [(paired[c] > paired.legacy).sum() for c in summary.index]
    summary.to_csv(out / "summary.csv", encoding="utf-8-sig")
    pd.DataFrame(predictions).to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(support).to_csv(out / "training_support.csv", index=False, encoding="utf-8-sig")
    if any(sha(p) != v for p, v in hashes.items()):
        raise ValueError("Frozen inputs changed")
    method.update(status="complete", baseline_reproduced_predictions=9600,
                  limitations=["Repeated CV shares data and is not independent confidence evidence", "Only training target design changed, no new held-out data", "No final model or deployment change"])
    (out / "report.json").write_text(json.dumps(method, indent=2), encoding="utf-8")
    (out / "report.html").write_text('<!doctype html><meta charset="utf-8"><h1>Fixed-label conditional binary training ablation</h1><p>Original320,24capture groups, frozen5folds×30. Legacy9600 predictions reproduced. No threshold tuning or new labels. CV repeats overlap; no independent confidence claim.</p>'+summary.round(5).to_html(), encoding="utf-8")
    print(summary[["macro_f1", "f1_1", "recall_2", "delta_vs_legacy", "improved_repeats"]].round(5).to_string(), flush=True)
    return method


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.out)
