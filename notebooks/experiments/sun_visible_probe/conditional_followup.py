"""Held-camera/reused-AIHub stress checks and conditional frozen-OOF bootstrap."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import conditional_training_probe as M
from ordinal_compare import metrics

METHODS = ["legacy", "binary_A_only", "positive_only_B", "binary_A_positive_B"]


def run(data, root):
    data, root = Path(data).resolve(), Path(root).resolve()
    initial = json.loads((root / "report.json").read_text(encoding="utf-8"))
    if initial["status"] != "complete" or initial["baseline_reproduced_predictions"] != 9600:
        raise ValueError("Completed reproduced CV required before followup")
    full = pd.read_csv(data / "road_visibility_20261104/run30/training_features.csv", dtype={"id": str})
    train = full[full.camera.ne("AI Hub")].reset_index(drop=True)
    if len(train) != 320 or train.group.nunique() != 24:
        raise ValueError("Original320/24groups required")
    inputs = [data / "road_visibility_20261104/run30/training_features.csv", root / "predictions.csv", Path(M.__file__), Path(__file__)]
    hashes = {str(p): M.sha(p) for p in inputs}
    specification = dict(methods=METHODS, bootstrap=2000, seed=0,
                         comparisons=["reused_AIHub", *["held_camera:"+str(c) for c in train.camera.unique()]],
                         selection="All fixed methods retained; no tuning after stress results",
                         uncertainty="Capture-group composition of frozen OOF predictions, no refitting; repeats are dependent")
    specification_file = root / "followup_specification.json"
    if specification_file.exists():
        raise ValueError("New followup output required; do not overwrite existing result")
    specification_file.write_text(json.dumps(specification, ensure_ascii=False, indent=2), encoding="utf-8")
    scores = []
    comparisons = [("reused_AIHub", train, full[full.camera.eq("AI Hub")])]
    comparisons += [("held_camera:"+str(c), train[train.camera.ne(c)], train[train.camera.eq(c)]) for c in train.camera.unique()]
    for name, tr, va in comparisons:
        for method, p in M.fold_predictions(tr, va).items():
            scores.append(dict(test=name, method=method, rows=len(va), **metrics(va.label.to_numpy(), M.R.C.decision(p))))
    # Unique target rows are never relabelled; only their frozen OOF predictions
    # are resampled by complete capture groups, paired across all methods.
    predictions = pd.read_csv(root / "predictions.csv", dtype={"id": str})
    if len(predictions) != 320*30*4 or predictions.duplicated(["repeat", "method", "id"]).any():
        raise ValueError("Expected complete four-method OOF predictions")
    metadata = train.set_index("id")
    groups = sorted(train.group.unique())
    group_index = {g: i for i, g in enumerate(groups)}
    method_index = {m: i for i, m in enumerate(METHODS)}
    confusion = np.zeros((len(groups), 30, 4, 3, 3), int)
    for row in predictions.itertuples():
        sample = metadata.loc[row.id]
        confusion[group_index[sample.group], row.repeat, method_index[row.method], int(sample.label), row.pred] += 1
    rng = np.random.default_rng(0)
    deltas = []
    for _ in range(2000):
        c = confusion[rng.integers(0, len(groups), len(groups))].sum(axis=0)
        diagonal = np.diagonal(c, axis1=-2, axis2=-1)
        denominator = c.sum(axis=-2)+c.sum(axis=-1)
        f = np.divide(2*diagonal, denominator, out=np.zeros(diagonal.shape, float), where=denominator != 0).mean(axis=-1).mean(axis=0)
        deltas.append(f-f[0])
    deltas = np.asarray(deltas)
    intervals = [dict(method=m, low=float(np.quantile(deltas[:, i], .025)), high=float(np.quantile(deltas[:, i], .975))) for i, m in enumerate(METHODS)]
    result = dict(specification=specification, scores=scores, conditional_intervals=intervals, input_sha256=hashes)
    pd.DataFrame(scores).to_csv(root / "followup_reproduced_scores.csv", index=False, encoding="utf-8-sig")
    (root / "followup_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if any(M.sha(p) != value for p, value in hashes.items()):
        raise ValueError("Inputs changed during followup")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.data, args.root)
    print(json.dumps(result["conditional_intervals"], ensure_ascii=False))
