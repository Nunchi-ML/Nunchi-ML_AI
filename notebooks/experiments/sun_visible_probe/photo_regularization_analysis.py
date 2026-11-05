"""Posthoc composition intervals for frozen regularization predictions; no fits."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import road_visibility_probe as R

PAIRS = [("fixed_detail", "selected_detail"),
         ("selected_road", "selected_detail"), ("fixed_road", "selected_road")]


def macro_from_counts(counts):
    diagonal = np.diagonal(counts, axis1=-2, axis2=-1)
    denominator = counts.sum(axis=-2) + counts.sum(axis=-1)
    return np.divide(2 * diagonal, denominator, out=np.zeros(diagonal.shape, float),
                     where=denominator != 0).mean(axis=-1)


def paired_interval(frame, baseline, candidate, draws=2000):
    groups = sorted(frame.video.unique())
    if not groups or frame.video.isna().any():
        raise ValueError("Nonmissing video groups required")
    counts = np.zeros((len(groups), 2, 3, 3), int)
    for k, group in enumerate(groups):
        part = frame[frame.video.eq(group)]
        for j, name in enumerate([baseline, candidate]):
            np.add.at(counts[k, j], (part.label.to_numpy(int), part[name].to_numpy(int)), 1)
    rng = np.random.default_rng(0)
    differences = []
    for _ in range(draws):
        f1 = macro_from_counts(counts[rng.integers(0, len(groups), len(groups))].sum(axis=0))
        differences.append(f1[1] - f1[0])
    observed = macro_from_counts(counts.sum(axis=0))
    return dict(baseline=baseline, candidate=candidate, groups=len(groups),
                delta=float(observed[1] - observed[0]),
                low=float(np.quantile(differences, .025)), high=float(np.quantile(differences, .975)))


def run(root):
    root = Path(root).resolve()
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    if report["status"] != "complete" or report["predictions_reproduced"] != 654:
        raise ValueError("Completed run with reproduced baselines required")
    out = root / "analysis"
    if out.exists():
        raise ValueError("Use new analysis directory")
    source_hash = R.digest(root / "predictions.csv")
    out.mkdir()
    specification = dict(status="posthoc_after_evaluation", comparisons=PAIRS, draws=2000, seed=0,
                         limitation="Paired frozen-prediction video composition only; no refitting or independent validation",
                         predictions_sha256=source_hash)
    (out / "specification_before_analysis.json").write_text(json.dumps(specification, indent=2), encoding="utf-8")
    frame = pd.read_csv(root / "predictions.csv")
    rows = [dict(evaluation=regime, **paired_interval(part, a, b))
            for regime, part in frame.groupby("evaluation", sort=False) for a, b in PAIRS]
    pd.DataFrame(rows).to_csv(out / "paired_video_intervals.csv", index=False, encoding="utf-8-sig")
    if R.digest(root / "predictions.csv") != source_hash:
        raise ValueError("Frozen predictions changed")
    (out / "report.json").write_text(json.dumps(dict(specification=specification, intervals=rows), indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).round(5).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
