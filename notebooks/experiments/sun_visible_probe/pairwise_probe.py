"""Nested group thresholding for fixed-grade pairwise XGBoost comparisons.

Original320/24captures/frozen5folds x30. No independent pair annotations,
human probability target, deep learning, deployment or test-tuned thresholds.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning

import pairwise_glare as P
import ordinal_compare as O
import conditional_training_probe as M
import road_visibility_probe as R

NAMES = ["legacy", "conditional", "xgb_balanced_argmax", "xgb_unweighted_argmax", "xgb_unweighted_cuts", "rank_global", "rank_capture"]


def metrics(y, pred):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        return O.metrics(np.asarray(y), np.asarray(pred))


def predict(train, test, columns):
    prediction, score, diagnostics, inner = {}, {}, [], []
    probability = O.task_probabilities("xgboost", train[columns].to_numpy(), train.label.to_numpy(), test[columns].to_numpy(), True, 3)
    prediction["xgb_balanced_argmax"] = probability.argmax(axis=1)
    for mode in ["multiclass", *P.MODES]:
        pred, rank_score, info, oof = P.nested_rank(train, test, columns, mode)
        name = "xgb_unweighted_cuts" if mode == "multiclass" else "rank_" + mode
        if mode == "multiclass":
            prediction["xgb_unweighted_argmax"] = info.pop("argmax_predictions")
        prediction[name], score[name] = pred, rank_score
        diagnostics.append(dict(method=name, **info))
        inner.extend(dict(method=name, id=cid, score=float(s)) for cid, s in zip(train.id, oof))
    return prediction, score, diagnostics, inner


def run(data, out):
    data, out = Path(data).resolve(), Path(out).resolve()
    if out.exists() or not out.is_relative_to(data):
        raise ValueError("Use a new private output")
    base, prior = data / "road_visibility_20261104/run30", data / "conditional_training_20261105"
    files = [base / "training_features.csv", base / "splits.csv", prior / "predictions.csv", prior / "report.json",
        Path(__file__), Path(P.__file__), Path(O.__file__), Path(M.__file__), Path(R.__file__)]
    hashes = {str(p): M.sha(p) for p in files}
    previous = json.loads(files[3].read_text(encoding="utf-8"))
    if previous["status"] != "complete" or any(M.sha(p) != h for p, h in previous["hashes"].items()):
        raise ValueError("Frozen baseline inputs/source changed")
    out.mkdir()
    specification = dict(status="frozen_before_fit", methods=NAMES, features=R.G+R.P.REL_R,
        targets="Original0/1/2; label-derived pairs only, not new human pair judgments",
        outer="Exact saved320/24capture/5folds x30, no reselection", inner="3group folds/shuffle seed0 in outer-training only",
        ranker="rank:pairwise/200trees/depth3/lr.05/lambda1/min_child1/hist/seed0/n_jobs1/mean8pairsPerPhoto",
        controls="SameXGB200/depth3/.05: balanced argmax,unweighted argmax,and unweighted expected-grade score with same nested cuts",
        scoring="Model-specific training empirical percentile,ties midpoint,not probability",
        cut_points="Fixed .05 grid plus endpoints,ordered cuts; innerOOF MacroF1,severe recall,MAE,then near1/3,2/3 tie rule",
        budget="13XGB fits per outer fold x150 plus4followup regimes; no Optuna/threshold selection on outer test",
        baseline="All9600 x2 frozen predictions retained; representative seed0 legacy/conditional320 x2 predictions refitted exactly",
        limitations="Repeated development CV,dependent repeats; pair labels reuse old grades; rank score is not population probability",
        versions={n: importlib.metadata.version(n) for n in ["xgboost", "numpy", "pandas", "scikit-learn"]}, hashes=hashes)
    (out / "specification_before_run.json").write_text(json.dumps(specification, indent=2), encoding="utf-8")
    shutil.copyfile(__file__, out / "runner_before_run.py")
    shutil.copyfile(P.__file__, out / "ranking_before_run.py")
    full = pd.read_csv(files[0], dtype={"id": str})
    train = full[full.camera.ne("AI Hub")].reset_index(drop=True)
    if len(train) != 320 or train.group.nunique() != 24 or not train.id.is_unique or full.image_sha256.duplicated().any():
        raise ValueError("Frozen320/24capture data required")
    columns = R.G+R.P.REL_R
    if len(columns) != 18 or not np.isfinite(full[columns].to_numpy(float)).all():
        raise ValueError("Frozen18 features required")
    image_hashes = {}
    for path, expected in zip(R.C.image_paths(full, data), full.image_sha256):
        if M.sha(path) != expected:
            raise ValueError("Original image bytes changed")
        image_hashes[str(path)] = expected
    splits = pd.read_csv(files[1], dtype={"id": str})
    reference = pd.read_csv(files[2], dtype={"id": str})
    lookup = {}
    for old, name in [("legacy", "legacy"), ("binary_A_positive_B", "conditional")]:
        part = reference[reference.method.eq(old)]
        if len(part) != 9600 or part.duplicated(["repeat", "id"]).any():
            raise ValueError("Frozen reference coverage mismatch")
        lookup[name] = part.set_index(["repeat", "id"]).pred
    records, scores, thresholds, inner = [], [], [], []
    for repeat in range(30):
        part = splits[splits["repeat"].eq(repeat)]
        if len(part) != 320 or not part.id.is_unique or set(part.id) != set(train.id):
            raise ValueError("Frozen split coverage mismatch")
        predictions = {name: np.full(320, -1, int) for name in NAMES}
        rank_scores = {name: np.zeros(320) for name in ["xgb_unweighted_cuts", "rank_global", "rank_capture"]}
        for name in lookup:
            predictions[name] = lookup[name].loc[repeat].loc[train.id].to_numpy()
        seen = np.zeros(320, int)
        for fold in range(5):
            mask = train.id.isin(part[part.fold.eq(fold)].id).to_numpy()
            a, b = train[~mask], train[mask]
            if not len(b) or set(a.group) & set(b.group):
                raise ValueError("Outer capture leakage")
            if repeat == 0:
                old = M.fold_predictions(a, b)
                for name, key in [("legacy", "legacy"), ("conditional", "binary_A_positive_B")]:
                    if not np.array_equal(R.C.decision(old[key]), predictions[name][mask]):
                        raise ValueError("Representative640 baseline prediction reproduction failed")
            pred, raw_score, info, oof = predict(a, b, columns)
            for name, value in pred.items():
                predictions[name][mask] = value
            for name, value in raw_score.items():
                rank_scores[name][mask] = value
            thresholds.extend(dict(repeat=repeat, fold=fold, **row) for row in info)
            inner.extend(dict(repeat=repeat, fold=fold, **row) for row in oof)
            seen[mask] += 1
        if not np.all(seen == 1):
            raise ValueError("OOF coverage mismatch")
        for name, pred in predictions.items():
            if not np.isin(pred, [0, 1, 2]).all():
                raise ValueError("Invalid grade prediction")
            aux = P.pair_accuracy(train.label, rank_scores[name]) if name in rank_scores else None
            scores.append(dict(test="group_cv320", repeat=repeat, method=name, pair_accuracy=aux, **metrics(train.label, pred)))
            records.extend(dict(test="group_cv320", repeat=repeat, id=cid, group=group, method=name, label=int(y), pred=int(p))
                           for cid, group, y, p in zip(train.id, train.group, train.label, pred))
        pd.DataFrame(scores).to_csv(out / "scores.csv", index=False, encoding="utf-8-sig")
        print("Pairwise outer repeat", repeat+1, "/30", flush=True)
    # Prespecified followups; no method/threshold retuning from their outcomes.
    regimes = [("reused_AIHub", train, full[full.camera.eq("AI Hub")])]
    regimes += [("held_camera:"+str(camera), train[train.camera.ne(camera)], train[train.camera.eq(camera)]) for camera in train.camera.unique()]
    for test_name, a, b in regimes:
        if set(a.group) & set(b.group) or set(a.image_sha256) & set(b.image_sha256):
            raise ValueError("Followup capture/image overlap")
        pred, raw_score, info, oof = predict(a, b, columns)
        old = M.fold_predictions(a, b)
        pred.update(legacy=R.C.decision(old["legacy"]), conditional=R.C.decision(old["binary_A_positive_B"]))
        for name, value in pred.items():
            scores.append(dict(test=test_name, repeat=-1, method=name, **metrics(b.label, value)))
            records.extend(dict(test=test_name, repeat=-1, id=cid, group=group, method=name, label=int(y), pred=int(p))
                           for cid, group, y, p in zip(b.id, b.group, b.label, value))
        thresholds.extend(dict(repeat=-1, fold=test_name, **row) for row in info)
        inner.extend(dict(repeat=-1, fold=test_name, **row) for row in oof)
        print("Pairwise followup complete", test_name, flush=True)
    predictions = pd.DataFrame(records)
    predictions.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(inner).to_csv(out / "inner_oof_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(thresholds).to_csv(out / "inner_selected_thresholds.csv", index=False, encoding="utf-8-sig")
    frame = pd.DataFrame(scores)
    frame.to_csv(out / "scores.csv", index=False, encoding="utf-8-sig")
    numeric = [c for c in frame.select_dtypes("number") if c != "repeat"]
    summary = frame.groupby(["test", "method"], sort=False)[numeric].mean().reset_index()
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    # Posthoc paired capture composition intervals, preserving all30 dependent repeats.
    units = sorted(train.group.unique())
    counts = np.zeros((24, 30, len(NAMES), 3, 3), int)
    index = {g: i for i, g in enumerate(units)}
    for row in predictions[predictions.test.eq("group_cv320")].itertuples():
        counts[index[row.group], row.repeat, NAMES.index(row.method), row.label, row.pred] += 1
    draws = []
    rng = np.random.default_rng(0)
    for _ in range(2000):
        c = counts[rng.integers(0,24,24)].sum(axis=0)
        diag = np.diagonal(c, axis1=-2, axis2=-1)
        denom = c.sum(axis=-2)+c.sum(axis=-1)
        f1 = np.divide(2*diag, denom, out=np.zeros(diag.shape, float), where=denom != 0).mean(axis=-1).mean(axis=0)
        draws.append(f1-f1[NAMES.index("conditional")])
    intervals = [dict(method=name, baseline="conditional", low=float(np.quantile(np.asarray(draws)[:, i], .025)),
                     high=float(np.quantile(np.asarray(draws)[:, i], .975))) for i, name in enumerate(NAMES)]
    pd.DataFrame(intervals).to_csv(out / "paired_capture_intervals.csv", index=False, encoding="utf-8-sig")
    if any(M.sha(p) != h for p, h in (hashes|image_hashes).items()):
        raise ValueError("Source/data/images changed during run")
    result = dict(status="complete", representative_baseline_refit_matches=640, frozen_baseline_predictions=19200,
        original_rows=320, groups=24, repeats=30, inner_folds=3, intervals=intervals,
        limitations=specification["limitations"], bootstrap="posthoc2000 seed0,frozenOOF composition only,no refitting",
        hashes=hashes, image_hashes=image_hashes)
    (out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "report.html").write_text('<!doctype html><meta charset="utf-8"><title>Fixed-grade pairwise ranking</title><style>body{font:16px system-ui;margin:24px}td,th{padding:5px;border:1px solid #ccc}</style><h1>Nested capture-safe ranking vs classification</h1><p>Same original320/frozen5folds x30. Labels unchanged. CDF scores are not probabilities. Inner threshold selection only; repeated development data.</p>'+summary.round(4).to_html(index=False)+'<h2>Posthoc capture intervals</h2>'+pd.DataFrame(intervals).round(4).to_html(index=False), encoding="utf-8")
    print(summary[["test", "method", "macro_f1", "f1_1", "recall_2"]].round(5).to_string(index=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.out)
