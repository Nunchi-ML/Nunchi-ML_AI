"""Budgeted fixed-A B-capture-weight pilot on reused development predictions.

No threshold/subset selection. Primary sqrt weighting and expansion gate are
declared before fitting. All fixed variants and equal-data rule controls remain.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import conditional_augmentation_probe as A
import rule_ml_validation as V
from nunchi_ml.glare import evaluation as E

POWERS = {"class": 0., "sqrt": .5, "inverse": 1.}
NAMES = [f"{pool}_{kind}_{policy}" for pool in ["source", "aug"]
         for kind in ["B", "rule"] for policy in POWERS] + ["source_conditional_G"]
REPEATS = [0, 1, 2]


def capture_weights(frame, power):
    if power not in POWERS.values() or not len(frame) or frame.group.isna().any():
        raise ValueError("Nonempty declared capture weighting required")
    count = frame.groupby("group").id.transform("size").to_numpy(float)
    weights = count ** (-power)
    return weights / weights.mean()


def balanced_capture_weights(positive, power):
    if not len(positive) or not positive.label.isin([1, 2]).all():
        raise ValueError("Only nonempty original positive grades1/2 allowed")
    weights = capture_weights(positive, power)
    labels = positive.label.to_numpy()
    # Equal class mass and total n keep capture weighting distinct from a prior
    # change or a global sample-weight change of LogisticRegression's C.
    for label in np.unique(labels):
        mask = labels == label
        weights[mask] *= len(positive) / (len(np.unique(labels)) * weights[mask].sum())
    return weights


def weighted_B(positive, test, power):
    if set(positive.group) & set(test.group):
        raise ValueError("B capture leakage")
    weights = balanced_capture_weights(positive, power)
    target = positive.label.eq(2).to_numpy(int)
    if len(np.unique(target)) < 2:
        return np.full(len(test), float(target[0])), weights
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000))
    columns = V.M.R.G + V.EXTRA
    # The same unweighted train-only scaler as the reference. Change B loss
    # weights only; no simultaneous normalization intervention.
    model.fit(positive[columns], target, logisticregression__sample_weight=weights)
    return model.predict_proba(test[columns])[:, list(model.classes_).index(1)], weights


def weighted_macro(y, predictions, weights):
    y, predictions, weights = np.asarray(y), np.asarray(predictions), np.asarray(weights, float)
    if (y.ndim != 1 or predictions.ndim != 2 or predictions.shape[1] != len(y)
            or weights.shape != y.shape or not np.isfinite(weights).all() or (weights <= 0).any()):
        raise ValueError("Matched positive finite rule weights required")
    values = []
    for label in range(3):
        tp = (((predictions == label) & (y == label)) * weights).sum(axis=1)
        den = ((predictions == label) * weights).sum(axis=1) + weights[y == label].sum()
        values.append(np.divide(2*tp, den, out=np.zeros(len(predictions)), where=den != 0))
    return np.mean(values, axis=0)


def rule_candidates(pool, test):
    alt, gap, cloud = (pool[c].to_numpy(float) for c in ["sun_alt", "gap", "cloud"])
    grid = [(a, g, c, s) for a in range(15, 50, 5) for g in range(15, 65, 5)
            for c in range(4, 10) for s in range(5, 30, 5) if s <= g]
    candidates = []
    for a, g, c, s in grid:
        on = (alt > 0) & (alt <= a) & (gap <= g) & (cloud <= c)
        candidates.append(np.where(on, np.where(gap <= s, 2, 1), 0))
    candidates = np.asarray(candidates)
    outputs, choices = {}, {}
    for policy, power in POWERS.items():
        weights = capture_weights(pool, power)
        values = weighted_macro(pool.label.to_numpy(), candidates, weights)
        selected = int(np.argmax(values))  # identical declared grid/tie order
        a, g, c, s = grid[selected]
        on = (test.sun_alt > 0) & (test.sun_alt <= a) & (test.gap <= g) & (test.cloud <= c)
        outputs[policy] = np.where(on, np.where(test.gap <= s, 2, 1), 0)
        choices[policy] = dict(alt=a, gap=g, cloud=c, severe_gap=s, weighted_training_macro_f1=float(values[selected]))
    return outputs, choices


def fit_candidates(train, auxiliary, test):
    A.training_pool(train, auxiliary, test)
    a = V.binary(train, test, V.M.R.G+V.M.R.P.REL_R, train.label.ge(1), "conditional_lr")
    outputs, masses, choices = {}, [], {}
    for name, extra in [("source", auxiliary.iloc[:0]), ("aug", auxiliary)]:
        pool = A.training_pool(train, extra, test)
        positive = pool[pool.label.ge(1)]
        rules, choices[name] = rule_candidates(pool, test)
        for policy, power in POWERS.items():
            b, weights = weighted_B(positive, test, power)
            outputs[f"{name}_B_{policy}"] = (V.decide(a, b), a, b)
            outputs[f"{name}_rule_{policy}"] = (rules[policy], None, None)
            mass = positive[["group", "label"]].copy()
            mass["weight"] = weights
            for (group, label), part in mass.groupby(["group", "label"]):
                masses.append(dict(pool=name, policy=policy, group=group, label=int(label),
                                   rows=len(part), weight_mass=float(part.weight.sum())))
    return outputs, masses, choices


def expansion_gate(scores):
    cv = scores[scores.test.eq("group_cv320")]
    if set(cv["repeat"]) != set(REPEATS):
        raise ValueError("All three predeclared pilot repeats required")
    pivot = cv.pivot(index="repeat", columns="method", values="macro_f1")
    if len(pivot) != 3 or pivot.isna().any().any():
        raise ValueError("Complete paired pilot methods required")
    mean = cv.groupby("method").mean(numeric_only=True)
    candidate, baseline = mean.loc["aug_B_sqrt"], mean.loc["aug_B_class"]
    delta = pivot.aug_B_sqrt - pivot.aug_B_class
    checks = dict(mean_f1_gain_ge_005=bool(delta.mean() >= .005),
        worst_repeat_loss_le_01=bool(delta.min() >= -.01),
        mean_severe_recall_loss_le_01=bool(candidate.recall_2 >= baseline.recall_2-.01),
        mean_extreme_errors_not_increased=bool(candidate.count_0_to_2+candidate.count_2_to_0 <= baseline.count_0_to_2+baseline.count_2_to_0+1e-12),
        beats_source_rule_by_005=bool(candidate.macro_f1 >= mean.loc["source_rule_class", "macro_f1"]+.005),
        beats_equal_pool_sqrt_rule_by_005=bool(candidate.macro_f1 >= mean.loc["aug_rule_sqrt", "macro_f1"]+.005))
    stress = scores[~scores.test.eq("group_cv320")]
    guard = []
    for regime in stress.test.unique():
        rows = stress[stress.test.eq(regime)].set_index("method")
        guard.append(bool(rows.loc["aug_B_sqrt", "recall_2"] >= rows.loc["aug_B_class", "recall_2"]-.01))
    checks["all_held_camera_severe_recall_loss_le_01"] = bool(len(guard) == 3 and all(guard))
    return dict(expand_primary=all(checks.values()), checks=checks,
                mean_delta=float(delta.mean()), worst_repeat_delta=float(delta.min()),
                independent_proof=False, inverse_is_diagnostic_not_a_test_selected_replacement=True)


def run(data, out, seconds=180):
    data, out = Path(data).resolve(), Path(out).resolve()
    if out.exists() or out.parent != data:
        raise ValueError("Use a new private pilot output directly under data")
    if seconds <= 0 or seconds > 180:
        raise ValueError("Pilot budget must be positive and at most180seconds")
    prior, base = data/"conditional_sources_20261105", data/"road_visibility_20261104/run30"
    files = [prior/"report.json", prior/"predictions.csv", base/"training_features.csv", base/"splits.csv"]
    sources = [Path(__file__), Path(A.__file__), Path(V.__file__), Path(E.__file__)]
    hashes = {str(p): V.M.sha(p) for p in files+sources}
    previous = json.loads(files[0].read_text(encoding="utf-8"))
    if previous["status"] != "complete" or any(V.M.sha(p) != h for p, h in previous["input_sha256"].items()):
        raise ValueError("Prior frozen model/input/source provenance changed")
    image_hashes = previous["image_sha256"]
    if len(image_hashes) != 624 or any(V.M.sha(p) != h for p, h in image_hashes.items()):
        raise ValueError("Frozen624 image mismatch")
    full = V.geometry(pd.read_csv(files[2], dtype={"id": str}))
    original, auxiliary = full[full.camera.ne("AI Hub")].reset_index(drop=True), full[full.camera.eq("AI Hub")]
    if len(original) != 320 or original.group.nunique() != 24 or len(auxiliary) != 304:
        raise ValueError("Frozen320/24 and training-onlyAIHub304 required")
    A.training_pool(original.iloc[:0], auxiliary, original)
    frozen, splits = pd.read_csv(files[1], dtype={"id": str}), pd.read_csv(files[3], dtype={"id": str})
    out.mkdir(); snapshots = out/"source_snapshot"; snapshots.mkdir()
    for path in sources:
        shutil.copyfile(path, snapshots/path.name)
    spec = dict(status="frozen_before_fit", base="d117679 / PR43 merged", primary="aug_B_sqrt vs aug_B_class and source_rule_class/equal-pool aug_rule_sqrt MacroF1",
        design="Same original320/24/frozen repeats0,1,2;3held-original-cameras. Original human grades unchanged; AIHub304 training-only.",
        fixed_A="Original-training-only balancedLR C1 G+relative12; unchanged for every B variant",
        fixed_B="G+gap/theta/calculatedlogLv; positive original(+AIHub) only; train-only unweighted scaler; LR C1/.5 boundaries",
        weights="Capture-positive n_g^-p, p0/.5/1. Rebalance each class to total n/2, total n; only B loss weights change. Singleclass constant.",
        rules="Same source/aug pools; same exact geometry-rule grid/order. For each p, normalize n_g^-p over all rule training grades; maximize capture-weighted3gradeMacroF1 on training only. No fake probabilities.",
        names=NAMES, budget=dict(repeats=REPEATS, held_cameras=3, outer_splits=18, A_fits=18, maximum_B_fits=108, rule_grid_passes=36, seconds=seconds),
        gate="Primary sqrt only: meanF1 gain>=.005, worst-repeat delta>=-.01, mean severe recall loss<=.01, no increase0<->2errors, >=.005above source unweighted rule and equal-pool sqrt rule, each held-camera severe recall loss<=.01. Any fail => no30repeat expansion.",
        selection="No fitted weight/threshold/subset selection; allfixed variants retained. Inverse diagnostic cannot replace failed primary by outer-test-best picking.",
        limits="Exploratory screening on reused development data. Gate uses development outcomes; any later same-data full run is not independent confirmation. No final selection before incoming17000photo analysis/allremaining tasks. No oracle result used for fitting.",
        input_sha256=hashes, versions={n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scikit-learn"]})
    (out/"specification_before_run.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    start = time.monotonic(); records, scores, mass_rows, rule_choices, manifests = [], [], [], [], []
    regimes = [("group_cv320", r, original) for r in REPEATS]
    regimes += [("held_camera:"+str(c), -1, original[original.camera.eq(c)]) for c in original.camera.unique()]
    reproduced = 0
    for regime, repeat, test in regimes:
        reference = frozen[frozen.test.eq(regime)&frozen["repeat"].eq(repeat)]
        if repeat >= 0:
            part = splits[splits["repeat"].eq(repeat)]
            if len(part) != 320 or not part.id.is_unique or set(part.id) != set(original.id):
                raise ValueError("Frozen split coverage mismatch")
            folds = [(f, original[~original.id.isin(part[part.fold.eq(f)].id)], original[original.id.isin(part[part.fold.eq(f)].id)]) for f in range(5)]
        else:
            folds = [(-1, original[~original.id.isin(test.id)], test)]
        values = {name: dict(pred=np.full(len(test), -1, int), a=np.full(len(test), np.nan), b=np.full(len(test), np.nan)) for name in NAMES}
        fold_ids, seen = np.full(len(test), -1, int), np.zeros(len(test), int)
        for fold, train, valid in folds:
            if time.monotonic()-start > seconds:
                (out/"report.json").write_text(json.dumps(dict(status="stopped_budget", seconds=seconds, completed_regimes=len(scores)//len(NAMES), final_selection=False)), encoding="utf-8")
                raise TimeoutError("Pilot budget reached; preserve partial output, do not expand")
            outputs, masses, choices = fit_candidates(train, auxiliary, valid)
            mask = test.id.isin(valid.id).to_numpy()
            for name, (pred, a, b) in outputs.items():
                values[name]["pred"][mask] = pred
                if a is not None:
                    values[name]["a"][mask], values[name]["b"][mask] = a, b
            fold_ids[mask], seen[mask] = fold, seen[mask]+1
            mass_rows.extend(dict(test=regime, repeat=repeat, fold=fold, **row) for row in masses)
            rule_choices.append(dict(test=regime, repeat=repeat, fold=fold, choices=choices))
            manifests.append(dict(test=regime, repeat=repeat, fold=fold, train_ids=train.id.tolist(), test_ids=valid.id.tolist(), auxiliary_ids=auxiliary.id.tolist()))
        if not np.all(seen == 1):
            raise ValueError("Exactly one original evaluation prediction required")
        for name, old in [("source_B_class", "B_000"), ("aug_B_class", "B_111"), ("source_rule_class", "rule_000"), ("aug_rule_class", "rule_111")]:
            ref = reference[reference.method.eq(old)]
            if not ref.id.is_unique or set(ref.id) != set(test.id):
                raise ValueError("Reference identity mismatch")
            ref = ref.set_index("id").loc[test.id]
            if not np.array_equal(ref.label, test.label) or not np.array_equal(ref.pred, values[name]["pred"]):
                raise ValueError("Unweighted reference reproduction failed")
            if "_B_" in name and not np.allclose(ref.b, values[name]["b"], atol=1e-12, rtol=0):
                raise ValueError("Class-balanced B probability reproduction failed")
            reproduced += len(test)
        ref_a = reference[reference.method.eq("B_111")].set_index("id").loc[test.id].a.to_numpy()
        ref = reference[reference.method.eq("source_conditional_G")].set_index("id").loc[test.id]
        values["source_conditional_G"].update(pred=ref.pred.to_numpy(), a=ref.a.to_numpy(), b=ref.b.to_numpy())
        for name, item in values.items():
            has_probability = "_B_" in name or name == "source_conditional_G"
            a, b = (item["a"], item["b"]) if has_probability else (None, None)
            if has_probability and not np.allclose(a, ref_a, atol=1e-12, rtol=0):
                raise ValueError("Fixed original-only A changed")
            probability = np.column_stack([1-a, a*(1-b), a*b]) if has_probability else None
            scores.append(dict(test=regime, repeat=repeat, method=name, **E.evaluate_grades(test.label, item["pred"], probability, b)))
            records.extend(dict(test=regime, repeat=repeat, method=name, fold=int(fold_ids[i]), id=cid, group=g, label=int(y), pred=int(item["pred"][i]),
                a=float(a[i]) if has_probability else None, b=float(b[i]) if has_probability else None,
                **{f"p{k}":float(probability[i,k]) if has_probability else None for k in range(3)}, policy_source="predeclared_fixed_capture_power_no_test_selection")
                for i, (cid, g, y) in enumerate(zip(test.id, test.group, test.label)))
        pd.DataFrame(scores).to_csv(out/"scores.csv", index=False, encoding="utf-8-sig")
        print("Capture-weight pilot regime complete", len(scores)//len(NAMES), "/6", flush=True)
    frame = pd.DataFrame(scores)
    summary = frame.groupby(["test", "method"], sort=False)[[c for c in frame.select_dtypes("number") if c != "repeat"]].mean().reset_index()
    summary.to_csv(out/"summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(records).to_csv(out/"predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(mass_rows).to_csv(out/"B_capture_class_weight_mass.csv", index=False, encoding="utf-8-sig")
    for filename, obj in [("rule_choices.json", rule_choices), ("split_manifest.json", manifests)]:
        (out/filename).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    gate = expansion_gate(frame)
    (out/"expansion_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    if any(V.M.sha(p) != h for p, h in (hashes | image_hashes).items()):
        raise ValueError("Frozen source/input/image changed during pilot")
    report = dict(status="complete_pilot", seconds=float(time.monotonic()-start), metric_rows=len(scores), prediction_rows=len(records),
        reproduced_predictions=reproduced, expansion=gate, final_selection=False, aihub_test=False, input_sha256=hashes, image_sha256=image_hashes, limits=spec["limits"])
    (out/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out/"report.html").write_text('<!doctype html><meta charset="utf-8"><h1>Capture-weight exploratory pilot</h1><p>Reused development data / three repeats only / no final selection.</p>'+summary.round(5).to_html(index=False), encoding="utf-8")
    print(summary[summary.test.eq("group_cv320")][["method", "macro_f1", "recall_2", "f1_1", "multiclass_log_loss"]].round(6).to_string(index=False), flush=True)
    print(json.dumps(gate, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=180)
    args = parser.parse_args(); run(args.data, args.out, args.seconds)
