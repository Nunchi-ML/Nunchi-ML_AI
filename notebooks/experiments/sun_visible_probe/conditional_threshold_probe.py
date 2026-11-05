"""Nested capture-safe thresholds for the fixed augmented geometry-B model.

Original labels/320 evaluation rows/frozen5fold x30; AIHub304 training only.
Probabilities are unchanged across threshold policies, never called calibrated.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import conditional_augmentation_probe as A
import rule_ml_validation as V
from nunchi_ml.glare.evaluation import evaluate_grades
from nunchi_ml.glare import evaluation as E

GRID = [.25, .35, .45, .5, .55, .65, .75]
POLICIES = ["B_threshold_inner", "AB_threshold_inner", "recall_guard_inner"]
RULES = ["source_rule_tuned", "aug_rule_tuned"]
CONTROLS = RULES + ["source_conditional", "B_geometry_aug"]
NAMES = CONTROLS + ["rule_inner_selected"] + POLICIES


def decision_stats(y, pred):
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    severe = y == 2
    return dict(macro_f1=float(V.macro_matrix(y, pred[None, :])[0]),
                severe_recall=float((pred[severe] == 2).mean()) if severe.any() else 0.,
                extreme_errors=int((abs(pred-y) == 2).sum()))


def choose_thresholds(y, a, b, policy):
    if policy not in POLICIES:
        raise ValueError("Unknown threshold policy")
    y = np.asarray(y)
    if y.shape != np.asarray(a).shape or not len(y) or not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Matched original training labels required")
    baseline = decision_stats(y, V.decide(a, b))
    best, chosen = None, None
    for ta in ([.5] if policy == "B_threshold_inner" else GRID):
        for tb in GRID:
            stats = decision_stats(y, V.decide(a, b, ta, tb))
            if policy == "recall_guard_inner" and (
                stats["severe_recall"] < baseline["severe_recall"] - 1e-12
                or stats["extreme_errors"] > baseline["extreme_errors"]
            ):
                continue
            key = (stats["macro_f1"], -stats["extreme_errors"], stats["severe_recall"],
                   -abs(ta-.5)-abs(tb-.5), -tb, -ta)
            if best is None or key > best:
                best = key
                chosen = dict(policy=policy, ta=ta, tb=tb, **stats, baseline=baseline)
    if chosen is None:
        raise ValueError("Fixed .5/.5 must remain a feasible candidate")
    return chosen


def fit_fixed_pair(train, auxiliary, test):
    pool = A.training_pool(train, auxiliary, test)
    a = V.binary(train, test, V.M.R.G+V.M.R.P.REL_R, train.label.ge(1), "conditional_lr")
    positive = pool[pool.label.ge(1)]
    b = V.binary(positive, test, V.M.R.G+V.EXTRA, positive.label.eq(2), "geometry_lr")
    return a, b


def inner_rule_predictions(train, auxiliary, test):
    pool = A.training_pool(train, auxiliary, test)
    return dict(source_rule_tuned=V.rule_predictions(train, test)["rule_tuned"],
                aug_rule_tuned=V.rule_predictions(pool, test)["rule_tuned"])


def predict(train, auxiliary, test):
    oof_a, oof_b = np.full(len(train), np.nan), np.full(len(train), np.nan)
    rule_oof = {name: np.full(len(train), -1, int) for name in RULES}
    inner_fold, seen = np.full(len(train), -1, int), np.zeros(len(train), int)
    for fold, (a, b) in enumerate(V.inner_splits(train)):
        tr, va = train.iloc[a], train.iloc[b]
        if set(tr.group) & set(va.group):
            raise ValueError("Inner capture leakage")
        oof_a[b], oof_b[b] = fit_fixed_pair(tr, auxiliary, va)
        for name, value in inner_rule_predictions(tr, auxiliary, va).items():
            rule_oof[name][b] = value
        inner_fold[b], seen[b] = fold, seen[b]+1
    if not np.all(seen == 1):
        raise ValueError("Every original training row needs exactly one inner OOF")
    choices = [choose_thresholds(train.label.to_numpy(), oof_a, oof_b, policy) for policy in POLICIES]
    chosen_rule = max(RULES, key=lambda k: (decision_stats(train.label, rule_oof[k])["macro_f1"], -RULES.index(k)))
    a, b = fit_fixed_pair(train, auxiliary, test)
    predictions = {c["policy"]: V.decide(a, b, c["ta"], c["tb"]) for c in choices}
    traces = [dict(id=cid, group=g, label=int(y), inner_fold=int(f), a=float(pa), b=float(pb),
                   source_rule=int(ps), aug_rule=int(pt))
              for cid, g, y, f, pa, pb, ps, pt in zip(train.id, train.group, train.label, inner_fold,
                  oof_a, oof_b, rule_oof[RULES[0]], rule_oof[RULES[1]])]
    return predictions, (a, b), dict(policies=choices, chosen_rule=chosen_rule,
                                    severe_original_training_support=int(train.label.eq(2).sum()),
                                    severe_auxiliary_training_support=int(auxiliary.label.eq(2).sum())), traces


def run(data, out):
    data, out = Path(data).resolve(), Path(out).resolve()
    if out.exists() or out.parent != data:
        raise ValueError("Use a new private output directly under data")
    prior, base = data/"conditional_augmentation_20261105", data/"road_visibility_20261104/run30"
    files = [prior/"report.json", prior/"predictions.csv", prior/"binary_probabilities.csv",
             base/"training_features.csv", base/"splits.csv", Path(__file__), Path(A.__file__),
             Path(V.__file__), Path(V.M.__file__), Path(V.Q.__file__), Path(E.__file__)]
    hashes = {str(p): V.M.sha(p) for p in files}
    previous = json.loads(files[0].read_text(encoding="utf-8"))
    if previous["status"] != "complete" or any(V.M.sha(p) != h for p, h in previous["input_sha256"].items()):
        raise ValueError("Reference model/input provenance changed")
    full = V.geometry(pd.read_csv(files[3], dtype={"id": str}))
    train = full[full.camera.ne("AI Hub")].reset_index(drop=True)
    aux = full[full.camera.eq("AI Hub")].reset_index(drop=True)
    if (len(train) != 320 or train.group.nunique() != 24 or len(aux) != 304
            or aux.group.nunique() != 3 or not full.id.is_unique):
        raise ValueError("Frozen320/24 and auxiliary304/3 required")
    A.training_pool(train.iloc[:0], aux, train)
    image_hashes = {}
    for path, expected in zip(V.M.R.C.image_paths(full, data), full.image_sha256):
        if V.M.sha(path) != expected:
            raise ValueError("Original image bytes changed")
        image_hashes[str(path)] = expected
    frozen = pd.read_csv(files[1], dtype={"id": str})
    raw = pd.read_csv(files[2], dtype={"id": str})
    splits = pd.read_csv(files[4], dtype={"id": str})
    out.mkdir()
    specification = dict(status="frozen_before_fit", primary="B_threshold_inner vs fixed B_geometry_aug and both tuned rule controls MacroF1",
        names=NAMES, original="Same320photos/24captures/frozen5folds x30; original human0/1/2 unchanged",
        model="Fixed original-only balanced LR C1 A(G+relative12); augmented positive-only LR C1 B(G+gap/theta/logLv); train-only scaling",
        auxiliary="Same AIHub304 always training-only, also supplied to augmented tuned rule; never an AIHub test",
        inner="3 capture folds shuffle seed0 on outer-original-training; inner holdouts excluded from all fits/scalers; same auxiliary pool available throughout",
        threshold_grid=GRID, policies=dict(B_threshold_inner="Fix A=.5, select B only on inner original OOF MacroF1",
            AB_threshold_inner="Choose both thresholds on same inner OOF",
            recall_guard_inner="Max inner MacroF1 subject to severe recall >= inner .5/.5 AND extreme errors <= inner .5/.5"),
        tie="MacroF1,fewer extreme errors,higher severe recall,near.5,lower B/lower A; same declared grid for all policies",
        rule_selection="Choose source-tuned or augmented-tuned rule by the same inner original OOF MacroF1; tie source rule; outer rule predictions reused after provenance checks",
        followups="All fixed policies on3heldoriginalcameras; no threshold changes after stress outcomes",
        budget="2LR fits x(3inner+1outer)x150 +3camera stress regimes; no Optuna/calibration/label changes",
        limits="Sequential development experiment on reused data, not independent external proof. Recall guard is inner-only, no outer guarantee. All candidates retained; no test-best selection.",
        input_sha256=hashes, versions={p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn"]})
    (out/"specification_before_run.json").write_text(json.dumps(specification, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(__file__, out/"source_before_run.py")
    shutil.copyfile(E.__file__, out/"evaluation_before_run.py")
    records, scores, selections, inner, manifests = [], [], [], [], []
    regimes = [("group_cv320", r, train) for r in range(30)]
    regimes += [("held_camera:"+str(c), -1, train[train.camera.eq(c)]) for c in train.camera.unique()]
    for regime, repeat, test in regimes:
        predictions = {name: np.full(len(test), -1, int) for name in NAMES}
        for name in CONTROLS:
            old = frozen[frozen.test.eq(regime) & frozen["repeat"].eq(repeat) & frozen.method.eq(name)]
            if not old.id.is_unique or set(old.id) != set(test.id):
                raise ValueError("Reference coverage mismatch")
            old = old.set_index("id").loc[test.id]
            if not np.array_equal(old.label.to_numpy(), test.label.to_numpy()):
                raise ValueError("Original labels changed")
            predictions[name] = old.pred.to_numpy()
        source = raw[raw.test.eq(regime) & raw["repeat"].eq(repeat)].set_index("id")
        if not source.index.is_unique or set(source.index) != set(test.id):
            raise ValueError("Reference probability coverage mismatch")
        source = source.loc[test.id]
        probabilities = np.full((len(test), 2), np.nan)
        seen, outer_fold = np.zeros(len(test), int), np.full(len(test), -1, int)
        if repeat >= 0:
            part = splits[splits["repeat"].eq(repeat)]
            if len(part) != 320 or not part.id.is_unique or set(part.id) != set(train.id):
                raise ValueError("Frozen split coverage mismatch")
            folds = [(f, train[~train.id.isin(part[part.fold.eq(f)].id)], train[train.id.isin(part[part.fold.eq(f)].id)]) for f in range(5)]
        else:
            folds = [(-1, train[~train.id.isin(test.id)], test)]
        for fold, a, b in folds:
            if set(a.group) & set(b.group):
                raise ValueError("Outer capture leakage")
            pred, pair, choices, trace = predict(a, aux, b)
            mask = test.id.isin(b.id).to_numpy()
            fixed = V.decide(*pair)
            if (not np.array_equal(fixed, predictions["B_geometry_aug"][mask])
                    or not np.allclose(pair[0], source.loc[b.id, "source_a"], atol=1e-12, rtol=0)
                    or not np.allclose(pair[1], source.loc[b.id, "geometry_b"], atol=1e-12, rtol=0)):
                raise ValueError("Fixed model probability/prediction reproduction failed")
            for name, value in pred.items():
                predictions[name][mask] = value
            predictions["rule_inner_selected"][mask] = predictions[choices["chosen_rule"]][mask]
            probabilities[mask] = np.column_stack(pair)
            outer_fold[mask], seen[mask] = fold, seen[mask]+1
            selections.append(dict(test=regime, repeat=repeat, fold=fold, **choices))
            inner.extend(dict(test=regime, repeat=repeat, fold=fold, **r) for r in trace)
            manifests.append(dict(test=regime, repeat=repeat, fold=fold, train_ids=a.id.tolist(),
                                  test_ids=b.id.tolist(), auxiliary_ids=aux.id.tolist()))
        if not np.all(seen == 1):
            raise ValueError("Each evaluation row needs exactly one prediction")
        pa, pb = probabilities.T
        for name, values in predictions.items():
            if name in POLICIES+ ["B_geometry_aug"]:
                a, b = pa, pb
            elif name == "source_conditional":
                a, b = source.source_a.to_numpy(), source.source_b.to_numpy()
            else:
                a = b = None
            p = np.column_stack([1-a, a*(1-b), a*b]) if a is not None else None
            result = evaluate_grades(test.label.to_numpy(), values, p, b)
            scores.append(dict(test=regime, repeat=repeat, method=name, **result))
            for i, (cid, group, y, value) in enumerate(zip(test.id, test.group, test.label, values)):
                records.append(dict(test=regime, repeat=repeat, method=name, id=cid, group=group,
                    label=int(y), pred=int(value), fold=int(outer_fold[i]),
                    a=float(a[i]) if a is not None else None, b=float(b[i]) if b is not None else None,
                    **{f"p{c}":float(p[i,c]) if p is not None else None for c in range(3)}))
        pd.DataFrame(scores).to_csv(out/"scores.csv", index=False, encoding="utf-8-sig")
        print("Nested fixed-model thresholds", regime, repeat+1, flush=True)
    frame = pd.DataFrame(scores)
    summary = frame.groupby(["test", "method"], sort=False)[[c for c in frame.select_dtypes("number") if c != "repeat"]].mean().reset_index()
    summary.to_csv(out/"summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(records).to_csv(out/"predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(inner).to_csv(out/"inner_oof.csv", index=False, encoding="utf-8-sig")
    (out/"inner_selections.json").write_text(json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf-8")
    (out/"split_manifest.json").write_text(json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8")
    names = NAMES; groups = sorted(train.group.unique()); group_index = {g:i for i,g in enumerate(groups)}
    counts = np.zeros((24,30,len(names),3,3),int)
    for r in pd.DataFrame(records).query("test == 'group_cv320'").itertuples():
        counts[group_index[r.group],r.repeat,names.index(r.method),r.label,r.pred] += 1
    rng = np.random.default_rng(0); draws = []
    for _ in range(2000):
        c = counts[rng.integers(0,24,24)].sum(axis=0)
        diag, den = np.diagonal(c,axis1=-2,axis2=-1), c.sum(axis=-2)+c.sum(axis=-1)
        draws.append(np.divide(2*diag,den,out=np.zeros(diag.shape,float),where=den!=0).mean(axis=-1).mean(axis=0))
    draws = np.asarray(draws)
    intervals = [dict(method=k,baseline=base,low=float(np.quantile(draws[:,i]-draws[:,names.index(base)],.025)),
                      high=float(np.quantile(draws[:,i]-draws[:,names.index(base)],.975)))
                 for base in ["B_geometry_aug", *RULES, "rule_inner_selected"] for i,k in enumerate(names)]
    pd.DataFrame(intervals).to_csv(out/"paired_capture_intervals.csv",index=False,encoding="utf-8-sig")
    if any(V.M.sha(p) != h for p,h in (hashes|image_hashes).items()):
        raise ValueError("Frozen source/input/image changed during run")
    result = dict(status="complete",fixed_model_reproduced=9920,inner_original_rows=len(inner),
        limits=specification["limits"],interval_scope="posthoc fixed-OOF capture composition only, no refitting",input_sha256=hashes,image_sha256=image_hashes)
    (out/"report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"report.html").write_text('<!doctype html><meta charset="utf-8"><title>Nested fixed-model thresholds</title><h1>Original320 / fixed model / inner-group thresholds</h1><p>Same probabilities across threshold policies. Reused development data. All policies retained, no final model selection.</p>'+summary.round(5).to_html(index=False),encoding="utf-8")
    print(summary[["test","method","macro_f1","recall_2","macro_ovr_auc"]].round(5).to_string(index=False),flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("data")); parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args(); run(args.data,args.out)
