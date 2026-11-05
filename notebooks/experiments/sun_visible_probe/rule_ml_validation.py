"""Same-grade, same-capture nested F1 comparison against strong tuned rules.

No test-driven thresholds, new labels, neural features, or deployment changes.
All fixed candidates and four stress conditions are retained, including losses.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import conditional_training_probe as M
import ordinal_compare as O
import rule_vs_ml as Q

KINDS = ["conditional_lr", "geometry_lr", "geometry_cat"]
RULES = ["rule_hand", "rule_tuned", "rule_physics"]
NAMES = RULES + ["rule_inner_selected", "legacy", *KINDS, "ml_inner_selected"]
THRESHOLDS = [.25, .35, .45, .5, .55, .65, .75]
EXTRA = ["gap", "theta", "lv"]


def metric(y, p):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        return O.metrics(np.asarray(y, int), np.asarray(p, int))


def decide(a, b, ta=.5, tb=.5):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 1 or a.shape != b.shape or not np.isfinite([a, b]).all() or ((a < 0) | (a > 1) | (b < 0) | (b > 1)).any():
        raise ValueError("Paired finite binary probabilities required")
    if not (0 <= ta <= 1 and 0 <= tb <= 1):
        raise ValueError("Valid thresholds required")
    return np.where(a >= ta, np.where(b >= tb, 2, 1), 0)


def macro_matrix(y, predictions):
    y, predictions = np.asarray(y, int), np.asarray(predictions, int)
    values = []
    for c in range(3):
        tp = ((predictions == c) & (y == c)).sum(axis=1)
        denominator = (predictions == c).sum(axis=1) + (y == c).sum()
        values.append(np.divide(2*tp, denominator, out=np.zeros(len(predictions)), where=denominator != 0))
    return np.asarray(values).mean(axis=0)


def rule_predictions(train, test):
    """Vectorized exact grids/order/tie handling of existing rule_vs_ml.py."""
    alt, gap, cloud = (train[c].to_numpy(float) for c in ("sun_alt", "gap", "cloud"))
    params = [(a, g, c, s) for a in range(15, 50, 5) for g in range(15, 65, 5)
              for c in range(4, 10) for s in range(5, 30, 5) if s <= g]
    values = []
    for a, g, c, s in params:
        on = (alt > 0) & (alt <= a) & (gap <= g) & (cloud <= c)
        values.append(np.where(on, np.where(gap <= s, 2, 1), 0))
    chosen = params[int(np.argmax(macro_matrix(train.label, np.asarray(values))))]
    a, g, c, s = chosen
    on = (test.sun_alt > 0) & (test.sun_alt <= a) & (test.gap <= g) & (test.cloud <= c)
    tuned = np.where(on, np.where(test.gap <= s, 2, 1), 0)
    qs = np.unique(np.quantile(train.lv, np.linspace(.05, .95, 37)))
    cuts = [(lo, hi) for i, lo in enumerate(qs) for hi in qs[i:]]
    predictions = np.asarray([np.where(train.lv >= hi, 2, np.where(train.lv >= lo, 1, 0)) for lo, hi in cuts])
    lo, hi = cuts[int(np.argmax(macro_matrix(train.label, predictions)))]
    physics = np.where(test.lv >= hi, 2, np.where(test.lv >= lo, 1, 0))
    return dict(rule_hand=Q.rule_hand(train, test).argmax(axis=1), rule_tuned=tuned, rule_physics=physics)


def geometry(frame):
    out = frame.copy()
    alt = np.radians(out.sun_alt.to_numpy(float))
    out["theta"] = np.degrees(np.arccos(np.clip(np.cos(alt)*np.cos(np.radians(out.gap)), -1, 1)))
    out["lv"] = Q.lv_score(out)
    return out


def binary(train, test, columns, target, kind):
    target = np.asarray(target, int)
    if len(target) != len(train) or not np.isin(target, [0, 1]).all():
        raise ValueError("Matched binary target required")
    if len(np.unique(target)) < 2:
        return np.full(len(test), float(target[0]) if len(target) else 0.)
    if kind == "geometry_cat":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(iterations=200, depth=3, learning_rate=.05, l2_leaf_reg=5,
                                   auto_class_weights="Balanced", random_seed=0, thread_count=1,
                                   verbose=False, allow_writing_files=False)
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight="balanced", max_iter=3000))
    model.fit(train[columns], target)
    return model.predict_proba(test[columns])[:, list(model.classes_).index(1)]


def fit_pair(train, test, kind):
    if kind not in KINDS:
        raise ValueError("Unknown fixed candidate")
    extra = [] if kind == "conditional_lr" else EXTRA
    a = binary(train, test, M.R.G+M.R.P.REL_R+extra, train.label.ge(1), kind)
    positive = train[train.label.ge(1)]
    b = binary(positive, test, M.R.G+extra, positive.label.eq(2), kind)
    return a, b


def inner_splits(train):
    groups = train.group.to_numpy()
    if len(np.unique(groups)) < 2:
        raise ValueError("At least two inner capture groups required")
    return list(GroupKFold(min(3, len(np.unique(groups))), shuffle=True, random_state=0).split(train, groups=groups))


def choose(y, oof):
    """Optimize one predeclared procedure using only outer-training OOF."""
    best, selection, diagnostics = None, None, []
    for order, kind in enumerate(KINDS):
        a, b = oof[kind]
        best_candidate = None
        for ta in THRESHOLDS:
            for tb in THRESHOLDS:
                p = decide(a, b, ta, tb)
                f1 = float(macro_matrix(y, p[None, :])[0])
                errors = int((((np.asarray(y) == 0) & (p == 2)) | ((np.asarray(y) == 2) & (p == 0))).sum())
                key = (f1, -errors, -abs(ta-.5)-abs(tb-.5), -order)
                row = dict(kind=kind, ta=ta, tb=tb, inner_macro_f1=f1, extreme_errors=errors)
                if best is None or key > best:
                    best, selection = key, row
                if best_candidate is None or key > best_candidate[0]:
                    best_candidate = key, row
        diagnostics.append(best_candidate[1])
    return selection, diagnostics


def predict(train, test):
    oof = {k: (np.full(len(train), np.nan), np.full(len(train), np.nan)) for k in KINDS}
    rule_oof = {k: np.full(len(train), -1, int) for k in RULES}
    inner_fold = np.full(len(train), -1, int)
    for fold, (a, b) in enumerate(inner_splits(train)):
        tr, va = train.iloc[a], train.iloc[b]
        if set(tr.group) & set(va.group):
            raise ValueError("Inner capture leakage")
        for kind in KINDS:
            p, q = fit_pair(tr, va, kind)
            oof[kind][0][b], oof[kind][1][b] = p, q
        for name, p in rule_predictions(tr, va).items():
            rule_oof[name][b] = p
        inner_fold[b] = fold
    selected, diagnostics = choose(train.label.to_numpy(), oof)
    chosen_rule = max(RULES, key=lambda k: (macro_matrix(train.label, rule_oof[k][None, :])[0], -RULES.index(k)))
    predictions = rule_predictions(train, test)
    predictions["rule_inner_selected"] = predictions[chosen_rule].copy()
    for kind in KINDS:
        a, b = fit_pair(train, test, kind)
        predictions[kind] = decide(a, b)
        if kind == selected["kind"]:
            predictions["ml_inner_selected"] = decide(a, b, selected["ta"], selected["tb"])
    traces = [dict(id=cid, inner_fold=int(fold), kind=kind, a=float(a), b=float(b))
              for kind, pair in oof.items() for cid, fold, a, b in zip(train.id, inner_fold, *pair)]
    return predictions, dict(**selected, selected_rule=chosen_rule, candidate_best=diagnostics), traces


def run(data, out):
    data, out = Path(data).resolve(), Path(out).resolve()
    if out.exists() or out.parent != data:
        raise ValueError("Use a new output directly under private data")
    base, prior = data / "road_visibility_20261104/run30", data / "conditional_training_20261105"
    files = [base/"training_features.csv", base/"splits.csv", prior/"predictions.csv", prior/"report.json",
             Path(__file__), Path(M.__file__), Path(Q.__file__), Path(O.__file__)]
    hashes = {str(p): M.sha(p) for p in files}
    previous = json.loads(files[3].read_text(encoding="utf-8"))
    if previous["status"] != "complete" or any(M.sha(p) != h for p, h in previous["hashes"].items()):
        raise ValueError("Original baseline provenance changed")
    full = geometry(pd.read_csv(files[0], dtype={"id": str}))
    train = full[full.camera.ne("AI Hub")].reset_index(drop=True)
    if len(train) != 320 or train.group.nunique() != 24 or not full.id.is_unique or full.image_sha256.duplicated().any():
        raise ValueError("Original320/24capture required")
    if not np.isfinite(full[M.R.G+M.R.P.REL_R+EXTRA].to_numpy(float)).all():
        raise ValueError("Finite features required, no row deletion")
    image_hashes = {}
    for path, expected in zip(M.R.C.image_paths(full, data), full.image_sha256):
        if M.sha(path) != expected:
            raise ValueError("Original image bytes changed")
        image_hashes[str(path)] = expected
    out.mkdir()
    specification = dict(status="frozen_before_fit", names=NAMES, primary="ml_inner_selected MacroF1 vs rule_tuned and rule_inner_selected",
        original="Same320/24capture/frozen5folds x30; same original0/1/2 labels; all methods retained",
        features=dict(base_A=M.R.G+M.R.P.REL_R, base_B=M.R.G, geometry_extra=EXTRA),
        candidates="Balanced conditional LR C1; same LR+geometry; CatBoost conditional200/depth3/.05/l2=5/balanced/thread1",
        inner="3group folds shuffle seed0 in outer-training; same folds for rule/model selection",
        selection=dict(thresholds=THRESHOLDS, tie="MacroF1,fewer0<->2errors,near.5 thresholds,declaredcandidateorder"),
        rules="Existing fixed rule A, exact train-fit grid B, train-quantile physics C; also innerOOF rule selection",
        followups="All methods: held3originalcameras and reusedAIHub304; no selection/retuning from stress outcomes",
        budget="3candidates x(3inner+1outer) x2binary fits per outerfold x150 plus4followups; constants skip fit; no Optuna",
        limits="Repeated development CV and reused stress data; no independent external proof. FixedOOF cluster interval is composition uncertainty only.",
        input_sha256=hashes, versions={n: importlib.metadata.version(n) for n in ["catboost", "scikit-learn", "numpy", "pandas"]})
    (out/"specification_before_run.json").write_text(json.dumps(specification, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(__file__, out/"source_before_run.py")
    splits = pd.read_csv(files[1], dtype={"id": str})
    frozen = pd.read_csv(files[2], dtype={"id": str})
    references = {name: frozen[frozen.method.eq(old)].set_index(["repeat", "id"]).pred
                  for name, old in [("legacy", "legacy"), ("conditional_lr", "binary_A_positive_B")]}
    if any(len(v) != 9600 or not v.index.is_unique for v in references.values()):
        raise ValueError("Baseline prediction coverage changed")
    records, scores, selections, traces = [], [], [], []
    for repeat in range(30):
        part = splits[splits["repeat"].eq(repeat)]
        if len(part) != 320 or not part.id.is_unique or set(part.id) != set(train.id):
            raise ValueError("Frozen split coverage mismatch")
        predictions = {k: np.full(320, -1, int) for k in NAMES}
        predictions["legacy"] = references["legacy"].loc[repeat].loc[train.id].to_numpy()
        seen = np.zeros(320, int)
        for fold in range(5):
            mask = train.id.isin(part[part.fold.eq(fold)].id).to_numpy()
            a, b = train[~mask], train[mask]
            if not len(b) or set(a.group) & set(b.group):
                raise ValueError("Outer capture leakage")
            pred, choice, trace = predict(a, b)
            for name, values in pred.items():
                predictions[name][mask] = values
            selections.append(dict(test="group_cv320", repeat=repeat, fold=fold, **choice))
            traces.extend(dict(test="group_cv320", repeat=repeat, fold=fold, **r) for r in trace)
            seen[mask] += 1
            if repeat == 0:
                for name, actual in rule_predictions(a, b).items():
                    fn = dict(rule_hand=Q.rule_hand, rule_tuned=Q.rule_tuned, rule_physics=Q.rule_physics)[name]
                    if not np.array_equal(actual, fn(a, b).argmax(axis=1)):
                        raise ValueError("Existing rule implementation reproduction failed")
        if not np.all(seen == 1) or not np.array_equal(predictions["conditional_lr"], references["conditional_lr"].loc[repeat].loc[train.id].to_numpy()):
            raise ValueError("OOF coverage or conditional baseline reproduction failed")
        for name, values in predictions.items():
            scores.append(dict(test="group_cv320", repeat=repeat, method=name, **metric(train.label, values)))
            records.extend(dict(test="group_cv320", repeat=repeat, method=name, id=cid, group=g, label=int(y), pred=int(p))
                           for cid, g, y, p in zip(train.id, train.group, train.label, values))
        pd.DataFrame(scores).to_csv(out/"scores.csv", index=False, encoding="utf-8-sig")
        print("Rule vs ML nested F1 repeat", repeat+1, "/30", flush=True)
    regimes = [("reused_AIHub", train, full[full.camera.eq("AI Hub")])]
    regimes += [("held_camera:"+str(c), train[train.camera.ne(c)], train[train.camera.eq(c)]) for c in train.camera.unique()]
    for regime, a, b in regimes:
        if set(a.group) & set(b.group) or set(a.image_sha256) & set(b.image_sha256):
            raise ValueError("Stress capture/image overlap")
        pred, choice, trace = predict(a, b)
        pred["legacy"] = M.R.C.decision(M.fold_predictions(a, b)["legacy"])
        for name, values in pred.items():
            scores.append(dict(test=regime, repeat=-1, method=name, **metric(b.label, values)))
            records.extend(dict(test=regime, repeat=-1, method=name, id=cid, group=g, label=int(y), pred=int(p))
                           for cid, g, y, p in zip(b.id, b.group, b.label, values))
        selections.append(dict(test=regime, repeat=-1, fold=-1, **choice))
        traces.extend(dict(test=regime, repeat=-1, fold=-1, **r) for r in trace)
        print("Rule vs ML stress complete", regime, flush=True)
    frame = pd.DataFrame(scores)
    summary = frame.groupby(["test", "method"], sort=False)[[c for c in frame.select_dtypes("number") if c != "repeat"]].mean().reset_index()
    summary.to_csv(out/"summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(records).to_csv(out/"predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selections).to_json(out/"inner_selections.json", orient="records", indent=2, force_ascii=False)
    pd.DataFrame(traces).to_csv(out/"inner_oof_probabilities.csv", index=False, encoding="utf-8-sig")
    frame.to_csv(out/"scores.csv", index=False, encoding="utf-8-sig")
    groups = sorted(train.group.unique()); lookup = {g: i for i, g in enumerate(groups)}
    counts = np.zeros((24, 30, len(NAMES), 3, 3), int)
    for r in pd.DataFrame(records).query("test == 'group_cv320'").itertuples():
        counts[lookup[r.group], r.repeat, NAMES.index(r.method), r.label, r.pred] += 1
    rng = np.random.default_rng(0); draws = []
    for _ in range(2000):
        c = counts[rng.integers(0,24,24)].sum(axis=0)
        denominator = c.sum(axis=-2)+c.sum(axis=-1); diagonal = np.diagonal(c, axis1=-2, axis2=-1)
        draws.append(np.divide(2*diagonal, denominator, out=np.zeros(diagonal.shape, float), where=denominator != 0).mean(axis=-1).mean(axis=0))
    draws = np.asarray(draws)
    intervals = [dict(method=k, baseline=base, low=float(np.quantile(draws[:,i]-draws[:,NAMES.index(base)],.025)),
                      high=float(np.quantile(draws[:,i]-draws[:,NAMES.index(base)],.975)))
                 for base in ["rule_tuned", "rule_inner_selected", "conditional_lr"] for i,k in enumerate(NAMES)]
    pd.DataFrame(intervals).to_csv(out/"paired_capture_intervals.csv", index=False, encoding="utf-8-sig")
    if any(M.sha(p) != h for p,h in (hashes|image_hashes).items()):
        raise ValueError("Input/source/image changed during run")
    result = dict(status="complete", original_photos=320, groups=24, repeats=30,
                  conditional_baseline_reproduced=9600, original_rule_seed0_reproduced=3*320,
                  limitations=specification["limits"], input_sha256=hashes, image_sha256=image_hashes)
    (out/"report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"report.html").write_text('<!doctype html><meta charset="utf-8"><title>Rules vs ML: nested F1</title><h1>Original320 / 24captures / frozen5fold x30</h1><p>No new labels. Selection uses inner grouped OOF only. Reused development evidence, not independent external proof.</p>'+summary.round(5).to_html(index=False)+'<h2>Frozen-OOF paired capture composition intervals</h2>'+pd.DataFrame(intervals).round(5).to_html(index=False),encoding="utf-8")
    print(summary[["test","method","macro_f1","f1_1","recall_2"]].round(5).to_string(index=False),flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.out)
