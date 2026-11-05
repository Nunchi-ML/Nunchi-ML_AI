"""Capture-source ablation with fixed A, fixed geometry-B and nested subset choice.

All eight auxiliary subsets and equal-data tuned rules are retained. Outer labels
are only scored, never used to choose sources, thresholds or hyperparameters.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import conditional_augmentation_probe as A
import conditional_calibration_probe as C
import rule_ml_validation as V
from nunchi_ml.glare import evaluation as E

MASKS = [f"{i:03b}" for i in range(8)]
NAMES = [f"{kind}_{mask}" for kind in ["B", "rule"] for mask in MASKS]
NAMES += ["B_inner_subset", "rule_inner_subset", "source_conditional_G"]
BASELINES = ["B_111", "B_000", "rule_000", "rule_111", "rule_inner_subset", "source_conditional_G"]


def subset(auxiliary, mask, groups):
    if mask not in MASKS or len(groups) != 3 or len(set(groups)) != 3 or set(auxiliary.group) != set(groups):
        raise ValueError("Exactly three declared auxiliary captures and a binary mask required")
    return auxiliary[auxiliary.group.isin([g for bit, g in zip(mask, groups) if bit == "1"])]


def fit_candidates(train, auxiliary, test, groups):
    # Validate the complete pool even when the selected subset is empty.
    A.training_pool(train, auxiliary, test)
    a = V.binary(train, test, V.M.R.G + V.M.R.P.REL_R, train.label.ge(1), "conditional_lr")
    outputs, cache = {}, {}
    for mask in MASKS:
        extra = subset(auxiliary, mask, groups)
        pool = A.training_pool(train, extra, test)
        positive = pool[pool.label.ge(1)]
        key = tuple(positive.id)
        if key not in cache:
            cache[key] = V.binary(positive, test, V.M.R.G + V.EXTRA, positive.label.eq(2), "geometry_lr")
        b = cache[key]
        rule = V.rule_predictions(pool, test)["rule_tuned"]
        outputs[mask] = dict(b=b, rule=rule, prediction=V.decide(a, b),
                             auxiliary_rows=len(extra), auxiliary_positive=int(extra.label.ge(1).sum()),
                             auxiliary_severe=int(extra.label.eq(2).sum()))
    return a, outputs


def choose_subset(y, outputs, kind):
    if kind not in ["B", "rule"]:
        raise ValueError("Unknown source-selection procedure")
    diagnostics = []
    for mask in MASKS:
        item = outputs[mask]
        pred = item["prediction" if kind == "B" else "rule"]
        score = float(V.macro_matrix(y, np.asarray(pred)[None, :])[0])
        # Same parsimony principle: prefer fewer rows actually used in this fit.
        rows = item["auxiliary_positive" if kind == "B" else "auxiliary_rows"]
        diagnostics.append(dict(mask=mask, inner_macro_f1=score, auxiliary_fit_rows=rows))
    chosen = max(diagnostics, key=lambda r: (r["inner_macro_f1"], -r["auxiliary_fit_rows"], -int(r["mask"], 2)))
    return chosen["mask"], diagnostics


def predict(train, auxiliary, test, groups):
    oof_a = np.full(len(train), np.nan)
    inner_fold, seen = np.full(len(train), -1, int), np.zeros(len(train), int)
    oof = {mask: dict(b=np.full(len(train), np.nan), rule=np.full(len(train), -1, int)) for mask in MASKS}
    for fold, (tr, va) in enumerate(V.inner_splits(train)):
        a, outputs = fit_candidates(train.iloc[tr], auxiliary, train.iloc[va], groups)
        oof_a[va], inner_fold[va], seen[va] = a, fold, seen[va] + 1
        for mask, item in outputs.items():
            oof[mask]["b"][va], oof[mask]["rule"][va] = item["b"], item["rule"]
            for key in ["auxiliary_rows", "auxiliary_positive", "auxiliary_severe"]:
                oof[mask][key] = item[key]
    if not np.all(seen == 1) or not np.isfinite(oof_a).all():
        raise ValueError("Exactly one capture-safe OOF per original training row required")
    for item in oof.values():
        item["prediction"] = V.decide(oof_a, item["b"])
    choices, diagnostics = {}, {}
    for kind in ["B", "rule"]:
        choices[kind], diagnostics[kind] = choose_subset(train.label.to_numpy(), oof, kind)
    a, outputs = fit_candidates(train, auxiliary, test, groups)
    result = {f"B_{mask}": (item["prediction"], a, item["b"], mask) for mask, item in outputs.items()}
    result.update({f"rule_{mask}": (item["rule"], None, None, mask) for mask, item in outputs.items()})
    for kind, mask in choices.items():
        result[kind + "_inner_subset"] = result[f"{kind}_{mask}"]
    traces = [dict(id=cid, group=g, label=int(y), inner_fold=int(f), mask=mask, a=float(pa), b=float(pb), rule=int(pr))
              for mask, item in oof.items()
              for cid, g, y, f, pa, pb, pr in zip(train.id, train.group, train.label, inner_fold, oof_a, item["b"], item["rule"])]
    return result, dict(selected=choices, candidate_scores=diagnostics), traces


def capture_intervals(records, original):
    index = {g: i for i, g in enumerate(sorted(original.group.unique()))}
    counts = np.zeros((24, 30, len(NAMES), 3, 3), int)
    for row in records[records.test.eq("group_cv320")].itertuples():
        counts[index[row.group], row.repeat, NAMES.index(row.method), row.label, row.pred] += 1
    rng, draws = np.random.default_rng(0), []
    for _ in range(2000):
        cm = counts[rng.integers(0, 24, 24)].sum(axis=0)
        diag, den = np.diagonal(cm, axis1=-2, axis2=-1), cm.sum(axis=-2) + cm.sum(axis=-1)
        draws.append(np.divide(2*diag, den, out=np.zeros(diag.shape, float), where=den != 0).mean(axis=-1).mean(axis=0))
    draws = np.asarray(draws)
    return [dict(method=name, baseline=base,
                 low=float(np.quantile(draws[:, i] - draws[:, NAMES.index(base)], .025)),
                 high=float(np.quantile(draws[:, i] - draws[:, NAMES.index(base)], .975)))
            for base in BASELINES for i, name in enumerate(NAMES)]


def run(data, out):
    data, out = Path(data).resolve(), Path(out).resolve()
    if out.exists() or out.parent != data:
        raise ValueError("Use a new private output directly under data")
    prior, base = data/"conditional_threshold_20261105", data/"road_visibility_20261104/run30"
    files = [prior/"report.json", prior/"predictions.csv", base/"training_features.csv", base/"splits.csv",
             data/"conditional_augmentation_20261105/binary_probabilities.csv"]
    sources = [Path(__file__), Path(A.__file__), Path(C.__file__), Path(V.__file__), Path(V.M.__file__),
               Path(V.Q.__file__), Path(V.O.__file__), Path(E.__file__)]
    hashes = {str(p): V.M.sha(p) for p in files + sources}
    previous = json.loads(files[0].read_text(encoding="utf-8"))
    if previous["status"] != "complete" or any(V.M.sha(p) != h for p, h in previous["input_sha256"].items()):
        raise ValueError("Frozen previous input/source provenance changed")
    full = V.geometry(pd.read_csv(files[2], dtype={"id": str}))
    original = full[full.camera.ne("AI Hub")].reset_index(drop=True)
    auxiliary = full[full.camera.eq("AI Hub")].reset_index(drop=True)
    groups = sorted(auxiliary.group.unique())
    if len(original) != 320 or original.group.nunique() != 24 or len(auxiliary) != 304 or len(groups) != 3:
        raise ValueError("Frozen original320/24 and AIHub304/3 required")
    if not np.isfinite(full[V.M.R.G + V.M.R.P.REL_R + V.EXTRA].to_numpy(float)).all():
        raise ValueError("Finite frozen features required; no row deletion")
    A.training_pool(original.iloc[:0], auxiliary, original)
    image_hashes = previous["image_sha256"]
    if len(image_hashes) != 624 or any(V.M.sha(p) != h for p, h in image_hashes.items()):
        raise ValueError("Original624 image bytes changed")
    frozen = pd.read_csv(files[1], dtype={"id": str})
    raw = pd.read_csv(files[4], dtype={"id": str})
    splits = pd.read_csv(files[3], dtype={"id": str})
    out.mkdir()
    specification = dict(status="frozen_before_fit", main_base="077318f / PR42 merged",
        primary="B_inner_subset MacroF1 vs fixed B_111 and strong rules rule_000, rule_111, rule_inner_subset",
        secondary="All8fixed B source subsets and all8equal-data tuned rules; inclusion/exclusion effects are descriptive, no outer-test-best adoption",
        names=NAMES, mask_bit_groups=groups, original="Original human0/1/2 unchanged; same320/24captures/frozen5folds x30",
        model="A original-only G+relative12; B positive-only original+declaredAIHubsubset G+gap/theta/logLv; balanced LR C1, train-only scaling, fixed.5/.5",
        source_selection="Same3innercapturefolds seed0 on outer-original-training. Max originalOOF MacroF1; ties fewer auxiliary fit rows (B positives/rule all), then numeric mask. No threshold/calibration/weight/feature search.",
        rules="Exact existing train-tuned geometry rule grid on same original+auxiliarysubset; separate inner source selection with identical folds/tie principle",
        auxiliary="Always training-only; no AIHub test even in empty-subset control. No fake G on human50/AI277.",
        budget="153outer splits x4fit passes: up to4896geometry-B fits +612A fits +4896exact rule grids; positive-pool identity cached within a pass; no Optuna/neural features",
        stress="Same three held-original-camera regimes, no selection from their outcomes",
        uncertainty="30repeats reuse same320photos; win counts descriptive. 2000capturebootstrap preserves30OOF repeats; fixed-prediction composition intervals, no model refitting or independent-trial CI.",
        limits="Sequential reused-development comparison, severe71of73inoneoriginalcapture; subset identity confounded with label mix/sample count/scene. No causal camera attribution, final selection or API changes.",
        input_sha256=hashes, versions={n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scikit-learn"]})
    (out/"specification_before_run.json").write_text(json.dumps(specification, ensure_ascii=False, indent=2), encoding="utf-8")
    snapshots = out/"source_snapshot"
    snapshots.mkdir()
    for path in sources:
        shutil.copyfile(path, snapshots/path.name)
    full.groupby(["camera", "group", "label"]).size().unstack(fill_value=0).to_csv(out/"capture_label_support.csv", encoding="utf-8-sig")
    records, scores, choices, traces, manifests = [], [], [], [], []
    reproduced = 0
    regimes = [("group_cv320", r, original) for r in range(30)]
    regimes += [("held_camera:" + str(c), -1, original[original.camera.eq(c)]) for c in original.camera.unique()]
    for regime, repeat, test in regimes:
        source = raw[raw.test.eq(regime) & raw["repeat"].eq(repeat)]
        if not source.id.is_unique or set(source.id) != set(test.id):
            raise ValueError("Frozen binary probability coverage mismatch")
        source = source.set_index("id").loc[test.id]
        reference = frozen[frozen.test.eq(regime) & frozen["repeat"].eq(repeat)]
        if repeat >= 0:
            part = splits[splits["repeat"].eq(repeat)]
            if len(part) != 320 or not part.id.is_unique or set(part.id) != set(original.id) or set(part.fold) != set(range(5)):
                raise ValueError("Frozen split mismatch")
            folds = [(f, original[~original.id.isin(part[part.fold.eq(f)].id)], original[original.id.isin(part[part.fold.eq(f)].id)]) for f in range(5)]
        else:
            folds = [(-1, original[~original.id.isin(test.id)], test)]
        values = {name: dict(pred=np.full(len(test), -1, int), a=np.full(len(test), np.nan), b=np.full(len(test), np.nan),
                            selected_mask=np.full(len(test), "", object)) for name in NAMES}
        outer_fold, seen = np.full(len(test), -1, int), np.zeros(len(test), int)
        for fold, train, valid in folds:
            outputs, selection, trace = predict(train, auxiliary, valid, groups)
            mask = test.id.isin(valid.id).to_numpy()
            for name, (pred, a, b, selected) in outputs.items():
                item = values[name]
                item["pred"][mask], item["selected_mask"][mask] = pred, selected
                if a is not None:
                    item["a"][mask], item["b"][mask] = a, b
            outer_fold[mask], seen[mask] = fold, seen[mask] + 1
            choices.append(dict(test=regime, repeat=repeat, fold=fold, **selection))
            traces.extend(dict(test=regime, repeat=repeat, fold=fold, **row) for row in trace)
            manifests.append(dict(test=regime, repeat=repeat, fold=fold, train_ids=train.id.tolist(), test_ids=valid.id.tolist(),
                original_B_positive_ids=train[train.label.ge(1)].id.tolist(), auxiliary_ids=auxiliary.id.tolist(),
                original_severe_support=int(train.label.eq(2).sum()), original_severe_captures=int(train[train.label.eq(2)].group.nunique())))
        if not np.all(seen == 1):
            raise ValueError("Outer OOF coverage mismatch")
        values["source_conditional_G"].update(pred=V.decide(source.source_a, source.source_b),
            a=source.source_a.to_numpy(), b=source.source_b.to_numpy(), selected_mask=np.full(len(test), "reference_G", object))
        for name, old in [("B_111", "B_geometry_aug"), ("rule_000", "source_rule_tuned"), ("rule_111", "aug_rule_tuned"), ("source_conditional_G", "source_conditional")]:
            ref = reference[reference.method.eq(old)]
            if not ref.id.is_unique or set(ref.id) != set(test.id):
                raise ValueError("Reference prediction coverage mismatch")
            ref = ref.set_index("id").loc[test.id]
            if not np.array_equal(ref.label, test.label) or not np.array_equal(ref.pred, values[name]["pred"]):
                raise ValueError("Reference grade/label reproduction failed")
            reproduced += len(test)
        if not np.allclose(values["B_111"]["b"], source.geometry_b, atol=1e-12, rtol=0):
            raise ValueError("Full-source B probability reproduction failed")
        for name, item in values.items():
            has_probability = name.startswith("B_") or name == "source_conditional_G"
            a, b = (item["a"], item["b"]) if has_probability else (None, None)
            if has_probability and not np.allclose(a, source.source_a, atol=1e-12, rtol=0):
                raise ValueError("Original-only A must be fixed across all source subsets")
            p = np.column_stack([1-a, a*(1-b), a*b]) if has_probability else None
            scores.append(dict(test=regime, repeat=repeat, method=name, **E.evaluate_grades(test.label, item["pred"], p, b)))
            records.extend(dict(test=regime, repeat=repeat, method=name, fold=int(outer_fold[i]), id=cid, group=g, label=int(y), pred=int(item["pred"][i]),
                selected_mask=item["selected_mask"][i], selection_source="original_outer_training_3captureOOF" if "inner_subset" in name else "predeclared_fixed",
                a=float(a[i]) if has_probability else None, b=float(b[i]) if has_probability else None,
                **{f"p{k}": float(p[i, k]) if has_probability else None for k in range(3)})
                for i, (cid, g, y) in enumerate(zip(test.id, test.group, test.label)))
        pd.DataFrame(scores).to_csv(out/"scores.csv", index=False, encoding="utf-8-sig")
        print("Source ablation", regime, repeat + 1, flush=True)
    frame, predictions = pd.DataFrame(scores), pd.DataFrame(records)
    summary = frame.groupby(["test", "method"], sort=False)[[c for c in frame.select_dtypes("number") if c != "repeat"]].mean().reset_index()
    summary.to_csv(out/"summary.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out/"predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(traces).to_csv(out/"inner_oof.csv", index=False, encoding="utf-8-sig")
    for filename, obj in [("inner_selections.json", choices), ("split_manifest.json", manifests)]:
        (out/filename).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([row for baseline in BASELINES for row in C.paired_repeats(frame, baseline)]).to_csv(out/"paired_repeat_deltas.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(capture_intervals(predictions, original)).to_csv(out/"paired_capture_intervals.csv", index=False, encoding="utf-8-sig")
    # This capture diagnostic preserves the predeclared grade metric and is not a selector.
    capture_scores = [dict(test=regime, repeat=int(repeat), method=name, group=group,
                          **E.evaluate_grades(part.label, part.pred, part[["p0", "p1", "p2"]].to_numpy() if part.a.notna().all() else None,
                                              part.b.to_numpy() if part.a.notna().all() else None))
                      for (regime, repeat, name, group), part in predictions.groupby(["test", "repeat", "method", "group"], sort=False)]
    pd.DataFrame(capture_scores).to_csv(out/"capture_scores.csv", index=False, encoding="utf-8-sig")
    if any(V.M.sha(p) != h for p, h in (hashes | image_hashes).items()):
        raise ValueError("Frozen source/input/image changed during run")
    result = dict(status="complete", original_photos=320, auxiliary_training_only=304, mask_bit_groups=groups,
        evaluation_rows=len(predictions), inner_oof_rows=len(traces), outer_selections=len(choices), reproduced_predictions=reproduced,
        final_selection=False, aihub_test=False, input_sha256=hashes, image_sha256=image_hashes, limits=specification["limits"])
    (out/"report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out/"report.html").write_text('<!doctype html><meta charset="utf-8"><title>Capture source ablation</title><h1>Fixed A / eight B source subsets / equal-data rules</h1><p>Reused original320 development benchmark; AIHub training only. Nested choice uses original outer-training only. No final selection.</p>' + summary.round(5).to_html(index=False), encoding="utf-8")
    print(summary[summary.test.eq("group_cv320")][["method", "macro_f1", "recall_2", "macro_ovr_auc"]].round(6).to_string(index=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.out)
