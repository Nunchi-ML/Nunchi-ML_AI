"""Frozen photo training-weight ablation, selected using training groups only.

No test-label selection, neural features, relabeling, sensor synthesis or export.
Group/camera inverse counts are followed by class-mass balancing; this is not
exact group balancing when class mixtures differ between groups.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

import photo_regularization_probe as P
import photo_regularization_analysis as A
from ordinal_compare import metrics

POLICIES = ["class_only", "capture_inverse", "capture_sqrt", "camera_inverse"]
PARAMETERS = {"base": P.GRID[0], "regularized": next(p for p in P.GRID if p["id"] == "l7_m10_r1")}


def training_weights(frame, policy):
    if (policy not in POLICIES or frame.empty or frame[["label", "group", "camera"]].isna().any().any()
            or set(frame.label) != {0, 1, 2}):
        raise ValueError("Training rows with three classes and capture/camera metadata required")
    if policy == "class_only":
        return compute_sample_weight("balanced", frame.label)
    unit = "camera" if policy == "camera_inverse" else "group"
    exponent = .5 if policy == "capture_sqrt" else 1.
    count = frame[unit].map(frame[unit].value_counts()).to_numpy(float)
    weight = count ** -exponent
    for label in [0, 1, 2]:
        mask = frame.label.eq(label).to_numpy()
        weight[mask] *= len(frame) / (3 * weight[mask].sum())
    if not np.isfinite(weight).all() or (weight <= 0).any() or not np.isclose(weight.mean(), 1):
        raise ValueError("Invalid training weights")
    return weight


def fit_predict(train, test, columns, parameters, policy):
    # Explicit weights contain the class correction; do not multiply it twice.
    estimator = P.model(parameters).set_params(class_weight=None)
    estimator.fit(train[columns], train.label, sample_weight=training_weights(train, policy))
    probability = P.R.validate_probabilities(estimator.predict_proba(test[columns]))
    return estimator.classes_[probability.argmax(axis=1)].astype(int)


def choose(records):
    # Selection inputs are training624 OOF metrics only, never photo-test scores.
    return max(records, key=lambda r: (r["macro_f1"], r["recall_2"], -POLICIES.index(r["policy"])))


def run(data, out):
    data, out = Path(data).resolve(), Path(out).resolve()
    if out.exists() or not out.is_relative_to(data):
        raise ValueError("Use a new private output directory")
    old, fd = data / "road_visibility_20261104/run30", data / "forward_detail_20261105"
    previous = data / "photo_regularization_20261105"
    files = [old / "training_features.csv", fd / "train_features.csv", fd / "specification_before_run.json",
             previous / "training_splits.csv", previous / "training_oof_predictions.csv",
             previous / "predictions.csv", previous / "choices_before_evaluation.json", previous / "report.json",
             old / "human_features.csv", fd / "human50_features.csv", fd / "other277_features.csv",
             data / "other_photos_20261105/result_v1/predictions.csv",
             data / "new_cameras/human_review_50/labels_성진.csv", data / "new_cameras/labels_claude.csv",
             Path(__file__), Path(P.__file__), Path(A.__file__), Path(P.R.__file__),
             Path(P.R.lightgbm_model.__code__.co_filename)]
    hashes = {str(p): P.R.digest(p) for p in files}
    out.mkdir()
    specification = dict(status="frozen_before_any_training", policies=POLICIES, parameters=PARAMETERS,
                         variants=["road", "detail"], budget="4policies x2fixed settings x2variants x5folds=80fits",
                         selection="pooled3class trainingOOF MacroF1, severe recall, then simpler weighting",
                         weights="inverse count (or square root) per training fold, then equal class mass; class_weight=None",
                         exclusions="No test selection, calibration, threshold/label change, neural features, sensors, deployment",
                         limitations="Exploratory reused development samples; training OOF is selection score, not original320CV",
                         hashes=hashes, versions={n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scikit-learn", "lightgbm"]})
    (out / "specification_before_run.json").write_text(json.dumps(specification, indent=2), encoding="utf-8")
    shutil.copyfile(__file__, out / "source_before_run.py")
    train = pd.read_csv(files[0], dtype={"id": str}).merge(pd.read_csv(files[1], dtype={"id": str}), on="id", validate="one_to_one")
    if len(train) != 624 or train.group.nunique() != 27 or train.image_sha256.duplicated().any():
        raise ValueError("Frozen624/27 training data required")
    extra = json.loads(files[2].read_text(encoding="utf-8"))["features"]
    if len(set(extra)) != 18 or not all(c.startswith("fd_") for c in extra):
        raise ValueError("Frozen forward18 schema required")
    base = P.R.PHOTO_METHODS["screen_road_lgb"][0]
    variants = dict(road=base, detail=base + extra)
    if not np.isfinite(train[variants["detail"]].to_numpy(float)).all():
        raise ValueError("Nonfinite training features")
    saved = pd.read_csv(files[3], dtype={"id": str}).set_index("id")
    splits = P.grouped_splits(train)
    diagnostics, composition = [], []
    for fold, (tr, va) in enumerate(splits):
        if not saved.loc[train.iloc[va].id, "fold"].eq(fold).all():
            raise ValueError("Previous training split mismatch")
        for partition, indices in [("train", tr), ("validation", va)]:
            part = train.iloc[indices]
            for label in [0, 1, 2]:
                local = part[part.label.eq(label)]
                composition.append(dict(fold=fold, partition=partition, label=label, rows=len(local), captures=local.group.nunique()))
        for policy in POLICIES:
            part = train.iloc[tr]
            w = training_weights(part, policy)
            for camera, group in part.assign(weight=w).groupby("camera", sort=False):
                diagnostics.append(dict(fold=fold, policy=policy, camera=camera, rows=len(group), weight=float(group.weight.sum())))
    pd.DataFrame(composition).to_csv(out / "fold_class_composition.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(diagnostics).to_csv(out / "fold_weight_mass.csv", index=False, encoding="utf-8-sig")
    train.groupby(["camera", "label"]).agg(rows=("id", "size"), captures=("group", "nunique")).to_csv(out / "training_composition.csv", encoding="utf-8-sig")
    old_oof = pd.read_csv(files[4], dtype={"id": str})
    choices, scores, oof = {}, [], []
    for variant, columns in variants.items():
        for parameter_name, parameters in PARAMETERS.items():
            candidates = []
            for policy in POLICIES:
                pred = np.full(len(train), -1, int)
                for tr, va in splits:
                    pred[va] = fit_predict(train.iloc[tr], train.iloc[va], columns, parameters, policy)
                if policy == "class_only":
                    expected = old_oof[old_oof.variant.eq(variant) & old_oof.parameters.eq(parameters["id"])].set_index("id").loc[train.id, "pred"]
                    if not np.array_equal(pred, expected.to_numpy()):
                        raise ValueError("Class-only trainingOOF reproduction failed")
                name = f"{variant}_{parameter_name}_{policy}"
                row = dict(method=name, variant=variant, parameters=parameter_name, policy=policy, **metrics(train.label.to_numpy(), pred))
                candidates.append(row)
                scores.append(row)
                oof.extend(dict(method=name, id=cid, pred=int(value)) for cid, value in zip(train.id, pred))
                print("training-only", name, "complete", flush=True)
            choices[f"{variant}_{parameter_name}"] = choose(candidates)
    pd.DataFrame(scores).to_csv(out / "training_only_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(oof).to_csv(out / "training_oof_predictions.csv", index=False, encoding="utf-8-sig")
    (out / "choices_before_evaluation.json").write_text(json.dumps(choices, indent=2), encoding="utf-8")
    # Only after training-only choices are frozen, load evaluation photos/labels.
    human = pd.read_csv(files[8], dtype={"id": str}).merge(pd.read_csv(files[9], dtype={"id": str}), on="id", validate="one_to_one")
    metadata = pd.read_csv(files[11], dtype={"id": str})
    other = metadata[metadata.evaluation.eq("outside277_AI_provisional")][["id", "video", "label", "image_sha256"]].merge(pd.read_csv(files[10], dtype={"id": str}), on="id", validate="one_to_one")
    if (len(human), len(other)) != (50, 277):
        raise ValueError("Evaluation counts changed")
    test = pd.concat([human.assign(evaluation="human50"), other.assign(evaluation="other277_AI")], ignore_index=True)
    if (not test.id.is_unique or test.image_sha256.duplicated().any() or set(test.image_sha256) & set(train.image_sha256)
            or not np.isfinite(test[variants["detail"]].to_numpy(float)).all()):
        raise ValueError("Invalid/overlapping evaluation photos")
    expected = pd.read_csv(files[5], dtype={"id": str}).set_index("id").loc[test.id]
    if not np.array_equal(expected.label.to_numpy(), test.label.to_numpy()):
        raise ValueError("Reference labels changed")
    predictions = test[["id", "video", "label", "evaluation"]].copy()
    evaluation, videos = [], []
    for variant, columns in variants.items():
        for parameter_name, parameters in PARAMETERS.items():
            for policy in POLICIES:
                name = f"{variant}_{parameter_name}_{policy}"
                pred = fit_predict(train, test, columns, parameters, policy)
                predictions[name] = pred
                if policy == "class_only":
                    old_name = ("fixed_" if parameter_name == "base" else "selected_") + variant
                    if not np.array_equal(pred, expected[old_name].to_numpy()):
                        raise ValueError("Baseline1308 evaluation reproduction failed")
    for key, choice in choices.items():
        predictions["selected_" + key] = predictions[choice["method"]]
    for name in predictions.columns[4:]:
        for regime in test.evaluation.unique():
            mask = test.evaluation.eq(regime)
            evaluation.append(dict(method=name, evaluation=regime, **metrics(test.loc[mask, "label"].to_numpy(), predictions.loc[mask, name].to_numpy())))
            for video, part in predictions.loc[mask].groupby("video", sort=False):
                videos.append(dict(method=name, evaluation=regime, video=video, **metrics(part.label.to_numpy(), part[name].to_numpy())))
    predictions.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(evaluation).to_csv(out / "scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(videos).to_csv(out / "per_video.csv", index=False, encoding="utf-8-sig")
    pairs = [(f"{key}_class_only", "selected_" + key) for key in choices]
    # Posthoc intervals: frozen predictions only, after point evaluation.
    (out / "interval_specification.json").write_text(json.dumps(dict(status="posthoc", pairs=pairs, draws=2000, seed=0,
        limitation="Frozen-prediction video-composition uncertainty only; no refitting or independent final test"), indent=2), encoding="utf-8")
    intervals = [dict(evaluation=regime, **A.paired_interval(part, a, b))
                 for regime, part in predictions.groupby("evaluation", sort=False) for a, b in pairs]
    pd.DataFrame(intervals).to_csv(out / "paired_video_intervals.csv", index=False, encoding="utf-8-sig")
    if any(P.R.digest(p) != value for p, value in hashes.items()):
        raise ValueError("Inputs/source changed during run")
    result = dict(status="complete", choices=choices, training_baselines_reproduced=2496,
                  evaluation_baselines_reproduced=1308, intervals=intervals, hashes=hashes,
                  limitations=specification["limitations"])
    (out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "report.html").write_text('<!doctype html><meta charset="utf-8"><title>Photo training weights</title><style>body{font:16px system-ui;margin:24px}td,th{padding:5px;border:1px solid #ccc}</style><h1>Training-only weight selection</h1><p>Exploratory reused data. Selection OOF is not independent performance or original320 CV.</p>'+pd.DataFrame(scores).round(4).to_html(index=False)+'<h2>Evaluation</h2>'+pd.DataFrame(evaluation).round(4).to_html(index=False)+'<h2>Posthoc conditional intervals</h2>'+pd.DataFrame(intervals).round(4).to_html(index=False), encoding="utf-8")
    print("CHOICES", {k: v["policy"] for k, v in choices.items()}, flush=True)
    print(pd.DataFrame(evaluation)[["method", "evaluation", "macro_f1", "f1_1", "recall_2"]].round(5).to_string(index=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.out)
