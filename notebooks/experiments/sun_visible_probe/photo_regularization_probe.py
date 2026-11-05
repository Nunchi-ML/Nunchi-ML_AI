"""Choose a small fixed LightGBM grid using training624 capture groups only.

Equal budgets with/without frozen forward detail. Human50/AI277 are loaded for
evaluation only AFTER choices are written. This remains exploratory development
evaluation, not an independent final test or original320 CV improvement.
"""
import argparse
from itertools import product
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import road_visibility_probe as R
from ordinal_compare import metrics

GRID = [dict(id="base", num_leaves=7, min_child_samples=5, reg_lambda=0.)]
GRID += [dict(id=f"l{leaves}_m{minimum}_r{penalty}", num_leaves=leaves,
              min_child_samples=minimum, reg_lambda=float(penalty))
         for leaves, minimum, penalty in product([3, 7], [10, 30], [1, 10])]


def grouped_splits(frame):
    if not frame.id.is_unique or frame.group.isna().any() or not frame.label.isin([0, 1, 2]).all():
        raise ValueError("Unique IDs, valid labels and capture groups required")
    splits = list(GroupKFold(5, shuffle=True, random_state=0).split(frame, frame.label, frame.group))
    seen = np.zeros(len(frame), int)
    for tr, va in splits:
        if set(frame.iloc[tr].group) & set(frame.iloc[va].group):
            raise ValueError("Training capture leakage")
        seen[va] += 1
    if not np.all(seen == 1):
        raise ValueError("Training OOF coverage mismatch")
    return splits


def select(records):
    # Input is training OOF metrics only. No test frame/metric is accepted here.
    return max(records, key=lambda r: (r["macro_f1"], r["recall_2"], -r["num_leaves"],
                                       r["min_child_samples"], r["reg_lambda"]))


def model(parameters):
    return R.lightgbm_model().set_params(**{k: parameters[k] for k in ["num_leaves", "min_child_samples", "reg_lambda"]})


def run(data, out):
    data, out = Path(data).resolve(), Path(out).resolve()
    if out.exists() or not out.is_relative_to(data):
        raise ValueError("Use new private output")
    old = data / "road_visibility_20261104/run30"
    detail = data / "forward_detail_20261105"
    others = data / "other_photos_20261105/result_v1"
    files = [old / "training_features.csv", old / "human_features.csv",
             detail / "train_features.csv", detail / "human50_features.csv", detail / "other277_features.csv",
             detail / "specification_before_run.json", detail / "photo_predictions.csv", detail / "report.json",
             others / "predictions.csv", data / "new_cameras/human_review_50/labels_성진.csv", data / "new_cameras/labels_claude.csv",
             Path(__file__), Path(R.__file__)]
    hashes = {str(p): R.digest(p) for p in files}
    out.mkdir()
    specification = dict(status="frozen_before_selection", grid=GRID, variants=["road", "detail"], folds=5, seed=0,
                         selection="Pooled3class trainingOOF MacroF1; training severe recall and simpler settings break ties",
                         budget="Same9 settings x5 training folds for each feature variant",
                         rules="All624 training only; no human50/AI277 selection, no calibration/threshold/label change",
                         limitations="Designed after development errors; evaluation photos/videos reused, not final independent validation",
                         hashes=hashes)
    (out / "specification_before_run.json").write_text(json.dumps(specification, indent=2), encoding="utf-8")
    shutil.copyfile(__file__, out / "source_before_run.py")
    feature_spec = json.loads(files[5].read_text(encoding="utf-8"))
    extra = feature_spec["features"]
    if len(extra) != 18 or len(set(extra)) != 18 or not all(x.startswith("fd_") for x in extra):
        raise ValueError("Frozen18 detail features required")
    train = pd.read_csv(files[0], dtype={"id": str}).merge(pd.read_csv(files[2], dtype={"id": str}), on="id", validate="one_to_one")
    if len(train) != 624 or train.group.nunique() != 27 or train.image_sha256.duplicated().any():
        raise ValueError("Original624/27capture groups required")
    base_features = R.PHOTO_METHODS["screen_road_lgb"][0]
    variants = dict(road=base_features, detail=base_features+extra)
    splits = grouped_splits(train)
    split_records = [dict(fold=k, id=train.iloc[i].id, group=train.iloc[i].group) for k, (_, va) in enumerate(splits) for i in va]
    pd.DataFrame(split_records).to_csv(out / "training_splits.csv", index=False, encoding="utf-8-sig")
    choices, tuning, oof = {}, [], []
    for variant, cols in variants.items():
        if not np.isfinite(train[cols].to_numpy(float)).all():
            raise ValueError("Nonfinite training features")
        candidates = []
        for parameters in GRID:
            prediction = np.full(624, -1, int)
            for tr, va in splits:
                estimator = model(parameters).fit(train.iloc[tr][cols], train.iloc[tr].label)
                prediction[va] = estimator.predict(train.iloc[va][cols]).astype(int)
            if not np.isin(prediction, [0, 1, 2]).all():
                raise ValueError("Invalid training OOF prediction")
            row = dict(variant=variant, **parameters, **metrics(train.label.to_numpy(), prediction))
            candidates.append(row)
            tuning.append(row)
            oof.extend(dict(variant=variant, parameters=parameters["id"], id=cid, pred=int(p)) for cid, p in zip(train.id, prediction))
            print("training-only", variant, parameters["id"], "OOF complete", flush=True)
        choices[variant] = select(candidates)
    pd.DataFrame(tuning).to_csv(out / "training_only_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(oof).to_csv(out / "training_oof_predictions.csv", index=False, encoding="utf-8-sig")
    # Freeze choices before opening evaluation labels/features below.
    (out / "choices_before_evaluation.json").write_text(json.dumps(choices, indent=2), encoding="utf-8")
    human = pd.read_csv(files[1], dtype={"id": str}).merge(pd.read_csv(files[3], dtype={"id": str}), on="id", validate="one_to_one")
    reference = pd.read_csv(files[8], dtype={"id": str})
    other = reference[reference.evaluation.eq("outside277_AI_provisional")][["id", "video", "label", "image_sha256"]].merge(pd.read_csv(files[4], dtype={"id": str}), on="id", validate="one_to_one")
    if (len(human), len(other)) != (50, 277):
        raise ValueError("Frozen evaluation counts changed")
    test = pd.concat([human.assign(evaluation="human50"), other.assign(evaluation="other277_AI")], ignore_index=True)
    if not test.id.is_unique or test.image_sha256.duplicated().any() or set(test.image_sha256) & set(train.image_sha256):
        raise ValueError("Evaluation photo overlap")
    expected = pd.read_csv(files[6], dtype={"id": str}).set_index("id").loc[test.id]
    if not np.array_equal(expected.label.to_numpy(), test.label.to_numpy()):
        raise ValueError("Evaluation reference labels changed")
    specs = {"fixed_road": ("road", GRID[0]), "fixed_detail": ("detail", GRID[0]),
             "selected_road": ("road", choices["road"]), "selected_detail": ("detail", choices["detail"]),
             "road_params_on_detail": ("detail", choices["road"]), "detail_params_on_road": ("road", choices["detail"])}
    cache, scores, per_video = {}, [], []
    predictions = test[["id", "video", "label", "evaluation"]].copy()
    for name, (variant, parameters) in specs.items():
        key = variant, parameters["id"]
        if key not in cache:
            estimator = model(parameters).fit(train[variants[variant]], train.label)
            cache[key] = estimator.predict(test[variants[variant]]).astype(int)
        pred = cache[key]
        predictions[name] = pred
        if name in ["fixed_road", "fixed_detail"]:
            old_name = "screen_road_lgb" if name == "fixed_road" else "screen_road_detail_lgb"
            if not np.array_equal(expected[old_name].to_numpy(), pred):
                raise ValueError("Fixed654 baseline predictions changed")
        for regime in test.evaluation.unique():
            mask = test.evaluation.eq(regime).to_numpy()
            scores.append(dict(method=name, evaluation=regime, rows=int(mask.sum()), **metrics(test.loc[mask, "label"].to_numpy(), pred[mask])))
            for video in test.loc[mask, "video"].unique():
                local = mask & test.video.eq(video).to_numpy()
                per_video.append(dict(method=name, evaluation=regime, video=video, rows=int(local.sum()), **metrics(test.loc[local, "label"].to_numpy(), pred[local])))
    predictions.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(scores).to_csv(out / "scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_video).to_csv(out / "per_video.csv", index=False, encoding="utf-8-sig")
    result = dict(status="complete", train_rows=624, train_groups=27, training_folds=5, tested_settings_per_variant=9,
                  selected_parameters=choices, predictions_reproduced=654,
                  evaluation_references="Preserved single-human50 and AI-provisional277; reused development samples",
                  selection_metric="Training624 OOF score is a selection score, not original320 CV or unbiased performance",
                  hashes=hashes)
    if any(R.digest(p) != v for p, v in hashes.items()):
        raise ValueError("Inputs/source changed during experiment")
    (out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "report.html").write_text('<!doctype html><meta charset="utf-8"><title>Training-only photo regularization</title><style>body{font:16px system-ui;margin:24px}table{border-collapse:collapse}td,th{padding:5px;border:1px solid #ccc}</style><h1>Training624 group-only settings selection</h1><p>Equal9setting budgets,5training capture folds. Choices frozen before test evaluation. Selection score is not unbiased validation or original320 CV. Original labels unchanged; samples reused.</p><h2>Training selection</h2>'+pd.DataFrame(tuning).round(4).to_html(index=False)+'<h2>Evaluation only after selection</h2>'+pd.DataFrame(scores).round(4).to_html(index=False), encoding="utf-8")
    print("CHOICES", {k: v["id"] for k, v in choices.items()}, flush=True)
    print(pd.DataFrame(scores)[["method", "evaluation", "macro_f1", "f1_1", "recall_2"]].round(5).to_string(index=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.out)
