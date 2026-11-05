"""Fixed CORAL ablation with held-out capture/video evaluation, no relabeling.

Two development protocols: other-camera source -> disjoint target captures;
source624 -> disjoint calibration/evaluation videos in reused newcamera327.
These are not the original320 repeated CV or independent new test data.
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
from sklearn.preprocessing import StandardScaler

import coral_alignment as C
import road_visibility_probe as R
import photo_regularization_analysis as A
from ordinal_compare import metrics

METHODS = ["source_only", "mean_only", "coral"]
FAMILIES = ["lightgbm", "logistic"]


def estimator(family):
    if family == "lightgbm":
        return R.lightgbm_model()
    if family == "logistic":
        return LogisticRegression(C=1., max_iter=3000, class_weight="balanced", random_state=0)
    raise ValueError("Unknown family")


def score(y, pred):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        return metrics(np.asarray(y), np.asarray(pred))


def fit_predictions(source, calibration_features, evaluation_features, columns, family):
    # Only source labels are accepted. Calibration/evaluation are feature arrays.
    scaler = StandardScaler().fit(source[columns])
    xs = scaler.transform(source[columns])
    xt = scaler.transform(C.matrix(calibration_features, len(columns)))
    xe = scaler.transform(C.matrix(evaluation_features, len(columns)))
    aligner = C.CoralAlignment(ridge=1.).fit(xs, xt)
    train_inputs = dict(source_only=xs,
                        mean_only=xs - xs.mean(axis=0) + xt.mean(axis=0),
                        coral=aligner.transform_source(xs))
    predictions = {}
    for method, values in train_inputs.items():
        model = estimator(family).fit(values, source.label)
        probabilities = R.validate_probabilities(model.predict_proba(xe))
        predictions[method] = model.classes_[probabilities.argmax(axis=1)].astype(int)
    source_cov, target_cov = np.cov(xs, rowvar=False), np.cov(xt, rowvar=False)
    diagnostics = dict(covariance_error_before=float(np.linalg.norm(source_cov-target_cov)),
        covariance_error_after=float(np.linalg.norm(np.cov(train_inputs["coral"], rowvar=False)-target_cov)),
        mean_shift=float(np.linalg.norm(xt.mean(axis=0)-xs.mean(axis=0))))
    return predictions, diagnostics


def run(data, out):
    data, out = Path(data).resolve(), Path(out).resolve()
    if out.exists() or not out.is_relative_to(data):
        raise ValueError("Use a new private output")
    old, fd = data / "road_visibility_20261104/run30", data / "forward_detail_20261105"
    other = data / "other_photos_20261105/result_v1/predictions.csv"
    inputs = [old / "training_features.csv", fd / "train_features.csv", old / "human_features.csv",
        fd / "human50_features.csv", fd / "other277_features.csv", fd / "specification_before_run.json",
        fd / "photo_predictions.csv", other, Path(__file__), Path(C.__file__), Path(A.__file__),
        Path(R.__file__), Path(R.lightgbm_model.__code__.co_filename)]
    hashes = {str(p): R.digest(p) for p in inputs}
    out.mkdir()
    spec = dict(status="frozen_before_fit", families=FAMILIES, methods=METHODS,
        features="Frozen screen12+road12+forward18; no new extraction/selection", ridge=1.,
        scaler="Source-only StandardScaler; covariance plus identity; centered source-to-calibration transform",
        classifier="Existing200tree/.05/leaves7/min_child5 balancedLGB; balancedLogisticC1/max_iter3000",
        splits="Every eligible capture/video held out once; calibration is every other target group",
        rules="Calibration labels never used; evaluation excluded from preprocessing/alignment; no tuning/selection/threshold changes",
        limitations="Reused development data; adaptation from other captures/videos, not original320CV or fully unseen domain performance",
        hashes=hashes, versions={n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scikit-learn", "lightgbm"]})
    (out / "specification_before_run.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    shutil.copyfile(__file__, out / "source_before_run.py")
    shutil.copyfile(C.__file__, out / "alignment_before_run.py")
    extra = json.loads(inputs[5].read_text(encoding="utf-8"))["features"]
    columns = R.PHOTO_METHODS["screen_road_lgb"][0] + extra
    if len(extra) != 18 or len(columns) != 42 or len(set(columns)) != 42:
        raise ValueError("Frozen42 feature order required")
    train = pd.read_csv(inputs[0], dtype={"id": str}).merge(pd.read_csv(inputs[1], dtype={"id": str}), on="id", validate="one_to_one")
    human = pd.read_csv(inputs[2], dtype={"id": str}).merge(pd.read_csv(inputs[3], dtype={"id": str}), on="id", validate="one_to_one")
    reference = pd.read_csv(other, dtype={"id": str})
    others = reference[reference.evaluation.eq("outside277_AI_provisional")][["id", "file", "video", "label", "image_sha256"]].merge(pd.read_csv(inputs[4], dtype={"id": str}), on="id", validate="one_to_one")
    target = pd.concat([human.assign(evaluation="human50"), others.assign(evaluation="other277_AI")], ignore_index=True)
    if len(train) != 624 or train.group.nunique() != 27 or (len(human), len(others)) != (50, 277):
        raise ValueError("Frozen sample counts changed")
    for frame in [train, target]:
        if not frame.id.is_unique or frame.image_sha256.duplicated().any() or not np.isfinite(frame[columns].to_numpy(float)).all():
            raise ValueError("Invalid frozen features/photos")
    paths = R.C.image_paths(train, data) + [(data / "new_cameras/label_pack" / f).resolve() for f in target.file]
    expected_hashes = pd.concat([train.image_sha256, target.image_sha256]).tolist()
    image_hashes = {}
    for path, expected_hash in zip(paths, expected_hashes):
        if not path.is_relative_to(data) or R.digest(path) != expected_hash:
            raise ValueError("Original image bytes changed")
        image_hashes[str(path)] = expected_hash
    if set(train.image_sha256) & set(target.image_sha256):
        raise ValueError("Source/new-camera photo overlap")
    jobs, skipped = [], []
    for camera, part in train.groupby("camera", sort=True):
        part = part.reset_index(drop=True)
        if part.group.nunique() < 2:
            skipped.append(dict(camera=camera, rows=len(part), reason="Only one capture; disjoint calibration/evaluation impossible"))
            continue
        source = train[train.camera.ne(camera)].reset_index(drop=True)
        for group, calibration, evaluation in C.separated_folds(part, "group"):
            jobs.append(dict(protocol="camera_adaptation", domain=camera, group=group, source=source,
                calibration=part.iloc[calibration], test=part.iloc[evaluation], unit="group"))
    for group, calibration, evaluation in C.separated_folds(target, "video"):
        jobs.append(dict(protocol="newvideo_adaptation", domain="reused_target_pool", group=group, source=train,
            calibration=target.iloc[calibration], test=target.iloc[evaluation], unit="video"))
    manifest = []
    for job in jobs:
        C.verify_disjoint(job["source"], job["calibration"], job["test"], job["unit"])
        manifest.append(dict(protocol=job["protocol"], domain=job["domain"], group=job["group"],
            source_ids=job["source"].id.tolist(), calibration_ids=job["calibration"].id.tolist(), evaluation_ids=job["test"].id.tolist()))
    (out / "splits_before_fit.json").write_text(json.dumps(dict(jobs=manifest, skipped=skipped), indent=2), encoding="utf-8")
    # Gate old baseline reproduction independently of the scaled ablation.
    expected = pd.read_csv(inputs[6], dtype={"id": str}).set_index("id").loc[target.id]
    if not np.array_equal(expected.label.to_numpy(), target.label.to_numpy()):
        raise ValueError("Reference labels changed")
    raw = R.lightgbm_model().fit(train[columns], train.label).predict(target[columns]).astype(int)
    if not np.array_equal(raw, expected.screen_road_detail_lgb.to_numpy()):
        raise ValueError("Frozen327 photo baseline reproduction failed")
    records, diagnostics = [], []
    for k, job in enumerate(jobs):
        source, cal, test = job["source"], job["calibration"], job["test"]
        for family in FAMILIES:
            predictions, diagnostic = fit_predictions(source, cal[columns].to_numpy(), test[columns].to_numpy(), columns, family)
            diagnostics.append(dict(protocol=job["protocol"], domain=job["domain"], group=job["group"], family=family,
                                    source_rows=len(source), calibration_rows=len(cal), evaluation_rows=len(test), **diagnostic))
            for method, prediction in predictions.items():
                for row, value in zip(test.to_dict("records"), prediction):
                    records.append(dict(protocol=job["protocol"], domain=job["domain"], capture=job["group"], id=row["id"],
                        evaluation=row.get("evaluation", "original_camera"), label=int(row["label"]), family=family, method=method, pred=int(value)))
        print(f'Completed{ k+1}/{len(jobs)} {job["protocol"]} capture', flush=True)
    predictions = pd.DataFrame(records)
    if predictions.duplicated(["protocol", "id", "family", "method"]).any():
        raise ValueError("Evaluation photo assigned twice")
    predictions.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(diagnostics).to_csv(out / "alignment_diagnostics.csv", index=False, encoding="utf-8-sig")
    scores, intervals = [], []
    for (protocol, regime, family), part in predictions.groupby(["protocol", "evaluation", "family"], sort=False):
        wide = part.pivot(index=["id", "capture", "label"], columns="method", values="pred").reset_index().rename(columns={"capture": "video"})
        for method in METHODS:
            scores.append(dict(protocol=protocol, evaluation=regime, domain="pooled", family=family, method=method, rows=len(wide), **score(wide.label, wide[method])))
        for candidate in ["mean_only", "coral"]:
            intervals.append(dict(protocol=protocol, evaluation=regime, family=family, **A.paired_interval(wide, "source_only", candidate)))
    for (domain, family, method), part in predictions[predictions.protocol.eq("camera_adaptation")].groupby(["domain", "family", "method"]):
        scores.append(dict(protocol="camera_adaptation", evaluation="original_camera", domain=domain, family=family, method=method, rows=len(part), **score(part.label, part.pred)))
    for regime, part in target.groupby("evaluation", sort=False):
        scores.append(dict(protocol="legacy_raw_reproduction", evaluation=regime, domain="pooled", family="lightgbm", method="raw_source", rows=len(part), **score(part.label, raw[part.index])))
        scores.append(dict(protocol="trivial_reference", evaluation=regime, domain="pooled", family="always_normal", method="always_normal", rows=len(part), **score(part.label, np.zeros(len(part), int))))
    pd.DataFrame(scores).to_csv(out / "scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(intervals).to_csv(out / "paired_group_intervals.csv", index=False, encoding="utf-8-sig")
    baseline = predictions[predictions.protocol.eq("newvideo_adaptation") & predictions.family.eq("lightgbm") & predictions.method.eq("source_only")].set_index("id").loc[target.id]
    scaled_differences = int((baseline.pred.to_numpy() != raw).sum())
    if any(R.digest(p) != value for p, value in (hashes | image_hashes).items()):
        raise ValueError("Source/inputs/images changed during run")
    result = dict(status="complete", jobs=len(jobs), skipped=skipped, baseline_reproduced=327,
        scaled_vs_raw_baseline_differences=scaled_differences, intervals=intervals,
        limitations=spec["limitations"], interval_status="Posthoc frozen-prediction capture composition,2000resamples seed0,no refitting",
        hashes=hashes, image_hashes=image_hashes)
    (out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "report.html").write_text('<!doctype html><meta charset="utf-8"><title>Disjoint CORAL development ablation</title><style>body{font:16px system-ui;margin:24px}td,th{padding:5px;border:1px solid #ccc}</style><h1>Disjoint calibration/evaluation captures</h1><p>Reused development data,not original320CV. Source-only scaling,ridge1,no target-label selection. Single-capture domain skipped.</p>'+pd.DataFrame(scores).round(4).to_html(index=False)+'<h2>Posthoc composition intervals</h2>'+pd.DataFrame(intervals).round(4).to_html(index=False), encoding="utf-8")
    print(pd.DataFrame(scores)[["protocol", "evaluation", "domain", "family", "method", "macro_f1", "recall_2"]].round(5).to_string(index=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.out)
