"""Frozen-region detail ablation: same labels, folds and score decisions.

No semantic detector, neural model, enhancement, tuning or deployment changes.
Both legacy and conditional OOF predictions must reproduce exactly.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pandas as pd

import conditional_training_probe as M
import forward_detail_features as F
import road_visibility_probe as R
from other_photo_compare import verify_reproduction
from ordinal_compare import metrics
from nunchi_ml.glare.features import extract_array, FEATURES as SCREEN
from rel_features import rel_array

NAMES = ["legacy", "conditional", "legacy_A_detail", "conditional_A_detail",
         "conditional_B_detail", "conditional_AB_detail"]


def predict_fold(train, test):
    a = R.G + R.P.REL_R
    pf, pg = R.C.probabilities(train, test, a), R.C.probabilities(train, test, R.G)
    p_extra = R.C.probabilities(train, test, a+F.FEATURES)
    old_a, old_b = pf[:, 1:].sum(1), pg[:, 2]/np.maximum(pg[:, 1:].sum(1), 1e-9)
    binary_a = M.binary_probability(train, test, a, train.label.ge(1).astype(int))
    extra_a = M.binary_probability(train, test, a+F.FEATURES, train.label.ge(1).astype(int))
    positive = train[train.label.ge(1)]
    binary_b = M.binary_probability(positive, test, R.G, positive.label.eq(2).astype(int))
    extra_b = M.binary_probability(positive, test, R.G+F.FEATURES, positive.label.eq(2).astype(int))
    return dict(legacy=M.compose(old_a, old_b), conditional=M.compose(binary_a, binary_b),
                legacy_A_detail=M.compose(p_extra[:, 1:].sum(1), old_b),
                conditional_A_detail=M.compose(extra_a, binary_b),
                conditional_B_detail=M.compose(binary_a, extra_b),
                conditional_AB_detail=M.compose(extra_a, extra_b))


def attach(frame, paths, out, name, full=False):
    if len(frame) != len(paths) or not frame.id.is_unique:
        raise ValueError("Unique matching image IDs required")

    def one(pair):
        row, path = pair
        raw = path.read_bytes()
        if M.sha(path) != row.image_sha256:
            raise ValueError("Photo bytes changed")
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        values = F.extract_array(image)
        if full:
            values.update(extract_array(image))
            values.update({"r_"+k: v for k, v in rel_array(image).items()})
            values.update(R.R.extract_array(image))
        return dict(id=row.id, **values)

    with ThreadPoolExecutor(max_workers=4) as pool:
        features = pd.DataFrame(pool.map(one, zip(frame.itertuples(), paths)))
    features.to_csv(out / (name+"_features.csv"), index=False, encoding="utf-8-sig")
    result = frame.merge(features, on="id", how="left", validate="one_to_one", sort=False)
    if not np.isfinite(result[F.FEATURES].to_numpy(float)).all():
        raise ValueError("Nonfinite detail features")
    return result


def group_intervals(train, records):
    lookup = train.set_index("id")
    groups = sorted(train.group.unique())
    gi, ni = {g: i for i, g in enumerate(groups)}, {n: i for i, n in enumerate(NAMES)}
    counts = np.zeros((24, 30, len(NAMES), 3, 3), int)
    for row in records:
        sample = lookup.loc[row["id"]]
        counts[gi[sample.group], row["repeat"], ni[row["method"]], int(sample.label), row["pred"]] += 1
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(2000):
        c = counts[rng.integers(0, 24, 24)].sum(axis=0)
        diagonal = np.diagonal(c, axis1=-2, axis2=-1)
        denominator = c.sum(axis=-2)+c.sum(axis=-1)
        f = np.divide(2*diagonal, denominator, out=np.zeros(diagonal.shape, float), where=denominator != 0).mean(axis=-1).mean(axis=0)
        draws.append(f-f[1])
    draws = np.asarray(draws)
    return [dict(method=n, baseline="conditional", low=float(np.quantile(draws[:, i], .025)), high=float(np.quantile(draws[:, i], .975))) for i, n in enumerate(NAMES)]


def run(data, out):
    data, out = Path(data).resolve(), Path(out).resolve()
    if out.exists() or not out.is_relative_to(data):
        raise ValueError("Use a new private output")
    old = data / "road_visibility_20261104/run30"
    extended = data / "other_photos_20261105/result_v1"
    inputs = [old / "training_features.csv", old / "human_features.csv", old / "splits.csv", old / "predictions.csv",
              data / "conditional_training_20261105/predictions.csv", extended / "predictions.csv",
              data / "new_cameras/human_review_50/labels_성진.csv", data / "new_cameras/labels_claude.csv"]
    sources = [Path(__file__), Path(F.__file__), Path(M.__file__), Path(R.__file__), Path(R.C.__file__), Path(R.R.__file__)]
    fingerprints = {str(p): M.sha(p) for p in inputs+sources}
    out.mkdir()
    for i, path in enumerate(sources):
        shutil.copyfile(path, out / f"source_{i:02d}_{path.name}")
    specification = dict(status="frozen_before_extraction_and_fit", version=F.VERSION, regions=F.REGIONS, features=F.FEATURES,
                         methods=NAMES, labels="Original labels unchanged", thresholds=[.5, .5], folds="Original frozen320/24capture,5folds x30",
                         tuning=False, neural_features=False, semantic_detector=False,
                         photo_methods=["screen_road_lgb", "screen_road_detail_lgb", "rel_road_logistic", "rel_road_detail_logistic"],
                         bootstrap="2000 seed0 capture resamples of frozenOOF; no refitting, conditional only",
                         hashes=fingerprints)
    (out / "specification_before_run.json").write_text(json.dumps(specification, indent=2), encoding="utf-8")
    full = pd.read_csv(inputs[0], dtype={"id": str})
    human = pd.read_csv(inputs[1], dtype={"id": str})
    previous = pd.read_csv(inputs[5], dtype={"id": str})
    others = previous[previous.evaluation.eq("outside277_AI_provisional")][["id", "file", "video", "label", "image_sha256"]].reset_index(drop=True)
    if (len(full), len(human), len(others)) != (624, 50, 277):
        raise ValueError("Preserved sample counts changed")
    combined_hashes = pd.concat([full.image_sha256, human.image_sha256, others.image_sha256])
    if combined_hashes.duplicated().any():
        raise ValueError("Image byte overlap")
    cv2.setNumThreads(1)
    F.masks()
    pack = (data / "new_cameras/label_pack").resolve()

    def photos(frame):
        result = [(pack / f).resolve() for f in frame.file]
        if any(not p.is_relative_to(pack) for p in result):
            raise ValueError("Image escapes source pack")
        return result

    full = attach(full, R.C.image_paths(full, data), out, "train")
    human = attach(human, photos(human), out, "human50")
    others = attach(others, photos(others), out, "other277", full=True)
    print("Verified951 image bytes and18 region features", flush=True)
    train = full[full.camera.ne("AI Hub")].reset_index(drop=True)
    if len(train) != 320 or train.group.nunique() != 24:
        raise ValueError("Original grouped training set changed")
    splits = pd.read_csv(inputs[2], dtype={"id": str})
    legacy = pd.read_csv(inputs[3], dtype={"id": str})
    legacy = legacy[legacy.test.eq("group_cv320") & legacy.method.eq("legacy")]
    conditional = pd.read_csv(inputs[4], dtype={"id": str})
    conditional = conditional[conditional.method.eq("binary_A_positive_B")]
    scores, records = [], []
    for repeat in range(30):
        part = splits[splits["repeat"].eq(repeat)]
        if len(part) != 320 or not part.id.is_unique or set(part.id) != set(train.id):
            raise ValueError("Frozen split mismatch")
        probabilities, seen = {}, np.zeros(320, int)
        for fold in range(5):
            mask = train.id.isin(part.loc[part.fold.eq(fold), "id"]).to_numpy()
            tr, va = train[~mask], train[mask]
            if set(tr.group) & set(va.group):
                raise ValueError("Capture group leakage")
            for name, p in predict_fold(tr, va).items():
                R.validate_probabilities(p)
                probabilities.setdefault(name, np.zeros((320, 3)))[mask] = p
            seen[mask] += 1
        if not np.all(seen == 1):
            raise ValueError("OOF coverage mismatch")
        for name, p in probabilities.items():
            pred = R.C.decision(p)
            if name in ["legacy", "conditional"]:
                ref = legacy if name == "legacy" else conditional
                expected = ref[ref["repeat"].eq(repeat)].set_index("id").loc[train.id, "pred"].to_numpy()
                if not np.array_equal(pred, expected):
                    raise ValueError("Required legacy/conditional baseline reproduction failed")
            scores.append(dict(test="group_cv320", repeat=repeat, method=name, rows=320, **metrics(train.label.to_numpy(), pred)))
            records.extend(dict(repeat=repeat, method=name, id=cid, pred=int(v)) for cid, v in zip(train.id, pred))
        pd.DataFrame(scores).to_csv(out / "scores.csv", index=False, encoding="utf-8-sig")
        print("forward detail repeat", repeat+1, "/30", flush=True)
    intervals = group_intervals(train, records)
    tests = [("reused_AIHub", train, full[full.camera.eq("AI Hub")])]
    tests += [("held_camera:"+str(c), train[train.camera.ne(c)], train[train.camera.eq(c)]) for c in train.camera.unique()]
    for name, tr, va in tests:
        for method, p in predict_fold(tr, va).items():
            scores.append(dict(test=name, repeat=-1, method=method, rows=len(va), **metrics(va.label.to_numpy(), R.C.decision(p))))
    test = pd.concat([human.assign(evaluation="human50"), others.assign(evaluation="other277_AI")], ignore_index=True)
    photo_predictions = {}
    for prefix, cols, family in [("screen_road", SCREEN+R.R.FEATURES, "lgb"), ("rel_road", R.P.REL_R+R.R.FEATURES, "logistic")]:
        for detail in [False, True]:
            name = prefix+("_detail" if detail else "")+("_lgb" if family == "lgb" else "_logistic")
            features = cols+(F.FEATURES if detail else [])
            if family == "lgb":
                model = R.lightgbm_model().fit(full[features], full.label)
                pred = model.predict(test[features]).astype(int)
            else:
                pred = R.C.decision(R.C.probabilities(full, test, features))
            photo_predictions[name] = pred
            if not detail:
                expected = previous.set_index("id").loc[test.id, name].to_numpy()
                if not np.array_equal(pred, expected):
                    raise ValueError("Required photo baseline reproduction failed")
            for regime in test.evaluation.unique():
                mask = test.evaluation.eq(regime).to_numpy()
                scores.append(dict(test=regime, repeat=-1, method=name, rows=int(mask.sum()), **metrics(test.loc[mask, "label"].to_numpy(), pred[mask])))
    frame = pd.DataFrame(scores)
    frame.to_csv(out / "scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(records).to_csv(out / "cv_predictions.csv", index=False, encoding="utf-8-sig")
    photo = test[["id", "video", "label", "evaluation"]].copy()
    for name, values in photo_predictions.items():
        photo[name] = values
    photo.to_csv(out / "photo_predictions.csv", index=False, encoding="utf-8-sig")
    numeric = [c for c in frame.select_dtypes("number") if c not in ["repeat", "rows"]]
    summary = frame.groupby(["test", "method"], sort=False)[numeric].mean().reset_index()
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    result = dict(status="complete", train_images=624, human_images=50, other_AI_images=277, new_features=18,
                  baseline_reproductions=dict(legacy_OOF=9600, conditional_OOF=9600, photo_predictions=654),
                  conditional_capture_intervals=intervals, hashes=fingerprints,
                  limitations=["Fixed regions/contours/corners do not identify object types or prove object existence/visibility",
                               "Low detail can mean no object, sky, road surface, exposure, blur or solar washout",
                               "Original320,AIHub304,human50,other277 are reused development data",
                               "FrozenOOF bootstrap reflects group composition, not refit uncertainty",
                               "Human/other photos have no G and are separate photo-only comparisons",
                               "No labels, thresholds, calibration, deployment or final model changed"])
    if any(M.sha(p) != v for p, v in fingerprints.items()):
        raise ValueError("Frozen input/source changed during experiment")
    (out / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report.html").write_text('<!doctype html><meta charset="utf-8"><title>Forward region detail ablation</title><style>body{font:16px system-ui;margin:24px}table{border-collapse:collapse}td,th{padding:5px;border:1px solid #ccc}</style><h1>Region detail, not semantic object visibility</h1><p>Same labels/folds/decisions. Legacy19200 OOF and654 photo predictions reproduced. All data reused; no final model.</p>'+summary.round(4).to_html(index=False)+'<h2>Capture composition intervals vs conditional</h2>'+pd.DataFrame(intervals).round(5).to_html(index=False), encoding="utf-8")
    print(summary[["test", "method", "macro_f1", "f1_1", "recall_2"]].round(5).to_string(index=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.out)
