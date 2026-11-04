"""Fixed-label road-detail ablation, separate from driver physiology/adaptation.

No neural features, label changes, calibration, tuning, deployment or export.
Compare the .6229 legacy protocol on the original 320 rows, plus photo-only
624->human50 development evaluation. Human50 has no G inputs. All feature sets
and decisions are specified here before execution; no selection on human50.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "glare_video_source"))

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

import ordinal_compare as O
import road_visibility_features as R
from new_camera_audit import load_labels
from nunchi_ml.glare.features import FEATURES, extract_array
from nunchi_ml.glare.pilot_eval import lightgbm_model
from nunchi_ml.glare.ordinal import CumulativeOrdinalClassifier
from rel_features import REL, rel_array

C, P, G = O.C, O.P, O.G_COLUMNS
METHODS = {
    "legacy": (P.REL_R, [], "legacy"),
    "road_alert": (P.REL_R + R.FEATURES, [], "legacy"),
    "road_severity": (P.REL_R, R.FEATURES, "legacy"),
    "road_both": (P.REL_R + R.FEATURES, R.FEATURES, "legacy"),
    "old_road_severity": (P.REL_R, C.V.ROAD, "legacy"),
    "ordinal_rel": (G + P.REL_R, [], "ordinal"),
    "ordinal_rel_road": (G + P.REL_R + R.FEATURES, [], "ordinal"),
}
PHOTO_METHODS = {
    "screen_lgb": (FEATURES, "lightgbm"),
    "screen_road_lgb": (FEATURES + R.FEATURES, "lightgbm"),
    "rel_logistic": (P.REL_R, "logistic"),
    "rel_road_logistic": (P.REL_R + R.FEATURES, "logistic"),
    "road_logistic": (R.FEATURES, "logistic"),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_probabilities(p):
    p = np.asarray(p, float)
    if (p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all()
            or (p < -1e-12).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-10)):
        raise ValueError("Invalid three-class probabilities")
    return p


def attach_road(data, paths, cache):
    if len(data) != len(paths) or not data.id.is_unique or data.id.isna().any():
        raise ValueError("Expected unique IDs and exactly one image per row")
    source = digest(R.__file__)
    previous = pd.read_csv(cache, dtype={"id": str}) if cache.exists() else pd.DataFrame()
    if len(previous) and (not previous.id.is_unique or previous.id.isna().any()):
        raise ValueError("Invalid road feature cache IDs")
    previous = previous.set_index("id").to_dict("index") if len(previous) else {}

    def one(pair):
        image_id, path = pair
        raw = path.read_bytes()
        identity = dict(image_sha256=hashlib.sha256(raw).hexdigest(),
                        road_source_sha256=source, road_version=R.VERSION)
        old = previous.get(image_id, {})
        if (all(old.get(k) == v for k, v in identity.items())
                and all(c in old for c in R.FEATURES)):
            return dict(id=image_id, **old)
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        return dict(id=image_id, **identity, **R.extract_array(image))

    with ThreadPoolExecutor(max_workers=4) as pool:
        fresh = pd.DataFrame(pool.map(one, zip(data.id, paths)))
    if not np.isfinite(fresh[R.FEATURES].to_numpy(float)).all():
        raise ValueError("Nonfinite road features; no row dropping")
    if "image_sha256" in data and not np.array_equal(data.image_sha256, fresh.image_sha256):
        raise ValueError("Image bytes changed between extractors")
    cache.parent.mkdir(parents=True, exist_ok=True)
    fresh.to_csv(cache, index=False, encoding="utf-8-sig")
    result = data.merge(fresh[["id", *R.FEATURES]], on="id", how="left", validate="one_to_one", sort=False)
    pd.testing.assert_frame_equal(result[list(data)], data.reset_index(drop=True))
    return result


def human_data(pack, label_path, training_hashes):
    sheet = pd.read_csv(pack / "label_pack/label_sheet_blind.csv", dtype={"id": str})[["id", "file"]]
    labels, _, info = load_labels([label_path], sheet.id, "human_single")
    fmap = pd.read_csv(pack / "private_frame_map.csv", dtype={"id": str})[["id", "video"]]
    if not fmap.id.is_unique or fmap.id.isna().any() or fmap.video.isna().any():
        raise ValueError("Invalid video map")
    data = sheet.merge(labels.reset_index(), on="id", validate="one_to_one").merge(
        fmap, on="id", how="left", validate="one_to_one")
    if len(data) != len(labels) or data.video.isna().any():
        raise ValueError("Every human label needs a mapped photo/video")
    paths, rows = [], []
    root = (pack / "label_pack").resolve()
    for d in data.itertuples():
        path = (root / d.file).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Photo path escapes blind pack")
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha in training_hashes:
            raise ValueError("Human evaluation image also appears in training")
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        rows.append(dict(image_sha256=sha, **extract_array(image),
                         **{"r_" + k: v for k, v in rel_array(image).items()}))
        paths.append(path)
    data = pd.concat([data.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    if data.image_sha256.duplicated().any():
        raise ValueError("Duplicate human evaluation photos")
    return data, paths, info


def legacy_fold(train, test):
    cache = {}
    def prob(columns):
        key = tuple(columns)
        if key not in cache:
            cache[key] = C.probabilities(train, test, list(columns))
        return cache[key]
    result = {}
    for name, (a, b, kind) in METHODS.items():
        violation = 0
        if kind == "ordinal":
            model = CumulativeOrdinalClassifier(O.make_estimator("logistic", 2), class_weight="balanced")
            model.fit(train[a], train.label.to_numpy())
            q = model.predict_cumulative_proba(test[a], corrected=False)
            violation = int((q[:, 0] < q[:, 1]).sum())
            p = model.predict_proba(test[a])
        else:
            pa, pb = prob(G + a), prob(G + b)
            alert = pa[:, 1:].sum(1)
            severe = pb[:, 2] / np.maximum(pb[:, 1:].sum(1), 1e-9)
            p = np.column_stack([1-alert, alert*(1-severe), alert*severe])
        result[name] = (validate_probabilities(p), violation)
    return result


def group_partitions(data, repeats):
    for seed in range(repeats):
        folds = list(GroupKFold(5, shuffle=True, random_state=seed).split(data, data.label, data.group))
        seen = np.zeros(len(data), int)
        for tr, va in folds:
            if set(data.iloc[tr].group) & set(data.iloc[va].group):
                raise AssertionError("Capture group leakage")
            seen[va] += 1
        if not np.all(seen == 1):
            raise AssertionError("Each row must have exactly one OOF prediction")
        yield seed, folds


def grouped_bootstrap(data, predictions, repeats):
    rng = np.random.default_rng(0)
    units = [np.flatnonzero(data.video.to_numpy() == v) for v in data.video.unique()]
    pairs = [("screen_lgb", "screen_road_lgb"), ("rel_logistic", "rel_road_logistic")]
    differences = {pair: [] for pair in pairs}
    y = data.label.to_numpy()
    for _ in range(repeats):
        ids = np.concatenate([units[i] for i in rng.integers(len(units), size=len(units))])
        for pair in pairs:
            base, added = [f1_score(y[ids], predictions[n][ids], labels=[0, 1, 2], average="macro", zero_division=0) for n in pair]
            differences[pair].append(added-base)
    return [dict(baseline=a, candidate=b, delta=float(f1_score(y, predictions[b], labels=[0, 1, 2], average="macro", zero_division=0)-f1_score(y, predictions[a], labels=[0, 1, 2], average="macro", zero_division=0)),
                 low=float(np.quantile(v, .025)), high=float(np.quantile(v, .975))) for (a, b), v in differences.items()]


def run(data, human, out, repeats, bootstrap):
    train = data[data.camera.ne("AI Hub")].reset_index(drop=True)
    stress = data[data.camera.eq("AI Hub")].reset_index(drop=True)
    scores, predictions, splits = [], [], []
    def record(name, kind, seed, subset, p, violations=0, pred=None):
        if pred is None:
            p = validate_probabilities(p)
            pred = p.argmax(1) if name.startswith("ordinal_") or name.endswith("lgb") else C.decision(p)
        row = dict(method=name, test=kind, repeat=seed, rows=len(subset), **O.metrics(subset.label, pred))
        row.update(alert_accuracy=float(np.mean((subset.label.to_numpy() >= 1) == (pred >= 1))),
                   raw_violations=violations, violation_rate=violations/len(subset))
        scores.append(row)
        for i, d in enumerate(subset.itertuples()):
            predictions.append(dict(method=name, test=kind, repeat=seed, id=d.id, label=int(d.label), pred=int(pred[i]),
                                    **({f"p{k}": float(p[i, k]) for k in range(3)} if p is not None else {})))
        return pred

    for seed, folds in group_partitions(train, repeats):
        probabilities = {name: np.zeros((len(train), 3)) for name in METHODS}
        violations = {name: 0 for name in METHODS}
        for fold, (tr, va) in enumerate(folds):
            for index in va:
                d = train.iloc[index]
                splits.append(dict(repeat=seed, fold=fold, id=d.id, group=d.group))
            for name, (p, count) in legacy_fold(train.iloc[tr], train.iloc[va]).items():
                probabilities[name][va] = p
                violations[name] += count
        for name in METHODS:
            record(name, "group_cv320", seed, train, probabilities[name], violations[name])
        print(f"group repeat {seed+1}/{repeats} complete", flush=True)
        pd.DataFrame(scores).to_csv(out / "scores.csv", index=False, encoding="utf-8-sig")

    for kind, tr, te in [("reused_AIHub", train, stress), *[("held_camera:"+cam, train[train.camera.ne(cam)], train[train.camera.eq(cam)]) for cam in train.camera.unique()]]:
        for name, (p, count) in legacy_fold(tr, te).items():
            record(name, kind, -1, te, p, count)

    photo_predictions = {}
    for name, (cols, family) in PHOTO_METHODS.items():
        if family == "lightgbm":
            model = lightgbm_model().fit(data[cols], data.label)
            p = np.zeros((len(human), 3))
            p[:, model.classes_] = model.predict_proba(human[cols])
        else:
            p = C.probabilities(data, human, cols)
        photo_predictions[name] = record(name, "human50_photo_only", -1, human, p)
    photo_predictions["always_normal"] = record("always_normal", "human50_photo_only", -1, human, None, pred=np.zeros(len(human), int))
    pd.DataFrame(grouped_bootstrap(human, photo_predictions, bootstrap)).to_csv(out / "human_paired_bootstrap.csv", index=False)
    scores = pd.DataFrame(scores)
    scores.to_csv(out / "scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(predictions).to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(splits).to_csv(out / "splits.csv", index=False, encoding="utf-8-sig")
    numeric = [c for c in scores.select_dtypes("number") if c not in ["repeat", "rows"]]
    summary = scores.groupby(["test", "method"], sort=False)[numeric].mean().reset_index()
    summary["macro_f1_sd"] = scores.groupby(["test", "method"], sort=False).macro_f1.std(ddof=0).to_numpy()
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    cv = scores[scores.test.eq("group_cv320")].pivot(index="repeat", columns="method", values="macro_f1")
    paired = pd.DataFrame([dict(method=n, mean_delta=float((cv[n]-cv.legacy).mean()), min_delta=float((cv[n]-cv.legacy).min()),
                                max_delta=float((cv[n]-cv.legacy).max()), improved_repeats=int((cv[n]>cv.legacy).sum())) for n in METHODS])
    paired.to_csv(out / "cv_paired.csv", index=False)
    (out / "report.html").write_text('<!doctype html><meta charset="utf-8"><title>Road detail ablation</title><style>body{font:15px system-ui;margin:32px}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ddd}</style><h1>Fixed-label road information-loss ablation</h1><p>Not measured driver vision, pupil dynamics, adaptation time or accident probability. Human50 is reused single-rater development evaluation; no G inputs there. No tuning or label changes. Repeated CV splits overlap; paired repeat differences are not independent confidence intervals.</p>' + summary.round(5).to_html(index=False) + '<h2>Paired CV differences vs legacy</h2>' + paired.round(5).to_html(index=False) + '<h2>Human50 paired video bootstrap</h2>' + pd.read_csv(out / "human_paired_bootstrap.csv").round(5).to_html(index=False), encoding="utf-8")
    print(summary[["test", "method", "macro_f1", "macro_f1_sd", "recall_2", "f1_1", "alert_accuracy"]].round(4).to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("data/road_visibility_20261104/run30"))
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args()
    if a.repeats < 1 or a.bootstrap < 1 or not a.out.resolve().is_relative_to(a.data_root.resolve()):
        ap.error("Positive repeats/bootstrap and private output inside data-root required")
    a.out.mkdir(parents=True, exist_ok=True)
    snapshot = a.data_root / "analysis_20261104/model2_snapshot.csv"
    rel = a.data_root / "sun_visible/rel_features.csv"
    cache = a.data_root / "analysis_20261104/step3_visibility/image_features.csv"
    pack = a.data_root / "new_cameras"
    labels = pack / "human_review_50/labels_성진.csv"
    files = [snapshot, rel, labels, pack / "label_pack/label_sheet_blind.csv", pack / "private_frame_map.csv"]
    before = {str(p): digest(p) for p in files}
    sources = [Path(__file__), Path(R.__file__), Path(O.__file__), Path(C.__file__), Path(P.__file__),
               HERE / "rel_features.py", HERE / "new_camera_audit.py", HERE / "ordinal_image_features.py",
               HERE.parent / "glare_video_source/g_weather_time_probe.py",
               Path(sys.modules[extract_array.__module__].__file__), Path(sys.modules[lightgbm_model.__module__].__file__),
               Path(sys.modules[CumulativeOrdinalClassifier.__module__].__file__), Path(C.V.__file__)]
    archived = a.out / "source_snapshot"
    archived.mkdir(exist_ok=True)
    for i, path in enumerate(sources):
        shutil.copyfile(path, archived / f"{i:02d}_{path.name}")
    method = dict(status="running", target="existing photo-based glare labels, not measured driver vision",
                  repeats=a.repeats, bootstrap=a.bootstrap, methods=METHODS, photo_methods=PHOTO_METHODS,
                  files=before, sources={str(p): digest(p) for p in sources},
                  versions={"opencv": cv2.__version__, **{n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scikit-learn", "lightgbm"]}},
                  limitations=["No relabeling; proxy target is not newly validated ground truth", "Human50 reused development set, single rater, no G", "AIHub reused development/stress set", "Fixed trapezoid is not semantic road segmentation", "Camera exposure/weather/road texture can confound features", "CV repeats overlap; no independent confidence claim"])
    def save():
        (a.out / "method.json").write_text(json.dumps(method, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    save()
    try:
        data = O.load_data(snapshot, rel, a.data_root, cache)
        data = attach_road(data, C.image_paths(data, a.data_root), a.out.parent / "road_train_cache.csv")
        human, paths, info = human_data(pack, labels, set(data.image_sha256))
        human = attach_road(human, paths, a.out.parent / "road_human_cache.csv")
        method.update(rows=len(data), cv_rows=int(data.camera.ne("AI Hub").sum()), cv_groups=int(data.loc[data.camera.ne("AI Hub"), "group"].nunique()), human_rows=len(human), human_videos=int(human.video.nunique()), label_evidence=info)
        data.to_csv(a.out / "training_features.csv", index=False, encoding="utf-8-sig")
        human.to_csv(a.out / "human_features.csv", index=False, encoding="utf-8-sig")
        print(f"Data verified: {len(data)} train catalogue, {len(human)} human rows", flush=True)
        run(data, human, a.out, a.repeats, a.bootstrap)
        if before != {str(p): digest(p) for p in files}:
            raise AssertionError("Input/label files changed during experiment")
        method["status"] = "complete"
        save()
    except BaseException as error:
        method.update(status="failed", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()
