"""Frozen-row image-proxy ablation. No threshold search, HTTP or model export.

Compare G, G+upper peak, G+road, G+both and G+old screen12. Every model uses
the SAME multinomial balanced logistic and .5/.5 rule; no argmax baseline.
Actual occlusion/legibility annotations are absent and never invented here.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nunchi_ml.glare import visibility as V
from nunchi_ml.glare.features import FEATURES, extract_array as old_features
from g_weather_time_probe import G_COLUMNS, metrics, replay

METHODS = {"G": G_COLUMNS, "G+sun_proxy": G_COLUMNS+V.SUN,
           "G+road_proxy": G_COLUMNS+V.ROAD, "G+both": G_COLUMNS+V.SUN+V.ROAD,
           "G+screen12_control": G_COLUMNS+FEATURES}


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def image_paths(snapshot, data_root):
    daegu = pd.concat([pd.read_csv(data_root / "glare_mapillary" / name, dtype={"id": str})[["id", "file"]]
                       for name in ("labels_drive.csv", "labels_dash.csv")])
    if not daegu.id.is_unique:
        raise ValueError("Ambiguous Daegu frame IDs")
    lookup = daegu.set_index("id").file.to_dict()
    paths = []
    for d in snapshot.itertuples():
        if d.camera == "AI Hub":
            path = data_root / "aihub_special" / d.file
        elif d.camera == "서울 폰":
            path = data_root / "glare_mapillary_seoul" / d.file
        elif d.camera in {"도로 광각", "블랙박스"}:
            path = data_root / "glare_mapillary" / lookup[d.id]
        else:
            raise ValueError(f"Unknown camera: {d.camera}")
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


def features(snapshot, paths, cache):
    """Check actual image bytes + feature source/version before reusing a row."""
    if len(paths) != len(snapshot):
        raise ValueError("Expected exactly one image path per snapshot row")
    source = sha256(Path(V.__file__).read_bytes())
    old_source = sha256((Path(V.__file__).parent / "features.py").read_bytes())
    previous = pd.read_csv(cache, dtype={"id": str}) if cache.exists() else pd.DataFrame()
    if len(previous) and not previous.id.is_unique:
        raise ValueError("Duplicate IDs in feature cache")
    previous = previous.set_index("id").to_dict("index") if len(previous) else {}

    def one(item):
        image_id, path = item
        raw = path.read_bytes()
        identity = dict(image_sha256=sha256(raw), feature_source_sha256=source,
                        old_feature_source_sha256=old_source, feature_version=V.VERSION)
        existing = previous.get(image_id, {})
        if all(existing.get(k) == value for k, value in identity.items()):
            return dict(id=image_id, **existing)
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        return dict(id=image_id, **identity, **V.extract_array(image), **old_features(image))

    with ThreadPoolExecutor(max_workers=4) as pool:
        frame = pd.DataFrame(pool.map(one, zip(snapshot.id, paths)))
    if not np.isfinite(frame[V.SUN+V.ROAD+V.DIAGNOSTICS+FEATURES].to_numpy(float)).all():
        raise ValueError("Missing/nonfinite image features; do not drop photos")
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False, encoding="utf-8-sig")
    # Original screen control is recomputed from the same images as new inputs.
    merged = snapshot.drop(columns=FEATURES).merge(frame, on="id", how="left", validate="one_to_one", sort=False)
    pd.testing.assert_frame_equal(merged[["id", "label", "group", "camera"]], snapshot[["id", "label", "group", "camera"]])
    return merged


def probabilities(train, test, columns):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    model.fit(train[columns], train.label)
    p = np.zeros((len(test), 3))
    p[:, model.classes_] = model.predict_proba(test[columns])
    return p


def decision(p):
    return np.where(p[:, 1:].sum(1) >= .5, np.where(p[:, 2] >= p[:, 1], 2, 1), 0)


def score(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    return {**metrics(y, prediction), "alert_recall": float((prediction[y >= 1] >= 1).mean()) if (y >= 1).any() else np.nan,
            "normal_count": int((y == 0).sum()), "glare_count": int((y >= 1).sum()),
            "false_alert_count": int(((y == 0) & (prediction >= 1)).sum())}


def evaluate(data, repeats):
    train = data[data.camera.ne("AI Hub")].reset_index(drop=True)
    test = data[data.camera.eq("AI Hub")].reset_index(drop=True)
    rows, predictions = [], []

    def record(method, kind, repeat, subset, p):
        pred = decision(p)
        rows.append(dict(method=method, test=kind, repeat=repeat, **score(subset.label, pred)))
        predictions.extend(dict(method=method, test=kind, repeat=repeat, id=d.id,
                                label=int(d.label), pred=int(q), p0=float(prob[0]), p1=float(prob[1]), p2=float(prob[2]))
                           for d, q, prob in zip(subset.itertuples(), pred, p))
        if kind == "reused_AIHub":
            for area in test.area.unique():
                mask = test.area.eq(area).to_numpy()
                rows.append(dict(method=method, test=f"AIHub_area:{area}", repeat=-1,
                                 **score(subset.loc[mask, "label"], pred[mask])))

    for method, columns in METHODS.items():
        record(method, "reused_AIHub", -1, test, probabilities(train, test, columns))
        for camera in train.camera.unique():
            held = train.camera.eq(camera)
            record(method, f"held_camera:{camera}", -1, train[held], probabilities(train[~held], train[held], columns))
        for seed in range(repeats):
            p = np.zeros((len(train), 3))
            for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(train, train.label, train.group):
                if set(train.iloc[tr].group) & set(train.iloc[te].group):
                    raise AssertionError("Capture group leakage")
                p[te] = probabilities(train.iloc[tr], train.iloc[te], columns)
            record(method, "group_cv", seed, train, p)
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=30)
    args = p.parse_args()
    if args.repeats < 1:
        p.error("--repeats must be positive")
    snapshot = pd.read_csv(args.snapshot, dtype={"id": str})
    if snapshot.id.isna().any() or not snapshot.id.is_unique:
        raise ValueError("Snapshot IDs must be present and unique")
    snapshot = replay(snapshot, args.data_root / "asos", "previous")
    if not np.isfinite(snapshot[G_COLUMNS].to_numpy(float)).all():
        raise ValueError("Missing G inputs; do not drop rows")
    paths = image_paths(snapshot, args.data_root)
    data = features(snapshot, paths, args.out / "image_features.csv")
    print(f"Frozen data: {len(data)} photos, {data.group.nunique()} groups, image features complete", flush=True)
    scores, predictions = evaluate(data, args.repeats)
    scores.to_csv(args.out / "scores.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(args.out / "predictions.csv", index=False, encoding="utf-8-sig")
    info = dict(snapshot_sha256=sha256(args.snapshot.read_bytes()), version=V.VERSION, methods=METHODS,
                thresholds=[.5, .5], weather="previous", projection="frozen_legacy", repeats=args.repeats,
                sklearn=sklearn.__version__, opencv=cv2.__version__, rows=len(data), groups=int(data.group.nunique()),
                sources={p.name: sha256(p.read_bytes()) for p in [Path(__file__), Path(V.__file__),
                         Path(__file__).parent / "g_weather_time_probe.py"]},
                weather_sha256={p.name: sha256(p.read_bytes()) for p in sorted((args.data_root / "asos").glob("*.csv"))},
                image_manifest_sha256=sha256((args.out / "image_features.csv").read_bytes()),
                limitation="Image proxies only, no occlusion/legibility truth. AIHub is a reused development set. No model selection/export.")
    (args.out / "method.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(scores.groupby(["method", "test"])[["macro_f1", "alert_accuracy", "false_alert_rate", "alert_recall",
                                               "severe_precision", "severe_recall"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
