"""Offline paired camera ablation on a frozen snapshot; no cache mutation/HTTP.

Keep the 6 G inputs, previous-hour weather, photos, groups and .5/.5 thresholds
fixed. Lens-only and SfM-pose variants are diagnostics, not production models.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold

from nunchi_ml.camera import PROJECTION_VERSION, mapillary_calibration, read_aihub_intrinsics
import fov_probe as F
from g_weather_time_probe import G_COLUMNS, metrics, predict, replay


def enrich(snapshot, metadata_dir, aihub_dir):
    """Use photo IDs and each frame's actual sibling calib.txt, never labels."""
    rows = []
    for _, d in snapshot.iterrows():
        if d.camera == "AI Hub":
            # Frame path: images/<area>/<...>/<drive>/image_0/<frame>.jpg
            calib = (aihub_dir / d.file).parent.parent / "calib.txt"
            values = read_aihub_intrinsics(calib)
            values["camera_type"] = values.pop("model")
            values.update(width=int(d.width), height=int(d.height))
        else:
            values = mapillary_calibration(json.loads(
                (metadata_dir / f"{int(d.id)}.json").read_text(encoding="utf-8")))
            if values.pop("id") != int(d.id):
                raise ValueError(f"Metadata ID mismatch for {d.id}")
            if (values["width"], values["height"]) != (d.width, d.height):
                raise ValueError(f"Snapshot/raw image size mismatch for {d.id}")
            diff = abs((values["pose_heading"] - d.heading + 180) % 360 - 180)
            # Heading convention check, fixed independently of labels/scores.
            if diff > 1e-5:
                raise ValueError(f"SfM/recorded heading mismatch for {d.id}")
        rows.append(values)
    meta = pd.DataFrame(rows, index=snapshot.index)
    return pd.concat([snapshot.drop(columns=snapshot.columns.intersection(meta.columns)), meta], axis=1)


def variants(snapshot, metadata_dir, aihub_dir):
    if snapshot.id.isna().any() or not snapshot.id.is_unique:
        raise ValueError("Snapshot IDs must be present and unique")
    calibrated = enrich(snapshot, metadata_dir, aihub_dir)
    result = {"legacy": snapshot.copy()}
    for mode in ("pinhole", "calibrated", "pose"):
        source = snapshot.assign(camera_type="perspective", k1=0., k2=0.) if mode == "pinhole" else calibrated
        features = F.fov_features(source, projection="pose" if mode == "pose" else "calibrated")
        d = calibrated.drop(columns=calibrated.columns.intersection(features.columns)).join(features)
        if d.heading_suspect.any():
            raise ValueError("Resolve suspect headings before this fixed-row ablation")
        # Outside the forward hemisphere: explicit fixed legacy G sentinel.
        # Unknown/invalid lens metadata raises above, rather than being imputed.
        d[["u", "v"]] = d[["u", "v"]].fillna({"u": 9., "v": -9.})
        result[mode] = d
    for mode, d in result.items():
        if not np.isfinite(d[G_COLUMNS].to_numpy(float)).all():
            raise ValueError(f"Incomplete G inputs: {mode}")
        pd.testing.assert_frame_equal(d[["id", "label", "group", "camera"]], snapshot[["id", "label", "group", "camera"]])
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--metadata-dir", type=Path, required=True)
    p.add_argument("--aihub-dir", type=Path, default=Path("data/aihub_special"))
    p.add_argument("--weather-dir", type=Path, default=Path("data/asos"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=30)
    args = p.parse_args()
    if args.repeats < 1:
        p.error("--repeats must be positive")
    snapshot = replay(pd.read_csv(args.snapshot), args.weather_dir, "previous")
    ds = variants(snapshot, args.metadata_dir, args.aihub_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    rows, predictions, audit = [], [], []
    for mode, d in ds.items():
        train = d[d.camera.ne("AI Hub")].reset_index(drop=True)
        test = d[d.camera.eq("AI Hub")]
        pred = predict(train, test)
        rows.append(dict(projection=mode, test="reused_AIHub", **metrics(test.label, pred)))
        predictions.extend(dict(projection=mode, test="reused_AIHub", repeat=-1,
                                id=i, label=int(y), pred=int(q)) for i, y, q in zip(test.id, test.label, pred))
        for camera in train.camera.unique():
            held = train.camera.eq(camera)
            pred = predict(train[~held], train[held])
            rows.append(dict(projection=mode, test=f"held_camera:{camera}", **metrics(train.loc[held, "label"], pred)))
            predictions.extend(dict(projection=mode, test=f"held_camera:{camera}", repeat=-1,
                                    id=i, label=int(y), pred=int(q))
                               for i, y, q in zip(train.loc[held, "id"], train.loc[held, "label"], pred))
        for seed in range(args.repeats):
            pred = np.zeros(len(train), dtype=int)
            for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(train, train.label, train.group):
                pred[te] = predict(train.iloc[tr], train.iloc[te])
            rows.append(dict(projection=mode, test="group_cv", repeat=seed, **metrics(train.label, pred)))
            predictions.extend(dict(projection=mode, test="group_cv", repeat=seed,
                                    id=i, label=int(y), pred=int(q)) for i, y, q in zip(train.id, train.label, pred))
        cols = [c for c in ("id", "camera", "camera_model", "pose_source", "pose_pitch", "u", "v", "in_fov",
                            "projection_valid", "projection_fingerprint") if c in d]
        d[cols].to_csv(args.out / f"geometry_{mode}.csv", index=False, encoding="utf-8-sig")
        audit.append(dict(projection=mode, rows=len(d), groups=int(d.group.nunique()),
                          in_fov=int(d.in_fov.sum()), changed_fov=int(d.in_fov.ne(snapshot.in_fov).sum()),
                          changed_coordinates=int((~np.isclose(d.u, snapshot.u) | ~np.isclose(d.v, snapshot.v)).sum())))
    scores = pd.DataFrame(rows)
    scores.to_csv(args.out / "scores.csv", index=False)
    pd.DataFrame(predictions).to_csv(args.out / "predictions.csv", index=False)
    info = dict(snapshot_sha256=digest(args.snapshot), sklearn=sklearn.__version__, repeats=args.repeats,
                weather="previous", thresholds=[.5, .5], audit=audit,
                projection_version=PROJECTION_VERSION,
                source_sha256={str(p.name): digest(p) for p in (
                    Path(__file__), Path(F.__file__),
                    Path(__file__).resolve().parents[3] / "src/nunchi_ml/camera.py")},
                metadata_sha256={p.name: digest(p) for p in sorted(args.metadata_dir.glob("*.json"))},
                calib_sha256={str(p.relative_to(args.aihub_dir)): digest(p) for p in sorted(args.aihub_dir.rglob("calib.txt"))},
                weather_sha256={p.name: digest(p) for p in sorted(args.weather_dir.glob("*.csv"))},
                limitation="Reused development set. SfM pose offline only; AIHub level assumed. No live validation.")
    (args.out / "method.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(scores.groupby(["projection", "test"])[list(metrics([0, 1, 2], np.array([0, 1, 2])))].mean().round(4).to_string())
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
