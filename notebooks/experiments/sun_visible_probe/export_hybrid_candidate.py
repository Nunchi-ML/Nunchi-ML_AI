"""Freeze existing photo+G candidates; verify export against saved experiments.

No model selection, new thresholds, roadside accuracy claim or server change.
Outputs contain fitted parameters and stay in the existing private data tree.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "glare_video_source"))

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import conditional_augmentation_probe as A
from rel_features import REL, rel_array as experiment_relative
from nunchi_ml.glare import hybrid as H
from nunchi_ml.glare.relative import rel_array


def fit(frame, columns, target):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight="balanced", max_iter=3000))
    model.fit(frame[columns], target)
    return model


def conditional(train, positive, geometry_b=False):
    a_cols, b_cols = H.G + H.RELATIVE, H.G + (H.EXTRA if geometry_b else [])
    a = fit(train, a_cols, train.label.ge(1))
    b = fit(positive, b_cols, positive.label.eq(2))
    return dict(kind="conditional", a=H.pack_linear(a, a_cols), b=H.pack_linear(b, b_cols))


def run(data, out):
    started = time.perf_counter()
    data, out = Path(data).resolve(), Path(out).resolve()
    if out.exists() or not out.is_relative_to(data):
        raise ValueError("Use a new private output under data")
    runtime_hashes = {name: H.digest(Path(H.__file__).with_name(name)) for name in ("hybrid.py", "relative.py")}
    base = data / "road_visibility_20261104/run30"
    previous = data / "conditional_augmentation_20261105"
    report = json.loads((previous / "report.json").read_text(encoding="utf-8"))
    hashes = report["input_sha256"]
    if report["status"] != "complete" or any(H.digest(p) != v for p, v in hashes.items()):
        raise ValueError("Saved reference inputs or experiment source changed")
    raw = pd.read_csv(base / "training_features.csv", dtype={"id": str})
    full = H.geometry(raw)
    original = full[full.camera.ne("AI Hub")].reset_index(drop=True)
    auxiliary = full[full.camera.eq("AI Hub")].reset_index(drop=True)
    if len(original) != 320 or len(auxiliary) != 304 or original.group.nunique() != 24:
        raise ValueError("Frozen original320 and auxiliary304 required")
    A.training_pool(original, auxiliary, original.iloc[:0])
    print("1/4 checking original photo bytes and all624 relative feature rows", flush=True)
    max_feature_error = 0.
    paths = A.V.M.R.C.image_paths(full, data)
    image_hashes = {}
    for row, path in zip(full.itertuples(), paths):
        expected_hash = row.image_sha256
        if H.digest(path) != expected_hash:
            raise ValueError("Original photo bytes changed")
        image_hashes[str(path)] = expected_hash
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Cannot decode frozen original photo")
        packaged, old = rel_array(image), experiment_relative(image)
        a, b = np.array([packaged[c] for c in REL]), np.array([old[c] for c in REL])
        c = np.array([getattr(row, "r_" + name) for name in REL])
        if not np.array_equal(a, b) or not np.allclose(a, c, atol=1e-10, rtol=1e-10):
            raise ValueError("Packaged extraction differs from frozen experiment")
        max_feature_error = max(max_feature_error, float(abs(a - c).max()))
    np.testing.assert_allclose(full[H.EXTRA], A.V.geometry(raw)[H.EXTRA], atol=1e-12, rtol=0)
    print("2/4 checking5 saved OOF folds and3 held-camera exports", flush=True)
    saved = pd.read_csv(previous / "binary_probabilities.csv", dtype={"id": str})
    frozen = pd.read_csv(previous / "predictions.csv", dtype={"id": str})
    splits = pd.read_csv(base / "splits.csv", dtype={"id": str})
    part = splits[splits["repeat"].eq(0)]
    if len(part) != 320 or not part.id.is_unique or set(part.id) != set(original.id):
        raise ValueError("Saved OOF split coverage mismatch")
    folds = [("group_cv320", 0, f, original[~original.id.isin(part[part.fold.eq(f)].id)],
              original[original.id.isin(part[part.fold.eq(f)].id)]) for f in range(5)]
    folds += [("held_camera:" + camera, -1, -1, original[original.camera.ne(camera)],
               original[original.camera.eq(camera)]) for camera in original.camera.unique()]
    max_probability_error, reproduced_rows = 0., 0
    for regime, repeat, fold, train, test in folds:
        pool = A.training_pool(train, auxiliary, test)
        model = conditional(train, pool[pool.label.ge(1)], geometry_b=True)
        predicted = H.predict_variant(test, model)
        old = saved[saved.test.eq(regime) & saved["repeat"].eq(repeat) & saved.fold.eq(fold)]
        if not old.id.is_unique or set(old.id) != set(test.id):
            raise ValueError("Saved probability coverage mismatch")
        old = old.set_index("id").loc[test.id]
        for name, column in [("alert_score", "source_a"), ("conditional_severe_score", "geometry_b")]:
            error = float(abs(predicted[name] - old[column].to_numpy()).max())
            max_probability_error = max(max_probability_error, error)
            np.testing.assert_allclose(predicted[name], old[column], atol=1e-10, rtol=0)
        reference = frozen[frozen.test.eq(regime) & frozen["repeat"].eq(repeat)
                           & frozen.method.eq("B_geometry_aug")].set_index("id").loc[test.id]
        if not np.array_equal(predicted["level"], reference.pred.to_numpy()):
            raise ValueError("Saved final decisions changed during export")
        reproduced_rows += len(test)
    print("3/4 fitting unchanged candidates and controls, exporting JSON parameters", flush=True)
    g = fit(original, H.G, original.label)
    models = dict(g_only=dict(kind="multiclass", a=H.pack_linear(g, H.G)),
                  conditional_original=conditional(original, original[original.label.ge(1)]),
                  B_geometry_aug=conditional(original, full[full.label.ge(1)], geometry_b=True))
    if (any(H.digest(p) != v for p, v in hashes.items())
            or any(H.digest(p) != v for p, v in image_hashes.items())
            or any(H.digest(Path(H.__file__).with_name(p)) != v for p, v in runtime_hashes.items())):
        raise ValueError("Frozen source/data/photo changed during export")
    out.mkdir(parents=True)
    (out / "parameters.json").write_text(json.dumps(models, indent=2, allow_nan=False), encoding="utf-8")
    meta = dict(format=H.VERSION, status="candidate", roadside_validated=False,
                parameters_sha256=H.digest(out / "parameters.json"),
                runtime_sha256=runtime_hashes,
                versions={p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "opencv-python-headless")},
                decision="A>=.5, then B>=.5; never argmax of joint scores",
                training=dict(A_original_rows=320, A_original_groups=24,
                              augmented_B_positive_rows=int(full.label.ge(1).sum()),
                              original_B_positive_rows=int(original.label.ge(1).sum())),
                preserved_input_sha256=hashes,
                extraction=dict(photos_checked=624, source_vs_package_exact=True,
                                cache_max_absolute_error=max_feature_error),
                export_replay=dict(OOF_repeats=[0], OOF_folds=5, held_cameras=3,
                                   rows=reproduced_rows, max_probability_error=max_probability_error),
                limitations=["Export equivalence is not new performance improvement or an independent test",
                             "Roadside images, direction/coverage, current/future validation and human final test unavailable",
                             "No final selection, serving replacement or calibrated driver-risk probabilities"],
                elapsed_seconds=time.perf_counter() - started)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    bundle = H.HybridCandidate(out)
    for name in H.VARIANTS:
        expected = H.predict_variant(full, models[name])
        actual = bundle.predict_features(full, name)
        np.testing.assert_array_equal(expected["level"], actual["level"])
        np.testing.assert_allclose(expected["joint_scores"], actual["joint_scores"], atol=0, rtol=0)
    print("4/4 roundtrip verified:3 variants x624 rows; candidate status only", flush=True)
    print(json.dumps({k: meta[k] for k in ("status", "roadside_validated", "extraction", "export_replay", "elapsed_seconds")}, ensure_ascii=False), flush=True)
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    run(args.data, args.out)
