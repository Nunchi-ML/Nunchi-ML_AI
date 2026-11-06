"""Frozen notebook model3 export: Ridge10 + OOF brightness + balanced LR .01.

No tuning, new labels, exit validation, physiological/driver-risk claim, or API.
Only known declarations are loaded from the versioned notebook, never its cells
that run searches, read unrelated files, or write experiments.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "model3-ridge-two-stage-candidate-v1"
VARIANT = "ridge_estimated_luminance_two_stage"
HERE = Path(__file__).resolve().parent
NOTEBOOKS = HERE.parent / "tunnel_photo_brightness"
LABELS = np.array(["하", "중", "상"])
FEATURES = ["sky_mean", "blue_sky", "white_ratio", "shadow_contrast", "road_mean",
            "sky_road_ratio", "portal_ratio", "saturation", "spread", "mean_all", "sun_alt", "cloud"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def selected_source(path, names):
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    chosen = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            name = node.name if isinstance(node, ast.FunctionDef) else (
                node.targets[0].id if isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) else None)
            if name in names:
                chosen.append(node)
    found = {node.name if isinstance(node, ast.FunctionDef) else node.targets[0].id for node in chosen}
    if found != set(names) or len(found) != len(chosen):
        raise ValueError("Frozen notebook declaration missing or duplicated")
    return ast.unparse(ast.Module(body=chosen, type_ignores=[])) + "\n"


def notebook_contract(notebook, formula):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    names = ["FEATURES", "PLUS", "gr", "X", "two_stage", "logit", "ridge", "add_est", "MAIN_COLS", "main_fp"]
    code = selected_source(formula, ["ratio_from_L20", "grade"]) + selected_source(notebook, names)
    env = dict(np=np, pd=pd, SimpleImputer=SimpleImputer, LogisticRegression=LogisticRegression,
               Ridge=Ridge, GroupKFold=GroupKFold, make_pipeline=make_pipeline, StandardScaler=StandardScaler)
    exec(compile(code, "<versioned-model3-declarations>", "exec"), env)
    if env["PLUS"] != FEATURES or env["MAIN_COLS"] != FEATURES + ["est_L"]:
        raise ValueError("Frozen model3 feature contract changed")
    return env, code


def pack(pipeline):
    imputer, scaler, model = pipeline.steps[0][1], pipeline.steps[1][1], pipeline.steps[2][1]
    return dict(median=imputer.statistics_.tolist(), mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                coef=np.asarray(model.coef_).reshape(-1).tolist(), intercept=float(np.asarray(model.intercept_).reshape(-1)[0]))


def linear(values, model):
    n = values.shape[1]
    arrays = {key: np.asarray(model[key], float) for key in ["median", "mean", "scale", "coef"]}
    if any(v.shape != (n,) or not np.isfinite(v).all() for v in arrays.values()) or (arrays["scale"] <= 0).any():
        raise ValueError("Invalid model3 parameters")
    if not np.isfinite(model["intercept"]) or np.isinf(values).any():
        raise ValueError("Invalid model3 numeric input")
    filled = np.where(np.isnan(values), arrays["median"], values)
    z = ((filled - arrays["mean"]) / arrays["scale"]) @ arrays["coef"] + model["intercept"]
    if not np.isfinite(z).all():
        raise ValueError("Model3 overflow")
    return z


def predict_parameters(frame, parameters, features):
    if not isinstance(frame, pd.DataFrame) or not len(frame) or not frame.columns.is_unique or not set(features) <= set(frame):
        raise ValueError("Explicit nonempty model3 features required")
    x = frame[features].to_numpy(float)
    for name, low, high in [("sun_alt", -90, 90), ("cloud", 0, 10)]:
        values = frame[name].to_numpy(float)
        if ((values < low) | (values > high)).any():
            raise ValueError("Model3 context feature out of range")
    estimated = linear(x, parameters["ridge"])
    z1 = linear(np.column_stack([x, estimated]), parameters["stage1"])
    z2 = linear(np.column_stack([x, estimated]), parameters["stage2"])
    a, b = [np.exp(-np.logaddexp(0, -z)) for z in [z1, z2]]
    with np.errstate(over="ignore", under="ignore"):
        luminance = 10. ** estimated
    if not np.isfinite(luminance).all() or (luminance <= 0).any():
        raise ValueError("Estimated luminance out of numeric range")
    return dict(level=np.where(z1 > 0, np.where(z2 > 0, 2, 1), 0),
                estimated_log10_luminance=estimated, estimated_luminance_cd_m2=luminance,
                alert_score=a, conditional_high_score=b,
                joint_scores=np.column_stack([1-a, a*(1-b), a*b]),
                missing_feature_count=np.isnan(x).sum(axis=1))


def fit_locked(train, notebook, formula):
    env, code = notebook_contract(notebook, formula)
    features = env["PLUS"]
    if not set(features + ["y", "cls", "scene"]) <= set(train) or train.scene.nunique() < 5:
        raise ValueError("Training features, targets and five scene groups required")
    if train.scene.isna().any() or not np.isfinite(train.y.to_numpy(float)).all() or train.cls.eq("출구").any():
        raise ValueError("Invalid frozen entrance/road training rows")
    if np.isinf(train[features].to_numpy(float)).any() or train[features].isna().all().any():
        raise ValueError("Missing training column or infinite feature")
    # Original add_est generates train brightness with scene-held-out folds.
    # main_fp trains both classifiers on those OOF brightness values.
    reference, stage1, stage2 = env["main_fp"](.01)(train, train, return_models=True)
    ridge = env["ridge"]().fit(train[features], train.y)
    if stage1[-1].classes_.tolist() != [False, True] or stage2[-1].classes_.tolist() != [False, True]:
        raise ValueError("Unexpected binary class order")
    parameters = dict(ridge=pack(ridge), stage1=pack(stage1), stage2=pack(stage2))
    predicted = predict_parameters(train, parameters, features)
    reference_est = ridge.predict(train[features])
    test_frame = train.copy()
    test_frame["est_L"] = reference_est
    ref_scores = [m.predict_proba(test_frame[env["MAIN_COLS"]].to_numpy(float))[:, 1] for m in [stage1, stage2]]
    if not np.array_equal(LABELS[predicted["level"]], reference):
        raise ValueError("Notebook and portable model3 grades differ")
    np.testing.assert_allclose(predicted["estimated_log10_luminance"], reference_est, rtol=0, atol=1e-10)
    errors = []
    for key, expected in zip(["alert_score", "conditional_high_score"], ref_scores):
        np.testing.assert_allclose(predicted[key], expected, rtol=0, atol=1e-10)
        errors.append(float(abs(predicted[key] - expected).max()))
    return parameters, features, dict(rows=len(train), all_grades_equal=True,
        max_log10_luminance_error=float(abs(predicted["estimated_log10_luminance"] - reference_est).max()),
        max_probability_error=max(errors), training_parity_is_external_accuracy=False), code


class Model3Candidate:
    def __init__(self, directory):
        directory = Path(directory)
        self.meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if (self.meta["format"] != VERSION or self.meta["variant"] != VARIANT
                or self.meta["status"] != "candidate" or self.meta["supported_zones"] != ["entry"]
                or self.meta["ridge_alpha"] != 10 or self.meta["logistic_C"] != .01
                or self.meta["brightness_training"] != "scene_group_5fold_OOF"):
            raise ValueError("Model3 candidate contract mismatch")
        if (digest(directory / "parameters.json") != self.meta["parameters_sha256"]
                or source_digest(__file__) != self.meta["runtime_source_sha256"]
                or digest(directory / "preprocess.py") != self.meta["preprocess_sha256"]):
            raise ValueError("Model3 artifact/source checksum mismatch")
        self.parameters = json.loads((directory / "parameters.json").read_text(encoding="utf-8"))
        self.features = self.meta["features"]
        if self.features != FEATURES:
            raise ValueError("Model3 feature contract mismatch")
        import cv2
        self.preprocess = dict(np=np, pd=pd, cv2=cv2)
        exec(compile((directory / "preprocess.py").read_text(encoding="utf-8"), "<frozen-model3-preprocess>", "exec"), self.preprocess)
        predict_parameters(pd.DataFrame([{k: 0. for k in self.features}]), self.parameters, self.features)

    def predict_features(self, frame, *, zone):
        if zone != "entry":
            raise ValueError("Model3 only supports explicitly identified entrance photos")
        return dict(model_version=VERSION, variant=VARIANT, status="candidate", supported_zone="entry",
                    driver_visibility_validated=False, adaptation_time_validated=False,
                    **predict_parameters(frame, self.parameters, self.features))

    def predict_image(self, path, metadata, *, zone):
        # Use the original model3 NOAA approximation and image extractor. Do not
        # replace them with model2 pvlib/relative features or physical JPEG lux.
        if zone != "entry":
            raise ValueError("Model3 only supports explicitly identified entrance photos")
        for key in ["lat", "lon", "cloud"]:
            value = metadata[key]
            if isinstance(value, (bool, str)) or not np.isfinite(value):
                raise ValueError("Explicit model3 photo-time context required")
        if not -90 <= metadata["lat"] <= 90 or not -180 <= metadata["lon"] <= 180 or not 0 <= metadata["cloud"] <= 10:
            raise ValueError("Model3 context out of range")
        at = pd.Timestamp(metadata["captured_at"])
        if pd.isna(at) or at.tzinfo is None:
            raise ValueError("Explicit photo timezone required")
        features = self.preprocess["photo_features"](path)
        if features is None:
            raise ValueError("Cannot read model3 photo")
        local = at.tz_convert("Asia/Seoul").tz_localize(None)
        altitude = self.preprocess["sun_altitude"](pd.Series([local]), [metadata["lat"]], [metadata["lon"]])[0]
        features.update(sun_alt=float(altitude), cloud=float(metadata["cloud"]))
        return self.predict_features(pd.DataFrame([features]), zone=zone)


def export(train_path, source_notebook, out):
    train_path, out = Path(train_path), Path(out)
    notebook, formula = NOTEBOOKS / "10_등급2단계_로지스틱.ipynb", NOTEBOOKS / "07_밝기추정_규칙vsML.ipynb"
    # The supplied notebook may append structure comparisons. Its training
    # declarations must remain exactly the committed representative model.
    names = ["FEATURES", "PLUS", "gr", "X", "two_stage", "logit", "ridge", "add_est", "MAIN_COLS", "main_fp"]
    if selected_source(source_notebook, names) != selected_source(notebook, names):
        raise ValueError("Supplied and committed representative training code differ")
    before = {str(p): digest(p) for p in [train_path, Path(source_notebook), notebook, formula]}
    train = pd.read_csv(train_path)
    if len(train) != 3624 or train.test_ok.sum() != 1927 or train.duplicated(["source_zip", "file_name"]).any():
        raise ValueError("Frozen 3624 training rows and 1927 entrance rows required")
    parameters, features, parity, code = fit_locked(train, notebook, formula)
    if any(digest(p) != value for p, value in before.items()):
        raise ValueError("Model3 source changed during export")
    out.mkdir(parents=True, exist_ok=False)
    (out / "parameters.json").write_text(json.dumps(parameters, indent=2, allow_nan=False), encoding="utf-8")
    preprocess = selected_source(formula, ["photo_features", "sun_altitude"])
    (out / "preprocess.py").write_text(preprocess, encoding="utf-8", newline="\n")
    meta = dict(format=VERSION, variant=VARIANT, status="candidate", features=features, supported_zones=["entry"],
                parameters_sha256=digest(out / "parameters.json"), preprocess_sha256=digest(out / "preprocess.py"),
                runtime_source_sha256=source_digest(__file__), training_input_sha256=before,
                training_rows=len(train), training_scene_groups=int(train.scene.nunique()),
                ridge_alpha=10, logistic_C=.01, brightness_training="scene_group_5fold_OOF",
                missing_features="training_median_from_original_notebook", portability=parity,
                external_66_reference_reproduced=False, independent_joint_model_accuracy=False,
                versions={n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scikit-learn", "opencv-python-headless"]})
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    restored = Model3Candidate(out).predict_features(train, zone="entry")
    assert len(restored["level"]) == 3624
    print(json.dumps(dict(status="candidate_exported", portability=parity, external_reference_reproduced=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--source-notebook", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to((HERE.parents[2] / "data").resolve()):
        parser.error("Keep model3 coefficients in a new private AI/data directory")
    export(args.train, args.source_notebook, args.out)
