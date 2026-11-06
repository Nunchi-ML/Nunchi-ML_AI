"""Portable model3 parity; synthetic inputs are never an accuracy benchmark."""
import importlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("model3_export_probe")


@pytest.fixture
def training(module):
    rng = np.random.default_rng(9)
    frame = pd.DataFrame(rng.uniform(.1, .9, (90, 12)), columns=module.FEATURES)
    frame["sun_alt"] *= 70
    frame["cloud"] *= 10
    frame["y"] = np.log10(np.resize([500., 1000., 5000.], len(frame)))
    frame["cls"] = "입구"
    frame["scene"] = np.arange(len(frame)) // 3
    frame.loc[::7, "portal_ratio"] = np.nan
    return frame


@pytest.fixture
def bundle(module, training, tmp_path):
    notebook = module.NOTEBOOKS / "10_등급2단계_로지스틱.ipynb"
    formula = module.NOTEBOOKS / "07_밝기추정_규칙vsML.ipynb"
    parameters, features, parity, _ = module.fit_locked(training, notebook, formula)
    out = tmp_path / "model3"
    out.mkdir()
    (out / "parameters.json").write_text(json.dumps(parameters), encoding="utf-8")
    (out / "preprocess.py").write_text(module.selected_source(formula, ["photo_features", "sun_altitude"]), encoding="utf-8")
    meta = dict(format=module.VERSION, variant=module.VARIANT, status="candidate", features=features,
                ridge_alpha=10, logistic_C=.01, brightness_training="scene_group_5fold_OOF",
                supported_zones=["entry"], parameters_sha256=module.digest(out / "parameters.json"),
                preprocess_sha256=module.digest(out / "preprocess.py"), runtime_source_sha256=module.source_digest(module.__file__))
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return out, parameters, parity


def test_frozen_notebook_and_json_inference_are_equivalent(module, training, bundle):
    out, parameters, parity = bundle
    expected = module.predict_parameters(training, parameters, module.FEATURES)
    restored = module.Model3Candidate(out).predict_features(training, zone="entry")
    assert parity["all_grades_equal"] and parity["max_probability_error"] < 1e-10
    assert not parity["training_parity_is_external_accuracy"]
    for key, value in expected.items():
        np.testing.assert_array_equal(restored[key], value)
    np.testing.assert_allclose(restored["joint_scores"].sum(axis=1), 1)
    assert restored["missing_feature_count"].max() == 1


@pytest.mark.parametrize("name", ["parameters.json", "preprocess.py"])
def test_corrupted_bundle_rejected_before_prediction(module, bundle, name):
    out, _, _ = bundle
    with (out / name).open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="checksum"):
        module.Model3Candidate(out)


@pytest.mark.parametrize("key,value", [("features", list(reversed(range(12)))), ("ridge_alpha", 1),
    ("logistic_C", 1), ("brightness_training", "in_sample"), ("supported_zones", ["entry", "exit"]),
    ("runtime_source_sha256", "0" * 64)])
def test_wrong_feature_training_or_source_contract_rejected(module, bundle, key, value):
    out, _, _ = bundle
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    meta[key] = value
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError):
        module.Model3Candidate(out)


def test_exit_and_bad_numeric_inputs_cannot_be_normal(module, training, bundle):
    out, _, _ = bundle
    candidate = module.Model3Candidate(out)
    for zone in ["exit", "road", "unknown"]:
        with pytest.raises(ValueError, match="entrance"):
            candidate.predict_features(training, zone=zone)
    for frame in [training.iloc[:0], training.drop(columns=[module.FEATURES[0]]),
                  training.assign(sun_alt=np.inf), training.assign(sun_alt=91), training.assign(cloud=-1)]:
        with pytest.raises(ValueError):
            candidate.predict_features(frame, zone="entry")


def test_same_image_uses_original_photo_extractor_and_model3_sun_formula(module, bundle, tmp_path):
    out, _, _ = bundle
    image = np.full((240, 400, 3), 180, np.uint8)
    image[80:190, 150:250] = 15
    path = tmp_path / "entrance.jpg"
    assert cv2.imwrite(str(path), image)
    candidate = module.Model3Candidate(out)
    metadata = dict(lat=35.8, lon=128.6, cloud=3., captured_at="2026-11-06T09:00:00+09:00")
    actual = candidate.predict_image(path, metadata, zone="entry")
    features = candidate.preprocess["photo_features"](path)
    altitude = candidate.preprocess["sun_altitude"](pd.Series([pd.Timestamp("2026-11-06 09:00:00")]), [35.8], [128.6])[0]
    features.update(sun_alt=altitude, cloud=3.)
    expected = candidate.predict_features(pd.DataFrame([features]), zone="entry")
    for key in ["level", "estimated_luminance_cd_m2", "alert_score", "conditional_high_score", "joint_scores"]:
        np.testing.assert_array_equal(actual[key], expected[key])
    for changes in [dict(cloud=True), dict(lat=91), dict(captured_at="2026-11-06T09:00:00")]:
        with pytest.raises(ValueError):
            candidate.predict_image(path, {**metadata, **changes}, zone="entry")


def test_only_selected_notebook_declarations_execute(module):
    notebook = module.NOTEBOOKS / "10_등급2단계_로지스틱.ipynb"
    code = module.selected_source(notebook, ["FEATURES", "PLUS", "ridge"])
    assert "read_csv" not in code and "to_csv" not in code and "C_MAIN" not in code
    with pytest.raises(ValueError, match="missing"):
        module.selected_source(notebook, ["nonexistent"])
