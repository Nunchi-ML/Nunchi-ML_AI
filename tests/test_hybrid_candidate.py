import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nunchi_ml.glare import hybrid as H


@pytest.mark.parametrize("classes", [2, 3])
def test_json_linear_export_matches_sklearn_on_heldout_inputs(classes):
    rng = np.random.default_rng(8)
    train = pd.DataFrame(rng.normal(size=(90, 4)), columns=list("abcd"))
    labels = np.arange(90) % classes
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
    model.fit(train, labels)
    held = pd.DataFrame(rng.normal(size=(17, 4)) * 4, columns=train.columns)
    packed = json.loads(json.dumps(H.pack_linear(model, list(train))))
    np.testing.assert_allclose(H.linear_probabilities(held, packed), model.predict_proba(held), atol=1e-14, rtol=0)


def constant(columns, score):
    return dict(features=columns, mean=[0.] * len(columns), scale=[1.] * len(columns),
                coefficients=[[0.] * len(columns)], intercept=[float(np.log(score / (1 - score)))],
                classes=[0, 1])


def test_conditional_threshold_is_not_joint_argmax():
    model = dict(kind="conditional", a=constant(["a"], .6), b=constant(["b"], .6))
    result = H.predict_variant(pd.DataFrame([dict(a=1., b=2.)]), model)
    assert result["level"].tolist() == [2]
    assert result["joint_scores"].argmax(axis=1).tolist() == [0]
    np.testing.assert_allclose(result["joint_scores"], [[.4, .24, .36]])


@pytest.mark.parametrize("change", [dict(scale=[0.]), dict(intercept=[float("nan")]),
                                    dict(classes=[0, 2]), dict(features=["missing"])])
def test_corrupt_model_or_incomplete_features_fail_closed(change):
    model = constant(["a"], .7)
    model.update(change)
    with pytest.raises(ValueError):
        H.linear_probabilities(pd.DataFrame([dict(a=1.)]), model)


def test_missing_or_nonfinite_geometry_is_not_a_normal_prediction():
    for values in ([0, 0, 0, 0, np.nan, 0], [0, 0, 0, 0, 11, 0], [0, 0, 0, .5, 0, 0]):
        with pytest.raises(ValueError):
            H.matrix(pd.DataFrame([values], columns=H.G), H.G)


def make_bundle(directory):
    models = {}
    for name, (a, b, kind) in H.VARIANTS.items():
        first = constant(a, .6)
        if kind == "multiclass":
            first.update(classes=[0, 1, 2], coefficients=[[0.] * len(a)] * 3, intercept=[0., 1., 2.])
        models[name] = dict(kind=kind, a=first, b=constant(b, .6))
    directory.mkdir()
    (directory / "parameters.json").write_text(json.dumps(models), encoding="utf-8")
    meta = dict(format=H.VERSION, status="candidate", roadside_validated=False,
                parameters_sha256=H.digest(directory / "parameters.json"),
                runtime_sha256={p: H.digest(Path(H.__file__).with_name(p)) for p in ("hybrid.py", "relative.py")})
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return directory


def test_bundle_roundtrip_tamper_and_final_promotion_are_rejected(tmp_path):
    directory = make_bundle(tmp_path / "bundle")
    bundle = H.HybridCandidate(directory)
    frame = pd.DataFrame([{c: 0. for c in H.G + H.RELATIVE + H.EXTRA}])
    result = bundle.predict_features(frame)
    assert result["status"] == "candidate" and result["roadside_validated"] is False
    path = directory / "meta.json"
    original = path.read_text(encoding="utf-8")
    path.write_text(original.replace('"candidate"', '"final"'), encoding="utf-8")
    with pytest.raises(ValueError, match="unvalidated"):
        H.HybridCandidate(directory)
    path.write_text(original, encoding="utf-8")
    with (directory / "parameters.json").open("a", encoding="utf-8") as f:
        f.write(" ")
    with pytest.raises(ValueError, match="checksum"):
        H.HybridCandidate(directory)
