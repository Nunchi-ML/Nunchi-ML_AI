"""Portable, non-pickle photo+G candidates. Existing pilot serving is unchanged.

This runtime reproduces the fixed .5/.5 conditional experiment. A candidate
bundle is deliberately never marked final by fitting it on all available rows.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .relative import REL, rel_array

VERSION = "photo-g-candidate-v1"
G = ["sun_alt", "u", "v", "in_fov", "cloud", "sunshine"]
RELATIVE = ["r_" + name for name in REL]
EXTRA = ["gap", "theta", "lv"]
VARIANTS = {
    "g_only": (G, G, "multiclass"),
    "conditional_original": (G + RELATIVE, G, "conditional"),
    "B_geometry_aug": (G + RELATIVE, G + EXTRA, "conditional"),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry(frame):
    """Same derived inputs as frozen rule_ml_validation.geometry, without fitting."""
    out = frame.copy()
    alt = np.radians(out.sun_alt.to_numpy(float).clip(-90, 90))
    cos_t = np.cos(alt) * np.cos(np.radians(out.gap.to_numpy(float)))
    theta = np.degrees(np.arccos(np.clip(cos_t, -1, 1)))
    out["theta"] = theta
    okta = out.cloud.to_numpy(float).clip(0, 10) * .8
    e = 1000 * np.sin(alt.clip(min=0)) * (1 - .75 * (okta / 8) ** 3.4)
    out["lv"] = np.log10(10 * e / theta.clip(min=1) ** 2 + 1e-3)
    return out


def matrix(frame, columns):
    if not isinstance(frame, pd.DataFrame) or not len(frame) or not frame.columns.is_unique:
        raise ValueError("Nonempty feature table with unique columns required")
    if not set(columns) <= set(frame):
        raise ValueError("Missing model features: " + ", ".join(sorted(set(columns) - set(frame))))
    values = frame[columns].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Missing/nonfinite model inputs are not normal observations")
    for name, low, high in [("sun_alt", -90, 90), ("cloud", 0, 10),
                            ("sunshine", 0, 1), ("in_fov", 0, 1), ("gap", 0, 180)]:
        if name in columns and not frame[name].between(low, high).all():
            raise ValueError("Model input out of range: " + name)
    if "in_fov" in columns and not frame.in_fov.isin([0, 1]).all():
        raise ValueError("in_fov must be 0 or 1")
    return values


def pack_linear(pipeline, columns):
    scaler, model = pipeline.steps[0][1], pipeline.steps[-1][1]
    return dict(features=list(columns), mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                coefficients=model.coef_.tolist(), intercept=model.intercept_.tolist(),
                classes=model.classes_.astype(int).tolist())


def linear_probabilities(frame, model):
    cols = model["features"]
    x = matrix(frame, cols)
    mean, scale = np.asarray(model["mean"], float), np.asarray(model["scale"], float)
    coefficients, intercept = np.asarray(model["coefficients"], float), np.asarray(model["intercept"], float)
    classes = model["classes"]
    k = 1 if classes == [0, 1] else 3
    if (classes not in ([0, 1], [0, 1, 2]) or mean.shape != (len(cols),)
            or scale.shape != mean.shape or coefficients.shape != (k, len(cols))
            or intercept.shape != (k,) or (scale <= 0).any()
            or not all(np.isfinite(v).all() for v in (mean, scale, coefficients, intercept))):
        raise ValueError("Invalid linear model parameters")
    score = ((x - mean) / scale) @ coefficients.T + intercept
    if not np.isfinite(score).all():
        raise ValueError("Inputs overflow the fitted linear model")
    if k == 1:
        # Stable sigmoid, including large finite logits.
        positive = np.exp(-np.logaddexp(0, -score[:, 0]))
        return np.column_stack([1 - positive, positive])
    score -= score.max(axis=1, keepdims=True)
    probability = np.exp(score)
    return probability / probability.sum(axis=1, keepdims=True)


def predict_variant(frame, model):
    if model["kind"] == "multiclass":
        probability = linear_probabilities(frame, model["a"])
        a = probability[:, 1:].sum(axis=1)
        b = np.divide(probability[:, 2], a, out=np.zeros(len(a)), where=a > 0)
        pred = np.where(a >= .5, np.where(probability[:, 2] >= probability[:, 1], 2, 1), 0)
    elif model["kind"] == "conditional":
        a, b = (linear_probabilities(frame, model[name])[:, 1] for name in ("a", "b"))
        probability = np.column_stack([1 - a, a * (1 - b), a * b])
        pred = np.where(a >= .5, np.where(b >= .5, 2, 1), 0)
    else:
        raise ValueError("Unknown model kind")
    return dict(level=pred, joint_scores=probability, alert_score=a, conditional_severe_score=b)


class HybridCandidate:
    def __init__(self, directory):
        directory = Path(directory)
        self.meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if (self.meta.get("format") != VERSION or self.meta.get("status") != "candidate"
                or self.meta.get("roadside_validated") is not False):
            raise ValueError("Only explicitly unvalidated candidate bundles are supported")
        if digest(directory / "parameters.json") != self.meta["parameters_sha256"]:
            raise ValueError("Model parameter checksum mismatch")
        for name in ("hybrid.py", "relative.py"):
            if digest(Path(__file__).with_name(name)) != self.meta["runtime_sha256"][name]:
                raise ValueError("Candidate runtime source mismatch")
        self.models = json.loads((directory / "parameters.json").read_text(encoding="utf-8"))
        if set(self.models) != set(VARIANTS):
            raise ValueError("Missing candidate/control variants")
        for name, (a, b, kind) in VARIANTS.items():
            model = self.models[name]
            if model["kind"] != kind or model["a"]["features"] != a:
                raise ValueError("Variant feature contract mismatch")
            if kind == "conditional" and model["b"]["features"] != b:
                raise ValueError("Variant B feature contract mismatch")
            if model["a"]["classes"] != ([0, 1, 2] if kind == "multiclass" else [0, 1]):
                raise ValueError("Variant A class contract mismatch")
            if kind == "conditional" and model["b"]["classes"] != [0, 1]:
                raise ValueError("Variant B class contract mismatch")
            predict_variant(pd.DataFrame([{col: 0. for col in set(a + b)}]), model)

    def predict_features(self, frame, variant="B_geometry_aug"):
        if variant not in self.models:
            raise ValueError("Unknown candidate variant")
        result = predict_variant(frame, self.models[variant])
        return dict(model_version=VERSION, status="candidate", roadside_validated=False,
                    variant=variant, **result)

    def predict_image(self, image, g_features, variant="B_geometry_aug"):
        """Use G derived for THIS photo. Route/ETA matching is a separate operation."""
        if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3
                or image.dtype != np.uint8 or min(image.shape[:2]) < 1):
            raise ValueError("Nonempty uint8 BGR photo required")
        values = dict(g_features)
        # Photo features cannot be injected by caller or replaced with label values.
        values.update({"r_" + k: v for k, v in rel_array(image).items()})
        return self.predict_features(geometry(pd.DataFrame([values])), variant)


def main():
    import argparse
    import cv2
    from .roadside import photo_g

    parser = argparse.ArgumentParser(description="Offline photo+G candidate inference; no serving changes")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--variant", choices=list(VARIANTS), default="B_geometry_aug")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.image.read_bytes()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode photo")
    context = json.loads(args.context.read_text(encoding="utf-8-sig"))
    prepared = photo_g(context, image.shape)
    result = HybridCandidate(args.bundle).predict_image(image, prepared["features"], args.variant)
    result = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in result.items()}
    result.update(image_sha256=hashlib.sha256(raw).hexdigest(), geometry_provenance=prepared,
                  interpretation="Current photo candidate; not future route or driver-risk probability")
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
