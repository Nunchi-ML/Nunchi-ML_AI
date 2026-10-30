"""API와 export 검수가 같이 쓰는 추론. 팀이 만든 로컬 모델만 로드한다."""
import hashlib
import json
import importlib.metadata
import importlib.resources
from pathlib import Path

import joblib
import cv2
import numpy as np
import pandas as pd

from .features import FEATURES, PREPROCESSING, extract_array

LABELS = ["정상", "눈부심", "심한 눈부심"]
SCORE_KEYS = ["normal", "glare", "severe"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def features_sha256():
    source = importlib.resources.files(__package__).joinpath("features.py").read_bytes()
    return hashlib.sha256(source).hexdigest()


class GlarePredictor:
    def __init__(self, directory):
        directory = Path(directory)
        self.meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if (self.meta["features"] != FEATURES or self.meta["classes"] != [0, 1, 2]
                or self.meta["preprocessing"] != PREPROCESSING
                or self.meta["status"] != "pilot"):
            raise ValueError("Incompatible model metadata")
        model_path = directory / "model2.joblib"
        if sha256(model_path) != self.meta["model_sha256"]:
            raise ValueError("Model checksum mismatch")
        if features_sha256() != self.meta["features_sha256"]:
            raise ValueError("Feature implementation mismatch")
        for package in ["numpy", "pandas", "scikit-learn", "lightgbm", "joblib"]:
            if importlib.metadata.version(package) != self.meta["versions"][package]:
                raise ValueError(f"Runtime version mismatch: {package}")
        if cv2.__version__ != self.meta["versions"]["opencv"]:
            raise ValueError("Runtime version mismatch: opencv")
        # joblib is executable serialization; never load user uploads or arbitrary URLs.
        self.model = joblib.load(model_path)
        if (list(self.model.classes_) != [0, 1, 2]
                or list(self.model.feature_name_) != FEATURES):
            raise ValueError("Incompatible model classes or feature order")
        self.predict(np.zeros((32, 48, 3), dtype=np.uint8))

    def predict(self, image):
        features = extract_array(image)
        frame = pd.DataFrame([[features[k] for k in FEATURES]], columns=FEATURES)
        scores = self.model.predict_proba(frame)[0]
        if scores.shape != (3,) or not np.isfinite(scores).all():
            raise ValueError("Invalid model output")
        level = int(np.argmax(scores))
        return {
            "model_version": self.meta["model_version"], "status": "pilot",
            "level": level, "label": LABELS[level],
            "scores": {k: float(v) for k, v in zip(SCORE_KEYS, scores)},
            "brightest": {"x": float(features["brightest_x"]),
                          "y": float(features["brightest_y"])},
        }
