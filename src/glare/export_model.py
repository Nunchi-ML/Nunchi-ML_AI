"""재현 가능한 시범 모델·메타데이터·전송 기준 JPEG 3장을 export한다.

python -m src.glare.export_model --images ../../data/glare --features ../../data/glare/pilot_glare/glare_features.csv --labels ../../data/glare/pilot_glare/pilot_labels.csv --out models/model2-pilot-v1
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import struct

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupKFold

from .features import FEATURES, PREPROCESSING, extract, extract_array
from .inference import GlarePredictor, features_sha256, sha256
from .pilot_eval import brightness_rule, cross_val, lightgbm_model, load


def transport_jpeg(image):
    """FE 규약: 전체 화면, 긴 변 1280 이하, JPEG quality 0.95, crop 없음."""
    height, width = image.shape[:2]
    scale = min(1.0, 1280 / max(width, height))
    if scale < 1:
        image = cv2.resize(image, (round(width * scale), round(height * scale)))
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok or encoded.nbytes > 1024 * 1024:
        raise ValueError("Reference JPEG exceeds transport limits")
    return encoded


def metrics(y, pred):
    return {"macro_f1": float(f1_score(y, pred, average="macro")),
            "severe_recall": float(recall_score(y, pred, labels=[2], average="macro")),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()}


def export(images, features, labels, out):
    d = load(features, labels)
    files = sorted((images / "Images").glob("*/*/*.png"))
    index = {(p.parts[-3], p.name): p for p in files}
    if len(index) != len(files):
        raise ValueError("Duplicate image keys on disk")
    fresh, jpeg_rows, reference_bytes = [], [], []
    for row in d.itertuples():
        path = index[(row.vid, row.file)]
        fresh.append(extract(path))
        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        encoded = transport_jpeg(image)
        reference_bytes.append(encoded.tobytes())
        jpeg_rows.append(extract_array(cv2.imdecode(encoded, cv2.IMREAD_COLOR)))
    actual = pd.DataFrame(fresh)[FEATURES]
    if not np.allclose(actual, d[FEATURES], atol=1e-7, rtol=1e-7):
        raise ValueError("Stored features differ from current full-frame preprocessing")
    transported = pd.DataFrame(jpeg_rows)[FEATURES]
    y = d.label.to_numpy()
    raw_pred = np.zeros(len(d), dtype=int)
    jpeg_pred = np.zeros(len(d), dtype=int)
    folds = []
    for tr, te in GroupKFold(5).split(d, y, d.vid):
        m = lightgbm_model().fit(d.iloc[tr][FEATURES], y[tr])
        raw_pred[te] = m.predict(d.iloc[te][FEATURES])
        jpeg_pred[te] = m.predict(transported.iloc[te])
        folds.append({"train_videos": sorted(d.iloc[tr].vid.unique().tolist()),
                      "test_videos": sorted(d.iloc[te].vid.unique().tolist()),
                      "test_image_keys": d.iloc[te][["vid", "file"]].to_dict("records")})
    baseline = cross_val(d, brightness_rule("mean_all"))
    model = lightgbm_model().fit(d[FEATURES], y)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model2.joblib")
    dimensions = Counter()
    for path in files:
        with path.open("rb") as stream:
            header = stream.read(24)
        w, h = struct.unpack(">II", header[16:24])
        dimensions[f"{w}x{h}"] += 1
    versions = {}
    for package in ["numpy", "pandas", "scikit-learn", "lightgbm", "joblib"]:
        versions[package] = importlib.metadata.version(package)
    versions["opencv"] = cv2.__version__
    versions["python"] = platform.python_version()
    meta = {
        "model_version": "model2-pilot-v1", "status": "pilot",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "features": FEATURES, "classes": [0, 1, 2], "preprocessing": PREPROCESSING,
        "transport": {"format": "JPEG", "quality": 0.95, "max_long_edge": 1280,
                      "max_bytes": 1048576, "crop": "none", "mirror": False},
        "source": {"dataset": "GLARE", "country": "US", "license": "CC BY 4.0",
                   "url": "https://github.com/NicholasCG/GLARE_Dataset",
                   "image_count": len(files), "image_dimensions": dict(dimensions)},
        "labels": {"count": len(d), "videos": int(d.vid.nunique()),
                   "class_counts": {str(k): int((y == k).sum()) for k in [0, 1, 2]},
                   "origin": "Claude-assisted pilot labels; independent human validation pending",
                   "sha256": sha256(labels)},
        "validation": {"method": "GroupKFold", "group": "vid", "n_splits": 5,
                       "raw": metrics(y, raw_pred), "jpeg95": metrics(y, jpeg_pred),
                       "brightness_baseline": metrics(y, baseline), "folds": folds},
        "training": {"fit": "all pilot labels after out-of-fold evaluation",
                     "parameters": model.get_params(), "features_csv_sha256": sha256(features)},
        "versions": versions, "scores_calibrated": False,
        "features_sha256": features_sha256(),
        "model_sha256": sha256(out / "model2.joblib"),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    predictor = GlarePredictor(out)
    refs = out / "reference"
    refs.mkdir(exist_ok=True)
    expected = []
    for label in [0, 1, 2]:
        i = int(np.flatnonzero(y == label)[0])
        path = refs / f"class-{label}.jpg"
        path.write_bytes(reference_bytes[i])
        decoded = cv2.imdecode(np.frombuffer(reference_bytes[i], np.uint8), cv2.IMREAD_COLOR)
        expected.append({"file": path.name, "sha256": sha256(path), "source_label": label,
                         "expected": predictor.predict(decoded)})
    (refs / "expected.json").write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "validation": {k: meta["validation"][k]
                      for k in ["raw", "jpeg95", "brightness_baseline"]}}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["images", "features", "labels", "out"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    export(args.images, args.features, args.labels, args.out)
