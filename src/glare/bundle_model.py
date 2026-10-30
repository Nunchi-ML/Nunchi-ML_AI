"""Export와 공통 wheel을 SHA-256으로 고정한 배포용 zip으로 묶는다."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np

from .inference import GlarePredictor, sha256


def bundle(directory, output):
    directory, output = Path(directory), Path(output)
    predictor = GlarePredictor(directory)
    wheels = list(directory.glob("nunchi_glare-*-py3-none-any.whl"))
    if len(wheels) != 1:
        raise ValueError("Exactly one nunchi-glare wheel is required")
    with zipfile.ZipFile(wheels[0]) as wheel:
        features_hash = hashlib.sha256(wheel.read("nunchi_glare/features.py")).hexdigest()
    if features_hash != predictor.meta["features_sha256"]:
        raise ValueError("Wheel preprocessing differs from exported model")
    references = json.loads((directory / "reference/expected.json").read_text(encoding="utf-8"))
    if len(references) != 3:
        raise ValueError("Three reference images are required")
    files = [directory / "model2.joblib", directory / "meta.json", wheels[0],
             directory / "reference/expected.json"]
    for item in references:
        name = item["file"]
        if Path(name).name != name or not name.endswith(".jpg"):
            raise ValueError("Invalid reference name")
        path = directory / "reference" / name
        if sha256(path) != item["sha256"]:
            raise ValueError("Reference checksum mismatch")
        actual = predictor.predict(cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR))
        if actual != item["expected"]:
            raise ValueError("Reference prediction mismatch")
        files.append(path)
    manifest = {
        "format": "nunchi-model2-bundle-v1", "model_version": predictor.meta["model_version"],
        "wheel": wheels[0].name,
        "files": {p.relative_to(directory).as_posix(): sha256(p) for p in files},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for path in files:
            archive.write(path, path.relative_to(directory).as_posix())
    digest = sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    print(json.dumps({"bundle": str(output), "sha256": digest,
                      "model_version": manifest["model_version"], "files": len(files)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    bundle(args.directory, args.output)
