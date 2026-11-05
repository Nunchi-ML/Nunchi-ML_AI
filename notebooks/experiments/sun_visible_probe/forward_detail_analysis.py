"""Posthoc paired photo-group intervals and fixed-region diagnostic images.

No refitting, tuning or label changes. Captures composition uncertainty only;
the existing human/AI development samples are not independent new test data.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import forward_detail_features as F
import road_visibility_probe as R


def run(data, root):
    data, root = Path(data).resolve(), Path(root).resolve()
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    if report["status"] != "complete":
        raise ValueError("Completed frozen model outputs required")
    out = root / "analysis"
    if out.exists():
        raise ValueError("New analysis output required")
    pairs = [("screen_road_lgb", "screen_road_detail_lgb"),
             ("rel_road_logistic", "rel_road_detail_logistic"),
             ("rel_road_logistic", "screen_road_detail_lgb")]
    out.mkdir()
    specification = dict(status="posthoc_after_point_scores", comparisons=pairs, bootstrap=2000, seed=0,
                         limitation="Frozen prediction/video composition conditional interval, no refitting; development samples reused",
                         diagnostic_selection="First catalogue sample per training camera, not model-error selection")
    (out / "specification_before_analysis.json").write_text(json.dumps(specification, indent=2), encoding="utf-8")
    frame = pd.read_csv(root / "photo_predictions.csv", dtype={"id": str})
    intervals = []
    for regime in frame.evaluation.unique():
        part = frame[frame.evaluation.eq(regime)]
        units = sorted(part.video.unique())
        group_index = {g: i for i, g in enumerate(units)}
        for baseline, candidate in pairs:
            counts = np.zeros((len(units), 2, 3, 3), int)
            for row in part.to_dict("records"):
                for j, method in enumerate([baseline, candidate]):
                    counts[group_index[row["video"]], j, int(row["label"]), int(row[method])] += 1
            rng = np.random.default_rng(0)
            draws = []
            for _ in range(2000):
                c = counts[rng.integers(0, len(units), len(units))].sum(axis=0)
                diagonal = np.diagonal(c, axis1=-2, axis2=-1)
                denominator = c.sum(axis=-2)+c.sum(axis=-1)
                f1 = np.divide(2*diagonal, denominator, out=np.zeros(diagonal.shape, float), where=denominator != 0).mean(axis=-1)
                draws.append(f1[1]-f1[0])
            intervals.append(dict(evaluation=regime, baseline=baseline, candidate=candidate, groups=len(units),
                                  low=float(np.quantile(draws, .025)), high=float(np.quantile(draws, .975))))
    pd.DataFrame(intervals).to_csv(out / "paired_photo_intervals.csv", index=False, encoding="utf-8-sig")
    training = pd.read_csv(data / "road_visibility_20261104/run30/training_features.csv", dtype={"id": str})
    selected = training.groupby("camera", sort=False).head(1)
    paths = R.C.image_paths(selected, data)
    canvas = np.full((len(selected)*408, 1280, 3), 245, np.uint8)
    for i, path in enumerate(paths):
        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, F.SIZE, interpolation=cv2.INTER_LINEAR)
        edge = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 25, 75)
        display = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        for name, rectangles in F.REGIONS.items():
            color = (255, 160, 0) if name == "front" else (0, 160, 255)
            for x0, y0, x1, y1 in rectangles:
                p, q = (round(x0*640), round(y0*368)), (round(x1*640), round(y1*368))
                cv2.rectangle(image, p, q, color, 2)
                cv2.rectangle(display, p, q, color, 2)
        canvas[i*408+40:i*408+408, :640] = image
        canvas[i*408+40:i*408+408, 640:] = display
        cv2.putText(canvas, f'Sample{i+1}: fixed regions / raw Canny, NO object detector', (10, i*408+28), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)
    cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(out / "region_diagnostics.jpg")
    result = dict(specification=specification, intervals=intervals, diagnostic_source_ids=selected.id.tolist(),
                  original_model_report_sha256=R.digest(root / "report.json"))
    (out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(pd.DataFrame(intervals).round(5).to_string(index=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.root)
