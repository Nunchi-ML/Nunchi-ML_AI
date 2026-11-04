import importlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from nunchi_ml.glare import visibility as V
from nunchi_ml.glare.features import FEATURES


@pytest.mark.parametrize("value", [0, 127, 255])
def test_uniform_image_is_finite_and_has_no_blob_or_road_structure(value):
    f = V.extract_array(np.full((368, 640, 3), value, dtype=np.uint8))
    assert np.isfinite(list(f.values())).all()
    assert f["peak_area"] == f["peak_roundness"] == f["road_gradient_relative"] == f["road_laplacian_relative"] == 0
    assert f["road_clipped_fraction"] == (value == 255)


def test_circle_and_strip_have_different_roundness():
    circle = np.full((368, 640, 3), 30, dtype=np.uint8)
    strip = circle.copy()
    cv2.circle(circle, (320, 110), 25, (230, 230, 230), -1)
    cv2.rectangle(strip, (80, 100), (560, 120), (230, 230, 230), -1)
    a, b = V.extract_array(circle), V.extract_array(strip)
    assert a["peak_roundness"] > .8 and a["peak_roundness"] > b["peak_roundness"]
    # A flat saturated disc has multiple maxima: the peak need not be its center.
    assert np.hypot(a["peak_x"]*640-320, a["peak_y"]*368-110) <= 25


def test_relative_road_features_reduce_gain_effect_and_respond_to_blur():
    im = np.full((368, 640, 3), 40, dtype=np.uint8)
    for x in range(128, 512, 16):
        im[220:313, x:x+8] = 80
    original = V.extract_array(im)
    gain = V.extract_array(im*2)
    blur = V.extract_array(cv2.GaussianBlur(im, (25, 25), 4))
    assert gain["road_contrast_relative"] == pytest.approx(original["road_contrast_relative"], abs=.003)
    assert gain["road_gradient_relative"] == pytest.approx(original["road_gradient_relative"], rel=.01)
    assert blur["road_laplacian_relative"] < original["road_laplacian_relative"]


@pytest.mark.parametrize("bad", [None, np.zeros((0, 2, 3), np.uint8), np.zeros((2, 2), np.uint8), np.zeros((2, 2, 3))])
def test_invalid_image_rejected(bad):
    with pytest.raises(ValueError):
        V.extract_array(bad)


def probe(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "notebooks/experiments/glare_video_source"
    monkeypatch.syspath_prepend(str(path))
    return importlib.import_module("g_visibility_probe")


def test_decision_uses_fixed_thresholds_not_argmax(monkeypatch):
    p = probe(monkeypatch)
    scores = np.array([[.4, .3, .3], [.5, .4, .1], [.6, .2, .2], [.2, .5, .3]])
    assert p.decision(scores).tolist() == [2, 1, 0, 1]


def test_image_cache_recomputes_changed_bytes_and_preserves_ids(monkeypatch, tmp_path):
    p = probe(monkeypatch)
    path, cache = tmp_path / "frame.png", tmp_path / "cache.csv"
    cv2.imwrite(str(path), np.zeros((368, 640, 3), np.uint8))
    snapshot = pd.DataFrame([dict(id="exact_id", label=0, group="trip", camera="phone", **dict.fromkeys(FEATURES, 0.))])
    before = p.features(snapshot, [path], cache)
    cv2.imwrite(str(path), np.full((368, 640, 3), 255, np.uint8))
    after = p.features(snapshot, [path], cache)
    assert before.image_sha256.iloc[0] != after.image_sha256.iloc[0]
    assert before.road_clipped_fraction.iloc[0] == 0 and after.road_clipped_fraction.iloc[0] == 1
    assert after.id.tolist() == snapshot.id.tolist()
    pd.testing.assert_frame_equal(p.features(snapshot, [path], cache), after, check_dtype=False)


def test_probabilities_support_missing_severe_class_in_training_fold(monkeypatch):
    p = probe(monkeypatch)
    train = pd.DataFrame(dict(x=[0., .1, 1., 1.1], label=[0, 0, 1, 1]))
    result = p.probabilities(train, pd.DataFrame(dict(x=[.5, 2.])), ["x"])
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result.sum(1), 1)
    assert (result[:, 2] == 0).all()


def test_absent_positive_class_is_not_reported_as_perfect_recall(monkeypatch):
    p = probe(monkeypatch)
    assert np.isnan(p.score([0, 0], [0, 0])["alert_recall"])


def test_blind_bundle_contains_no_source_ids_labels_or_camera_metadata(monkeypatch, tmp_path):
    import json
    import re
    probe(monkeypatch)
    review = importlib.import_module("prepare_visibility_review")
    image = tmp_path / "original.png"
    cv2.imwrite(str(image), np.full((60, 100, 3), 40, np.uint8))
    snapshot = pd.DataFrame(dict(id=["secret_id1", "secret_id2"], camera=["cameraA", "cameraB"],
                                  group=["private_day1", "private_day2"], label=[0, 2]))
    monkeypatch.setattr(review, "image_paths", lambda rows, root: [image]*len(rows))
    bundle, count = review.build(snapshot, tmp_path, tmp_path / "out", per_camera=1)
    html = (bundle / "review.html").read_text(encoding="utf-8")
    payload = json.loads(re.search(r"const payload=(.*);", html).group(1))
    assert count == 2 and all(set(r) == {"review_id", "image"} for r in payload["records"])
    assert not any(value in html for value in ["secret_id", "private_day", "cameraA", "cameraB"])
    blank = pd.read_csv(bundle / "blank_annotations.csv")
    assert blank[review.ANSWER_COLUMNS].isna().all().all()
    assert len(list((bundle / "assets").glob("*.jpg"))) == 2
    assert (tmp_path / "out/private_review_mapping.csv").is_file()
    with pytest.raises(FileExistsError):
        review.build(snapshot, tmp_path, tmp_path / "out", per_camera=1)
    original_order = review.sample(snapshot, 1, 47).id.tolist()
    changed_labels_order = review.sample(snapshot.assign(label=[2, 0]), 1, 47).id.tolist()
    assert original_order == changed_labels_order
    first_manifest = re.search(r'const manifest="([^"]+)"', html).group(1)
    cv2.imwrite(str(image), np.full((60, 100, 3), 120, np.uint8))
    changed_bundle, _ = review.build(snapshot, tmp_path, tmp_path / "different_images", per_camera=1)
    changed_html = (changed_bundle / "review.html").read_text(encoding="utf-8")
    assert first_manifest != re.search(r'const manifest="([^"]+)"', changed_html).group(1)
