import importlib
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("forward_detail_features")


@pytest.mark.parametrize("brightness", [0, 127, 255])
def test_flat_regions_have_no_shape_evidence_but_do_not_produce_a_label(module, brightness):
    f = module.extract_array(np.full((368, 640, 3), brightness, np.uint8))
    assert list(f) == module.FEATURES and len(f) == 18
    assert f["fd_front_weak_edges"] == f["fd_front_corner_coverage"] == f["fd_front_contour_support"] == 0
    assert f["fd_front_bright_flat"] == int(brightness == 255)
    assert f["fd_front_dark_flat"] == int(brightness == 0)
    assert not {"label", "grade", "object_present"} & set(f)


def test_geometric_detail_is_measured_in_the_specified_region(module):
    image = np.full((368, 640, 3), 127, np.uint8)
    cv2.rectangle(image, (210, 130), (420, 230), (255, 255, 255), 3)
    f = module.extract_array(image)
    assert f["fd_front_weak_edges"] > 0 and f["fd_front_corner_coverage"] > 0
    assert f["fd_context_weak_edges"] == 0
    assert f["fd_front_context_edges"] > .99


@pytest.mark.parametrize("image", [np.zeros((0, 2, 3), np.uint8), np.zeros((2, 2), np.uint8), np.zeros((2, 2, 3), float)])
def test_invalid_image_rejected(module, image):
    with pytest.raises(ValueError):
        module.extract_array(image)


def test_features_are_deterministic_finite_and_bounded(module):
    image = np.random.default_rng(7).integers(0, 256, (400, 700, 3), np.uint8)
    a, b = module.extract_array(image), module.extract_array(image.copy())
    assert a == b and all(np.isfinite(v) and 0 <= v <= 1 for v in a.values())
