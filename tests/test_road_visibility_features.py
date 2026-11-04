import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def features(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("road_visibility_features")


@pytest.mark.parametrize("value", [0, 100, 255])
def test_uniform_images_have_no_invented_edges_or_contrast(features, value):
    d = features.extract_array(np.full((368, 640, 3), value, np.uint8))
    assert list(d) == features.FEATURES
    assert np.isfinite(list(d.values())).all()
    assert d["rv_edge_density"] == d["rv_long_line"] == d["rv_line_support"] == 0
    assert d["rv_contrast_p10"] == d["rv_contrast_p50"] == 0
    assert d["rv_entropy"] == 0
    assert d["rv_white_clip"] == (value == 255)
    assert d["rv_black_clip"] == (value == 0)


def test_bright_sky_does_not_itself_count_as_road_clipping(features):
    image = np.full((368, 640, 3), 100, np.uint8)
    image[:120] = 255
    sky = features.extract_array(image)
    assert sky["rv_white_clip"] == 0
    assert sky["rv_black_clip"] == 0
    image[features.road_mask().astype(bool)] = 255
    road = features.extract_array(image)
    assert road["rv_white_clip"] == 1


def test_texture_loss_changes_local_detail_without_changing_mean(features):
    image = np.full((368, 640, 3), 120, np.uint8)
    texture = image.copy()
    checker = np.indices((368, 640)).sum(axis=0) % 2
    texture[checker == 0] = 40
    texture[checker == 1] = 200
    plain, textured = features.extract_array(image), features.extract_array(texture)
    assert textured["rv_local_std9"] > plain["rv_local_std9"]
    assert textured["rv_contrast_p50"] > plain["rv_contrast_p50"]
    assert textured["rv_entropy"] > plain["rv_entropy"]


@pytest.mark.parametrize("bad", [None, np.zeros((3, 4)), np.zeros((3, 4, 3)), np.zeros((0, 4, 3), np.uint8)])
def test_invalid_bgr_rejected(features, bad):
    with pytest.raises(ValueError):
        features.extract_array(bad)
