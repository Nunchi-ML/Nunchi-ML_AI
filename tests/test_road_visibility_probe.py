import importlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def probe(monkeypatch):
    here = Path(__file__).resolve().parents[1] / "notebooks/experiments"
    monkeypatch.syspath_prepend(str(here / "glare_video_source"))
    monkeypatch.syspath_prepend(str(here / "sun_visible_probe"))
    return importlib.import_module("road_visibility_probe")


@pytest.mark.parametrize("p", [np.zeros((2, 2)), [[.5, .5, .1]], [[1.1, -.1, 0]], [[np.nan, 0, 1]]])
def test_invalid_class_probabilities_rejected(probe, p):
    with pytest.raises(ValueError):
        probe.validate_probabilities(p)


def test_group_folds_are_reproducible_and_complete(probe):
    data = pd.DataFrame(dict(group=np.repeat(np.arange(12), 3), label=np.tile([0, 1, 2], 12)))
    first, second = list(probe.group_partitions(data, 2)), list(probe.group_partitions(data, 2))
    for (seed, folds), (seed2, folds2) in zip(first, second):
        assert seed == seed2
        for (tr, va), (tr2, va2) in zip(folds, folds2):
            np.testing.assert_array_equal(tr, tr2)
            np.testing.assert_array_equal(va, va2)
            assert not set(data.iloc[tr].group) & set(data.iloc[va].group)
        np.testing.assert_array_equal(np.sort(np.concatenate([va for _, va in folds])), np.arange(len(data)))


def test_road_cache_matches_id_order_and_invalidates_old_source(probe, tmp_path):
    paths = [tmp_path / "bright.png", tmp_path / "dark.png"]
    for path, value in zip(paths, [255, 0]):
        ok, encoded = cv2.imencode(".png", np.full((368, 640, 3), value, np.uint8))
        assert ok
        path.write_bytes(encoded.tobytes())
    data = pd.DataFrame(dict(id=["b", "a"], label=[2, 0]))
    cache = tmp_path / "cache.csv"
    original = probe.attach_road(data, paths, cache)
    assert original.id.tolist() == ["b", "a"]
    assert original.rv_white_clip.tolist() == [1., 0.]
    previous = pd.read_csv(cache)
    previous["road_source_sha256"] = "stale"
    previous["rv_white_clip"] = 17
    previous.to_csv(cache, index=False)
    refreshed = probe.attach_road(data, paths, cache)
    pd.testing.assert_frame_equal(original, refreshed)
    wrong = data.assign(image_sha256="different-image")
    with pytest.raises(ValueError, match="Image bytes changed"):
        probe.attach_road(wrong, paths, cache)


def test_no_duplicate_or_unmatched_feature_rows(probe, tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        probe.attach_road(pd.DataFrame(dict(id=["a", "a"])), [tmp_path / "a", tmp_path / "b"], tmp_path / "cache")
    with pytest.raises(ValueError, match="exactly one"):
        probe.attach_road(pd.DataFrame(dict(id=["a"])), [], tmp_path / "cache")
