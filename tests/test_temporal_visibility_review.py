import importlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def review(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("temporal_visibility_review")


def test_mapping_preserves_selected_order_and_rejects_unmatched(review):
    sheet = pd.DataFrame(dict(id=["a", "b"], file=["a.jpg", "b.jpg"]))
    mapping = pd.DataFrame(dict(id=["b", "a"], video=["v2", "v1"], t=[5., 3.]))
    result = review.match_rows(sheet, mapping, pd.DataFrame(dict(id=["b", "a"])))
    assert result.id.tolist() == ["b", "a"] and result.video.tolist() == ["v2", "v1"]
    with pytest.raises(ValueError, match="missing"):
        review.match_rows(sheet, mapping, pd.DataFrame(dict(id=["z"])))
    with pytest.raises(ValueError, match="unique"):
        review.match_rows(sheet, mapping, pd.DataFrame(dict(id=["a", "a"])))
    with pytest.raises(ValueError, match="valid video/time"):
        review.match_rows(sheet, mapping.iloc[:1], pd.DataFrame(dict(id=["a"])))


def test_path_escape_is_rejected(review, tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        review.under(tmp_path, "../secret.csv")


@pytest.mark.parametrize("time", [-1, float("nan"), float("inf")])
def test_invalid_seek_times_fail_before_decoder(review, time):
    with pytest.raises(ValueError, match="Invalid frame time"):
        review.read_at(None, time)


def test_failed_seek_is_not_silently_substituted(review):
    class Bad:
        def set(self, *args):
            return False
    with pytest.raises(ValueError, match="seek failed"):
        review.read_at(Bad(), 3)


def test_prepare_keeps_center_bytes_and_answers_private(review, tmp_path, monkeypatch):
    root = tmp_path / "data/new_cameras"
    pack = root / "label_pack"
    pack.mkdir(parents=True)
    image = np.full((20, 30, 3), 100, np.uint8)
    _, encoded = cv2.imencode(".jpg", image)
    (pack / "a.jpg").write_bytes(encoded.tobytes())
    pd.DataFrame(dict(id=["a"], file=["a.jpg"])).to_csv(pack / "label_sheet_blind.csv", index=False)
    (root / "human_review_50").mkdir()
    pd.DataFrame(dict(id=["a"], label=[2])).to_csv(root / "human_review_50/labels_성진.csv", index=False)
    pd.DataFrame(dict(id=["a"], video=["v1"], t=[3.])).to_csv(root / "private_frame_map.csv", index=False)
    (root / "v1").mkdir()
    (root / "v1/video.mp4").write_bytes(b"fake video for mocked decoder")
    (root / "labels_claude.csv").write_text("UNCHANGED-AI-ANSWER", encoding="utf-8")
    for relative in ["ai_label_review_20261104/labels_codex.csv", "road_visibility_20261104/run30/predictions.csv"]:
        path = root.parent / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("UNCHANGED-ANSWER", encoding="utf-8")
    class Fake:
        def __init__(self, *args):
            self.time = 0
        def isOpened(self):
            return True
        def get(self, key):
            return {cv2.CAP_PROP_FPS: 10., cv2.CAP_PROP_FRAME_COUNT: 100.,
                    cv2.CAP_PROP_POS_MSEC: self.time, cv2.CAP_PROP_POS_FRAMES: self.time/100+1}[key]
        def set(self, key, value):
            self.time = value
            return True
        def read(self):
            return True, image.copy()
        def release(self):
            pass
    monkeypatch.setattr(review.cv2, "VideoCapture", Fake)
    out = review.prepare(root, root.parent / "temporal_test")
    assert (out / "review/T001/center.jpg").read_bytes() == (pack / "a.jpg").read_bytes()
    method = json.loads((out / "method.json").read_text(encoding="utf-8"))
    assert method["status"] == "complete" and method["frames"] == 5
    audit = pd.read_csv(out / "frame_audit.csv")
    assert audit.requested_seconds.tolist() == [1, 2, 3, 4, 5]
    assert audit.decoder_reported_seconds.tolist() == [1, 2, 3, 4, 5]
    page = (out / "review/review.html").read_text(encoding="utf-8")
    assert "UNCHANGED" not in page and "reference_human" not in page
    assert "center_sha256" in page and "localStorage" in page
    blank = pd.read_csv(out / "review/observations_blank.csv")
    assert blank.drop(columns="case_id").isna().all().all()
    with pytest.raises(ValueError, match="new private"):
        review.prepare(root, out)
