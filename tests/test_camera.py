from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from nunchi_ml.camera import (calibration_fingerprint, camera_features, mapillary_calibration,
                              project_ray, read_aihub_intrinsics, rotation_matrix, sun_camera_ray)


@pytest.mark.parametrize("model", ["perspective", "fisheye", "brown"])
def test_projection_matches_opencv(model):
    rays = np.array([[.2, -.1, 1.], [-.5, .3, 1.], [0., 0., 1.]])
    k = np.array([[800., 0, 650], [0, 810, 350], [0, 0, 1]])
    kwargs = dict(width=1280, height=720, model=model, k1=-.12, k2=.03)
    if model == "brown":
        kwargs.update(fx=800., fy=810., cx=650., cy=350., p1=.002, p2=-.001)
        distortion = np.array([-.12, .03, .002, -.001, 0.])
    else:
        kwargs["focal"] = .625
        k = np.array([[800., 0, 639.5], [0, 800, 359.5], [0, 0, 1]])
        distortion = np.array([-.12, .03, 0., 0.])
    if model == "fisheye":
        pixels, _ = cv2.fisheye.projectPoints(rays.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k, distortion)
    else:
        pixels, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), k, distortion)
    expected = (pixels.reshape(-1, 2) + .5) / [640., -360.] + [-1., 1.]
    actual = [[project_ray(ray, **kwargs)[key] for key in ("u", "v")] for ray in rays]
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_world_axes_rotation_and_pitch():
    np.testing.assert_allclose(sun_camera_ray(0, 90, 0), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(sun_camera_ray(0, 90, 90), [0, 0, 1], atol=1e-12)
    aa = [1., -.3, .8]
    np.testing.assert_allclose(rotation_matrix(aa), cv2.Rodrigues(np.array(aa))[0], atol=1e-12)
    level = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.]])
    pitch = rotation_matrix([np.radians(-10), 0, 0]) @ level
    np.testing.assert_allclose(sun_camera_ray(10, 0, 0, world_to_camera=pitch), [0, 0, 1], atol=1e-12)


@pytest.mark.parametrize("ray", [[0, 0, -1], [1, 0, 0]])
def test_behind_or_side_is_not_clamped_to_front(ray):
    result = project_ray(ray, width=100, height=100, focal=.5, model="perspective")
    assert not result["projection_valid"] and result["in_fov"] == 0 and np.isnan(result["u"])


@pytest.mark.parametrize("extra", [dict(model="unknown"), dict(k1=np.nan), dict(focal=0), dict(width=-1)])
def test_bad_lens_metadata_raises(extra):
    args = dict(width=100, height=80, focal=.5, model="perspective") | extra
    with pytest.raises(ValueError):
        project_ray([0, 0, 1], **args)


def test_metadata_orientation_and_missing_calibration():
    m = dict(id="123", width=100, height=80, camera_parameters=[.5, 0, 0],
             camera_type="perspective", exif_orientation=1)
    assert mapillary_calibration(m)["k1"] == 0
    for bad in (dict(exif_orientation=6), dict(camera_parameters=[.5]), dict(camera_parameters=[.5, np.inf, 0])):
        with pytest.raises(ValueError):
            mapillary_calibration(m | bad)
    d = pd.DataFrame([dict(sun_alt=10, sun_az=0, heading=0, heading_suspect=False) | mapillary_calibration(m)])
    out = camera_features(d)
    assert out.iloc[0].pose_source == "level_assumed" and out.iloc[0].in_fov == 1
    with pytest.raises(ValueError):
        camera_features(d.assign(k1=np.nan))
    with pytest.raises(ValueError, match="Incomplete SfM rotation"):
        camera_features(d.assign(rotation_y=1), use_pose=True)


def test_below_center_still_in_image_when_camera_is_tilted():
    out = project_ray([0, .1, 1], width=100, height=100, model="perspective", focal=.5)
    assert out["v"] < 0 and out["in_fov"] == 1


def test_aihub_intrinsics_do_not_become_world_pose(tmp_path):
    path = tmp_path / "calib.txt"
    path.write_text("Angle : LR, UD, Rot\n-6,1,2\nintrinsicCameraParameter\n100,101,50,40,-.1,.01,0,0\n")
    result = read_aihub_intrinsics(path)
    assert result == dict(model="brown", fx=100., fy=101., cx=50., cy=40., k1=-.1, k2=.01, p1=0., p2=0.)


def test_cache_identity_changes_with_lens_or_pose():
    a = calibration_fingerprint(dict(model="fisheye", k1=.1), [0, 0, 1])
    assert a != calibration_fingerprint(dict(model="fisheye", k1=0.), [0, 0, 1])
    assert a != calibration_fingerprint(dict(model="fisheye", k1=.1), [0, .1, 1])


def test_legacy_projection_unchanged_and_opt_in_mode(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "notebooks/experiments/glare_video_source"
    monkeypatch.syspath_prepend(str(path))
    import fov_probe
    d = pd.DataFrame([dict(focal=.5, width=100, height=80, sun_alt=10, sun_az=30,
                           heading=0, heading_suspect=False, camera_type="perspective", k1=0., k2=0.)])
    legacy = fov_probe.fov_features(d)
    corrected = fov_probe.fov_features(d, projection="calibrated")
    np.testing.assert_allclose(legacy[["u", "v", "r", "in_fov"]], corrected[["u", "v", "r", "in_fov"]])
    with pytest.raises(ValueError):
        fov_probe.fov_features(d, projection="typo")


def test_offline_metadata_preserves_all_ids_and_cache(monkeypatch, tmp_path):
    import json
    path = Path(__file__).resolve().parents[1] / "notebooks/experiments/glare_video_source"
    monkeypatch.syspath_prepend(str(path))
    import fov_probe
    import g_camera_probe
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    # North-facing, level world-to-camera rotation in OpenSfM angle-axis.
    m = dict(id="123", width=100, height=80, camera_parameters=[.5, 0, 0],
             camera_type="perspective", exif_orientation=1, computed_rotation=[np.pi/2, 0, 0])
    (metadata / "123.json").write_text(json.dumps(m))
    frame = tmp_path / "images/drive/image_0/frame.jpg"
    frame.parent.mkdir(parents=True)
    (frame.parent.parent / "calib.txt").write_text("intrinsicCameraParameter\n50,50,50,40,0,0,0,0\n")
    data = pd.DataFrame([dict(id="123", camera="phone", width=100, height=80, heading=0),
                         dict(id="aihub:drive/frame", camera="AI Hub", width=100, height=80,
                              heading=0, file="images/drive/image_0/frame.jpg")])
    enriched = g_camera_probe.enrich(data, metadata, tmp_path)
    assert enriched.id.tolist() == data.id.tolist()
    assert enriched.camera_type.tolist() == ["perspective", "brown"]
    monkeypatch.setattr(fov_probe.requests, "get", lambda *a, **k: pytest.fail("unexpected HTTP"))
    cached = fov_probe.camera_params([123], tmp_path, metadata_dir=metadata)
    reread = fov_probe.camera_params([123], tmp_path)
    assert len(cached) == len(reread) == 1 and reread.k1.iloc[0] == 0
    bad = dict(m, id="124")
    (metadata / "123.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="ID mismatch"):
        g_camera_probe.enrich(data, metadata, tmp_path)
