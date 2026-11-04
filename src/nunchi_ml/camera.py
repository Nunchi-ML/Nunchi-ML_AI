"""Sun-ray projection for OpenSfM perspective/fisheye and OpenCV Brown lenses.

ENU world: east, north, up. Camera: right, down, forward. Normalized outputs
u/v use +/-1 at image edges and v grows UP, matching the existing G features.
Sources: opensfm.org/docs/geometry.html; MapillaryJS geo/Geo.ts (ENU -> R C).
"""
import hashlib
import json

import numpy as np

PROJECTION_VERSION = "sun-camera-v1"


def rotation_matrix(angle_axis):
    """Rodrigues world-to-camera rotation; values are radians, not Euler angles."""
    v = np.asarray(angle_axis, dtype=float)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError("angle_axis must have three finite components")
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return np.eye(3)
    x, y, z = v / theta
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)


def sun_camera_ray(altitude, azimuth, heading, *, world_to_camera=None):
    """A unit ray. Without measured pose, explicitly assume a level camera."""
    if not np.isfinite([altitude, azimuth, heading]).all() or not -90 <= altitude <= 90:
        raise ValueError("Sun angles and heading must be finite, altitude within [-90, 90]")
    alt, az, h = np.radians([altitude, azimuth, heading])
    world = np.array([np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)])
    if world_to_camera is None:
        r = np.array([[np.cos(h), -np.sin(h), 0], [0, 0, -1], [np.sin(h), np.cos(h), 0]])
    else:
        r = np.asarray(world_to_camera, dtype=float)
        if (r.shape != (3, 3) or not np.isfinite(r).all()
                or not np.allclose(r @ r.T, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(r), 1, atol=1e-6)):
            raise ValueError("world_to_camera must be a proper rotation matrix")
    return r @ world


def project_ray(ray, *, width, height, model, focal=None, fx=None, fy=None,
                cx=None, cy=None, k1=0., k2=0., p1=0., p2=0.):
    """Project onto raw image pixels. No clipping of behind-camera rays.

    perspective/fisheye: focal is normalized by max(width,height), k1/k2 follow
    OpenSfM's model-specific radial formula. brown: fx/fy/cx/cy are pixels and
    k1/k2/p1/p2 follow OpenCV. The rear hemisphere is outside this FPV scope.
    """
    if model not in {"perspective", "fisheye", "brown"}:
        raise ValueError(f"Unsupported camera model: {model}")
    values = [width, height, k1, k2, p1, p2]
    if not np.isfinite(values).all() or min(width, height) <= 0:
        raise ValueError("Invalid image dimensions or distortion coefficients")
    if model != "brown":
        if focal is None or not np.isfinite(focal) or focal <= 0:
            raise ValueError("A positive normalized focal is required")
        fx = fy = focal * max(width, height)
        # OpenSfM/OpenCV pixel centers: top-left center=(0,0).
        cx, cy = (width - 1) / 2, (height - 1) / 2
    elif any(v is None for v in (fx, fy, cx, cy)) or not np.isfinite([fx, fy, cx, cy]).all() or min(fx, fy) <= 0:
        raise ValueError("Brown projection requires positive fx/fy and finite cx/cy")
    v = np.asarray(ray, dtype=float)
    if v.shape != (3,) or not np.isfinite(v).all() or np.linalg.norm(v) == 0:
        raise ValueError("ray must be a nonzero finite 3-vector")
    x, y, z = v / np.linalg.norm(v)
    invalid = {"u": np.nan, "v": np.nan, "r": np.nan, "in_fov": 0., "projection_valid": False}
    if z <= 1e-6:
        return invalid
    if model == "fisheye":
        rho = np.hypot(x, y)
        theta = np.arctan2(rho, z)
        scale = theta / rho if rho > 1e-12 else 1.
        radial = 1 + k1 * theta**2 + k2 * theta**4
        xn, yn = x * scale * radial, y * scale * radial
    else:
        xn, yn = x / z, y / z
        r2 = xn*xn + yn*yn
        radial = 1 + k1*r2 + k2*r2*r2
        xn, yn = (xn*radial + 2*p1*xn*yn + p2*(r2 + 2*xn*xn),
                  yn*radial + p1*(r2 + 2*yn*yn) + 2*p2*xn*yn)
    if radial <= 0 or not np.isfinite([xn, yn]).all():
        return invalid
    # Convert pixel centers to image-edge coordinates for existing G features.
    u = 2 * (fx*xn + cx + .5) / width - 1
    v = 1 - 2 * (fy*yn + cy + .5) / height
    return {"u": float(u), "v": float(v), "r": float(np.hypot(u, v)),
            "in_fov": float(abs(u) <= 1 and abs(v) <= 1), "projection_valid": True}


def calibration_fingerprint(parameters, ray):
    """Cache identity includes calibration, pose-derived ray and implementation."""
    content = {"version": PROJECTION_VERSION, "parameters": parameters, "ray": list(map(float, ray))}
    return hashlib.sha256(json.dumps(content, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_aihub_intrinsics(path):
    """AI Hub calib.txt intrinsics only; LiDAR extrinsics are NOT an ENU pose."""
    lines = path.read_text(encoding="utf-8").splitlines()
    values = list(map(float, lines[lines.index("intrinsicCameraParameter") + 1].split(",")))
    if len(values) != 8 or not np.isfinite(values).all():
        raise ValueError("Expected fx,fy,cx,cy,k1,k2,p1,p2 in AI Hub calibration")
    return {"model": "brown", **dict(zip(("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"), values))}


def mapillary_calibration(metadata):
    """Retain the lens and SfM pose; unsupported EXIF orientations fail closed."""
    cp = metadata.get("camera_parameters") or []
    if len(cp) != 3 or metadata.get("camera_type") not in {"perspective", "fisheye"}:
        raise ValueError("Expected OpenSfM perspective/fisheye [focal,k1,k2]")
    if metadata.get("exif_orientation") != 1:
        raise ValueError("Only EXIF orientation 1 is supported; rotate calibration first")
    out = dict(id=int(metadata["id"]), focal=cp[0], k1=cp[1], k2=cp[2],
               width=metadata["width"], height=metadata["height"],
               camera_type=metadata["camera_type"], exif_orientation=1)
    project_ray([0, 0, 1], model=out["camera_type"],
                **{k: out[k] for k in ("width", "height", "focal", "k1", "k2")})
    rotation = metadata.get("computed_rotation")
    if rotation is not None:
        r = rotation_matrix(rotation)
        forward = r.T @ [0, 0, 1]
        out.update(dict(zip(("rotation_x", "rotation_y", "rotation_z"), rotation)))
        out["pose_heading"] = float(np.degrees(np.arctan2(forward[0], forward[1])) % 360)
        out["pose_pitch"] = float(np.degrees(np.arcsin(np.clip(forward[2], -1, 1))))
    return out


def camera_features(data, *, use_pose=False):
    """Strict calibrated projection; never infer calibration from a missing value.

    SfM pose is an offline diagnostic, not measured phone attitude. Where no
    SfM rotation is present, use the explicitly marked level-camera assumption.
    """
    import pandas as pd

    rows = []
    for _, d in data.iterrows():
        model = d.get("camera_type")
        keys = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2") if model == "brown" else ("focal", "k1", "k2")
        params = {k: float(d[k]) for k in ("width", "height", *keys)}
        params["model"] = model
        r, pose_source = None, "level_assumed"
        pose = [d.get(k, np.nan) for k in ("rotation_x", "rotation_y", "rotation_z")]
        if use_pose and any(pd.notna(pose)):
            if not all(pd.notna(pose)):
                raise ValueError("Incomplete SfM rotation")
            if d.get("exif_orientation") != 1:
                raise ValueError("SfM pose requires EXIF orientation 1")
            r = rotation_matrix(pose)
            pose_source = "mapillary_sfm"
        ray = sun_camera_ray(float(d.sun_alt), float(d.sun_az), float(d.heading), world_to_camera=r)
        out = project_ray(ray, **params)
        out.update(pose_source=pose_source, camera_model=model,
                   projection_fingerprint=calibration_fingerprint(params, ray))
        if bool(d.heading_suspect):
            out.update(u=np.nan, v=np.nan, r=np.nan, in_fov=np.nan, projection_valid=False)
        rows.append(out)
    return pd.DataFrame(rows, index=data.index)
