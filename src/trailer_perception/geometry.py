"""Metric articulated-trailer geometry and calibrated image projection.

``base_link`` has X forward, Y left and Z up. The asset frame F has X
towards the trailer body, Y left and Z up; its zero-angle origin is the
hitch centre. State = [x_F_base_m, y_F_base_m, yaw_F_base_rad, q_rad].
F's height, roll and pitch are known priors, not estimated quantities.

The default optical frame has X image-right, Y image-down, Z forward,
and looks backwards from base_link. ``pitch_down_deg=0`` is horizontal.
All 2D coordinates are ORIGINAL image pixels, never network crop pixels.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2 * math.pi) - math.pi)


def axis_rotation(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if axis.shape != (3,) or not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("hinge_axis must be a finite nonzero 3-vector")
    return cv2.Rodrigues(axis / norm * angle)[0]


def camera_matrices(config: dict) -> tuple[np.ndarray, np.ndarray]:
    camera = config["camera"]
    intrinsic = np.array([[camera["fx"], 0, camera["cx"]],
                          [0, camera["fy"], camera["cy"]], [0, 0, 1]], dtype=float)
    distortion = np.asarray(camera.get("distortion_coefficients", [0] * 4), dtype=float)
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0 or not np.isfinite(intrinsic).all():
        raise ValueError("camera intrinsics must be finite with positive focal lengths")
    if camera.get("distortion_model", "equidistant") not in {"equidistant", "pinhole"}:
        raise ValueError("supported camera models: equidistant and pinhole")
    if distortion.shape != (4,) or not np.isfinite(distortion).all():
        raise ValueError("distortion_coefficients must contain four finite values")
    if camera.get("distortion_model") == "pinhole" and np.any(distortion):
        raise ValueError("pinhole requires zero distortion; Brown-Conrady is not implemented")
    return intrinsic, distortion


def camera_basis(config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return camera origin in base and rotation from base to optical frame.

    A measured ``rotation_base_from_optical`` 3x3 may override the default
    rear-view/pitch mounting. Translation always comes from ``offset_m``.
    """
    camera = config["camera"]
    origin = np.asarray(camera["offset_m"], dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("camera offset_m must contain three finite metre values")
    if "rotation_base_from_optical" in camera:
        rotation = np.asarray(camera["rotation_base_from_optical"], dtype=float)
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)):
            raise ValueError("rotation_base_from_optical must be a proper 3x3 rotation")
        return origin, rotation.T
    pitch = math.radians(float(camera.get("pitch_down_deg", 0.0)))
    return origin, np.array([[0, 1, 0], [math.sin(pitch), 0, -math.cos(pitch)],
                             [-math.cos(pitch), 0, -math.sin(pitch)]], dtype=float)


def landmark_points(config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return four box points in F and one hitch point in hinge-local H."""
    entries = config["landmarks"]
    if len(entries) != 5 or [p["part"] for p in entries] != ["body"] * 4 + ["drawbar"]:
        raise ValueError("expected four ordered body corners followed by one drawbar hitch")
    points = np.asarray([p["point_m"] for p in entries], dtype=float)
    if points.shape != (5, 3) or not np.isfinite(points).all():
        raise ValueError("landmark point_m values must form a finite 5x3 array")
    return points[:4], points[4:]


def state_transforms(state, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return ^base T_F and ^F T_H(q); H is the rotating hinge-local frame."""
    state = np.asarray(state, dtype=float)
    if state.shape != (4,) or not np.isfinite(state).all():
        raise ValueError("state must contain four finite values [x,y,yaw,q]")
    geometry = config["geometry"]
    body = np.eye(4)
    body[:3, :3] = rotation_z(state[2])
    body[:3, 3] = [state[0], state[1], geometry["asset_height_m"]]
    joint = np.eye(4)
    joint[:3, :3] = axis_rotation(geometry["hinge_axis"], state[3])
    joint[:3, 3] = geometry["hinge_center_m"]
    return body, joint


def landmarks_in_base(state, config: dict) -> np.ndarray:
    body_points, hitch_points = landmark_points(config)
    body, joint = state_transforms(state, config)
    hitch_fixed = hitch_points @ joint[:3, :3].T + joint[:3, 3]
    points_fixed = np.concatenate((body_points, hitch_fixed))
    return points_fixed @ body[:3, :3].T + body[:3, 3]


def project_base_points(points, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Project with equidistant distortion; behind-camera points become NaN."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    intrinsic, distortion = camera_matrices(config)
    origin, rotation = camera_basis(config)
    optical = (points - origin) @ rotation.T
    depth = optical[:, 2]
    xy = optical[:, :2] / np.where(depth > 1e-9, depth, 1.0)[:, None]
    if config["camera"].get("distortion_model", "equidistant") == "equidistant":
        radius = np.linalg.norm(xy, axis=1)
        theta = np.arctan(radius)
        theta2 = theta * theta
        theta_d = theta * (1 + distortion[0] * theta2 + distortion[1] * theta2**2
                           + distortion[2] * theta2**3 + distortion[3] * theta2**4)
        xy *= np.divide(theta_d, radius, out=np.ones_like(radius), where=radius > 1e-12)[:, None]
    pixels = xy * [intrinsic[0, 0], intrinsic[1, 1]] + intrinsic[:2, 2]
    pixels[depth <= 1e-9] = np.nan
    return pixels, depth


def project_landmarks(state, config: dict) -> np.ndarray:
    """Project ordered five landmarks; state angles are radians, output is 5x2."""
    return project_base_points(landmarks_in_base(state, config), config)[0]


def normalized_pixels(pixels, config: dict) -> np.ndarray:
    """Undo calibrated distortion to normalized optical rays (x,y,1)."""
    intrinsic, distortion = camera_matrices(config)
    pixels = np.ascontiguousarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    if config["camera"].get("distortion_model", "equidistant") == "equidistant":
        return cv2.fisheye.undistortPoints(pixels, intrinsic, distortion).reshape(-1, 2)
    return (pixels.reshape(-1, 2) - intrinsic[:2, 2]) / [intrinsic[0, 0], intrinsic[1, 1]]
