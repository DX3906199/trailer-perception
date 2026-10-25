"""Single-frame metric pose from box-four-corners + hitch-centre pixels.

Adapted from the project's experiment-three planar articulated solver:
fisheye inversion -> metric PnP seeds -> hinge-angle search -> joint Huber
least-squares over [x,y,yaw,q]. A known-height ray initializer complements
PnP for its near-planar front face. This is a constrained 4-DoF estimator,
not unconstrained six-DoF pose for two independent rigid bodies.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.optimize import least_squares

from .geometry import (
    camera_basis, landmark_points, landmarks_in_base, normalized_pixels,
    project_base_points, project_landmarks, rotation_z, state_transforms, wrap_angle,
)


def _failure(reason: str, **diagnostics) -> dict:
    # JSON null, not nonstandard JSON NaN/Infinity, denotes an unavailable estimate.
    return {"valid": False, "reason": reason, "frame_id": "base_link", "state": None,
            "hitch_base_m": None, "covariance": None, **diagnostics}


def _height_seed(body: np.ndarray, pixels: np.ndarray, config: dict) -> np.ndarray:
    origin, rotation = camera_basis(config)
    rays = np.column_stack((normalized_pixels(pixels, config), np.ones(4))) @ rotation
    heights = body[:, 2] + config["geometry"]["asset_height_m"]
    if np.any(np.abs(rays[:, 2]) < 1e-7):
        raise ValueError("height ray is parallel to its known-height plane")
    distances = (heights - origin[2]) / rays[:, 2]
    if np.any(distances <= 0):
        raise ValueError("height ray intersection lies behind camera")
    lifted = origin + distances[:, None] * rays
    fixed_delta = body[:, :2] - body[:, :2].mean(axis=0)
    lifted_delta = lifted[:, :2] - lifted[:, :2].mean(axis=0)
    dot = np.sum(fixed_delta * lifted_delta)
    cross = np.sum(fixed_delta[:, 0] * lifted_delta[:, 1] - fixed_delta[:, 1] * lifted_delta[:, 0])
    if math.hypot(dot, cross) < 1e-9:
        raise ValueError("body points do not constrain planar yaw")
    yaw = math.atan2(cross, dot)
    offset = lifted[:, :2].mean(axis=0) - rotation_z(yaw)[:2, :2] @ body[:, :2].mean(axis=0)
    return np.r_[offset, yaw]


def _initial_seeds(body: np.ndarray, pixels: np.ndarray, config: dict) -> list[np.ndarray]:
    seeds = []
    normalized = normalized_pixels(pixels, config).astype(np.float64)
    origin, base_to_camera = camera_basis(config)
    # SQPnP exposes all returned candidates, avoiding a silently chosen planar branch.
    try:
        result = cv2.solvePnPGeneric(body, normalized, np.eye(3), None, flags=cv2.SOLVEPNP_SQPNP)
        if result[0]:
            for rvec, tvec in zip(result[1], result[2]):
                rotation = base_to_camera.T @ cv2.Rodrigues(rvec)[0]
                translation = origin + base_to_camera.T @ tvec.reshape(3)
                seeds.append(np.r_[translation[:2], math.atan2(rotation[1, 0], rotation[0, 0])])
    except cv2.error:
        pass
    try:
        seeds.append(_height_seed(body, pixels, config))
    except ValueError:
        pass
    return [seed for seed in seeds if np.isfinite(seed).all()]


def solve_pose(keypoints, config: dict, confidence=None) -> dict:
    """Estimate metric pose and return a strictly JSON-serializable dictionary.

    ``keypoints`` must be the original-image 5x2 pixels, ordered exactly as
    ``config['landmarks']``. Confidence is optional: heatmap peaks are not
    calibrated visibility scores. Pass actual visibility/confidence when known;
    missing or invisible points must be NaN or score zero. All five are required.

    A valid output supplies ``state`` [x_m,y_m,yaw_rad,q_rad], ``hitch_base_m``,
    ``body_transform_base``, ``drawbar_transform_body`` and 4x4 ``covariance``
    in state order (metres/radians). Covariance is a local linearized estimate
    under an assumed pixel-noise floor; it excludes calibration/model error.
    Invalid estimates contain null state/hitch, never a planning-ready target.
    """
    settings = config.get("solver", {})
    pixels = np.asarray(keypoints, dtype=float)
    scores = np.ones(5) if confidence is None else np.asarray(confidence, dtype=float)
    if pixels.shape != (5, 2) or scores.shape != (5,):
        return _failure("invalid_input_shape")
    selected = (np.isfinite(pixels).all(axis=1) & np.isfinite(scores)
                & (scores > float(settings.get("minimum_confidence", 0.0))))
    if not selected[:4].all():
        return _failure("insufficient_body_landmarks")
    if not selected[4]:
        return _failure("missing_hitch")
    camera = config["camera"]
    if settings.get("require_inside_image", True):
        if (np.any(pixels < 0) or np.any(pixels[:, 0] >= camera["width"])
                or np.any(pixels[:, 1] >= camera["height"])):
            return _failure("keypoints_outside_image")
    if abs(cv2.contourArea(pixels[:4].astype(np.float32))) < float(settings.get("minimum_box_area_px2", 25)):
        return _failure("degenerate_box_projection")
    if not cv2.isContourConvex(pixels[:4].astype(np.float32)):
        return _failure("invalid_corner_order")
    body, _ = landmark_points(config)
    seeds = _initial_seeds(body, pixels[:4], config)
    if not seeds:
        return _failure("initialization_failed")
    q_min, q_max = np.deg2rad(config["geometry"]["angle_limits_deg"])
    if not q_min < q_max:
        raise ValueError("angle_limits_deg must be an increasing pair")
    step = math.radians(float(settings.get("angle_grid_step_deg", 2.0)))
    if step <= 0:
        raise ValueError("angle_grid_step_deg must be positive")
    angles = np.linspace(q_min, q_max, max(2, int(np.ceil((q_max - q_min) / step)) + 1))
    weights = np.sqrt(np.clip(scores, 1e-6, 1.0))[:, None]
    xy_bounds = np.asarray(settings.get("translation_bounds_m", [[-20, -10], [5, 10]]), dtype=float)
    if xy_bounds.shape != (2, 2) or np.any(xy_bounds[0] >= xy_bounds[1]):
        raise ValueError("translation_bounds_m must be [[xmin,ymin],[xmax,ymax]]")

    def residual(state):
        uv, depth = project_base_points(landmarks_in_base(state, config), config)
        if np.any(depth <= 1e-6):
            return np.full(10, 1e4)
        return np.nan_to_num(((uv - pixels) * weights).ravel(), nan=1e4, posinf=1e4, neginf=-1e4)

    candidates = []
    for seed in seeds:
        if np.any(seed[:2] < xy_bounds[0]) or np.any(seed[:2] > xy_bounds[1]):
            continue
        costs = [float(np.sum(residual(np.r_[seed, q]) ** 2)) for q in angles]
        initial = np.r_[seed, angles[int(np.argmin(costs))]]
        lower = np.r_[xy_bounds[0], seed[2] - math.pi / 2, q_min]
        upper = np.r_[xy_bounds[1], seed[2] + math.pi / 2, q_max]
        result = least_squares(residual, np.clip(initial, lower + 1e-9, upper - 1e-9),
                               bounds=(lower, upper), loss="huber",
                               f_scale=float(settings.get("huber_scale_px", 5.0)),
                               max_nfev=250, x_scale="jac")
        if result.success and np.isfinite(result.x).all():
            uv = project_landmarks(result.x, config)
            if np.isfinite(uv).all():
                # Threshold/report unweighted Euclidean error per keypoint.
                rms = float(np.sqrt(np.mean(np.sum((uv - pixels) ** 2, axis=1))))
                candidates.append((rms, result))
    if not candidates:
        return _failure("optimization_failed_or_outside_bounds")
    candidates.sort(key=lambda pair: pair[0])
    rms, best = candidates[0]
    state = best.x.copy()
    state[2] = wrap_angle(state[2])
    for other_rms, other in candidates[1:]:
        delta = other.x - state
        delta[2] = wrap_angle(delta[2])
        distinct = (np.linalg.norm(delta[:2]) > float(settings.get("ambiguity_translation_m", 0.10))
                    or np.max(np.abs(delta[2:])) > math.radians(float(settings.get("ambiguity_angle_deg", 5.0))))
        if distinct and other_rms - rms < float(settings.get("ambiguity_cost_margin_px", 0.5)):
            return _failure("ambiguous_pose", reprojection_rms_px=rms)
    # Numerical Jacobian of raw residuals, not scipy's Huber-rescaled Jacobian.
    epsilon = 1e-5
    jac = np.column_stack([(residual(state + np.eye(4)[i] * epsilon)
                            - residual(state - np.eye(4)[i] * epsilon)) / (2 * epsilon)
                           for i in range(4)])
    information = jac.T @ jac
    condition = float(np.linalg.cond(information))
    if not np.isfinite(condition) or condition > float(settings.get("maximum_condition_number", 1e10)):
        return _failure("ill_conditioned", reprojection_rms_px=rms)
    variance = max(float(settings.get("pixel_noise_floor_px", 1.0)) ** 2,
                   float(np.sum(residual(state) ** 2) / 6))
    covariance = np.linalg.pinv(information) * variance
    angle_std_deg = np.rad2deg(np.sqrt(np.maximum(np.diag(covariance)[2:], 0)))
    diagnostics = {"reprojection_rms_px": rms, "condition_number": condition,
                   "yaw_std_deg": float(angle_std_deg[0]), "articulation_std_deg": float(angle_std_deg[1]),
                   "valid_keypoints": 5}
    if rms > float(settings.get("maximum_reprojection_rms_px", 20.0)):
        return _failure("reprojection_too_large", **diagnostics)
    if np.max(angle_std_deg) > float(settings.get("maximum_angle_std_deg", 15.0)):
        return _failure("uncertain_pose", **diagnostics)
    body_transform, drawbar_transform = state_transforms(state, config)
    return {"valid": True, "reason": "ok", "frame_id": "base_link", "state": state.tolist(),
            "state_names": ["x_m", "y_m", "yaw_rad", "q_rad"],
            "hitch_base_m": landmarks_in_base(state, config)[4].tolist(),
            "body_transform_base": body_transform.tolist(),
            "drawbar_transform_body": drawbar_transform.tolist(),
            "covariance": covariance.tolist(), "projected_keypoints": project_landmarks(state, config).tolist(),
            **diagnostics}
