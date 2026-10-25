from pathlib import Path

import cv2
import numpy as np
import pytest

from trailer_perception.scene import (
    DEFAULT_ASSETS, FACE_ROTATIONS, articulated_gaussians, camera_view,
    cube_faces, fisheye_rays, load_gaussians, load_scene_config, scaled_camera,
)


def test_camera_matches_pose_solver():
    from trailer_perception.geometry import camera_basis
    cfg = load_scene_config()
    for yaw_deg in (0, 70, 180):
        centre, view = camera_view(cfg["camera"], -2, 0.3, yaw_deg)
        yaw = np.deg2rad(yaw_deg)
        base_world = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                               [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        origin, basis = camera_basis(cfg)
        np.testing.assert_allclose(centre, [-2, 0.3, 0] + base_world @ origin)
        np.testing.assert_allclose(view[:3, :3], basis @ base_world.T, atol=1e-7)


def test_fisheye_ray_projection_roundtrip():
    camera = scaled_camera(load_scene_config()["camera"], 240)
    rays = fisheye_rays(camera)
    # Include all forward points, including rays far outside an intermediate pinhole.
    valid = rays[..., 2] > 0.05
    k = np.array([[camera["fx"], 0, camera["cx"]], [0, camera["fy"], camera["cy"]], [0, 0, 1]])
    uv, _ = cv2.fisheye.projectPoints(rays[valid].astype(float)[:, None], np.zeros(3), np.zeros(3),
                                      k, np.asarray(camera["distortion_coefficients"]))
    yy, xx = np.nonzero(valid)
    np.testing.assert_allclose(uv[:, 0], np.column_stack((xx, yy)), atol=2e-5)
    np.testing.assert_allclose(np.linalg.norm(rays, axis=-1), 1.0, atol=1e-6)


def test_cubemap_complete_coverage_and_guard():
    camera = scaled_camera(load_scene_config()["camera"], 240)
    size, _, faces = cube_faces(camera)
    counts = np.zeros((camera["height"], camera["width"]), dtype=int)
    assert len(faces) > 1  # A single pinhole cannot represent this calibration.
    for face in faces:
        counts[face["region"]] += face["mask"]
        for key in ("map_x", "map_y"):
            values = face[key][face["mask"]]
            assert np.min(values) >= 2 and np.max(values) < size - 2
    assert np.all(counts == 1)
    for rotation in FACE_ROTATIONS:
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3))
        assert np.linalg.det(rotation) == 1


def test_gaussian_split_zero_angle_reconstructs_original():
    pytest.importorskip("plyfile")
    from scipy.spatial import cKDTree
    cfg = load_scene_config()
    full = load_gaussians(DEFAULT_ASSETS / "trailer_visual_3dgs_local.ply")
    split = articulated_gaussians(DEFAULT_ASSETS, cfg["geometry"], 0)
    full["xyz"][:, 2] += cfg["geometry"]["asset_height_m"]
    distances, index = cKDTree(full["xyz"]).query(split["xyz"])
    assert len(full["xyz"]) == len(split["xyz"]) == 48041
    assert np.max(distances) < 5e-7
    for key in ("scales", "opacity", "sh"):
        np.testing.assert_allclose(split[key], full[key][index], atol=1e-6)
    # q and -q are the same orientation; compare absolute quaternion dot product.
    np.testing.assert_allclose(np.abs(np.sum(split["quats"] * full["quats"][index], axis=1)), 1, atol=2e-6)


def test_rigid_articulation_preserves_internal_distances():
    pytest.importorskip("plyfile")
    cfg = load_scene_config()
    zero = articulated_gaussians(DEFAULT_ASSETS, cfg["geometry"], 0)
    moved = articulated_gaussians(DEFAULT_ASSETS, cfg["geometry"], 25)
    np.testing.assert_allclose(zero["xyz"][:41890], moved["xyz"][:41890])
    for points in (zero, moved):
        np.testing.assert_allclose(np.linalg.norm(points["quats"], axis=1), 1, atol=2e-6)
    np.testing.assert_allclose(np.linalg.norm(np.diff(zero["xyz"][41890::100], axis=0), axis=1),
                               np.linalg.norm(np.diff(moved["xyz"][41890::100], axis=0), axis=1), atol=1e-6)
    with pytest.raises(ValueError, match="outside"):
        articulated_gaussians(DEFAULT_ASSETS, cfg["geometry"], 31)


def test_mujoco_scene_is_static_complete_mesh():
    pytest.importorskip("mujoco")
    from trailer_perception.scene import load_mujoco_scene
    model, data = load_mujoco_scene(load_scene_config(), ego_x=-2.5)
    assert model.nmesh == 1
    assert model.nq == 0
    assert model.nmocap == 1
    np.testing.assert_allclose(data.mocap_pos[0], [-2.5, 0, 0])
