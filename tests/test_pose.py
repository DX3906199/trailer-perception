import copy
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from trailer_perception.geometry import camera_basis, camera_matrices, landmarks_in_base, project_landmarks, wrap_angle
from trailer_perception.pose import solve_pose


@pytest.fixture
def config():
    return yaml.safe_load((Path(__file__).parents[1] / "configs/geometry.yaml").read_text())


@pytest.mark.parametrize("state", [
    [-2.5, 0, math.pi, 0], [-3.2, 0.35, math.pi + 0.10, 0.30],
    [-1.5, -0.2, math.pi - 0.1, -0.3], [-4, 0.6, math.pi, math.radians(30)],
])
def test_exact_five_point_recovery(config, state):
    estimate = solve_pose(project_landmarks(state, config), config)
    assert estimate["valid"], estimate
    delta = np.asarray(estimate["state"]) - state
    delta[2] = wrap_angle(delta[2])
    assert np.max(np.abs(delta)) < 1e-5
    np.testing.assert_allclose(estimate["hitch_base_m"], landmarks_in_base(state, config)[4], atol=1e-5)
    assert np.trace(estimate["covariance"]) > 0  # Noise floor prevents false zero uncertainty.
    json.dumps(estimate, allow_nan=False)


def test_projection_matches_opencv_fisheye(config):
    points = landmarks_in_base([-2.7, 0.2, math.pi, 0.2], config)
    origin, rotation = camera_basis(config)
    optical = (points - origin) @ rotation.T
    intrinsic, distortion = camera_matrices(config)
    expected = cv2.fisheye.projectPoints(optical.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), intrinsic, distortion)[0]
    np.testing.assert_allclose(project_landmarks([-2.7, 0.2, math.pi, 0.2], config), expected.reshape(-1, 2), atol=1e-9)


def test_original_straight_dataset_fixture(config):
    # Independent labels from synthetic_STRAIGHT_pose_0100_A0_matched, generated
    # by the previous project before this package was written. Not a round-trip
    # through the new projector, and not an accuracy test of learned detection.
    pixels = [[1156.7586792215027, 817.7846982811875],
              [1154.5846104433258, 638.717837602276],
              [784.6116590958318, 633.366255467022],
              [776.871918378369, 811.1518414163447],
              [958.4697356892999, 979.660771735942]]
    result = solve_pose(pixels, config)
    assert result["valid"], result
    np.testing.assert_allclose(result["hitch_base_m"], [-1.7452394762650278, 0, 0.21], atol=1e-7)
    assert abs(result["state"][3]) < 1e-7


def test_noisy_observations(config):
    state = [-2.5, 0.15, math.pi + 0.05, 0.2]
    points = project_landmarks(state, config) + np.random.default_rng(11).normal(0, 0.5, (5, 2))
    estimate = solve_pose(points, config)
    assert estimate["valid"], estimate
    assert np.linalg.norm(np.asarray(estimate["hitch_base_m"]) - landmarks_in_base(state, config)[4]) < 0.05
    assert abs(estimate["state"][3] - state[3]) < math.radians(5)


@pytest.mark.parametrize("index", [0, 4])
def test_missing_point_rejected(config, index):
    points = project_landmarks([-2.5, 0, math.pi, 0], config)
    points[index] = np.nan
    result = solve_pose(points, config)
    assert not result["valid"]
    assert result["state"] is None and result["hitch_base_m"] is None
    json.dumps(result, allow_nan=False)


def test_invisible_hitch_rejected(config):
    result = solve_pose(project_landmarks([-2.5, 0, math.pi, 0], config), config, [1, 1, 1, 1, 0])
    assert result["reason"] == "missing_hitch"


def test_unobservable_joint_guard(config):
    # A hinge-local point ON the axis cannot constrain joint angle.
    config = copy.deepcopy(config)
    config["landmarks"][4]["point_m"] = [0, 0, 0]
    points = project_landmarks([-2.5, 0, math.pi, 0.1], config)
    result = solve_pose(points, config)
    assert not result["valid"]
    assert result["reason"] in {"ill_conditioned", "ambiguous_pose"}


def test_corner_order_and_out_of_frame_guard(config):
    points = project_landmarks([-2.5, 0, math.pi, 0], config)
    points[[1, 2]] = points[[2, 1]]
    assert not solve_pose(points, config)["valid"]
    points[0] = [-1, 100]
    assert solve_pose(points, config)["reason"] == "keypoints_outside_image"


def test_measured_extrinsic_override(config):
    state = [-2.5, 0, math.pi, -0.2]
    config["camera"]["pitch_down_deg"] = 12.0
    points = project_landmarks(state, config)
    _, rotation = camera_basis(config)
    config["camera"]["rotation_base_from_optical"] = rotation.T.tolist()
    config["camera"]["pitch_down_deg"] = 0
    np.testing.assert_allclose(project_landmarks(state, config), points)
    assert solve_pose(points, config)["valid"]
