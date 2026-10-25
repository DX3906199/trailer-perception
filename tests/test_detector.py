"""Coordinate and decoding contracts; these do not download model weights."""

import numpy as np
import pytest
import torch

from trailer_perception.detector import decode_heatmaps, preprocess_image


def test_original_to_input_to_original_with_crop():
    rgb = np.zeros((1536, 1920, 3), dtype=np.uint8)
    tensor, forward, inverse = preprocess_image(rgb, [192, 76, 1632, 1506])
    assert tensor.shape == (3, 896, 896)
    points = np.asarray([[192, 76], [1000.25, 920.5], [1632, 1506]])
    transformed = points @ forward[:, :2].T + forward[:, 2]
    recovered = transformed @ inverse[:, :2].T + inverse[:, 2]
    np.testing.assert_allclose(recovered, points, atol=1e-10)
    assert forward[0, 0] == forward[1, 1]  # preserve aspect ratio


def test_gaussian_subpixel_decoding_preserves_original_pixels():
    y, x = np.mgrid[:32, :32]
    center = np.array([14.3, 17.6])
    heatmap = np.exp(-((x-center[0])**2+(y-center[1])**2)/8)
    inverse = np.array([[2., 0., 13.], [0., 2., -7.]])
    points, scores = decode_heatmaps(heatmap[None], inverse)
    np.testing.assert_allclose(points[0], center * 8 + [13, -7], atol=1e-10)
    assert scores[0] > 0


def test_invalid_heatmaps_are_not_false_detections():
    maps = np.stack([np.zeros((8, 8)), np.ones((8, 8)), np.full((8, 8), np.nan)])
    points, scores = decode_heatmaps(maps, np.eye(2, 3))
    assert np.isnan(points).all()
    np.testing.assert_array_equal(scores, 0)


def test_boundary_peak_remains_at_boundary():
    maps = np.zeros((1, 4, 4))
    maps[0, 0, 3] = 2.
    points, scores = decode_heatmaps(maps, np.eye(2, 3))
    np.testing.assert_array_equal(points, [[12., 0.]])
    np.testing.assert_array_equal(scores, [2.])  # response can exceed one


def test_batched_decoding_matches_single():
    y, x = np.mgrid[:16, :16]
    maps = np.exp(-((x-7.2)**2+(y-8.3)**2)/4)[None]
    affine = np.array([[2., 0., 100.], [0., 2., 80.]])
    single = decode_heatmaps(maps, affine)
    batch = decode_heatmaps(torch.tensor(maps[None]), torch.tensor(affine[None]))
    np.testing.assert_allclose(single[0], batch[0][0], atol=1e-5)
    np.testing.assert_allclose(single[1], batch[1][0], atol=1e-5)


@pytest.mark.parametrize("roi", [[5, 0, 2, 5], [0, 0, 2, np.nan], [0, 0, 1]])
def test_invalid_crop_is_rejected(roi):
    with pytest.raises(ValueError):
        preprocess_image(np.zeros((8, 10, 3), np.uint8), roi)


def test_float_images_are_not_silently_rescaled():
    with pytest.raises(ValueError, match="uint8"):
        preprocess_image(np.zeros((8, 10, 3), np.float32))
