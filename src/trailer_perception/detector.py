"""Portable M0: DINOv2 ViT-S/14 and the trained five-landmark heatmap head.

The released checkpoint contains the entire backbone. Construction never
downloads pretrained weights. Image geometry, normalization and subpixel
decoding preserve the original M0 experiment's inference implementation.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


KEYPOINT_NAMES = (
    "body_front_right_lower",
    "body_front_right_upper_rectangle",
    "body_front_left_upper_rectangle",
    "body_front_left_lower",
    "hitch_center",
)
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(
    image_rgb: np.ndarray,
    crop_xyxy: Sequence[float] | None = None,
    image_size: int = 896,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Square, aspect-preserving RGB warp and original/input affine pair.

    A crop is expressed in ORIGINAL image pixel coordinates, xyxy, with
    exclusive right/bottom bounds. It may extend beyond the image. Its longest
    side fills the input; the shorter side receives symmetric context/padding.
    The mean-color padding and affine pixel convention match model training.
    """
    image_rgb = np.asarray(image_rgb)
    if (image_rgb.ndim != 3 or image_rgb.shape[2] != 3
            or image_rgb.dtype != np.uint8 or min(image_rgb.shape[:2]) < 1):
        raise ValueError("image_rgb must be a nonempty HxWx3 uint8 RGB array")
    height, width = image_rgb.shape[:2]
    crop = np.asarray([0, 0, width, height] if crop_xyxy is None else crop_xyxy,
                      dtype=np.float64)
    if crop.shape != (4,) or not np.isfinite(crop).all():
        raise ValueError("ROI must contain four finite pixel xyxy coordinates")
    crop_width, crop_height = crop[2:] - crop[:2]
    if (min(crop_width, crop_height) <= 0 or int(image_size) != image_size
            or image_size <= 0):
        raise ValueError("ROI and image_size must have positive dimensions")
    scale = float(image_size) / max(crop_width, crop_height)
    center = (crop[:2] + crop[2:]) / 2.0
    affine = np.asarray([
        [scale, 0.0, image_size / 2.0 - scale * center[0]],
        [0.0, scale, image_size / 2.0 - scale * center[1]],
    ], dtype=np.float64)
    inverse = cv2.invertAffineTransform(affine)
    warped = cv2.warpAffine(
        image_rgb, affine, (int(image_size), int(image_size)),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(float(v) for v in IMAGENET_MEAN * 255.0),
    )
    normalized = (warped.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))
    return tensor, affine, inverse


def decode_heatmaps(
    heatmaps: torch.Tensor | np.ndarray,
    inverse_affines: torch.Tensor | np.ndarray,
    heatmap_stride: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Original-pixel points and raw responses, with local log-peak fitting.

    Supports KxHxW with a 2x3 affine, or batched BxKxHxW/Bx2x3. A flat,
    nonfinite or nonpositive map yields NaN coordinates and a zero response.
    Responses are uncalibrated regression values, not visibility probabilities.
    """
    if heatmap_stride <= 0:
        raise ValueError("heatmap_stride must be positive")
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.detach().float().cpu().numpy()
    if isinstance(inverse_affines, torch.Tensor):
        inverse_affines = inverse_affines.detach().float().cpu().numpy()
    maps = np.asarray(heatmaps, dtype=np.float64)
    affines = np.asarray(inverse_affines, dtype=np.float64)
    single = maps.ndim == 3
    if single:
        maps = maps[None]
        if affines.shape == (2, 3):
            affines = affines[None]
    if (maps.ndim != 4 or min(maps.shape[-2:]) < 1
            or affines.shape != (len(maps), 2, 3)):
        raise ValueError("expected KxHxW + 2x3, or BxKxHxW + Bx2x3")
    if not np.isfinite(affines).all():
        raise ValueError("inverse affine must be finite")
    batch, count, height, width = maps.shape
    points = np.full((batch, count, 2), np.nan, dtype=np.float64)
    scores = np.zeros((batch, count), dtype=np.float64)
    for b in range(batch):
        for k in range(count):
            field = maps[b, k]
            if (not np.isfinite(field).all() or np.max(field) <= 0
                    or np.ptp(field) < 1e-10):
                continue
            y, x = np.unravel_index(np.argmax(field), field.shape)
            scores[b, k] = float(field[y, x])
            refined = np.asarray([x, y], dtype=np.float64)
            if 0 < x < width - 1 and 0 < y < height - 1:
                patch = np.log(np.maximum(field[y - 1:y + 2, x - 1:x + 2], 1e-12))
                dx = (patch[1, 2] - patch[1, 0]) / 2.0
                dy = (patch[2, 1] - patch[0, 1]) / 2.0
                dxx = patch[1, 2] - 2.0 * patch[1, 1] + patch[1, 0]
                dyy = patch[2, 1] - 2.0 * patch[1, 1] + patch[0, 1]
                dxy = (patch[2, 2] - patch[2, 0] - patch[0, 2] + patch[0, 0]) / 4.0
                hessian = np.asarray([[dxx, dxy], [dxy, dyy]])
                if (np.linalg.eigvalsh(hessian)[-1] < -1e-8
                        and np.linalg.cond(hessian) < 1e4):
                    delta = -np.linalg.solve(hessian, [dx, dy])
                    if np.all(np.abs(delta) <= 1.0):
                        refined += delta
            points[b, k] = (refined * heatmap_stride) @ affines[b, :, :2].T + affines[b, :, 2]
    return (points[0], scores[0]) if single else (points, scores)


def _conv_norm_act(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class _DINOHeatmap(nn.Module):
    def __init__(self):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0,
            img_size=518, dynamic_img_size=True, dynamic_img_pad=False,
        )
        if self.backbone.num_features != 384:
            raise RuntimeError("Expected a 384-channel DINOv2 ViT-S/14")
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.project = _conv_norm_act(384, 256)
        self.decode_middle = _conv_norm_act(256, 128)
        self.decode_final = nn.Sequential(_conv_norm_act(128, 64), nn.Conv2d(64, 5, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if (image.ndim != 4 or image.shape[1] != 3
                or image.shape[2] % 28 or image.shape[3] % 28):
            raise ValueError("Expected RGB BCHW with spatial sizes divisible by 28")
        height, width = image.shape[-2:]
        tokens = self.backbone.forward_features(image)
        patches = tokens[:, self.backbone.num_prefix_tokens:]
        features = patches.transpose(1, 2).reshape(
            image.shape[0], 384, height // 14, width // 14).contiguous()
        decoded = self.project(features)
        decoded = F.interpolate(decoded, scale_factor=2, mode="bilinear", align_corners=False)
        decoded = self.decode_middle(decoded)
        decoded = F.interpolate(decoded, size=(height // 4, width // 4),
                                mode="bilinear", align_corners=False)
        return self.decode_final(decoded)


class M0HeatmapModel(nn.Module):
    """Original state-dict names (``coarse.*``), with no experiment dependency."""

    def __init__(self):
        super().__init__()
        self.coarse = _DINOHeatmap()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.coarse(image)


class Detector:
    """Predict the five fixed-order landmarks in a single-trailer RGB image.

    ``roi`` is optional original-image pixel xyxy (not normalized coordinates).
    M0 always returns heatmap peaks; it does not have an object-presence head.
    ``scores`` are raw, uncalibrated responses. ``in_image`` only checks pixel
    bounds and does not certify landmark visibility or a correct detection.

    CPU uses FP32. CUDA uses BF16 autocast where supported, matching the
    released experiment. The caller chooses the device; no silent CPU fallback.
    """

    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Supported devices are cpu or cuda[:index]")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError("Expected an M0 checkpoint dictionary containing 'model'")
        # Historical internal M1 means public M0: resolution-only global head.
        if payload.get("variant", "M1") != "M1":
            raise ValueError("This detector loads the M0 global-head checkpoint only")
        self.image_size = int(payload.get("image_size", 896))
        if self.image_size != 896:
            raise ValueError("The released M0 inference contract requires image_size=896")
        self.model = M0HeatmapModel()
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        self.use_bfloat16 = (self.device.type == "cuda"
                             and torch.cuda.get_device_capability(self.device)[0] >= 8)

    @torch.inference_mode()
    def predict(self, image_rgb: np.ndarray, roi: Sequence[float] | None = None) -> dict:
        tensor, affine, inverse = preprocess_image(image_rgb, roi, self.image_size)
        precision = (torch.autocast("cuda", dtype=torch.bfloat16)
                     if self.use_bfloat16 else nullcontext())
        with precision:
            heatmaps = self.model(tensor[None].to(self.device))
        points, scores = decode_heatmaps(heatmaps[0], inverse)
        height, width = image_rgb.shape[:2]
        finite = np.isfinite(points).all(axis=1)
        in_image = (finite & (points[:, 0] >= 0) & (points[:, 0] < width)
                    & (points[:, 1] >= 0) & (points[:, 1] < height))
        # Invalid predictions are explicit JSON nulls, never clipped to an edge.
        serial_points = [[float(x), float(y)] if valid else [None, None]
                         for (x, y), valid in zip(points, finite)]
        crop = [0.0, 0.0, float(width), float(height)] if roi is None else list(map(float, roi))
        return {
            "keypoints": serial_points,
            "scores": scores.tolist(),
            "keypoint_names": list(KEYPOINT_NAMES),
            "finite": finite.tolist(),
            "in_image": in_image.tolist(),
            "image_width": width,
            "image_height": height,
            "input_size": self.image_size,
            "roi_xyxy": crop,
            "forward_affine": affine.tolist(),
            "inverse_affine": inverse.tolist(),
            "precision": "bfloat16" if self.use_bfloat16 else "float32",
            "score_semantics": "uncalibrated raw heatmap response",
            "model": "dinov2_m0_896",
        }
