# v0.1.0

Standalone M0 five-landmark inference, calibrated fisheye articulated pose,
MuJoCo whole-mesh viewer and optional articulated gsplat renderer.

Download the checkpoint with `python scripts/download_weights.py`; its SHA-256
is pinned in `weights/manifest.json`. It includes the pretrained DINOv2 backbone
and trained global heatmap head. No ROS or private experiment directory needed.

See README for CPU/CUDA installation, the example image, geometry-only check,
video inference, and calibrated simulation scenes.
