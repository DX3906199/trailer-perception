# Release validation

Checked on 2026-10-25. These are reproducibility/smoke checks, not a new model
benchmark or a claim of accuracy on all real scenes.

| Check | Result |
|---|---|
| Unit suite | 30 tests passed: 13 geometry/pose, 9 detector transforms/decoding, 6 scene, 2 CLI |
| Migration parity | Original vs portable M0, same 1920×1536 real frame and 896 input, CPU FP32: preprocessing, heatmaps, points and scores exactly equal |
| Independent known-pixel fixture | Historical straight-scene pixels recover hitch `[-1.7452394763,0,0.21] m`; projection difference below `3e-13 px` |
| Image → M0 → solver | Included straight synthetic image: valid; reprojection RMS 1.77 px |
| Articulated GS → M0 → solver | Fresh 1920×1536 render, 10° joint, ego X=−2 m: valid; reprojection RMS 1.17 px, hitch error 2.18 cm, joint error 0.35° |
| Gaussian asset reconstruction | Split body + movable part at zero angle matches all 48,041 original Gaussian centres and attributes |
| MuJoCo rendering | Real EGL mesh render at 960×768 completed |
| gsplat rendering | Real CUDA render at 1920×1536 completed; first run compiled extension; subsequent fresh process approximately 4.1 s including setup |

Tested detector runtime: Python 3.12, PyTorch 2.12.0+cu130 (CPU FP32 test),
timm 1.0.19, OpenCV 4.13, NumPy 1.26.4, SciPy 1.17.1.
Scene checks also used MuJoCo 3.12, gsplat 1.5.3 and a CUDA 13/PyTorch 2.13
environment. Tests check projection/geometry directly, independently from
rendering quality. Compatibility ranges in pyproject are broader than these
specific tested versions.

All scene truth is stored separately for evaluation. The network receives only
RGB, and the solver receives only predicted pixels plus calibrated shape and
camera priors. The geometry-only example intentionally uses known projected
pixels to isolate numerical behavior from detector error.
