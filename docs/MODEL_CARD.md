# DINOv2 M0 896 mixed-data checkpoint

Input: one uint8 RGB image, normally 1920×1536 equidistant fisheye.
Aspect-preserving mean-padded affine preprocessing maps it to 896×896.
Frozen ViT-S/14 features feed a convolutional decoder; the five 224×224
heatmaps use local log-peak interpolation and an inverse affine to recover
original-image coordinates. Output order is fixed in the main README.

The model contains the complete pretrained backbone and task-specific head.
`timm==1.0.19` recreates the architecture without downloading any weights.
CPU uses FP32; supported NVIDIA devices use BF16 autocast. These two precision
paths may produce slightly different predictions. The historical internal
checkpoint name `M1` means the public M0 global model in this experiment;
there is no local-refinement or structure branch in this release.

Training: 1,000 synthetic images, articulated angles −20° to +20°, varied
distance/appearance, horizontal rear camera. Mixed adaptation adds 100 reviewed
frames from one real acquisition. Twenty adaptation epochs, 24 synthetic and
8 real samples per batch, frozen backbone, final-epoch checkpoint selection.
Model metadata intentionally excludes machine paths and raw recorded images.

On the held-out 200-image synthetic test, before/after adaptation hitch median
error was 2.10/1.97 original pixels; PCK@10 was 100% for both. This is a
single-run comparison. On separate real recordings qualitative replay showed
reduced confusion with cones in some sequences, while cross-date and occlusion
cases remain challenging. Those real test recordings had no point labels,
so no real metric accuracy is reported.

This network always searches for five landmark peaks. Raw heatmap values can
exceed 1 and are not probabilities or proof that the target is present.
The released asset/camera combination supports a reproducible demo. A different
trailer or camera needs appropriate calibration and evaluation.

Weights: Meta DINOv2 ViT-S/14 backbone with this project's trained head.
Upstream DINOv2 Apache-2.0 terms apply to the backbone; see notices.
