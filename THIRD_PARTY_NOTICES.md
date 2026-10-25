# External components

This export adapts the owner's existing project modules into independent files;
it does not vendor the old experiment tree or third-party repositories.

| Component | Use | License/source |
|---|---|---|
| Meta DINOv2 | Pretrained ViT-S/14 backbone in checkpoint | [Apache-2.0](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) |
| timm 1.0.19 | Backbone implementation loaded as dependency | [Apache-2.0](https://github.com/huggingface/pytorch-image-models/blob/v1.0.19/LICENSE) |
| PyTorch / torchvision | Model runtime | [PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| OpenCV | Image transforms and fisheye/PnP operations | [Apache-2.0](https://github.com/opencv/opencv/blob/4.x/LICENSE) |
| SciPy / NumPy | Numerical optimization / arrays | Respective BSD licenses in installed distributions |
| MuJoCo | Optional mesh scene rendering/viewer | [Apache-2.0](https://github.com/google-deepmind/mujoco/blob/main/LICENSE) |
| gsplat 1.5.3 | Optional Gaussian renderer | [Apache-2.0](https://github.com/nerfstudio-project/gsplat/blob/main/LICENSE) |

The Apache-2.0 license text is included as `licenses/Apache-2.0.txt`.
Pretrained checkpoint portions retain their original authors' rights.

Source lineage: detector comes from the project's `base_models.DINOv2Heatmap`,
global `OptimizationModel` branch and `base_data` preprocessing/decoding.
Geometry is adapted from the calibrated experiment-three solver. Scene rendering
uses the project's full-fisheye cubemap construction and B3 part assets.
The exported sources remove server-specific imports and paths.
