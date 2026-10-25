"""Portable trailer scene: MuJoCo mesh or CUDA gsplat, calibrated fisheye output.

MuJoCo does not load Gaussian PLY files. The two backends share the camera
geometry, while gsplat separately renders the articulated Gaussian asset.
Six overlapping pinhole faces cover the full calibrated fisheye field of view,
including the upper cargo body that a single intermediate pinhole can clip.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS = ROOT / "assets" / "scene"
FACE_ROTATIONS = np.asarray([
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
    [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
    [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
    [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
], dtype=np.float32)


def load_scene_config(config: str | Path | dict | None = None) -> dict:
    """Load the same configuration used by the five-keypoint pose solver."""
    if isinstance(config, dict):
        return config
    path = Path(config) if config else ROOT / "configs" / "geometry.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def scaled_camera(camera: dict, width: int | None = None, height: int | None = None) -> dict:
    result = dict(camera)
    source_width, source_height = int(camera["width"]), int(camera["height"])
    if width is None and height is None:
        return result
    width = int(width or round(int(height) * source_width / source_height))
    height = int(height or round(width * source_height / source_width))
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    sx, sy = width / source_width, height / source_height
    result.update(width=width, height=height, fx=camera["fx"] * sx,
                  fy=camera["fy"] * sy, cx=camera["cx"] * sx, cy=camera["cy"] * sy)
    return result


def camera_view(camera: dict, x: float, y: float, yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """World-to-camera matrix with OpenCV right/down/forward camera axes."""
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    base = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    pitch = math.radians(camera.get("pitch_down_deg", 0.0))
    # Rear camera points along base -X, with its image-right along base +Y.
    local_basis = np.array([[0, 1, 0], [math.sin(pitch), 0, -math.cos(pitch)],
                            [-math.cos(pitch), 0, -math.sin(pitch)]])
    if "rotation_base_from_optical" in camera:
        local_basis = np.asarray(camera["rotation_base_from_optical"], dtype=float).T
        if local_basis.shape != (3, 3) or not np.allclose(local_basis @ local_basis.T, np.eye(3), atol=1e-6):
            raise ValueError("Invalid camera rotation")
    basis = local_basis @ base.T
    centre = np.array([x, y, 0.0]) + base @ np.asarray(camera["offset_m"], dtype=float)
    view = np.eye(4, dtype=np.float32)
    view[:3, :3], view[:3, 3] = basis, -basis @ centre
    return centre, view


def fisheye_rays(camera: dict) -> np.ndarray:
    """Invert the OpenCV equidistant polynomial without pinhole truncation."""
    if camera.get("distortion_model", "equidistant") != "equidistant":
        raise ValueError("This scene renderer expects an equidistant camera")
    x, y = np.meshgrid((np.arange(camera["width"]) - camera["cx"]) / camera["fx"],
                       (np.arange(camera["height"]) - camera["cy"]) / camera["fy"])
    radius = np.hypot(x, y)
    theta = radius.copy()
    coefficients = np.asarray(camera["distortion_coefficients"], dtype=float)
    if coefficients.shape != (4,):
        raise ValueError("Expected four equidistant distortion coefficients")
    for _ in range(12):
        powers = [theta ** (2 * i) for i in range(1, 5)]
        value = theta * (1 + sum(k * p for k, p in zip(coefficients, powers)))
        derivative = 1 + sum((2 * i + 3) * k * p for i, (k, p) in enumerate(zip(coefficients, powers)))
        if np.any(derivative <= 0):
            raise ValueError("Fisheye calibration is not monotonic on this image")
        theta -= (value - radius) / derivative
    ratio = np.divide(np.sin(theta), radius, out=np.ones_like(radius), where=radius > 1e-12)
    rays = np.stack((ratio * x, ratio * y, np.cos(theta)), axis=-1)
    if not np.all(np.isfinite(rays)) or np.any(np.abs(theta) > math.pi):
        raise ValueError("Fisheye calibration has invalid ray directions")
    return rays.astype(np.float32)


def cube_faces(camera: dict, backend: str = "gaussian") -> tuple[int, np.ndarray, list[dict]]:
    """Each pixel belongs to the face with the largest positive ray depth.

    gsplat uses corner-origin pixel coordinates. OpenCV remap uses array-centre
    coordinates; the half-pixel offset below matches the validated dataset renderer.
    MuJoCo centres a symmetric camera at (size-1)/2 in array coordinates.
    """
    if backend not in ("gaussian", "mesh"):
        raise ValueError("Unknown cubemap backend")
    size = int(math.ceil((2 * max(camera["fx"], camera["fy"]) + 9) / 32) * 32)
    focal, centre = (size - 9) / 2, (size - 1) / 2
    intrinsic = np.array([[focal, 0, centre], [0, focal, centre], [0, 0, 1]], np.float32)
    rays = fisheye_rays(camera)
    owner = np.argmax(rays @ FACE_ROTATIONS[:, 2, :].T, axis=-1)
    faces = []
    source_centre = centre - (0.5 if backend == "gaussian" else 0.0)
    for index, rotation in enumerate(FACE_ROTATIONS):
        ys, xs = np.nonzero(owner == index)
        if not len(ys):
            continue
        region = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
        mask = owner[region] == index
        directions = rays[region] @ rotation.T
        safe_depth = np.where(mask, directions[..., 2], 1.0)
        map_x = (source_centre + focal * directions[..., 0] / safe_depth).astype(np.float32)
        map_y = (source_centre + focal * directions[..., 1] / safe_depth).astype(np.float32)
        faces.append(dict(rotation=rotation, region=region, mask=mask, map_x=map_x, map_y=map_y))
    return size, intrinsic, faces


def _remap(destination: np.ndarray, image: np.ndarray, face: dict) -> None:
    values = cv2.remap(image, face["map_x"], face["map_y"], cv2.INTER_LINEAR)
    destination[face["region"]][face["mask"]] = values[face["mask"]]


def _axis_rotation(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    angle = math.radians(angle_deg)
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * cross @ cross


def load_gaussians(path: str | Path) -> dict[str, np.ndarray]:
    """Read our uncompressed SH-degree-zero 3DGS PLY, not ordinary point clouds."""
    from plyfile import PlyData
    vertex = PlyData.read(path)["vertex"]
    names = {prop.name for prop in vertex.properties}
    required = ["x", "y", "z", "opacity"] + [f"f_dc_{i}" for i in range(3)]
    required += [f"rot_{i}" for i in range(4)] + [f"scale_{i}" for i in range(3)]
    if not set(required) <= names:
        raise ValueError(f"Not a standard 3DGS PLY: {path}")
    if any(name.startswith("f_rest_") for name in names):
        raise ValueError("This portable example supports the supplied SH-degree-zero assets only")
    def columns(fields):
        return np.column_stack([vertex[name] for name in fields]).astype(np.float32)
    rotation = columns([f"rot_{i}" for i in range(4)])
    norm = np.linalg.norm(rotation, axis=1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("Zero-length Gaussian quaternion")
    opacity_logit = np.asarray(vertex["opacity"], dtype=np.float64)
    opacity = np.exp(-np.logaddexp(0, -opacity_logit)).astype(np.float32)
    result = dict(xyz=columns(["x", "y", "z"]), quats=rotation / norm,
                  scales=np.exp(columns([f"scale_{i}" for i in range(3)])), opacity=opacity,
                  sh=columns([f"f_dc_{i}" for i in range(3)])[:, None, :])
    if not all(np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError("Non-finite Gaussian attributes")
    return result


def articulated_gaussians(asset_dir: str | Path, geometry: dict, angle_deg: float) -> dict:
    """Apply one rigid transform to drawbar, front axle, and both front wheels."""
    low, high = geometry["angle_limits_deg"]
    if not low <= angle_deg <= high:
        raise ValueError(f"Drawbar angle {angle_deg} is outside [{low}, {high}]")
    asset_dir = Path(asset_dir)
    body = load_gaussians(asset_dir / "trailer_body_3dgs_local.ply")
    movable = load_gaussians(asset_dir / "trailer_movable_assembly_hinge_local.ply")
    axis = np.asarray(geometry["hinge_axis"], dtype=float)
    axis /= np.linalg.norm(axis)
    rotation = _axis_rotation(axis, angle_deg)
    movable["xyz"] = movable["xyz"] @ rotation.T + np.asarray(geometry["hinge_center_m"])
    half = math.radians(angle_deg) / 2
    a = np.r_[math.cos(half), axis * math.sin(half)]
    b = movable["quats"]
    w = a[0] * b[:, :1] - np.sum(a[1:] * b[:, 1:], axis=1, keepdims=True)
    v = a[0] * b[:, 1:] + b[:, :1] * a[1:] + np.cross(a[1:], b[:, 1:])
    movable["quats"] = np.column_stack((w, v)).astype(np.float32)
    result = {key: np.concatenate((body[key], movable[key]), axis=0) for key in body}
    result["xyz"][:, 2] += geometry["asset_height_m"]
    return result


def _background(camera: dict, centre: np.ndarray, view: np.ndarray, asset_dir: Path) -> np.ndarray:
    rays = fisheye_rays(camera) @ view[:3, :3]
    sky_blend = np.clip((rays[..., 2] + 0.12) / 0.85, 0, 1)[..., None]
    image = np.array([0.76, 0.78, 0.79]) * (1 - sky_blend) + np.array([0.55, 0.58, 0.61]) * sky_blend
    ground = rays[..., 2] < -1e-6
    distance = np.divide(-centre[2], rays[..., 2], out=np.zeros_like(rays[..., 2]), where=ground)
    ground &= distance > 0
    world = centre[:2] + distance[..., None] * rays[..., :2]
    texture = cv2.imread(str(asset_dir / "clean_ground_texture.png"))
    if texture is None:
        raise FileNotFoundError(asset_dir / "clean_ground_texture.png")
    texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    map_x = np.mod(world[..., 0] / 0.02, texture.shape[1]).astype(np.float32)
    map_y = np.mod(world[..., 1] / 0.02, texture.shape[0]).astype(np.float32)
    values = cv2.remap(texture, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    image[ground] = values[ground]
    return image.astype(np.float32)


def _render_gaussian(camera: dict, geometry: dict, assets: Path, view: np.ndarray,
                     centre: np.ndarray, angle_deg: float, device: str) -> np.ndarray:
    import torch
    try:
        from gsplat.rendering import rasterization
    except ImportError as exc:
        raise RuntimeError("Gaussian mode requires the optional gsplat CUDA renderer; see README") from exc
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise RuntimeError("Gaussian rendering requires a CUDA GPU; use mesh mode for a CPU preview")
    data = {key: torch.as_tensor(value, dtype=torch.float32, device=device).contiguous()
            for key, value in articulated_gaussians(assets, geometry, angle_deg).items()}
    size, intrinsic, faces = cube_faces(camera)
    foreground = np.zeros((camera["height"], camera["width"], 3), np.float32)
    alpha = np.zeros(foreground.shape[:2], np.float32)
    with torch.inference_mode():
        for face in faces:
            face_view = np.eye(4, dtype=np.float32)
            face_view[:3] = face["rotation"] @ view[:3]
            rgb, opacity, _ = rasterization(
                means=data["xyz"], quats=data["quats"], scales=data["scales"],
                opacities=data["opacity"], colors=data["sh"],
                viewmats=torch.as_tensor(face_view[None], device=device),
                Ks=torch.as_tensor(intrinsic[None], device=device),
                width=size, height=size, sh_degree=0, render_mode="RGB", packed=False,
            )
            _remap(foreground, rgb[0].cpu().numpy(), face)
            _remap(alpha, opacity[0, ..., 0].cpu().numpy(), face)
    return np.clip(foreground + (1 - alpha[..., None]) * _background(camera, centre, view, assets), 0, 1)


def load_mujoco_scene(config: dict, assets: str | Path = DEFAULT_ASSETS, ego_x: float = -2.0,
                      ego_y: float = 0.0, ego_yaw_deg: float = 180.0):
    """Build a static preview; no vehicle dynamics or insertion physics are implied."""
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("Mesh mode requires MuJoCo: pip install '.[simulation]'") from exc
    tree = ET.parse(ROOT / "configs" / "scene.xml")
    tree.find("compiler").set("meshdir", str(Path(assets).resolve()))
    tree.find("./worldbody/body[@name='trailer']").set("pos", f"0 0 {config['geometry']['asset_height_m']}")
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"))
    data = mujoco.MjData(model)
    ego_id = model.body("ego").mocapid[0]
    data.mocap_pos[ego_id] = [ego_x, ego_y, 0]
    half = math.radians(ego_yaw_deg) / 2
    data.mocap_quat[ego_id] = [math.cos(half), 0, 0, math.sin(half)]
    mujoco.mj_forward(model, data)
    return model, data


def _render_mesh(config: dict, camera: dict, assets: Path, centre: np.ndarray,
                 view: np.ndarray, ego_x: float, ego_y: float, ego_yaw_deg: float) -> np.ndarray:
    import mujoco
    model, data = load_mujoco_scene(config, assets, ego_x, ego_y, ego_yaw_deg)
    size, intrinsic, faces = cube_faces(camera, "mesh")
    model.vis.global_.offwidth = max(size, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(size, model.vis.global_.offheight)
    camera_id = model.camera("render_camera").id
    model.cam_pos[camera_id] = centre
    model.cam_fovy[camera_id] = math.degrees(2 * math.atan(size / (2 * intrinsic[1, 1])))
    result = np.zeros((camera["height"], camera["width"], 3), np.uint8)
    with mujoco.Renderer(model, height=size, width=size) as renderer:
        for face in faces:
            rotation_cv_world = face["rotation"] @ view[:3, :3]
            # Camera local axes: right, up, backward (MuJoCo looks down -Z).
            rotation_world_gl = rotation_cv_world.T @ np.diag([1, -1, -1])
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, rotation_world_gl.flatten())
            model.cam_quat[camera_id] = quat
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera_id)
            _remap(result, renderer.render(), face)
    return result.astype(np.float32) / 255


def render_scene(output: str | Path, *, config: str | Path | dict | None = None,
                 width: int | None = None, height: int | None = None,
                 ego_x: float = -2.0, ego_y: float = 0.0, ego_yaw_deg: float = 180.0,
                 drawbar_angle_deg: float = 0.0, mode: str = "mesh",
                 asset_dir: str | Path | None = None, device: str = "cuda:0") -> dict:
    """Write RGB image and a JSON sidecar with exact calibration and scene pose.

    Relative body pose theta=[x_base,y_base,yaw_base,q] is derived from the known
    simulator state for optional synthetic verification. Never use this sidecar
    as an initializer for evaluating the image-based solver.
    """
    started = time.monotonic()
    cfg = load_scene_config(config)
    camera = scaled_camera(cfg["camera"], width, height)
    centre, view = camera_view(camera, ego_x, ego_y, ego_yaw_deg)
    assets = Path(asset_dir) if asset_dir else DEFAULT_ASSETS
    if mode == "gaussian":
        image = _render_gaussian(camera, cfg["geometry"], assets, view, centre, drawbar_angle_deg, device)
    elif mode == "mesh":
        if abs(drawbar_angle_deg) > 1e-9:
            raise ValueError("The complete mesh is static. Use Gaussian mode for articulated angles.")
        image = _render_mesh(cfg, camera, assets, centre, view, ego_x, ego_y, ego_yaw_deg)
    else:
        raise ValueError("mode must be mesh or gaussian")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), cv2.cvtColor(np.round(image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write {output}")
    yaw = math.radians(ego_yaw_deg)
    body_xy = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]]) @ [-ego_x, -ego_y]
    state = [float(body_xy[0]), float(body_xy[1]), (math.pi - yaw) % (2 * math.pi) - math.pi,
             math.radians(drawbar_angle_deg)]
    report = dict(mode=mode, image=output.name, camera=camera, geometry_config={**cfg, "camera": camera},
                  asset_height_m=cfg["geometry"]["asset_height_m"],
                  simulator_truth_state_xy_yaw_q=state, camera_world_m=centre.tolist(),
                  ego_world=[ego_x, ego_y, ego_yaw_deg], drawbar_angle_deg=drawbar_angle_deg,
                  projection="equidistant_six_face_cubemap", elapsed_seconds=time.monotonic() - started)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def show_scene(*, config: str | Path | dict | None = None, ego_x: float = -2.0,
               ego_y: float = 0.0, ego_yaw_deg: float = 180.0, asset_dir: str | Path | None = None) -> None:
    """Open MuJoCo's interactive orbit viewer around the static whole-mesh scene."""
    import mujoco.viewer
    cfg = load_scene_config(config)
    model, data = load_mujoco_scene(cfg, asset_dir or DEFAULT_ASSETS, ego_x, ego_y, ego_yaw_deg)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.7, 0, 0.5]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 8.0, 130.0, -30.0
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)
