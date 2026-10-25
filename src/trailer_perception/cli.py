"""Small ROS-independent command-line interface; all paths are explicit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import yaml

from .geometry import project_landmarks
from .pose import solve_pose


def read_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def write_json(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def overlay(image_bgr, detection, pose=None):
    image = image_bgr.copy()
    colors = [(70,220,90), (255,175,30), (50,220,245), (220,120,40), (230,70,245)]
    for index, (point, color) in enumerate(zip(detection["keypoints"], colors)):
        if point[0] is None or not np.isfinite(point).all():
            continue
        pixel = tuple(np.rint(point).astype(int))
        cv2.circle(image, pixel, 7, (0,0,0), 3, cv2.LINE_AA)
        cv2.circle(image, pixel, 5, color, 2, cv2.LINE_AA)
        cv2.putText(image, str(index), (pixel[0]+10,pixel[1]-8), cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2, cv2.LINE_AA)
    if pose is not None:
        if pose["valid"]:
            x,y,z = pose["hitch_base_m"]
            message = f"hitch base: ({x:+.3f}, {y:+.3f}, {z:+.3f}) m | q={np.rad2deg(pose['state'][3]):+.2f} deg"
        else:
            message = "Pose invalid: " + pose["reason"]
        cv2.rectangle(image, (0,0), (image.shape[1],55), (24,24,24), -1)
        cv2.putText(image, message, (16,37), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2, cv2.LINE_AA)
    return image


def infer(image, detector, config, roi=None):
    height, width = image.shape[:2]
    if (width,height) != (config["camera"]["width"],config["camera"]["height"]):
        raise ValueError("Image dimensions differ from camera calibration. Update calibration or supply original-size images; do not silently resize PnP input.")
    detected = detector.predict(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), roi=roi)
    # Raw heatmap responses are NOT calibrated visibility probabilities.
    pose = solve_pose(detected["keypoints"], config)
    return {"detection": detected, "pose": pose}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("geometry-demo", help="Project known geometry and recover it; no model required")
    demo.add_argument("--config", default="configs/geometry.yaml")
    demo.add_argument("--state", type=float, nargs=4, default=[-2.5,0,np.pi,0], metavar=("X","Y","YAW_RAD","Q_RAD"))
    demo.add_argument("--output", default="outputs/geometry_demo.json")
    pnp = sub.add_parser("pnp", help="Solve original-image five-point JSON")
    pnp.add_argument("--points", required=True)
    pnp.add_argument("--config", default="configs/geometry.yaml")
    pnp.add_argument("--output", default="outputs/pose.json")
    for command in ("detect", "run", "video"):
        item = sub.add_parser(command, help={"detect":"Image to five pixels", "run":"Image to five pixels and metric pose", "video":"Video to overlays and per-frame JSONL"}[command])
        item.add_argument("--input", required=True)
        item.add_argument("--weights", default="weights/dinov2_m0_896_mixed.pt")
        item.add_argument("--device", default="cpu")
        item.add_argument("--threads", type=int, default=4)
        item.add_argument("--roi", type=float, nargs=4, metavar=("X0","Y0","X1","Y1"))
        item.add_argument("--output", default="outputs/"+command)
        item.add_argument("--config", default="configs/geometry.yaml")
    args = parser.parse_args(argv)
    if args.command == "geometry-demo":
        config = read_config(args.config)
        pixels = project_landmarks(args.state,config)
        if not np.isfinite(pixels).all():
            parser.error("Requested demo landmarks are behind the camera")
        result = {"known_state":args.state,"keypoints":pixels.tolist(),"estimate":solve_pose(pixels,config)}
        write_json(args.output,result)
        print(json.dumps(result,indent=2,allow_nan=False))
        return
    if args.command == "pnp":
        data = json.loads(Path(args.points).read_text())
        detection = data.get("detection",data) if isinstance(data,dict) else {"keypoints":data}
        result = solve_pose(detection["keypoints"],read_config(args.config),detection.get("visibility"))
        write_json(args.output,result)
        print(json.dumps(result,indent=2,allow_nan=False))
        return
    from .detector import Detector
    import torch
    if args.threads <= 0:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    detector = Detector(args.weights,args.device)
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    config = read_config(args.config)
    if args.command != "video":
        image = cv2.imread(args.input)
        if image is None:
            raise FileNotFoundError(args.input)
        start = time.monotonic()
        result = ({"detection":detector.predict(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),args.roi)}
                  if args.command == "detect" else infer(image,detector,config,args.roi))
        result["elapsed_seconds"] = time.monotonic()-start
        write_json(output/"prediction.json",result)
        if not cv2.imwrite(str(output/"overlay.jpg"),overlay(image,result["detection"],result.get("pose"))):
            raise OSError("Failed to write overlay")
        print(json.dumps(result,indent=2,allow_nan=False))
        return
    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        raise FileNotFoundError(args.input)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 15.0
    writer = None
    index = 0
    try:
        with (output/"predictions.jsonl").open("w") as stream:
            while True:
                ok,image = capture.read()
                if not ok: break
                result = infer(image,detector,config,args.roi)
                result.update(frame_index=index,video_seconds=index/fps)
                stream.write(json.dumps(result,allow_nan=False)+"\n")
                if writer is None:
                    writer = cv2.VideoWriter(str(output/"overlay.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),fps,(image.shape[1],image.shape[0]))
                    if not writer.isOpened(): raise OSError("OpenCV MP4 writer unavailable")
                writer.write(overlay(image,result["detection"],result["pose"]))
                index += 1
                if index%10 == 0: print(f"processed {index} frames",flush=True)
    finally:
        capture.release()
        if writer is not None: writer.release()
    if not index: raise ValueError("Video contains no decodable frames")
    write_json(output/"summary.json",{"frames":index,"fps":fps,"complete":True})
    print(f"Completed {index} frames: {output}")
