"""Render calibrated MuJoCo/3DGS images or open the MuJoCo scene viewer."""
import argparse
from pathlib import Path
import yaml
from trailer_perception.scene import render_scene,show_scene,load_scene_config

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mode',choices=['mesh','gaussian','viewer'],default='mesh')
parser.add_argument('--config',default='configs/geometry.yaml')
parser.add_argument('--x',type=float,default=-2.0,help='Ego world X in metres')
parser.add_argument('--y',type=float,default=0.0,help='Ego world Y in metres')
parser.add_argument('--yaw',type=float,default=180.0,help='Ego world yaw in degrees')
parser.add_argument('--angle',type=float,default=0.0,help='Drawbar joint angle in degrees (Gaussian mode)')
parser.add_argument('--width',type=int)
parser.add_argument('--height',type=int)
parser.add_argument('--device',default='cuda:0')
parser.add_argument('--output',default='outputs/scene.png')
args=parser.parse_args()
if args.mode=='viewer':
    if args.angle!=0: parser.error('Whole-mesh viewer only supports angle 0')
    show_scene(config=args.config,ego_x=args.x,ego_y=args.y,ego_yaw_deg=args.yaw)
else:
    report=render_scene(args.output,config=args.config,width=args.width,height=args.height,
        ego_x=args.x,ego_y=args.y,ego_yaw_deg=args.yaw,drawbar_angle_deg=args.angle,
        mode=args.mode,device=args.device)
    config=load_scene_config(args.config)
    config['camera']=report['camera']
    target=Path(args.output).with_suffix('.yaml')
    target.write_text(yaml.safe_dump(config,sort_keys=False))
    print('Image:',args.output,'\nCalibrated inference config:',target)
