import json
from pathlib import Path
from trailer_perception.cli import main

ROOT=Path(__file__).resolve().parents[1]

def test_geometry_cli(tmp_path):
    output=tmp_path/'demo.json'
    main(['geometry-demo','--config',str(ROOT/'configs/geometry.yaml'),'--output',str(output)])
    value=json.loads(output.read_text())
    assert value['estimate']['valid']
    assert abs(value['estimate']['hitch_base_m'][0]+2.5)<1e-7

def test_pnp_cli_missing_landmark(tmp_path):
    source=tmp_path/'points.json'
    source.write_text(json.dumps({'keypoints':[[None,None]]*5}))
    target=tmp_path/'pose.json'
    main(['pnp','--points',str(source),'--config',str(ROOT/'configs/geometry.yaml'),'--output',str(target)])
    assert json.loads(target.read_text())['hitch_base_m'] is None
