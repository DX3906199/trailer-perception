"""Export tensor-only M0 inference weights without private training metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True)
    parser.add_argument("--output",default="weights/dinov2_m0_896_mixed.pt")
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists(): raise FileExistsError(output)
    source=torch.load(args.source,map_location="cpu",weights_only=True)
    if source['variant']!='M1' or source['image_size']!=896: raise ValueError('Expected public M0 896')
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=source['model'],variant='M1',experiment_name='M0',image_size=896),output)
    checksum=hashlib.sha256(output.read_bytes()).hexdigest()
    manifest=dict(name='DINOv2 M0 mixed 896',file=output.name,bytes=output.stat().st_size,sha256=checksum,
        release_url='https://github.com/DX3906199/trailer-perception/releases/download/v0.1.0/'+output.name,
        architecture='Frozen DINOv2 ViT-S/14 + global five-heatmap decoder',
        training=dict(synthetic_images=1000,real_images=100,mixed_finetuning_epochs=20,
                      checkpoint_selection='final epoch 20'),
        keypoints=['body_front_right_lower','body_front_right_upper_rectangle','body_front_left_upper_rectangle','body_front_left_lower','hitch_center'])
    (output.parent/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
