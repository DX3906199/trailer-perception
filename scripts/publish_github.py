"""Publish this reviewed export to the explicitly requested public repository.

Requires GitHub CLI authentication as DX3906199. No token is accepted via
arguments or written into the repository. Existing unrelated repositories and
release assets are never overwritten.
"""
import argparse
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
REPOSITORY='DX3906199/trailer-perception'

def run(*args):
    return subprocess.check_output(args,cwd=ROOT,text=True).strip()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gh',default='gh')
    args=parser.parse_args()
    if run(args.gh,'api','user','--jq','.login').lower()!='dx3906199':
        raise RuntimeError('Authenticate GitHub CLI as DX3906199 before publishing')
    if run('git','status','--porcelain'):
        raise RuntimeError('Commit reviewed source changes before publishing')
    remote=subprocess.run(['git','remote','get-url','origin'],cwd=ROOT,text=True,capture_output=True)
    if remote.returncode:
        exists=subprocess.run([args.gh,'repo','view',REPOSITORY],cwd=ROOT,capture_output=True)
        if exists.returncode==0:
            raise RuntimeError('Repository already exists. Review it and attach its origin explicitly; refusing to replace it.')
        run(args.gh,'repo','create',REPOSITORY,'--public','--source',str(ROOT),'--remote','origin',
            '--description','DINOv2 five-keypoint trailer detection, fisheye articulated PnP, and MuJoCo/3DGS scenes')
    elif remote.stdout.strip().rstrip('/').removesuffix('.git') not in (
            'https://github.com/'+REPOSITORY,'git@github.com:'+REPOSITORY):
        raise RuntimeError('Unexpected git origin')
    run('git','push','-u','origin','main')
    run('git','push','origin','v0.1.0')
    release=subprocess.run([args.gh,'release','view','v0.1.0','--repo',REPOSITORY],cwd=ROOT,capture_output=True)
    if release.returncode:
        run(args.gh,'release','create','v0.1.0',str(ROOT/'weights/dinov2_m0_896_mixed.pt'),
            '--repo',REPOSITORY,'--verify-tag','--title','v0.1.0: M0 detection and articulated pose',
            '--notes-file',str(ROOT/'docs/RELEASE_NOTES.md'))
    else:
        print('Release already exists: verify its weight SHA against weights/manifest.json.')
    print('https://github.com/'+REPOSITORY)

if __name__=='__main__':main()
