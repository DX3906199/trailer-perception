"""Download the released M0 checkpoint and verify its pinned SHA-256."""
import hashlib
import json
from pathlib import Path
import urllib.request

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'weights/manifest.json').read_text())
target=root/'weights'/manifest['file']
if target.exists():
    if hashlib.sha256(target.read_bytes()).hexdigest()!=manifest['sha256']:
        raise ValueError('Existing weight checksum differs; move it aside before downloading')
    print('Weights already verified:',target)
else:
    partial=target.with_suffix('.download')
    try:
        request=urllib.request.Request(manifest['release_url'],headers={'User-Agent':'trailer-perception/0.1.0'})
        with urllib.request.urlopen(request,timeout=60) as response,partial.open('wb') as output:
            while chunk:=response.read(1024*1024): output.write(chunk)
        if partial.stat().st_size!=manifest['bytes'] or hashlib.sha256(partial.read_bytes()).hexdigest()!=manifest['sha256']:
            raise ValueError('Downloaded weight checksum/size mismatch')
        partial.replace(target)
        print('Downloaded and verified:',target)
    except Exception:
        if partial.exists(): partial.unlink()
        raise
