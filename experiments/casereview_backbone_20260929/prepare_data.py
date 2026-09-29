"""Download ONLY benchmark assets, safely extract and verify. Never downloads a model."""
import argparse,json,tarfile
from pathlib import Path
from common import EXPECTED,verify

def extract(root):
    with tarfile.open(root/'images.tar') as t:
        for m in t.getmembers():
            target=(root/m.name).resolve()
            if not target.is_relative_to(root.resolve()) or not m.isfile():
                raise ValueError(f'Unsafe/non-file archive member: {m.name}')
        t.extractall(root,filter='data')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--dataset',choices=list(EXPECTED)+['all'],default='all')
    p.add_argument('--revision',help='Exact HF dataset commit SHA (required for download)')
    p.add_argument('--local-only',action='store_true'); a=p.parse_args()
    names=list(EXPECTED) if a.dataset=='all' else [a.dataset]
    if not a.local_only:
        import re
        if not a.revision or not re.fullmatch('[0-9a-f]{40}',a.revision): p.error('Supply a pinned 40-character dataset commit, not main.')
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id='czty/SkinGPTX-Images',repo_type='dataset',revision=a.revision,
                          allow_patterns=[f'{x}/*' for x in names],local_dir=a.data_root)
    for name in names:
        root=a.data_root/name
        verify(root,images=False)
        extract(root); verify(root)
        print(name,'PASS',flush=True)

if __name__=='__main__':main()
