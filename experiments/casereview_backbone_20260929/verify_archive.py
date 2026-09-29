"""Validate packaged image bytes without extraction or loading any model."""
import argparse,hashlib,json,tarfile
from pathlib import Path
from common import EXPECTED,verify,read_rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--dataset',choices=list(EXPECTED)+['all'],default='all');a=p.parse_args()
    for name in EXPECTED if a.dataset=='all' else [a.dataset]:
        root=a.data_root/name;jobs=verify(root,images=False)
        expected={r['image']:r['sha256'] for r in read_rows(root/'manifest.jsonl')};seen=set()
        with tarfile.open(root/'images.tar') as archive:
            for member in archive:
                assert member.isfile() and member.name in expected and member.name not in seen
                f=archive.extractfile(member);h=hashlib.sha256()
                for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
                assert h.hexdigest()==expected[member.name],member.name
                seen.add(member.name)
        assert seen==set(expected)
        print(json.dumps(dict(dataset=name,status='PASS',archived_images=len(seen),jobs=len(jobs))),flush=True)

if __name__=='__main__':main()
