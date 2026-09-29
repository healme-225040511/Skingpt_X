import hashlib, json
from pathlib import Path

EXPECTED={'DermNet-Curated-A':(197,8512,2343),'DermNet-Curated-B':(333,13438,3812),'DermNet-Curated-C':(385,13089,3718)}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def read_rows(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]

def verify(root,images=True):
    root=Path(root); k,ntr,nte=EXPECTED[root.name]
    audit=json.loads((root/'audit.json').read_text())
    assert audit['status']=='PASS' and audit['num_classes']==k
    for name,digest in audit['checksums'].items():
        assert sha(root/name)==digest,('Checksum mismatch',root/name)
    labels=json.loads((root/'classes.json').read_text())
    manifest=read_rows(root/'manifest.jsonl'); jobs=read_rows(root/'case_jobs.jsonl')
    assert len(labels)==len(set(labels))==k and labels==sorted(labels)
    assert sum(r['split']=='train' for r in manifest)==ntr
    assert sum(r['split']=='test' for r in manifest)==nte
    assert {r['label'] for r in manifest if r['split']=='train'}==set(labels)
    assert len({r['sample_id'] for r in manifest})==len(manifest)==ntr+nte
    assert len({r['sha256'] for r in manifest})==len(manifest)
    assert len(jobs)==len({r['job_id'] for r in jobs})==nte
    m={r['sample_id']:r for r in manifest}
    assert {r['sample_id'] for r in jobs}=={r['sample_id'] for r in manifest if r['split']=='test'}
    for r in manifest:
        p=(root/r['image']).resolve()
        assert p.is_relative_to(root.resolve()) and r['label'] in labels
        if images: assert sha(p)==r['sha256'],('Image checksum mismatch',r['sample_id'])
    for j in jobs:
        assert j['dataset']==root.name and j['ground_truth']==m[j['sample_id']]['label']
        assert j['image']==m[j['sample_id']]['image'] and j['image_sha256']==m[j['sample_id']]['sha256']
        assert hashlib.sha256(j['prompt'].encode()).hexdigest()==j['prompt_sha256']
        assert len(j['candidate_labels'])==len(set(j['candidate_labels']))==5
        assert set(j['candidate_labels'])<=set(labels) and j['panderm_prediction'] in j['candidate_labels']
    return jobs

def parse_choice(response,candidates):
    # Historical finalize_ontology_benchmark.py semantics, with non-dict JSON treated as invalid.
    import re
    value=re.sub(r'^```(?:json)?\s*|\s*```$','',str(response).strip(),flags=re.I)
    match=re.search(r'\{.*\}',value,flags=re.S); obj=None
    for text in [value,match.group(0) if match else None]:
        if text:
            try: obj=json.loads(text); break
            except (ValueError,TypeError): pass
    v=str(obj.get('PrimaryDiagnosis') or '').strip().casefold() if isinstance(obj,dict) else ''
    exact={x.casefold():x for x in candidates}
    if v in exact: return exact[v]
    contained=[label for key,label in exact.items() if key in v or v in key]
    return contained[0] if len(contained)==1 else None
