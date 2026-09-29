"""Export audited refined cohorts and exact historical CaseReview prompts. CPU only."""
import argparse, csv, hashlib, json, tarfile
from collections import Counter
from pathlib import Path

SPECS = [('DermNet225_OntologyV1','DermNet-Curated-A',197,8512,2343),
         ('DermNet353_OntologyV1','DermNet-Curated-B',333,13438,3812),
         ('DermNet498_OntologyV1','DermNet-Curated-C',385,13089,3718)]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def read_jsonl(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]

def write_json(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def write_jsonl(p,rows):
    Path(p).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset-root',type=Path,required=True)
    ap.add_argument('--benchmark-root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists(): raise SystemExit('Use a new output directory; existing bundles are never overwritten.')
    a.output.mkdir(parents=True)
    jobs=[r for p in sorted((a.benchmark_root/'case_jobs').glob('*.jsonl')) for r in read_jsonl(p)]
    assert len(jobs)==9873 and len({r['job_id'] for r in jobs})==9873
    baseline=list(csv.DictReader((a.benchmark_root/'paired_predictions.csv').open()))
    assert len(baseline)==9873 and len({(r['dataset'],r['sample_id']) for r in baseline})==9873
    for old,new,k,ntrain,ntest in SPECS:
        dest=a.output/new; dest.mkdir()
        manifest=a.dataset_root/old/'manifest.csv'
        allrows=list(csv.DictReader(manifest.open()))
        primary=[r for r in allrows if r['endpoint_partition']=='primary_candidate']
        train=[r for r in primary if r['split']=='train']
        labels=sorted({r['canonical_diagnosis_candidate'] for r in train})
        test=[r for r in primary if r['split']=='test' and r['canonical_diagnosis_candidate'] in labels]
        assert (len(labels),len(train),len(test))==(k,ntrain,ntest),(new,len(labels),len(train),len(test))
        excluded=[{'sample_id':r['sample_id'],'label':r['canonical_diagnosis_candidate'],'reason':'test-only label absent from training'} for r in primary if r['split']=='test' and r['canonical_diagnosis_candidate'] not in labels]
        rows=train+test; ids={r['sample_id']:r for r in rows}
        assert len(ids)==len(rows)
        js=[r for r in jobs if r['dataset']==old]; bs=[r for r in baseline if r['dataset']==old]
        expected={r['sample_id'] for r in test}
        assert {r['sample_id'] for r in js}=={r['sample_id'] for r in bs}==expected
        bmap={r['sample_id']:r for r in bs}; exported=[]; hashes={'train':set(),'test':set()}
        with tarfile.open(dest/'images.tar','w') as tar:
            for r in rows:
                source=Path(r['materialized_path']); digest=sha(source)
                assert digest==r['sample_id'],('Image/sample hash mismatch',r['sample_id'])
                assert digest not in hashes['train']|hashes['test']
                hashes[r['split']].add(digest)
                rel=f"images/{r['split']}/{r['sample_id']}{source.suffix.lower()}"
                tar.add(source,arcname=rel,recursive=False)
                exported.append({'sample_id':r['sample_id'],'split':r['split'],'label':r['canonical_diagnosis_candidate'],
                                 'image':rel,'sha256':digest,'endpoint_partition':'primary_candidate'})
        imap={r['sample_id']:r for r in exported}; ej=[]
        for r in sorted(js,key=lambda x:x['job_id']):
            sid=r['sample_id']; b=bmap[sid]
            assert r['ground_truth']==b['ground_truth']==imap[sid]['label']
            assert r['candidate_labels']==json.loads(b['candidate_labels'])
            assert r['panderm_prediction']==b['panderm_prediction']
            assert len(r['candidate_labels'])==5 and set(r['candidate_labels'])<=set(labels)
            assert sha(r['image_path'])==imap[sid]['sha256']
            ej.append({'job_id':r['job_id'],'sample_id':sid,'dataset':new,'legacy_dataset':old,
                       'image':imap[sid]['image'],'image_sha256':imap[sid]['sha256'],
                       'ground_truth':r['ground_truth'],'candidate_labels':r['candidate_labels'],
                       'panderm_prediction':r['panderm_prediction'],'prompt':r['prompt'],
                       'prompt_sha256':hashlib.sha256(r['prompt'].encode()).hexdigest(),
                       'historical_8b_prediction':b['skingptx_prediction'],
                       'historical_8b_valid':b['valid_model_output'].lower()=='true'})
        write_jsonl(dest/'manifest.jsonl',exported); write_jsonl(dest/'case_jobs.jsonl',ej)
        write_json(dest/'classes.json',labels); write_json(dest/'excluded_test_only.json',excluded)
        tr=Counter(r['label'] for r in exported if r['split']=='train'); te=Counter(r['label'] for r in exported if r['split']=='test')
        write_json(dest/'class_support.json',[{'label':x,'train':tr[x],'test':te[x]} for x in labels])
        write_json(dest/'audit.json',{'status':'PASS','name':new,'legacy_name':old,'num_classes':k,
                   'train_images':ntrain,'test_images':ntest,'jobs':len(ej),'exact_train_test_overlap':0,
                   'source_manifest_sha256':sha(manifest),'prompt_policy':'byte-identical historical 8B case_jobs prompts',
                   'scope':'fixed evidence CaseReview only','dermatologist_review':'pending',
                   'checksums':{p.name:sha(p) for p in dest.iterdir() if p.is_file()}})
        (dest/'README.md').write_text(f'# {new}\n\nRefined strict benchmark derived from {old}.\n\n'
            f'{k} training-supported labels; {ntrain:,} train images; {ntest:,} test images.\n\n'
            'Only primary_candidate images are included. Ambiguous/review-required, modality holdout, excluded and unsupported test-only labels are not part of this benchmark.\n\n'
            'Engineering refinement is NOT completed dermatologist adjudication. Dataset names do not imply nesting. '
            'Image sources retain their original rights and restrictions; this export grants no new image license. '
            'No patient-level or pretraining-overlap independence claim is made.\n\n'
            'images.tar contains relative images/train and images/test paths. manifest.jsonl and classes.json are authoritative. '
            'case_jobs.jsonl preserves original model-visible prompts and frozen evidence. Ground truth is scoring metadata only. '
            'Training images are provided for provenance, not for retraining during this controlled experiment.\n')
        print(new,'PASS',flush=True)

if __name__=='__main__': main()
