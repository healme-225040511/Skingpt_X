"""Complete-cohort paired scoring with historical fallback and explicit failures."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,matthews_corrcoef,cohen_kappa_score
from common import EXPECTED,verify,read_rows,parse_choice,sha

def metrics(y,p):
    return {'n':len(y),'accuracy':accuracy_score(y,p),'balanced_accuracy':balanced_accuracy_score(y,p),
            'macro_f1':f1_score(y,p,average='macro',zero_division=0),
            'weighted_f1':f1_score(y,p,average='weighted',zero_division=0),
            'mcc':matthews_corrcoef(y,p),'kappa':cohen_kappa_score(y,p)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--outputs',type=Path,nargs='+',required=True)
    p.add_argument('--reference-outputs',type=Path,nargs='+',help='Optional matched-engine 8B runs; otherwise archived 8B predictions')
    p.add_argument('--report-dir',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=10000)
    a=p.parse_args(); assert a.bootstrap>=1000
    if a.report_dir.exists():raise SystemExit('Use a new report directory')
    jobs={j['job_id']:j for name in EXPECTED for j in verify(a.data_root/name,images=False)}
    def load(paths):
        rows=[]; ids=set(); model_configs=[]
        for path in paths:
            meta=json.loads(path.with_suffix('.meta.json').read_text())
            assert meta['jobs_sha256']==sha(a.data_root/meta['dataset']/'case_jobs.jsonl')
            import hashlib
            sig=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
            model_configs.append({k:v for k,v in meta.items() if k not in ['dataset','jobs_sha256','shard_index','num_shards','gpus']})
            for r in read_rows(path):
                assert r['job_id'] in jobs and r['job_id'] not in ids,'Duplicate/unexpected output'
                j=jobs[r['job_id']]; assert r['run_signature']==sig
                assert r['dataset']==j['dataset']==meta['dataset']
                assert r['prompt_sha256']==j['prompt_sha256'] and r['image_sha256']==j['image_sha256']
                ids.add(r['job_id']); rows.append(r)
        assert ids==set(jobs),f'Incomplete comparison: {len(ids)}/{len(jobs)}'
        assert all(m==model_configs[0] for m in model_configs),'Mixed model/settings across datasets'
        return {r['job_id']:r for r in rows},model_configs[0]
    outputs,target_meta=load(a.outputs)
    assert target_meta['model_id']=='Qwen/Qwen3-VL-30B-A3B-Instruct-FP8'
    reference,ref_meta=load(a.reference_outputs) if a.reference_outputs else (None,None)
    if ref_meta:
        assert ref_meta['model_id']=='Qwen/Qwen3-VL-8B-Instruct'
        assert ref_meta['generation']==target_meta['generation'] and ref_meta['versions']==target_meta['versions']
    pairs=[]; results={}; rng=np.random.default_rng(20260929)
    for name in EXPECTED:
        group=[]
        for j in jobs.values():
            if j['dataset']!=name:continue
            r=outputs[j['job_id']]; pred=parse_choice(r['response'],j['candidate_labels'])
            old=parse_choice(reference[j['job_id']]['response'],j['candidate_labels']) if reference else j['historical_8b_prediction']
            row={'dataset':name,'sample_id':j['sample_id'],'ground_truth':j['ground_truth'],
                 'reference_prediction':old or j['panderm_prediction'],'target_prediction':pred or j['panderm_prediction'],
                 'target_valid':pred is not None,'reference_valid':old is not None if reference else j['historical_8b_valid'],
                 'panderm_prediction':j['panderm_prediction'],'seconds':r['seconds'],
                 'input_tokens':r['input_tokens'],'output_tokens':r['output_tokens'],'finish_reason':r['finish_reason']}
            group.append(row);pairs.append(row)
        y=[r['ground_truth'] for r in group];old=[r['reference_prediction'] for r in group];new=[r['target_prediction'] for r in group]
        oc=np.array(y)==np.array(old);nc=np.array(y)==np.array(new);d=nc.astype(int)-oc.astype(int)
        draws=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(a.bootstrap)])
        rescue=int((~oc&nc).sum());harm=int((oc&~nc).sum())
        results[name]={'reference':metrics(y,old),'target':metrics(y,new),'panderm':metrics(y,[r['panderm_prediction'] for r in group]),
                       'accuracy_delta':float(d.mean()),'delta_95ci':np.quantile(draws,[.025,.975]).tolist(),
                       'rescue':rescue,'harm':harm,'mcnemar_p':float(binomtest(rescue,rescue+harm,.5).pvalue) if rescue+harm else 1.,
                       'target_invalid':sum(not r['target_valid'] for r in group),
                       'target_accuracy_invalid_as_wrong':sum(r['target_valid'] and r['target_prediction']==r['ground_truth'] for r in group)/len(group),
                       'reference_invalid':sum(not r['reference_valid'] for r in group),
                       'reference_accuracy_invalid_as_wrong':sum(r['reference_valid'] and r['reference_prediction']==r['ground_truth'] for r in group)/len(group),
                       'target_seconds_total':sum(r['seconds'] for r in group),
                       'target_output_tokens_total':sum(r['output_tokens'] for r in group)}
    ordered=sorted(results,key=lambda n:results[n]['mcnemar_p']); prev=0.
    for i,name in enumerate(ordered):
        prev=max(prev,min(1.,results[name]['mcnemar_p']*(len(ordered)-i)));results[name]['holm_p']=prev
    a.report_dir.mkdir(parents=True)
    with (a.report_dir/'paired_predictions.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    (a.report_dir/'metrics.json').write_text(json.dumps({'reference_mode':'matched_engine_8b' if reference else 'historical_transformers_8b',
        'target_metadata':target_meta,'reference_metadata':ref_meta,'datasets':results,
        'limitations':['Dense BF16 vs MoE FP8 changes architecture and precision, not parameter count alone.',
                       'Historical 8B comparison additionally changes inference engine; optional matched-engine 8B removes this extra confound.',
                       'Datasets share source images; do not pool them as independent cohorts.',
                       'Primary scoring retains historical PanDerm Top-1 fallback.']},indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
