"""Fixed-evidence CaseReview. Requires existing local model files; no model download."""
import argparse,hashlib,importlib.metadata,json,os,platform,time
from pathlib import Path
from common import EXPECTED,verify,sha,read_rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True); p.add_argument('--dataset',choices=EXPECTED,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--model-id',choices=['Qwen/Qwen3-VL-30B-A3B-Instruct-FP8','Qwen/Qwen3-VL-8B-Instruct'],required=True)
    p.add_argument('--model-revision',required=True,help='Record exact checkpoint revision downloaded externally')
    p.add_argument('--tensor-parallel-size',type=int,default=1)
    p.add_argument('--max-model-len',type=int,default=32768)
    p.add_argument('--gpu-memory-utilization',type=float,default=.85)
    p.add_argument('--num-shards',type=int,default=1); p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--preflight-only',action='store_true'); a=p.parse_args()
    assert a.num_shards>=1 and 0<=a.shard_index<a.num_shards
    root=a.data_root/a.dataset; jobs=verify(root)
    jobs=[j for i,j in enumerate(jobs) if i%a.num_shards==a.shard_index]
    if a.preflight_only:
        print(json.dumps({'status':'PASS','dataset':a.dataset,'shard_jobs':len(jobs),'model_loaded':False})); return
    import re
    assert re.fullmatch(r'[0-9a-f]{40}',a.model_revision),'Use the exact 40-character checkpoint commit'
    assert a.model_path.is_dir() and (a.model_path/'config.json').is_file(),'Local model snapshot required'
    cfg=json.loads((a.model_path/'config.json').read_text())
    if a.model_id.endswith('-FP8'):
        assert cfg.get('model_type')=='qwen3_vl_moe',cfg.get('model_type')
        assert 'fp8' in json.dumps(cfg.get('quantization_config',{})).lower(),'Not an FP8 checkpoint'
    else: assert cfg.get('model_type')=='qwen3_vl'
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM,SamplingParams
    assert torch.cuda.is_available()
    if a.model_id.endswith('-FP8'):
        assert all(torch.cuda.get_device_capability(i)>=(8,9) for i in range(torch.cuda.device_count())), 'FP8 requires suitable GPUs; no silent BF16 fallback.'
    meta={'model_id':a.model_id,'model_revision':a.model_revision,'model_config_sha256':sha(a.model_path/'config.json'),
          'snapshot_metadata':{x.name:sha(x) for x in sorted(a.model_path.iterdir()) if x.is_file() and x.suffix in ['.json','.jinja']},
          'weight_files':{x.name:x.stat().st_size for x in a.model_path.glob('*.safetensors')},
          'dataset':a.dataset,'jobs_sha256':sha(root/'case_jobs.jsonl'),'protocol':'fixed-evidence-casereview-v1',
          'generation':{'max_new_tokens':320,'temperature':0.0,'min_pixels':200704,'max_pixels':501760,'seed':0},
          'tensor_parallel_size':a.tensor_parallel_size,'max_model_len':a.max_model_len,
          'gpu_memory_utilization':a.gpu_memory_utilization,'num_shards':a.num_shards,'shard_index':a.shard_index,
          'python':platform.python_version(),'versions':{x:importlib.metadata.version(x) for x in ['torch','transformers','vllm','Pillow']},
          'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    assert meta['weight_files'],'No local safetensors found'
    signature=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True); mp=a.output.with_suffix('.meta.json')
    if mp.exists(): assert json.loads(mp.read_text())==meta,'Resume configuration changed; use a new output'
    else:
        assert not a.output.exists(),'Output exists without metadata'
        mp.write_text(json.dumps(meta,indent=2)+'\n')
    previous=read_rows(a.output) if a.output.exists() else []; done={r['job_id'] for r in previous}
    assert len(done)==len(previous) and done<={j['job_id'] for j in jobs}
    jmap={j['job_id']:j for j in jobs}
    for r in previous:
        assert r['run_signature']==signature and r['prompt_sha256']==jmap[r['job_id']]['prompt_sha256']
        assert r['image_sha256']==jmap[r['job_id']]['image_sha256']
    if len(done)==len(jobs): print('Already complete'); return
    processor=AutoProcessor.from_pretrained(a.model_path,local_files_only=True,min_pixels=200704,max_pixels=501760)
    llm=LLM(model=str(a.model_path.resolve()),tokenizer=str(a.model_path.resolve()),dtype='bfloat16',
            tensor_parallel_size=a.tensor_parallel_size,max_model_len=a.max_model_len,
            gpu_memory_utilization=a.gpu_memory_utilization,seed=0,generation_config='auto',
            mm_processor_kwargs={'min_pixels':200704,'max_pixels':501760},limit_mm_per_prompt={'image':1})
    sampling=SamplingParams(temperature=0,max_tokens=320,seed=0)
    with a.output.open('a') as out:
        for j in jobs:
            if j['job_id'] in done:continue
            # Scoring metadata (ground truth, baseline prediction) is NEVER sent to the model.
            text=processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':j['prompt']}]}],tokenize=False,add_generation_prompt=True)
            with Image.open(root/j['image']) as src: im=src.convert('RGB')
            start=time.monotonic()
            result=llm.generate([{'prompt':text,'multi_modal_data':{'image':im}}],sampling,use_tqdm=False)[0]
            response=processor.tokenizer.decode(result.outputs[0].token_ids,skip_special_tokens=True,clean_up_tokenization_spaces=True)
            r={'job_id':j['job_id'],'dataset':a.dataset,'sample_id':j['sample_id'],
               'prompt_sha256':j['prompt_sha256'],'image_sha256':j['image_sha256'],'run_signature':signature,
               'response':response,'seconds':time.monotonic()-start,'output_tokens':len(result.outputs[0].token_ids),
               'input_tokens':len(result.prompt_token_ids),'finish_reason':result.outputs[0].finish_reason}
            out.write(json.dumps(r,ensure_ascii=False)+'\n'); out.flush(); done.add(j['job_id'])
            print(f'{a.dataset}: {len(done)}/{len(jobs)}',flush=True)

if __name__=='__main__':main()
