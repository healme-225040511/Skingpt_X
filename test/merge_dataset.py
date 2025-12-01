import json, os

files = [
    '/Volumes/T7/SkinGPT-X-EvaluationResults/HAM10000/test4/Reasoning_output.json',
    '/Volumes/T7/SkinGPT-X-EvaluationResults/HAM10000/test3/Reasoning_output.json',
    '/Volumes/T7/SkinGPT-X-EvaluationResults/HAM10000/test2/Reasoning_output.json',
    '/Volumes/T7/SkinGPT-X-EvaluationResults/HAM10000/test1/Reasoning_output.json'
]

out_file = '/Volumes/T7/SkinGPT-X-EvaluationResults/HAM10000/Reasoning_output.json'

merged = {}
for fp in files:
    if not os.path.exists(fp):
        print('skip', fp)
        continue
    with open(fp, 'r', encoding='utf-8') as f:
        merged.update(json.load(f))   # 后读入的 key 覆盖先读入的

with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print('done →', out_file, f'({len(merged)} images)')