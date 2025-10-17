import json, csv
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


def json_to_csv(json_file, csv_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'pred'])  # 表头
        for filename, info in data.items():
            print(info.get('PrimaryDiagnosis', ''))
            writer.writerow([filename, info.get('PrimaryDiagnosis', '')])


def get_label_from_filename(json_path, csv_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data: Dict[str, dict] = json.load(f)

    rows: List[List[str]] = []
    for filename in data.keys():
        # 拆分 label/filename
        if '/' in filename:
            label, _ = filename.rsplit('/', 1)
        else:
            label = ''
        rows.append([filename, label])

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        writer.writerows(rows)
    return

def modify_filename(dataset_dir: str, json_path: str, out_json: str):
    # 1. 读 json，拿到所有文件名
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 建立 文件名 -> 相对路径（标签/文件名） 映射
    file_map = {}
    for root, _, files in os.walk(dataset_dir):
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), dataset_dir)
            file_map[name] = rel.replace('\\', '/')

    # 3. 原地回填：老 key -> 新 key，内容完全拷贝
    new_data = {}
    for old_key, content in data.items():
        new_key = file_map.get(old_key, old_key)  # 找不到就保持原文件名
        if new_key == old_key:
            print(f'Warning: {old_key} 未在数据集下找到，保持原 key', file=sys.stderr)
        new_data[new_key] = content

    # 4. 写回 json（保持中文缩进、排序与原来一致）
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

    print(f'Done！已生成 {out_json}')


if __name__ == '__main__':
    base_image_path = '/Volumes/T7/SkinGPT-X-Dataset/Dermnet/test'
    json_to_csv('/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/Reasoning_output_new.json',
                                  '/Volumes/T7/SkinGPT-X-EvaluationResults/Experiments/Diagnosis/SkinGPTX/Reasoning_output.csv')
    # get_label_from_filename('/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/Reasoning_output_new.json',
    #                         '/Volumes/T7/SkinGPT-X-EvaluationResults/Experiments/Diagnosis/SkinGPTX/filename_to_labels.csv')
    # modify_filename(json_path='/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/Reasoning_output_new.json',
    #                 out_json='/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/Reasoning_output_new.json',
    #                 dataset_dir='/Volumes/T7/SkinGPT-X-Dataset/Dermnet/evaluation_split2')
