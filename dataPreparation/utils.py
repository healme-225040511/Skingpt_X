import json, csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

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


def build_filename_map(json_file: str, path_file: str) -> Dict[str, str]:
    """同之前：裸文件名 → 完整路径"""
    with open(json_file, encoding='utf-8') as fj, open(path_file, encoding='utf-8') as fp:
        j_keys = json.load(fj).keys()
        p_keys = json.load(fp).keys()
    name_to_full = {Path(k).name: k for k in p_keys}
    mapping = {}
    for k in j_keys:
        bare = Path(k).name
        full = name_to_full.get(bare)
        if full is None:
            raise KeyError(f'在 path_file 中找不到文件名：{bare}')
        mapping[bare] = full
    return mapping


def remap_to_new_file(src_json: str, path_json: str, dst_json: str) -> None:
    """
    把 src_json 的 key 按 path_json 的映射规则替换后，写入全新文件 dst_json
    """
    # 1. 计算映射
    mapping = build_filename_map(src_json, path_json)

    # 2. 读取原数据
    with open(src_json, encoding='utf-8') as f:
        old_data = json.load(f)

    # 3. 替换 key
    new_data = {mapping[Path(k).name]: v for k, v in old_data.items()}

    # 4. 写入新文件
    with open(dst_json, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # base_image_path = '/Volumes/T7/SkinGPT-X-Dataset/Dermnet/test'
    # json_to_csv('/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/CaseReview_output_new.json',
    #                               '/Volumes/T7/SkinGPT-X-EvaluationResults/Experiments/Diagnosis/SkinGPTX/CaseReview_output.csv')
    get_label_from_filename('/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/CaseReview_output_new.json',
                            '/Volumes/T7/SkinGPT-X-EvaluationResults/Experiments/Diagnosis/SkinGPTX/filename_to_labels_CaseReview.csv')
    # modify_filename(json_path='/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/Reasoning_output_new.json',
    #                 out_json='/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/Reasoning_output_new.json',
    #                 dataset_dir='/Volumes/T7/SkinGPT-X-Dataset/Dermnet/evaluation_split2')
    # ----------------- 用法示例 -----------------
    # remap_to_new_file("/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/CaseReview_output.json",
    #                   "/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/Reasoning_output_new.json",
    #                   "/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/CaseReview_output_new.json")
    # print("已生成新文件：CaseReview_output_new.json")
