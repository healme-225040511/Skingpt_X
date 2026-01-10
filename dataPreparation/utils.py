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


def build_filename_map(json_file: str, path_file: str):
    """同之前：裸文件名 → 完整路径"""
    with open(json_file, encoding='utf-8') as fj, open(path_file, encoding='utf-8') as fp:
        j_keys = json.load(fj).keys()
        p_keys = json.load(fp).keys()
    name_to_full = {Path(k).name: k for k in p_keys}
    mapping = {}
    not_found_files = []
    for k in j_keys:
        bare = Path(k).name
        full = name_to_full.get(bare)
        if full is None:
            mapping[bare] = bare
            not_found_files.append(bare)
            print(f'Warning: {bare} 未在 path_file 中找到', file=sys.stderr)
            # raise KeyError(f'在 path_file 中找不到文件名：{bare}')
        else:
            mapping[bare] = full
    return mapping, not_found_files


def remap_to_new_file(src_json: str, path_json: str, dst_json: str) -> list[str]:
    """
    把 src_json 的 key 按 path_json 的映射规则替换后，写入全新文件 dst_json
    """
    # 1. 计算映射
    mapping, not_found_files = build_filename_map(src_json, path_json)

    # 2. 读取原数据
    with open(src_json, encoding='utf-8') as f:
        old_data = json.load(f)

    # 3. 替换 key
    new_data = {mapping[Path(k).name]: v for k, v in old_data.items()}

    # 4. 写入新文件
    with open(dst_json, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

    return not_found_files


def find_parent_directories(image_names, base_directory) -> Dict[str, str]:
    """找到给定的图片文件名的父目录（疾病名称）"""
    # 存储文件名与其父目录的映射关系
    results = {}

    # 遍历基目录及其子目录
    for root, dirs, files in os.walk(base_directory):
        for image_name in image_names:
            if image_name in files:
                # 找到对应文件后获取其父目录
                parent_directory = os.path.basename(root)
                results[image_name] = parent_directory

    return results

def update_file_parents(file_map, json_path: str, out_json: str):
    """
    查找不包含"/"的文件名，并更新它们的父目录到新的 JSON 文件中
    """
    # 1. 读取 JSON 数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 建立文件名与相对路径（父目录/文件名）映射
    new_data = {}
    # 3. 更新数据，将文件的父目录回填
    for old_key, content in data.items():
        if '/' not in old_key:  # 仅更新不包含"/"的文件名
            parent_dir = file_map.get(old_key, '')  # 如果未找到，则为 ''
            new_data[parent_dir+'/'+old_key] = content
        else:
            new_data[old_key] = content

    # 4. 写回新的 JSON 文件
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

    print(f'更新完成！已生成 {out_json}')

def capitalize_first_segment(key: str) -> str:
    """
    把 key 按 '/' 分割，对 [0] 段做首字母大写，and/other 保持小写
    """
    if '/' not in key:
        return key
    parts = key.split('/')
    segment = parts[0]

    # 按空格切词，跳过 and/other
    words = segment.split()
    new_words = [
        w.capitalize() if w.lower() not in {'and', 'other'} else w.lower()
        for w in words
    ]
    parts[0] = ' '.join(new_words)
    return '/'.join(parts)

def UpperFirstWord(file_path: Path):
    with file_path.open(encoding='utf-8') as f:
        data = json.load(f)

    # 假设顶层是 list[dict]；若是 dict，改成 data = {k: capitalize_first_segment(k): v ...}
    if isinstance(data, list):
        new_data = [{capitalize_first_segment(k): v for k, v in item.items()} for item in data]
    else:
        new_data = {capitalize_first_segment(k): v for k, v in data.items()}

    with file_path.open('w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    print(f'✅ 已覆盖保存 → {file_path}')

if __name__ == '__main__':
    base_image_path = '/Volumes/T7/SkinGPT-X-Dataset/HAM10000/test'
    json_to_csv('/Volumes/T7/SkinGPT-X-EvaluationResults/HAM10000/SkinGPT-X/Reasoning_results.json',
                                  '/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/Hulumed/Reasoning_results.csv')
    get_label_from_filename('/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/Hulumed/Reasoning_results.json',
                            '/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/Hulumed/Reasoning_labels.csv')
    # modify_filename(json_path='/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/Reasoning_output_new.json',
    #                 out_json='/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/Reasoning_output_new.json',
    #                 dataset_dir='/Volumes/T7/SkinGPT-X-Dataset/Dermnet/evaluation_split2')
    # ----------------- 用法示例 -----------------
    # not_found_files = remap_to_new_file("/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/RAG_output.json",
    #                   "/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/Reasoning_output_new.json",
    #                   "/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/RAG_output_new.json")
    # # print("已生成新文件：CaseReview_output_new.json")
    # base_directory = '/Volumes/T7/SkinGPT-X-Dataset/Dermnet/test'
    # parent_directories = find_parent_directories(not_found_files, base_directory)
    # update_file_parents(parent_directories, '/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/RAG_output_new.json', '/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/RAG_output_new.json')
    # UpperFirstWord(Path('/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/Reasoning_output_new.json'))