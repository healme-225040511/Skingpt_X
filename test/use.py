import json
import os


def merge_json_files(file_paths):
    # 用于存储合并后的数据
    merged_data = {}

    # 遍历每个文件
    for file_path in file_paths:
        # 加载JSON文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 遍历文件中的每个条目
        for key, value in data.items():
            # 如果该键尚未在合并数据中出现，则添加
            if key not in merged_data:
                merged_data[key] = value
            else:
                print(f"Warning: Duplicate key '{key}' found in {file_path}. Skipping this entry.")

    return merged_data


def save_merged_data(merged_data, output_file):
    # 将合并后的数据保存到新的JSON文件
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(merged_data, file, indent=4, ensure_ascii=False)
    print(f"Merged data saved to {output_file}")


# 示例：指定要合并的JSON文件路径
file_paths = [
    "/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/RAG_output.json",
    "/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/RAG_output (1).json",
    "/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/RAG_output（2）.json",
    "/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/RAG_output（3）.json"
    # 添加更多文件路径
]

# 合并数据
merged_data = merge_json_files(file_paths)

# 保存合并后的数据到新文件
output_file = "/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/WebSearch_merged_data.json"
save_merged_data(merged_data, output_file)

import json


def check_duplicate_keys(json_file):
    # 用于存储键及其出现次数
    key_count = {}

    # 加载JSON文件
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 遍历JSON数据中的每个键
    for key in data.keys():
        if key in key_count:
            key_count[key] += 1
        else:
            key_count[key] = 1

    # 检查是否有重复的键
    duplicates = {key: count for key, count in key_count.items() if count > 1}
    if duplicates:
        print(f"Duplicate keys found in {json_file}:")
        for key, count in duplicates.items():
            print(f"Key: '{key}' appears {count} times.")
    else:
        print(f"No duplicate keys found in {json_file}.")


# 示例：指定要检查的JSON文件路径
json_file_path ="/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/WebSearch_merged_data.json"

# 检查JSON文件内部是否有重复的键
check_duplicate_keys(json_file_path)

# import pandas as pd
# from pathlib import Path
# import csv
# import json   # pip install joblib
#
# # ========== 路径 ==========
# img_dir  = Path('/Volumes/T7/SkinGPT-X-Dataset/HAM10000/ISIC2018_Task3_Test_Input')
# prob_csv = Path('/Volumes/T7/SkinGPT-X-Dataset/HAM10000/ISIC2018_Task3_Test_GroundTruth.csv')
# out_csv  = Path('/Volumes/T7/SkinGPT-X-Dataset/HAM10000/HAM10000_test_label.csv')
#
# # ---------- 1. 读映射 ----------
# with open('/Volumes/T7/SkinGPT-X-Dataset/HAM10000//ham10000_class2idx.json') as f:
#     class2idx = json.load(f)          # 直接 {"akiec":0, "bcc":1, ...}
#
# # ---------- 2. 读概率表 ----------
# df = pd.read_csv(prob_csv)
#
# # ---------- 3. 文件存在性过滤 ----------
# SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
# id2path = {fp.stem: fp for fp in img_dir.glob('*') if fp.suffix.lower() in SUFFIXES}
# df = df[df['image'].isin(id2path)]
#
# # ---------- 4. one-hot → 类别名 → 索引 ----------
# CLASSES = list(class2idx.keys())          # 顺序与 CSV 列一致
# df['label'] = df[CLASSES].idxmax(axis=1).map(class2idx)   # 直接查表
#
# # ---------- 5. 写 CSV ----------
# with out_csv.open('w', newline='', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerow(['image', 'label', 'split'])
#     for _, row in df.iterrows():
#         csv_path = id2path[row['image']].relative_to(img_dir)
#         writer.writerow([csv_path, row['label'], 'test'])
#
# print(f'Test set converted: {len(df)} samples -> {out_csv.resolve()}')