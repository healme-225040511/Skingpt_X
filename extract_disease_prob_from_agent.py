#!/usr/bin/env python3
"""
extract_confidence_for_disease_list.py
从原始 JSON 提取 DISEASES 列表中 23 个疾病的预测概率
python extract_confidence_for_disease_list.py raw.json
"""

import json, re, sys, csv
from pathlib import Path

# 23 个疾病标签（与提问顺序一致）
DISEASES = [
    'Acne and Rosacea Photos',
    'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
    'Atopic Dermatitis Photos',
    'Bullous Disease Photos',
    'Cellulitis Impetigo and other Bacterial Infections',
    'Eczema Photos',
    'Exanthems and Drug Eruptions',
    'Hair Loss Photos Alopecia and other Hair Diseases',
    'Herpes HPV and other STDs Photos',
    'Light Diseases and Disorders of Pigmentation',
    'Lupus and other Connective Tissue diseases',
    'Melanoma Skin Cancer Nevi and Moles',
    'Nail Fungus and other Nail Disease',
    'Poison Ivy Photos and other Contact Dermatitis',
    'Psoriasis pictures Lichen Planus and related diseases',
    'Scabies Lyme Disease and other Infestations and Bites',
    'Seborrheic Keratoses and other Benign Tumors',
    'Systemic Disease',
    'Tinea Ringworm Candidiasis and other Fungal Infections',
    'Urticaria Hives',
    'Vascular Tumors',
    'Vasculitis Photos',
    'Warts Molluscum and other Viral Infections'
]

# 正则：捕获  "- **疾病名:** 概率%"
CONF_RE = re.compile(r"- \*\*(.+?):\*\*\s+(\d+(?:\.\d+)?)%")

def extract_one(text: str):
    """返回 dict: 疾病 -> float(概率)"""
    return {dis: float(pr) for dis, pr in CONF_RE.findall(text)}

def extract(json_path):
    in_file = Path(json_path)
    raw = json.loads(in_file.read_text(encoding="utf-8"))

    # 1. 提取概率
    extracted = {img: extract_one(txt) for img, txt in raw.items()}

    # 2. 补全缺失疾病为 0.0
    for img in extracted:
        for dis in DISEASES:
            extracted[img].setdefault(dis, 0.0)

    # 3. 写 CSV（行=病例，列=疾病）
    csv_file = Path("confidence_matrix.csv")
    with csv_file.open("w", newline='', encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image"] + DISEASES)
        for img, probs in extracted.items():
            w.writerow([img] + [probs[d] for d in DISEASES])

    # 4. 写 JSON（嵌套字典）
    json_file = Path("confidence_dict.json")
    json_file.write_text(json.dumps(extracted, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Done！\nCSV -> {csv_file}\nJSON -> {json_file}")

if __name__ == "__main__":
    extract()

import numpy as np
import pandas as pd

# # 读取 CSV 文件
# df = pd.read_csv('disease_probs.csv')
#
# # 提取图片路径（第一列）
# image_paths = df.iloc[:, 0]
#
# # 提取数值部分（所有列除了第一列）
# logits = df.iloc[:, 1:].values
#
# # 定义 softmax 函数（按行）
# def softmax(x):
#     exp = np.exp(x - np.max(x, axis=1, keepdims=True))  # 防止溢出
#     return exp / np.sum(exp, axis=1, keepdims=True)
#
# # 应用 softmax
# probabilities = softmax(logits)
#
# # 创建新的 DataFrame
# prob_df = pd.DataFrame(probabilities, columns=df.columns[1:])
# prob_df.insert(0, 'image', image_paths)
#
# # 保存结果
# prob_df.to_csv('disease_probs_softmax.csv', index=False)