# #!/usr/bin/env python3
# """
# extract_confidence_for_disease_list.py
# 从原始 JSON 提取 DISEASES 列表中 23 个疾病的预测概率
# python extract_confidence_for_disease_list.py raw.json
# """
#
# import json, re, sys, csv
# from pathlib import Path
#
# # 23 个疾病标签（与提问顺序一致）
# DISEASES = [
#     'Acne and Rosacea Photos',
#     'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
#     'Atopic Dermatitis Photos',
#     'Bullous Disease Photos',
#     'Cellulitis Impetigo and other Bacterial Infections',
#     'Eczema Photos',
#     'Exanthems and Drug Eruptions',
#     'Hair Loss Photos Alopecia and other Hair Diseases',
#     'Herpes HPV and other STDs Photos',
#     'Light Diseases and Disorders of Pigmentation',
#     'Lupus and other Connective Tissue diseases',
#     'Melanoma Skin Cancer Nevi and Moles',
#     'Nail Fungus and other Nail Disease',
#     'Poison Ivy Photos and other Contact Dermatitis',
#     'Psoriasis pictures Lichen Planus and related diseases',
#     'Scabies Lyme Disease and other Infestations and Bites',
#     'Seborrheic Keratoses and other Benign Tumors',
#     'Systemic Disease',
#     'Tinea Ringworm Candidiasis and other Fungal Infections',
#     'Urticaria Hives',
#     'Vascular Tumors',
#     'Vasculitis Photos',
#     'Warts Molluscum and other Viral Infections'
# ]
#
# # 正则：捕获  "- **疾病名:** 概率%"
# CONF_RE = re.compile(r"- \*\*(.+?):\*\*\s+(\d+(?:\.\d+)?)%")
#
# def extract_one(text: str):
#     """返回 dict: 疾病 -> float(概率)"""
#     return {dis: float(pr) for dis, pr in CONF_RE.findall(text)}
#
# def extract(json_path):
#     in_file = Path(json_path)
#     raw = json.loads(in_file.read_text(encoding="utf-8"))
#
#     # 1. 提取概率
#     extracted = {img: extract_one(txt) for img, txt in raw.items()}
#
#     # 2. 补全缺失疾病为 0.0
#     for img in extracted:
#         for dis in DISEASES:
#             extracted[img].setdefault(dis, 0.0)
#
#     # 3. 写 CSV（行=病例，列=疾病）
#     csv_file = Path("confidence_matrix.csv")
#     with csv_file.open("w", newline='', encoding="utf-8") as f:
#         w = csv.writer(f)
#         w.writerow(["image"] + DISEASES)
#         for img, probs in extracted.items():
#             w.writerow([img] + [probs[d] for d in DISEASES])
#
#     # 4. 写 JSON（嵌套字典）
#     json_file = Path("confidence_dict.json")
#     json_file.write_text(json.dumps(extracted, indent=2, ensure_ascii=False), encoding="utf-8")
#
#     print(f"Done！\nCSV -> {csv_file}\nJSON -> {json_file}")
#
# if __name__ == "__main__":
#     extract()

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

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# import re, csv
# from pathlib import Path
# import pandas as pd
#
# # 路径
# pred_csv = Path('/Volumes/T7/SkinGPT-X-EvaluationResults/HAM10000/Medgamma/results_HAM10000.csv')
# gt_csv   = Path('/Volumes/T7/SkinGPT-X-Dataset/HAM10000/ISIC2018_Task3_Test_GroundTruth.csv')
# out_csv  = Path('/Volumes/T7/SkinGPT-X-EvaluationResults/HAM10000/Medgamma/pred_with_label.csv')
#
# # 读取 GT
# def load_gt(path):
#     df = pd.read_csv(path, sep=None, engine='python')
#     cols = [c for c in df.columns if c.strip().lower() != 'image']
#     df['true_label'] = df[cols].idxmax(axis=1)
#     return dict(zip(df['image'], df['true_label']))
#
# gt_map = load_gt(gt_csv)
#
# # 只抓 *   **Primary Diagnosis:** Xxxx
# def extract_primary(text: str) -> str:
#     # 1) JSON  "PrimaryDiagnosis": "xxx"
#     m = re.search(r'"PrimaryDiagnosis"\s*:\s*"([^"]+)"', text, re.I)
#     if m:
#         return m.group(1).strip()
#
#     # 2) Markdown  **Primary Diagnosis:** xxx
#     m = re.search(r'\*\s*\*\*Primary Diagnosis:\*\*\s*(.+?)\s*$', text, re.M | re.I)
#     if m:
#         return m.group(1).strip()
#
#     # 3) 自然句  "the primary diagnosis appears to be a **melanoma**"
#     m = re.search(r'primary diagnosis appears to be (?:a|an)?\s*\*\*([^*]+)\*\*', text, re.I)
#     if m:
#         return m.group(1).strip()
#
#     # 4) 自然句  "the primary diagnosis is likely **Melanoma**"
#     m = re.search(r'primary diagnosis is likely (?:a|an)?\s*\*\*([^*]+)\*\*', text, re.I)
#     if m:
#         return m.group(1).strip()
#
#     # 5) 自然句  "the primary diagnosis is a **melanoma**"
#     m = re.search(r'primary diagnosis is (?:a|an)\s*\*\*([^*]+)\*\*', text, re.I)
#     if m:
#         return m.group(1).strip()
#
#     return 'N/A'
# # 主循环
# results = []
# with pred_csv.open(newline='', encoding='utf-8') as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         img_path  = row['image']
#         img_name  = Path(img_path).stem
#         true_lab  = gt_map.get(img_name, 'unknown')
#         pred_diag = extract_primary(row['pred'])
#         results.append({'image': img_path, 'pred_diag': pred_diag, 'true_label': true_lab})
#
# pd.DataFrame(results).to_csv(out_csv, index=False, encoding='utf-8')
# print(f'✅ 仅提取诊断完成 → {out_csv.resolve()}')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import re
from pathlib import Path

file_in = Path('/Volumes/T7/SkinGPT-X-EvaluationResults/HAM10000/Medgamma/pred_with_label.csv')   # 换成你的路径
file_out = file_in.with_suffix('.extracted.csv')        # 可改为 file_in 原地覆盖

# 万能正则：提取 **疾病名称**
star_re = re.compile(r'\*\*([^*]+)\*\*')

def pick_disease(s: str):
    hits = star_re.findall(s)
    return hits[0].strip() if hits else s          # 无 ** 时保留原串；若想置空改成 return ''

df = pd.read_csv(file_in, header=None, names=['image','pred','true_label'])
df['pred'] = df['pred'].astype(str).apply(pick_disease)
df.to_csv(file_out, index=False, header=False, encoding='utf-8')
print(f'✅ 提取完成 → {file_out}')