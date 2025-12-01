import csv
from typing import Dict

import numpy as np
from pathlib import Path

# ---------------- 1. 23 类固定顺序 ----------------
HAM10000_DISEASE_NAME = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
HAM10000_DISEASE_MAPPING_NAME = {"akiec": "Actinic keratoses and intraepithelial carcinoma / Bowen's disease",
                                 "bcc": "basal cell carcinoma",
                                 "bkl": "benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses)",
                                 "df": "dermatofibroma", "mel": "melanoma", "nv": "melanocytic nevi ",
                                 "vasc": "vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)"}
REVERSED_MAPPING = {
    "Actinic keratoses and intraepithelial carcinoma / Bowen's disease": "akiec",
    "basal cell carcinoma": "bcc",
    "benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses)": "bkl",
    "dermatofibroma": "df",
    "melanoma": "mel",
    "melanocytic nevi ": "nv",
    "vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)": "vasc"
}
MAPPING_NAME = [HAM10000_DISEASE_MAPPING_NAME[abbr] for abbr in HAM10000_DISEASE_NAME]
TARGET_SET = set(MAPPING_NAME)


# ---------------- 2. 自定义归一化函数 ----------------
def calculate_normalization(row):
    """对一行数据（数值部分）进行归一化，使其和为 1。"""
    x = np.array(row, dtype=float)
    x_sum = np.sum(x)

    if x_sum == 0:
        return np.repeat(np.nan, len(x))

    return x / x_sum


# ---------------- 3. 提取子文件夹名称 ----------------
def extract_subfolder(filename: str) -> str:
    """从文件路径中提取子文件夹名称"""
    path = Path(filename)
    return path.parent.name


# ---------- 1. 建立 image → 数字标签映射 ---------- #
# 顺序与 HAM10000_DISEASE_NAME 保持一致
HAM10000_DISEASE_NAME = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def build_label_map(ground_truth_csv: Path) -> Dict[str, int]:
    """返回 {image_id: label_int}"""
    label_map = {}
    with open(ground_truth_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row['image']  # 如 ISIC_0034524
            one_hot = [int(row[dis]) for dis in HAM10000_DISEASE_NAME]
            label = one_hot.index(1)  # 找到 1 所在的下标
            label_map[img_id] = label
    return label_map


# ---------- 2. 主处理函数（仅新增 label 列，不改概率） ---------- #
def normalize_csv(
        prob_csv: Path,
        ground_truth_csv: Path,
        out_csv: Path
):
    # 读取 label 映射
    label_map = build_label_map(ground_truth_csv)

    with open(prob_csv, newline='', encoding='utf-8') as in_f, \
            open(out_csv, 'w', newline='', encoding='utf-8') as out_f:
        reader = csv.DictReader(in_f)
        fieldnames = reader.fieldnames + ['label']
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            img_id = row.get('image_id') or row.get('filename', '')
            # 如果找不到对应 label，可设为 -1 或抛异常
            row['label'] = HAM10000_DISEASE_NAME[label_map.get(img_id.split('.')[0].split('/')[1], -1)]
            writer.writerow(row)

    print(f"✅ 已添加 ground-truth label → {out_csv}")


# ---------------- 4. 调用主函数 ----------------
input_csv_path = Path(
    "/Volumes/T7/SkinGPT-X-EvaluationResults/HAM10000/SkinGPT-X/Reasoning_output.csv")  # 替换为之前的 CSV 文件路径
output_csv_path = Path(
    "/Volumes/T7/SkinGPT-X-EvaluationResults/HAM10000/SkinGPT-X/Reasoning_output_softmax.csv")  # 替换为新的 CSV 文件路径
label_csv_path = Path(
    "/Volumes/T7/SkinGPT-X-Dataset/HAM10000/ISIC2018_Task3_Test_GroundTruth.csv")  # 替换为 ground-truth CSV 文件路径
normalize_csv(input_csv_path, label_csv_path, output_csv_path)
