import csv
import numpy as np
from pathlib import Path

# ---------------- 1. 23 类固定顺序 ----------------
DERMNET_DISEASE_NAME = [
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

# ---------------- 4. 主处理函数 ----------------
def normalize_csv(input_csv_path: Path, output_csv_path: Path):
    # 读取原始 CSV 文件
    with open(input_csv_path, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        headers = reader.fieldnames  # 获取列名
        rows = list(reader)  # 读取所有行

    # 添加 'label' 列
    headers.append('label')

    # 准备输出 CSV 文件
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=headers)
        writer.writeheader()

        for row in rows:
            # 提取概率值并转换为 numpy 数组
            probs = [float(row[disease]) for disease in DERMNET_DISEASE_NAME]
            # 应用自定义归一化
            normalized_probs = calculate_normalization(probs)
            # 更新行数据为归一化后的概率
            for disease, norm_prob in zip(DERMNET_DISEASE_NAME, normalized_probs):
                row[disease] = round(norm_prob, 4) if not np.isnan(norm_prob) else norm_prob
            # 提取子文件夹名称并添加到 'label' 列
            row['label'] = extract_subfolder(row['filename'])
            # 写入新的 CSV 文件
            writer.writerow(row)

    print(f"✅ 归一化完成，结果已写入 {output_csv_path}")

# ---------------- 4. 调用主函数 ----------------
input_csv_path = Path("/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/Reasoning_output.csv")  # 替换为之前的 CSV 文件路径
output_csv_path = Path("/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/Reasoning_output_softmax.csv")  # 替换为新的 CSV 文件路径

normalize_csv(input_csv_path, output_csv_path)