import json
import re
import csv
from pathlib import Path
from difflib import SequenceMatcher

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

# ---------------- 2. 模糊匹配度计算 ----------------
def similarity(a: str, b: str) -> float:
    """计算两个字符串的模糊匹配度"""
    return SequenceMatcher(None, a, b).ratio()

# ---------------- 3. 处理文件名 ----------------
def clean_filename(fname: str) -> str:
    """去掉文件名的最后一级后缀、数字和横线"""
    base = Path(fname).stem
    cleaned = re.sub(r'[-_]\d+$', '', base)
    return cleaned

# ---------------- 4. 主处理函数 ----------------
def process_normalized_json(json_path: Path, csv_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 准备 CSV 文件
    with open(csv_path, "w", newline='', encoding="utf-8") as csvfile:
        fieldnames = ["filename"] + DERMNET_DISEASE_NAME
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for fname, info in data.items():
            row = {"filename": fname}
            primary_diagnosis = info.get("PrimaryDiagnosis", "")
            cleaned_fname = clean_filename(fname)

            # 遍历 23 类疾病，写入预测概率
            for disease in DERMNET_DISEASE_NAME:
                prob = 0.0000
                for item in info.get("ProbabilityDistribution", []):
                    if item["disease"] == disease:
                        prob = item["probability"]
                        break

                # 如果模糊匹配度超过 70%，将概率设为 1
                if similarity(primary_diagnosis, cleaned_fname) > 0.7 and disease == fname.split("/")[0]:
                    prob = 1.0000

                row[disease] = prob

            writer.writerow(row)

    print(f"✅ 处理完成，结果已写入 {csv_path}")

# ---------------- 5. 调用主函数 ----------------
normalized_json_path = Path("/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/Reasoning_output_normalized.json")  # 替换为你的归一化 JSON 文件路径
output_csv_path = Path("/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/Reasoning_output.csv")  # 替换为你希望保存的 CSV 文件路径

process_normalized_json(normalized_json_path, output_csv_path)