import json
import re
import csv
from pathlib import Path
from difflib import SequenceMatcher

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
        fieldnames = ["filename"] + HAM10000_DISEASE_NAME
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for fname, info in data.items():
            row = {"filename": fname}
            primary_diagnosis = info.get("PrimaryDiagnosis", "")
            cleaned_fname = clean_filename(fname)

            for disease in HAM10000_DISEASE_NAME:
                prob = 0.0000
                for item in info.get("ProbabilityDistribution", []):
                    if item["disease"] == disease:
                        prob = item["probability"]
                        break

                row[disease] = prob

            writer.writerow(row)

    print(f"✅ 处理完成，结果已写入 {csv_path}")

# ---------------- 5. 调用主函数 ----------------
normalized_json_path = Path("/Volumes/T7/SkinGPT-X-EvaluationResults/HAM10000/SkinGPT-X/Reasoning_output_normalized.json")  # 替换为你的归一化 JSON 文件路径
output_csv_path = Path("/Volumes/T7/SkinGPT-X-EvaluationResults/HAM10000/SkinGPT-X/Reasoning_output.csv")  # 替换为你希望保存的 CSV 文件路径

process_normalized_json(normalized_json_path, output_csv_path)