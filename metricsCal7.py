"""
计算所有模型在数据集上的预测指标
"""

import re
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_fscore_support
)
# 导入 fuzzywuzzy 库
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from sklearn.metrics import confusion_matrix, classification_report

from Constants import DERMNET_DISEASE_NAME, REASONINGLAYER_EVALUATION_PATH, REASONING_LABELS_PATH, BASE_IMAGE_DIRECTORY, \
    REASONING_WORDHIT_OUTPUT, MEDGAMMA_EVALUATION_PATH, MEDGAMMA_LABELS_PATH, MEDGAMMA_WORDHIT_OUTPUT, \
    CASEREVIEW_LABELS_PATH, CASEREVIEW_WORDHIT_OUTPUT, CASEREVIEW_EVALUATION_PATH, SYNONYM_DICT_HAM10000
from dataPreparation.Panderm_Assessment import wCK_ci
from metricsCal import SYNONYM_DICT_HAM10000


# ---------- 2. 读文件 (修改为读取原始标签，不进行预处理) ----------
def read_pred_file_raw(path: str) -> dict:
    """
    返回 dict: filename -> raw_pred_disease
    """
    df = pd.read_csv(path)
    # 统一小写 key，防止列名大小写问题
    df.columns = [c.lower() for c in df.columns]
    return {
        row['filename']: row['pred']
        for _, row in df.iterrows()
    }


def read_label_file_raw(path: str) -> dict:
    """
    读取 filename_to_label.csv
    返回 dict: filename -> raw_true_label
    """
    df = pd.read_csv(path, dtype=str)  # 自动识别逗号分隔
    df.columns = [c.lower() for c in df.columns]
    return dict(zip(df["filename"], df["label"].str.lower().str.strip()))


# ---------- NEW FUNCTION: Scan Image Folders ----------
def scan_image_folders(base_directory: str) -> dict:
    """
    Scans subdirectories for image files.
    Extracts image base names (without extension) and maps them to their parent folder names.
    Both are normalized (lowercase, strip).

    Args:
        base_directory: The base path where subfolders containing images reside.

    Returns:
        dict: Mappings from image base names (normalized) to their parent folder names (normalized).
              Returns an empty dict if the base_directory is not found or invalid.
    """
    image_to_folder_mappings = {}

    if not Path(base_directory).is_dir():
        print(
            f"Warning: Base image directory '{base_directory}' not found or is not a directory. Skipping image folder scan.")
        return image_to_folder_mappings

    for folder_path in Path(base_directory).iterdir():
        if folder_path.is_dir():  # Only process actual directories
            folder_name_canonical = folder_path.name.strip().lower()

            for file_path in folder_path.iterdir():
                if file_path.is_file():  # Only process actual files
                    # Check if it's an image (simple check by extension, can be improved)
                    if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                        image_basename_raw = file_path.stem.strip().lower()  # Get name without extension
                        image_to_folder_mappings[image_basename_raw] = folder_name_canonical

    return image_to_folder_mappings

def _get_words(text: str) -> Set[str]:
    """小写+拆词+去重"""
    _WORD_RE = re.compile(r"[a-zA-Z]+")  # 仅保留字母
    _STOP_WORDS = {
        "and", "or", "the", "of", "in", "on", "at", "with",
        "jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"
    }
    words = _WORD_RE.findall(str(text).lower())
    return {w for w in words if w not in _STOP_WORDS}



# 反向索引：同义词 → 标准名
SYNONYM_DICT_HAM10000_REVERSE = {}
for std, synonyms in SYNONYM_DICT_HAM10000.items():
    for w in synonyms:
        SYNONYM_DICT_HAM10000_REVERSE[w] = std


def pred_to_class(pred_words) -> int:
    """
    使用模糊匹配将预测词集映射到 23 类 ID，优先级：
    1. 预测词直接命中
    2. 模糊匹配最接近的同义词
    3. 未命中返回 'unknown' 对应索引 23
    """
    # 将预测词集转换为一个字符串
    pred_str = pred_words.lower()

    # 1. 预测词直接命中
    if pred_str in SYNONYM_DICT_HAM10000_REVERSE:
        return list(SYNONYM_DICT_HAM10000.keys()).index(SYNONYM_DICT_HAM10000_REVERSE[pred_str])

    # 2. 模糊匹配最接近的同义词
    best_match = process.extractOne(pred_str, [value for values in SYNONYM_DICT_HAM10000.values() for value in values], scorer=process.fuzz.token_sort_ratio)
    print(pred_words)
    print(best_match)
    if best_match and best_match[1] > 50:  # 设置一个阈值，例如 50
        for key, values in SYNONYM_DICT_HAM10000.items():
            if best_match[0] in values:
                return list(SYNONYM_DICT_HAM10000.keys()).index(key)

    # 3. 未命中
    return 7  # 固定把 unknown 放最后一位


from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.utils import resample
import numpy as np
import pandas as pd

def _boot_ci(y_true, y_pred, metric_func, n_resample=1000, seed=42):
    rng = np.random.default_rng(seed)
    scores = [metric_func(np.array(y_true)[idx], np.array(y_pred)[idx])
              for idx in (resample(range(len(y_true))) for _ in range(n_resample))]
    return np.percentile(scores, [2.5, 97.5])

def word_hit_metrics(pred_file: str,
                     label_file: str,
                     img_dir: str = None,
                     out_csv: str = None) -> dict:
    preds = read_pred_file_raw(pred_file)
    labels = read_label_file_raw(label_file)
    filenames = sorted(preds.keys())
    y_true_id, y_pred_id = [], []

    for f in filenames:
        raw_pred  = preds[f]
        raw_label = labels[f]
        pred_words  = _get_words(str(raw_pred))
        y_true_id.append(list(SYNONYM_DICT_HAM10000.keys()).index(raw_label.lower()))
        y_pred_id.append(pred_to_class(str(raw_pred)))

    # 指标 + 95% CI
    acc, bacc, wf1 = (accuracy_score(y_true_id, y_pred_id),
                      balanced_accuracy_score(y_true_id, y_pred_id),
                      f1_score(y_true_id, y_pred_id, average='weighted', zero_division=0))
    acc_ci, bacc_ci, wf1_ci = (_boot_ci(y_true_id, y_pred_id, f) for f in
                               (accuracy_score, balanced_accuracy_score,
                                lambda yt, yp: f1_score(yt, yp, average='weighted', zero_division=0)))

    if out_csv:
        records = []
        for i, f in enumerate(filenames):
            pred_words  = _get_words(str(preds[f]))
            label_words = _get_words(str(labels[f]))
            hit = (y_pred_id[i] == y_true_id[i])
            records.append({
                "filename": f,
                "pred_words": " ".join(sorted(pred_words)),
                "label_words": " ".join(sorted(label_words)),
                "hit": hit,
                "y_true_id": y_true_id[i],
                "y_pred_id": y_pred_id[i]
            })
        pd.DataFrame(records).to_csv(out_csv, index=False)
    ck = cohen_kappa_score(y_true_id, y_pred_id)
    ck_ci = _boot_ci(y_true_id, y_pred_id, cohen_kappa_score)

    # 在返回字典中追加
    return {
        "ACC": f"{acc:.3f} ({acc_ci[0]:.3f}, {acc_ci[1]:.3f})",
        "BACC": f"{bacc:.3f} ({bacc_ci[0]:.3f}, {bacc_ci[1]:.3f})",
        "Weighted_F1": f"{wf1:.3f} ({wf1_ci[0]:.3f}, {wf1_ci[1]:.3f})",
        "Cohen_Kappa": f"{ck:.3f} ({ck_ci[0]:.3f}, {ck_ci[1]:.3f})"
    }


def calc_NPV_PNR(M: np.ndarray):
    """M: 24×24 int confusion matrix"""
    TP_c = np.diag(M)
    FP_c = M.sum(axis=0) - TP_c
    FN_c = M.sum(axis=1) - TP_c
    TN_c = M.sum() - (TP_c + FP_c + FN_c)

    total_pos = TP_c + FN_c
    total_neg = TN_c + FP_c
    PNR = total_pos.sum() / total_neg.sum()

    NPV_c = TN_c / (TN_c + FN_c)
    NPV_macro = np.nanmean(NPV_c)
    NPV_weighted = np.average(NPV_c, weights=total_pos + total_neg)

    return {'PNR': PNR,
            'NPV_macro': NPV_macro,
            'NPV_weighted': NPV_weighted}


from sklearn.metrics import f1_score, cohen_kappa_score, matthews_corrcoef
import numpy as np






# ---------- 5. 用法示例 (修改以演示模糊匹配) ----------
if __name__ == "__main__":
    # print(SYNONYM_DICT_DERMNET.values())
    # 示例用法：使用一个阈值来控制模糊匹配的宽松程度
    # FUZZY_THRESHOLD = 70  # 可以调整此阈值 (0-100)
    # test(FUZZY_THRESHOLD)
    # metrics = calcMetrics(
    #     pred_file=REASONINGLAYER_EVALUATION_PATH,
    #     label_file=SKINGPTX_LABELS_PATH,
    #     similarity_threshold=FUZZY_THRESHOLD, base_image_directory=BASE_IMAGE_DIRECTORY, using_re=False,
    #     output_fuzzy_csv_path='/Volumes/T7/SkinGPT-X-EvaluationResults/Experiments/Diagnosis/SkinGPTX/Reasoning_fuzzy_output.csv')
    # print(metrics)
    scores = word_hit_metrics('/Volumes/T7/SkinGPT-X-EvaluationResults/HAM10000/Hulumed/diagnosis_results.csv',
                              '/Volumes/T7/SkinGPT-X-EvaluationResults/HAM10000/Hulumed/diagnosis_results.csv',
                              img_dir=BASE_IMAGE_DIRECTORY,
                              out_csv='/Volumes/T7/SkinGPT-X-EvaluationResults/HAM10000/Hulumed/metrics_results.csv')
    print(f"{scores}")
    # df = pd.read_excel('/Volumes/T7/SkinGPT-X-EvaluationResults/Experiments/Diagnosis/ISIC/final_corrected_results_with_pred_and_true_label.xlsx')

    # 计算指标
    # print(metrics)
    # 保存结果到文件
    # results_df = pd.DataFrame([metrics])
    # results_df.to_excel('/Volumes/T7/SkinGPT-X-EvaluationResults/Experiments/Diagnosis/ISIC/evaluation_metrics.xlsx', index=False)
    # print(f"\n指标结果已保存到: evaluation_metrics.xlsx")
