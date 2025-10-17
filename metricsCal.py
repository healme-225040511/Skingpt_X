"""
计算所有模型在数据集上的预测指标
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, average_precision_score
)
# 导入 fuzzywuzzy 库
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# ---------- 1. 工具：从一段预测文本里抽疾病名 ----------
DIAGNOSE_RE = re.compile(
    r"the most likely diagnosis is\s+([^.]+?)(?:\.|\(|,|\||$)",
    flags=re.I
)
BOLD_RE = re.compile(r'\*\*(.*?)\*\*', flags=re.S)


def extractDiagnosisFromMedgamma(text: str) -> str:
    """
    先尝试提取 **疾病名(含括号)**；
    若失败再用旧正则提取。
    统一小写并压缩空格。
    """
    # 优先 **...**
    m = BOLD_RE.search(text)
    if m:
        disease = m.group(1).strip()
    else:
        # 兜底旧逻辑
        m = DIAGNOSE_RE.search(text)
        if not m:
            return ""
        disease = m.group(1).strip()
        # 旧逻辑：去掉括号内说明
        disease = re.sub(r"\s*\([^)]*\)", "", disease)

    # 统一清洗多余空格
    disease = re.sub(r"\s+", " ", disease).strip()
    return disease.lower()


# ---------- 2. 读文件 (修改为读取原始标签，不进行预处理) ----------
def read_pred_file_raw(path: str, using_re:bool = True) -> dict:
    """
    返回 dict: filename -> raw_pred_disease
    """
    df = pd.read_csv(path)
    # 统一小写 key，防止列名大小写问题
    df.columns = [c.lower() for c in df.columns]
    return {
        row['filename']: extractDiagnosisFromMedgamma(row['pred']) if using_re else (row['pred'])
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


# ---------- MODIFIED canonicalize_labels_fuzzy function ----------
def canonicalize_labels_fuzzy(
        raw_labels_to_map: list[str],
        pre_existing_canonical_map: dict = None,
        # Hard-coded raw label -> canonical label mappings (e.g., image_name -> folder_name)
        similarity_threshold: int = 85
) -> dict:
    """
    使用模糊匹配将原始标签规范化为标准标签。
    raw_labels_to_map: 包含所有需要被映射的原始标签字符串列表。
    pre_existing_canonical_map: 可选字典，提供原始标签到其规范标签的预定义硬映射。
                              这些映射将被优先考虑，并且它们的值将成为模糊匹配的潜在规范目标。
    similarity_threshold: 相似度阈值 (0-100)。
    返回一个字典，映射原始标签到其规范标签。
    """
    if pre_existing_canonical_map is None:
        pre_existing_canonical_map = {}

    # Step 1: Combine all unique raw labels, keys from pre_existing_map (e.g., image names),
    # and values from pre_existing_map (e.g., folder names) into a pool for fuzzy matching.
    all_unique_terms_in_pool = (
            set(raw_labels_to_map) |
            set(pre_existing_canonical_map.keys()) |
            set(pre_existing_canonical_map.values())
    )
    # Filter out empty strings which can cause issues with fuzzywuzzy if not desired.
    all_unique_terms_in_pool = {term for term in all_unique_terms_in_pool if term}

    # Sort for consistent processing order
    sorted_unique_terms_pool = sorted(list(all_unique_terms_in_pool))

    # working_map will store intermediate fuzzy match results
    # Default each term to map to itself
    working_map = {term: term for term in sorted_unique_terms_pool}

    # Step 2: Perform fuzzy matching for each term against the pool.
    # The goal is that if A is very similar to B, A should point to B (or vice versa).
    # We prioritize matching to a term that already has a higher "canonical power"
    # (e.e.g, folder names from pre_existing_canonical_map) or itself.

    # Iterate through all labels that we want to map/process
    for query_label in sorted(list(all_unique_terms_in_pool)):  # Iterate over all terms in the pool
        # Do not allow a term to match itself if it's the best option, unless no other good match
        # To avoid matching to itself when there's a better candidate:
        # Create targets excluding the query_label itself.
        fuzzy_targets = [term for term in sorted_unique_terms_pool if term != query_label]

        if fuzzy_targets:
            match_result = process.extractOne(
                query_label,
                fuzzy_targets,
                scorer=fuzz.token_set_ratio  # Robust for multi-word phrases
            )

            if match_result and match_result[1] >= similarity_threshold:
                best_fuzzy_match_candidate = match_result[0]
                # Update map: query_label now points to its best fuzzy match
                # This is an intermediate map, hard mappings will override later.
                working_map[query_label] = best_fuzzy_match_candidate
        # If no fuzzy match > threshold, it remains mapped to itself (default from initialization)

    # Step 3: Apply pre-existing hard mappings (e.g., image basename -> folder name).
    # These mappings take priority and overwrite any fuzzy matches for the image basenames.
    for raw_img_basename, folder_name in pre_existing_canonical_map.items():
        working_map[raw_img_basename] = folder_name

    # Step 4: Resolve chains of mappings (transitive closure).
    # Example: if A -> B and B -> C, then A should finally map to C.
    max_iterations = len(working_map) + 1  # At most N+1 iterations to resolve all chains
    for _ in range(max_iterations):
        changed = False
        for current_label, target_label in list(working_map.items()):
            # If the target_label itself maps to something else, update
            if target_label in working_map and working_map[target_label] != target_label:
                working_map[current_label] = working_map[target_label]
                changed = True
        if not changed:
            break  # No more changes, mapping is stable

    # Step 5: Filter the final map to only include the original `raw_labels_to_map` as keys
    # or relevant pre_existing_canonical_map keys. Expand to include all unique terms
    # for a comprehensive final map the user can inspect.
    final_canonical_map = {}
    for term in sorted_unique_terms_pool:  # Iterate through the full set of terms we considered
        final_canonical_map[term] = working_map.get(term, term)  # Ensure all processed terms are accounted for

    return final_canonical_map


# ---------- 4. 计算指标 (修改以使用规范化后的标签) ----------
def calcMetrics(pred_file: str, label_file: str, similarity_threshold: int = 85,
                base_image_directory: str = None, using_re: bool = False):
    """
    主函数：读文件 -> 对齐 -> 规范化 -> 算五个指标
    Args:
        pred_file (str): 预测结果文件路径。
        label_file (str): 真实标签文件路径。
        similarity_threshold (int): 模糊匹配相似度阈值 (0-100)。
        base_image_directory (str): 包含疾病图片分类文件夹的根目录路径，用于提取文件夹名作为规范标签。

    Returns:
        dict: 包含计算出的各项指标。
    """
    preds_raw = read_pred_file_raw(pred_file, using_re)
    labels_raw = read_label_file_raw(label_file)
    print(preds_raw)
    print(labels_raw)

    # 1. Image folder scanning (new)
    # This dict maps image_basename -> folder_name (canonical)
    image_to_folder_canonical_map = {}
    if base_image_directory:
        image_to_folder_canonical_map = scan_image_folders(base_image_directory)

    # 2. Align data from prediction and label files
    filenames = sorted(set(preds_raw) & set(labels_raw))
    y_true_raw_list = [labels_raw[f] for f in filenames]
    y_pred_raw_list = [preds_raw[f] for f in filenames]

    # 3. Aggregate all raw labels that need to be mapped (from files AND image names)
    all_raw_labels_for_processing = list(set(
        y_true_raw_list +
        y_pred_raw_list +
        list(image_to_folder_canonical_map.keys())  # Add image base names as raw labels for processing
    ))

    # 4. Perform fuzzy canonicalization
    final_canonical_mapping = canonicalize_labels_fuzzy(
        raw_labels_to_map=all_raw_labels_for_processing,
        pre_existing_canonical_map=image_to_folder_canonical_map,  # Pass the image -> folder map as pre-existing
        similarity_threshold=similarity_threshold
    )

    # 5. Apply the final canonicalization map to true and predicted labels
    # Use .get() with a fallback to raw_label itself if somehow not in mapping (shouldn't happen if processing is complete)
    y_true = [final_canonical_mapping.get(raw_label, raw_label) for raw_label in y_true_raw_list]
    y_pred = [final_canonical_mapping.get(raw_label, raw_label) for raw_label in y_pred_raw_list]

    # --- Debugging prints ---
    print("\n--- After Final Canonicalization ---")
    print("Full Canonical Mapping:\n", final_canonical_mapping)  # 打印完整的映射供调试
    print("Canonicalized True Labels (mapped):", y_true)
    print("Canonicalized Predicted Labels (mapped):", y_pred)

    # --- End Debugging prints ---

    # 6. Encode labels and calculate metrics
    all_labels = sorted(list(set(y_true) | set(y_pred)))
    label2id = {l: i for i, l in enumerate(all_labels)}
    y_true_id = np.array([label2id[l] for l in y_true])
    y_pred_id = np.array([label2id[l] for l in y_pred])

    # --- Debugging prints ---
    print("\n--- Encoding for Metrics ---")
    print("Unique Canonical Labels (sorted list):", all_labels)
    print("Number of Unique Canonical Labels (n_class):", len(all_labels))
    print("Canonicalized True Labels (IDs):", y_true_id)
    print("Canonicalized Pred Labels (IDs):", y_pred_id)
    print("------------------------------\n")
    # --- End Debugging prints ---

    metrics = {}
    metrics["ACC"] = accuracy_score(y_true_id, y_pred_id)
    metrics["BACC"] = balanced_accuracy_score(y_true_id, y_pred_id)
    metrics["W_F1"] = f1_score(y_true_id, y_pred_id, average="weighted", zero_division=0)

    n_class = len(all_labels)
    if n_class < 2:
        print(
            "Warning: Only one unique class found after canonicalization (n_class < 2). AUROC and AUPR are not well-defined and will be set to NaN.")
        metrics["AUROC"] = np.nan
        metrics["AUPR"] = np.nan
    else:
        y_true_bin = np.zeros((len(y_true), n_class))
        y_true_bin[np.arange(len(y_true)), y_true_id] = 1
        y_pred_bin = np.zeros((len(y_pred), n_class))
        y_pred_bin[np.arange(len(y_pred)), y_pred_id] = 1

        metrics["AUROC"] = roc_auc_score(y_true_bin, y_pred_bin, average="weighted", multi_class="ovr")
        metrics["AUPR"] = average_precision_score(y_true_bin, y_pred_bin, average="weighted")

    return metrics

def test(FUZZY_THRESHOLD):
    # 创建一些虚拟文件用于演示
    # filename_to_medgamma_pred.csv
    pred_data = {
        'filename': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg', 'img7.png'],
        'pred': [
            'the most likely diagnosis is **Nodular Basal Cell Carcinoma (BCC)**',  # 应匹配 melanoma
            'the most likely diagnosis is Squamous Cell Carcinoma (SCC), likely underlying a Cutaneous Horn',
            # 应匹配 atopic dermatitis (取决于阈值)
            'the most likely diagnosis is Squamous Cell Carcinoma (SCC), highly suspected',  # 应匹配 scc
            'the most likely diagnosis is Extramammary Paget\'s Disease (EMPD)',  # 应匹配 psoriasis
            'the most likely diagnosis is Pigmented Basal Cell Carcinoma"',  # 应匹配 tinea corporis
            'the most likely diagnosis is Pemphigus Foliaceus',  # 自身即规范
            'the most likely diagnosis is Malignant Melanoma, likely Superficial Spreading type'  # 应匹配 bcc
        ]
    }
    pred_df = pd.DataFrame(pred_data)
    pred_df.to_csv("filename_to_medgamma_pred.csv", index=False)

    # filename_to_label.csv
    label_data = {
        'filename': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg', 'img7.png'],
        'label': [
            'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
            'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
            'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',  # 真实标签也有差异
            'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
            'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
            'Acne and Rosacea Photo',  # 应匹配 atopic dermatitis (取决于阈值)
            'Tinea Ringworm Candidiasis and other Fungal Infections'  # 应匹配 bcc
        ]
    }
    label_df = pd.DataFrame(label_data)
    label_df.to_csv("filename_to_label.csv", index=False)

    # 运行指标计算
    res = calcMetrics(
        "filename_to_medgamma_pred.csv",
        "filename_to_label.csv",
        similarity_threshold=FUZZY_THRESHOLD,
        using_re=True
    )
    for k, v in res.items():
        print(f"{k}: {v:.4f}")

    # 清理虚拟文件
    import os
    os.remove("filename_to_medgamma_pred.csv")
    os.remove("filename_to_label.csv")

MEDGAMMA_EVALUATION_PATH = '/Volumes/T7/SkinGPT-X-EvaluationResults/Medgamma/filename_to_medgamma_pred.csv'
REASONINGLAYER_EVALUATION_PATH = '/Volumes/T7/SkinGPT-X-EvaluationResults/Experiments/Diagnosis/SkinGPTX/Reasoning_output.csv'
MEDGAMMA_LABELS_PATH = '/Volumes/T7/SkinGPT-X-EvaluationResults/Experiments/Diagnosis/filename_to_label.csv'
SKINGPTX_LABELS_PATH = '/Volumes/T7/SkinGPT-X-EvaluationResults/Experiments/Diagnosis/SkinGPTX/filename_to_labels.csv'
BASE_IMAGE_DIRECTORY = '/Volumes/T7/SkinGPT-X-Dataset/Dermnet/test'
# ---------- 5. 用法示例 (修改以演示模糊匹配) ----------
if __name__ == "__main__":
    # 示例用法：使用一个阈值来控制模糊匹配的宽松程度
    FUZZY_THRESHOLD = 70  # 可以调整此阈值 (0-100)
    # test(FUZZY_THRESHOLD)
    metrics = calcMetrics(
        pred_file=REASONINGLAYER_EVALUATION_PATH,
        label_file=SKINGPTX_LABELS_PATH,
        similarity_threshold=FUZZY_THRESHOLD, base_image_directory=BASE_IMAGE_DIRECTORY)
    print(metrics)