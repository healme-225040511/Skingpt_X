"""
计算所有模型在数据集上的预测指标
"""

import re
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
def read_pred_file_raw(path: str) -> dict:
    """
    返回 dict: filename -> raw_pred_disease
    """
    df = pd.read_csv(path)
    # 统一小写 key，防止列名大小写问题
    df.columns = [c.lower() for c in df.columns]
    return {
        row['filename']: extractDiagnosisFromMedgamma(row['medgamma_pred'])
        for _, row in df.iterrows()
    }

def read_label_file_raw(path: str) -> dict:
    """
    读取 filename_to_label.csv
    返回 dict: filename -> raw_true_label
    """
    df = pd.read_csv(path, dtype=str)          # 自动识别逗号分隔
    df.columns = [c.lower() for c in df.columns]
    return dict(zip(df["filename"], df["label"].str.lower().str.strip()))

# ---------- 3. 标签规范化函数 (新增) ----------
def canonicalize_labels_fuzzy(raw_labels: list[str], similarity_threshold: int = 85) -> dict:
    """
    使用模糊匹配将原始标签规范化为标准标签。
    raw_labels: 包含所有待规范化的原始标签字符串列表。
    similarity_threshold: 相似度阈值 (0-100)。
    返回一个字典，映射原始标签到其规范标签。
    """

    # 存储规范形式的标签
    canonical_labels_set = set() # 用set保证唯一性
    # 存储原始标签到规范标签的映射
    label_to_canonical_map = {}

    # 为了确保处理顺序的稳定性，先对原始标签进行排序
    sorted_raw_labels = sorted(list(set(raw_labels)))

    for raw_label in sorted_raw_labels:
        if raw_label in label_to_canonical_map: # 如果已经被处理过，跳过
            continue

        best_match = None

        # 尝试与已知的规范标签进行模糊匹配
        if canonical_labels_set:
            # 使用 token_set_ratio 对包含多个单词的标签效果更好，因为它不关心单词顺序和重复
            match_result = process.extractOne(raw_label, list(canonical_labels_set), scorer=fuzz.token_set_ratio)

            if match_result and match_result[1] >= similarity_threshold:
                best_match = match_result[0]

        if best_match:
            # 如果找到最佳匹配，则将当前原始标签映射到该规范标签
            label_to_canonical_map[raw_label] = best_match
        else:
            # 如果没有找到足够相似的规范标签，则当前原始标签成为一个新的规范标签
            canonical_labels_set.add(raw_label)
            label_to_canonical_map[raw_label] = raw_label

    # 需要对所有原始标签重新映射一次，因为在构建过程中，
    # 某个标签可能在靠后的迭代中才找到其规范
    final_canonical_map = {}
    for raw_label in set(raw_labels): # 再次遍历所有原始标签，确保都被映射
        best_match_label = raw_label

        if canonical_labels_set:
            match_result = process.extractOne(raw_label, list(canonical_labels_set), scorer=fuzz.token_set_ratio)
            if match_result and match_result[1] >= similarity_threshold:
                best_match_label = match_result[0]

        final_canonical_map[raw_label] = best_match_label

    return final_canonical_map

# ---------- 4. 计算指标 (修改以使用规范化后的标签) ----------
def calcMetricsMedgamma(pred_file: str, label_file: str, similarity_threshold: int = 85):
    """
    主函数：读文件 -> 对齐 -> 规范化 -> 算五个指标
    返回 dict
    """
    preds_raw = read_pred_file_raw(pred_file)
    labels_raw = read_label_file_raw(label_file)
    # 对齐
    filenames = sorted(set(preds_raw) & set(labels_raw))
    y_true_raw_list = [labels_raw[f] for f in filenames]
    y_pred_raw_list = [preds_raw[f] for f in filenames]

    # 结合所有原始标签，进行规范化
    all_raw_labels = list(set(y_true_raw_list + y_pred_raw_list))
    # 获取原始标签到规范标签的映射
    canonical_mapping = canonicalize_labels_fuzzy(all_raw_labels, similarity_threshold)
    print(canonical_mapping)

    # 应用规范化映射
    y_true = [canonical_mapping[raw_label] for raw_label in y_true_raw_list]
    y_pred = [canonical_mapping[raw_label] for raw_label in y_pred_raw_list]

    # 统一标签编码 (这部分与原代码一致，但现在操作的是规范化后的标签)
    all_labels = sorted(set(y_true) | set(y_pred))
    label2id = {l: i for i, l in enumerate(all_labels)}
    y_true_id = np.array([label2id[l] for l in y_true])
    y_pred_id = np.array([label2id[l] for l in y_pred])

    # 多分类概率格式：这里用 0/1 的 one-hot 近似
    # 对于 AUROC / AUPR 必须转成二值或多分类概率矩阵
    n_class = len(all_labels)
    y_true_bin = np.zeros((len(y_true), n_class))
    y_true_bin[np.arange(len(y_true)), y_true_id] = 1
    y_pred_bin = np.zeros((len(y_pred), n_class))
    y_pred_bin[np.arange(len(y_pred)), y_pred_id] = 1

    metrics = {
        "ACC": accuracy_score(y_true_id, y_pred_id),
        "BACC": balanced_accuracy_score(y_true_id, y_pred_id),
        "W_F1": f1_score(y_true_id, y_pred_id, average="weighted", zero_division=0),
        "AUROC": roc_auc_score(y_true_bin, y_pred_bin, average="weighted", multi_class="ovr"),
        "AUPR": average_precision_score(y_true_bin, y_pred_bin, average="weighted")
    }
    return metrics

def calMetricsReasoningLaye(pred_file: str, label_file: str, FUZZY_THRESHOLD: int):
    return
def test(FUZZY_THRESHOLD):
    # 创建一些虚拟文件用于演示
    # filename_to_medgamma_pred.csv
    pred_data = {
        'filename': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg', 'img7.png'],
        'medgamma_pred': [
            'the most likely diagnosis is **Nodular Basal Cell Carcinoma (BCC)**',  # 应匹配 melanoma
            'the most likely diagnosis is Squamous Cell Carcinoma (SCC), likely underlying a Cutaneous Horn',  # 应匹配 atopic dermatitis (取决于阈值)
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
            ' Acne and Rosacea Photo',  # 应匹配 atopic dermatitis (取决于阈值)
            'Tinea Ringworm Candidiasis and other Fungal Infections'  # 应匹配 bcc
        ]
    }
    label_df = pd.DataFrame(label_data)
    label_df.to_csv("filename_to_label.csv", index=False)

    # 运行指标计算
    res = calcMetricsMedgamma(
        "filename_to_medgamma_pred.csv",
        "filename_to_label.csv",
        similarity_threshold=FUZZY_THRESHOLD
    )
    for k, v in res.items():
        print(f"{k}: {v:.4f}")

    # 清理虚拟文件
    import os
    os.remove("filename_to_medgamma_pred.csv")
    os.remove("filename_to_label.csv")

# ---------- 5. 用法示例 (修改以演示模糊匹配) ----------
if __name__ == "__main__":
    # 示例用法：使用一个阈值来控制模糊匹配的宽松程度
    FUZZY_THRESHOLD = 50 # 可以调整此阈值 (0-100)
    metrics = calcMetricsMedgamma(pred_file='/Volumes/T7/SkinGPT-X-EvaluationResults/Medgamma/filename_to_medgamma_pred.csv', label_file='/Volumes/T7/SkinGPT-X-EvaluationResults/Medgamma/filename_to_label.csv', similarity_threshold=FUZZY_THRESHOLD)
    print(metrics)