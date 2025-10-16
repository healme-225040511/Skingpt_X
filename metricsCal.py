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

# ---------- 2. 读文件 ----------
def read_pred_file(path: str) -> dict:
    """
    返回 dict: filename -> pred_disease
    """
    df = pd.read_csv(path)
    # 统一小写 key，防止列名大小写问题
    df.columns = [c.lower() for c in df.columns]
    return {
        row['filename']: extractDiagnosisFromMedgamma(row['medgamma_pred'])
        for _, row in df.iterrows()
    }

def read_label_file(path: str) -> dict:
    """
    读取 filename_to_label.csv
    返回 dict: filename -> true_label
    """
    df = pd.read_csv(path, dtype=str)          # 自动识别逗号分隔
    df.columns = [c.lower() for c in df.columns]
    return dict(zip(df["filename"], df["label"].str.lower().str.strip()))

# ---------- 3. 计算指标 ----------
def calc_metrics(pred_file: str, label_file: str):
    """
    主函数：读文件 -> 对齐 -> 算五个指标
    返回 dict
    """
    preds = read_pred_file(pred_file)
    labels = read_label_file(label_file)
    # 对齐
    filenames = sorted(set(preds) & set(labels))
    y_true = [labels[f] for f in filenames]
    y_pred = [preds[f] for f in filenames]

    # 统一标签编码
    all_labels = sorted(set(y_true) | set(y_pred))
    label2id = {l: i for i, l in enumerate(all_labels)}
    y_true_id = np.array([label2id[l] for l in y_true])
    y_pred_id = np.array([label2id[l] for l in y_pred])
    print(y_true_id)
    print(y_pred_id)

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

# ---------- 4. 用法示例 ----------
if __name__ == "__main__":
    res = calc_metrics("/Volumes/T7/SkinGPT-X-EvaluationResults/Medgamma/filename_to_medgamma_pred.csv", "/Volumes/T7/SkinGPT-X-EvaluationResults/Medgamma/filename_to_label.csv")
    for k, v in res.items():
        print(f"{k}: {v:.4f}")