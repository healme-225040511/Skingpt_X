import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    cohen_kappa_score,
    classification_report,
    roc_auc_score,
)
from sklearn.utils import resample

# --- 配置参数 ---
N_BOOTSTRAPS = 1000  # 引导法的重复次数
ALPHA = 0.05  # 置信水平 1 - ALPHA = 95%
DATA_FILE = "Reasoning_output_softmax_normalized.csv"

# 读取上传的文件
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"错误: 找不到文件 {DATA_FILE}。请确保文件已上传。")
    exit()

# 提取标签和概率
# 假设 'label' 是真实标签列，其余列为概率列（最后一列除外）
y_true = df['label']
prob_columns = df.columns[1:-1]
probabilities = df[prob_columns]
y_pred = probabilities.idxmax(axis=1)  # 预测标签

# 获取所有类别名称
ALL_LABELS = list(probabilities.columns)


# --- 1. 定义置信区间计算函数（引导法）---
def bootstrap_ci(y_true, probabilities, metric_type, n_bootstraps=N_BOOTSTRAPS, alpha=ALPHA):
    """
    使用非参数引导法计算指定指标的置信区间。
    metric_type 必须是 'ACC', 'BACC', 'Weighted', 'AUC_ROC' 之一。
    'Weighted' 同时计算 F1, NPV (使用 Weighted Precision), 和 Kappa。
    """
    rng = np.random.RandomState(42)
    scores = []

    y_true_np = y_true.to_numpy()
    probabilities_np = probabilities.to_numpy()

    for _ in range(n_bootstraps):
        try:
            # 带有放回的抽样
            indices = np.arange(len(y_true_np))
            # 确保 resample_indices 始终使用同一个 RandomState
            resample_indices = resample(indices, replace=True, random_state=rng)

            y_true_resample = y_true_np[resample_indices]
            probabilities_resample = probabilities_np[resample_indices]

            y_pred_resample = [ALL_LABELS[i] for i in probabilities_resample.argmax(axis=1)]

            unique_labels = np.unique(y_true_resample)
            if len(unique_labels) < 2:
                # 样本子集类别太少，无法计算多分类指标
                continue

            if metric_type == 'ACC':
                score = accuracy_score(y_true_resample, y_pred_resample)

            elif metric_type == 'BACC':
                score = balanced_accuracy_score(y_true_resample, y_pred_resample)

            elif metric_type == 'Weighted':
                # 计算 Weighted F1, Weighted NPV (使用 Weighted Precision), Cohen's Kappa
                kappa = cohen_kappa_score(y_true_resample, y_pred_resample)

                f1_weighted = f1_score(y_true_resample, y_pred_resample, average='weighted', zero_division=0)
                # 使用 Weighted Precision 替代 Weighted NPV
                npv_weighted = precision_score(y_true_resample, y_pred_resample, average='weighted', zero_division=0)

                score = (f1_weighted, npv_weighted, kappa)

            elif metric_type == 'AUC_ROC':
                score = roc_auc_score(
                    y_true_resample,
                    probabilities_resample,
                    multi_class='ovr',
                    average='macro',
                    labels=ALL_LABELS
                )

            scores.append(score)

        except Exception as e:
            # 捕获并跳过计算失败的引导样本
            continue

    # --- 计算 CI ---
    if len(scores) < 30:
        print(f"⚠️ 警告：引导法 '{metric_type}' 只有 {len(scores)} 次成功计算。无法可靠计算 CI。")
        if metric_type == 'Weighted':
            return [np.nan] * 6
        return np.nan, np.nan

    # 对得分进行排序
    # 对于多指标元组，基于第一个指标 (Weighted F1) 排序
    scores.sort(key=lambda x: x if metric_type not in ['Weighted'] else x[0])

    lower_bound_index = int(alpha / 2 * len(scores))
    upper_bound_index = int((1 - alpha / 2) * len(scores))

    if metric_type == 'Weighted':
        # Scores 是 (Weighted F1, Weighted NPV, Kappa) 的元组列表
        f1_scores = [s[0] for s in scores]
        npv_scores = [s[1] for s in scores]
        kappa_scores = [s[2] for s in scores]

        f1_lower = f1_scores[lower_bound_index]
        f1_upper = f1_scores[upper_bound_index]
        n_lower = npv_scores[lower_bound_index]
        n_upper = npv_scores[upper_bound_index]
        k_lower = kappa_scores[lower_bound_index]
        k_upper = kappa_scores[upper_bound_index]

        return f1_lower, f1_upper, n_lower, n_upper, k_lower, k_upper

    else:
        # Scores 是单值列表
        lower_bound = scores[lower_bound_index]
        upper_bound = scores[upper_bound_index]
        return lower_bound, upper_bound


# 格式化函数
def format_ci(value, lower, upper):
    """将指标格式化为 '值 (CI下限, CI上限)'"""
    if np.isnan(value) or np.isnan(lower):
        return f"{value:.3f} (nan, nan)"
    return f"{value:.3f} ({lower:.3f}, {upper:.3f})"


# --- 2. 计算各项指标及其 95% CI ---

# 2.1 整体指标 (ACC, BACC)
accuracy = accuracy_score(y_true, y_pred)
acc_lower, acc_upper = bootstrap_ci(y_true, probabilities, 'ACC')

bacc = balanced_accuracy_score(y_true, y_pred)
bacc_lower, bacc_upper = bootstrap_ci(y_true, probabilities, 'BACC')

# 2.2 加权指标 (F1, NPV, Kappa)
weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
# 使用 Weighted Precision 作为 Weighted NPV 的代理
weighted_npv = precision_score(y_true, y_pred, average='weighted', zero_division=0)
kappa = cohen_kappa_score(y_true, y_pred)

ci_result_weighted = bootstrap_ci(y_true, probabilities, 'Weighted')

if len(ci_result_weighted) == 6:
    f1_lower, f1_upper, n_lower, n_upper, k_lower, k_upper = ci_result_weighted
else:
    f1_lower, f1_upper, n_lower, n_upper, k_lower, k_upper = [np.nan] * 6

# 2.3 AUC-ROC
try:
    macro_auc = roc_auc_score(y_true, probabilities, multi_class='ovr', average='macro', labels=ALL_LABELS)
    auc_lower, auc_upper = bootstrap_ci(y_true, probabilities, 'AUC_ROC')
except ValueError:
    macro_auc = np.nan
    auc_lower, auc_upper = np.nan, np.nan

# --- 3. 识别错误病例 ---
# 找出预测标签不等于真实标签的行
misclassified_indices = (y_true != y_pred)
error_df = df[misclassified_indices].copy()

# --- 核心修改：只提取样本 ID ---
# 假设第一个不是 'label' 或概率列的列是文件/样本 ID
ID_COLUMNS = [col for col in df.columns if col not in ['label'] + list(prob_columns)]

if ID_COLUMNS:
    # 选取第一个非标签/非概率列作为 ID
    ID_COLUMN_NAME = ID_COLUMNS[0]
    # 提取错误样本的 ID 列表，并将其格式化为字符串
    error_report_content = error_df[ID_COLUMN_NAME].to_string(index=False)
    error_report_title = f"以下是错误分类样本的 {ID_COLUMN_NAME} (文件/样本 ID) 列表："
else:
    # 如果没有额外的列作为 ID，则打印样本的 DataFrame 索引
    error_report_content = misclassified_indices[misclassified_indices].index.to_series().to_string(index=True)
    error_report_title = "以下是错误分类样本的索引 (Index) 列表："

error_report = f"""
## ❌ 错误病例分析 (Misclassified Cases)

共识别出 {len(error_df)} 个错误分类的样本。

{error_report_title}
{error_report_content}
"""

# --- 4. 格式化最终输出 ---
output = f"""
## 📊 模型评估指标

### 整体指标 (95% 置信区间)
* **准确率 (ACC)**: **{format_ci(accuracy, acc_lower, acc_upper)}**
* **平衡准确率 (BACC)**: **{format_ci(bacc, bacc_lower, bacc_upper)}**
* **加权 F1 分数 (Weighted F1)**: **{format_ci(weighted_f1, f1_lower, f1_upper)}**
* **加权 NPV (Weighted NPV)**: **{format_ci(weighted_npv, n_lower, n_upper)}**
* **宏平均 AUC-ROC**: **{format_ci(macro_auc, auc_lower, auc_upper)}**
* **Cohen's Kappa**: **{format_ci(kappa, k_lower, k_upper)}**

---

### 各类别详细分类报告 (Classification Report)

{classification_report(y_true, y_pred, zero_division=0)}

---
{error_report}
"""

print(output)