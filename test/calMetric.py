import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import io

# 读取上传的文件
df = pd.read_csv("Reasoning_output_softmax_normalized.csv")

# 最后一列是真实的标签（label）
y_true = df['label']

# 倒数第二列是原始的文件名（filename），倒数第一列是label
# 概率列是第1列到倒数第2列
prob_columns = df.columns[1:-1]
probabilities = df[prob_columns]

# 获取预测的标签：概率最大的列名即为预测的类别
# idxmax(axis=1) 返回每一行中最大值的列名
y_pred = probabilities.idxmax(axis=1)

# --- 计算指标 ---

# 1. 准确率 (Accuracy)
accuracy = accuracy_score(y_true, y_pred)

# 2. 完整的分类报告 (包含每个类别的 Precision, Recall, F1-Score 以及 Macro/Weighted Avg)
# zero_division='warn' 是默认行为，但此处明确指定以防万一
report = classification_report(y_true, y_pred, zero_division=0)

# 提取关键的宏平均指标
report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
macro_precision = report_dict['macro avg']['precision']
macro_recall = report_dict['macro avg']['recall']
macro_f1 = report_dict['macro avg']['f1-score']


# --- 结果输出 ---
output = f"""
## 📊 模型评估指标

### 整体指标
* **准确率 (Accuracy)**: **{accuracy:.4f}**
* **宏平均精确率 (Macro Avg Precision)**: **{macro_precision:.4f}**
* **宏平均召回率 (Macro Avg Recall)**: **{macro_recall:.4f}**
* **宏平均 F1 分数 (Macro Avg F1-Score)**: **{macro_f1:.4f}**

---

### 各类别详细分类报告 (Classification Report)

{report}
"""

print(output)

import pandas as pd
import io

# 读取上传的文件
df = pd.read_csv("Reasoning_output_softmax_normalized.csv")

# 1. 提取必要的列
# 假设 'filename' 是第一列，'label' 是最后一列
filename_col = df.columns[0]
label_col = df.columns[-1]

# 概率列是第1列到倒数第2列
prob_columns = df.columns[1:-1]
probabilities = df[prob_columns]

# 2. 获取模型的预测标签
# idxmax(axis=1) 返回每一行中最大值的列名，即为预测的类别
df['predicted_label'] = probabilities.idxmax(axis=1)

# 3. 筛选出预测错误的样本
# 预测错误的条件是：预测标签不等于真实标签
incorrect_predictions = df[df['predicted_label'] != df[label_col]].copy()

# 4. 提取所需信息
# 只保留文件名、真实标签和错误的预测标签
error_analysis_df = incorrect_predictions[[filename_col, label_col, 'predicted_label']]

# 5. 写入文件
# 假设将结果保存为 'incorrect_predictions.csv'
output_filename = "incorrect_predictions.csv"
error_analysis_df.to_csv(output_filename, index=False)

# 打印前几条错误记录作为预览
preview = error_analysis_df.head().to_markdown(index=False)
total_errors = len(error_analysis_df)
total_samples = len(df)

print(f"✅ 已成功将预测错误的 {total_errors} 个样本的文件名写入文件：`{output_filename}`。")
print(f"总样本数：{total_samples}，错误率：{(total_errors / total_samples):.4f}")
print("\n--- 错误记录预览 (前 5 行) ---")
print(preview)