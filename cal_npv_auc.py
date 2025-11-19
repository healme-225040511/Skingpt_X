from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# 1. 读 CSV（第一列是图片路径，其余是概率）
df = pd.read_csv('disease_probs.csv')
y_score = df.iloc[:, 1:].values   # 预测概率
y_true = []                       # 真实标签（父目录）

for p in df.iloc[:, 0]:
    y_true.append(Path(p).parts[0])

# 2. 二值化标签
classes = df.columns[1:].tolist()
y_true_bin = np.array([[1 if cls == true else 0 for cls in classes] for true in y_true])

# 3. 计算指标
macro_auc = roc_auc_score(y_true_bin, y_score, average='macro')
micro_auc = roc_auc_score(y_true_bin, y_score, average='micro')
macro_ap  = average_precision_score(y_true_bin, y_score, average='macro')
micro_ap  = average_precision_score(y_true_bin, y_score, average='micro')

print(f'Macro ROC-AUC : {macro_auc:.4f}')
print(f'Micro ROC-AUC : {micro_auc:.4f}')
print(f'Macro PR-AUC  : {macro_ap:.4f}')
print(f'Micro PR-AUC  : {micro_ap:.4f}')