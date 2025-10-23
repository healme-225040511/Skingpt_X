import pandas as pd, numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, cohen_kappa_score, confusion_matrix)

# 1. 读入预测概率表
df = pd.read_csv('/Volumes/T7/SkinGPT-X-EvaluationResults/PanDerm_Base_LP_result/Dermnet_predprob.csv')

# 2. 生成预测标签（已给出可直接用）
y_pred = df['pred_label'].values

# 3. 解析真实标签（示例：按文件夹名映射）
label_map = {
    'Acne and Rosacea Photos': 0,
    'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions': 1,
    'Atopic Dermatitis Photos': 2,
    'Bullous Disease Photos': 3,
    'Cellulitis Impetigo and other Bacterial Infections': 4,
    'Eczema Photos': 5,
    'Exanthems and Drug Eruptions': 6,
    'Hair Loss Photos Alopecia and other Hair Diseases': 7,
    'Herpes HPV and other STDs': 8,
    'Light Diseases and Disorders of Pigmentation': 9,
    'Lupus and other Connective Tissue diseases': 10,
    'Melanoma Skin Cancer Nevi and Moles': 11,
    'Nail Fungus and other Nail Disease': 12,
    'Poison Ivy Photos and other Contact Dermatitis': 13,
    'Psoriasis pictures Lichen Planus and related diseases': 14,
    'Scabies Lyme Disease and other Infestations and Bites': 15,
    'Seborrheic Keratoses and other Benign Tumors': 16,
    'Systemic Disease': 17,
    'Tinea Ringworm Candidiasis and other Fungal Infections': 18,
    'Urticaria Hives': 19,
    'Vascular Tumors': 20,
    'Vasculitis Photos': 21,
    'Warts Molluscum and other Viral Infections': 22
}
y_true = df['filename'].str.extract(r'^([^/]+)')[0].map(label_map).values
y_true = y_true.astype('int')
print(y_pred, y_true)

# ---------- 0. 数据 ----------
# y_true / y_pred 已经从 Dermnet_predprob.csv 解析好
# 这里仅示例，实际替换成自己的向量
# y_true = ...
# y_pred = ...

R = 1000               # bootstrap 次数
rng = np.random.RandomState(42)
n = y_true.shape[0]

# ---------- 1. 指标函数 ----------
def acc(y, p):   return accuracy_score(y, p)
def bacc(y, p):  return balanced_accuracy_score(y, p)
def wf1(y, p):   return f1_score(y, p, average='weighted', zero_division=0)

def wkappa(y, p):
    return cohen_kappa_score(y, p, weights='quadratic')

def wnpv(y, p):
    C = confusion_matrix(y, p)
    sup = C.sum(axis=1)          # 每个真实类的样本数
    K = C.shape[0]
    npv_list = []
    for i in range(K):
        # 非 i 类的真实索引
        other_true = np.r_[0:i, i+1:K]
        # 非 i 类的预测索引
        other_pred = np.r_[0:i, i+1:K]
        tn = C[np.ix_(other_true, other_pred)].sum()  # 保持二维
        fn = C[i, other_pred].sum()                   # i 类被判到其它类
        npv_list.append(tn / (tn + fn + 1e-8))
    return np.average(npv_list, weights=sup)

# ---------- 2. bootstrap ----------
def boot(func):
    dist = []
    for _ in range(R):
        idx = rng.choice(n, size=n, replace=True)
        dist.append(func(y_true[idx], y_pred[idx]))
    return np.percentile(dist, [2.5, 97.5])

acc_point   = acc(y_true, y_pred);   acc_ci   = boot(acc)
bacc_point  = bacc(y_true, y_pred);  bacc_ci  = boot(bacc)
wf1_point   = wf1(y_true, y_pred);   wf1_ci   = boot(wf1)
wCK_point   = wkappa(y_true, y_pred); wCK_ci  = boot(wkappa)
wNPV_point  = wnpv(y_true, y_pred);  wNPV_ci = boot(wnpv)

# ---------- 3. 组装 report ----------
report = {
    "ACC": f"{acc_point:.3f} ({acc_ci[0]:.3f}, {acc_ci[1]:.3f})",
    "BACC": f"{bacc_point:.3f} ({bacc_ci[0]:.3f}, {bacc_ci[1]:.3f})",
    "Weighted_F1": f"{wf1_point:.3f} ({wf1_ci[0]:.3f}, {wf1_ci[1]:.3f})",
    "Cohen_Kappa": f"{wCK_point:.3f} ({wCK_ci[0]:.3f}, {wCK_ci[1]:.3f})",
    "Weighted_NPV": f"{wNPV_point:.3f} ({wNPV_ci[0]:.3f}, {wNPV_ci[1]:.3f})"
}

print(report)