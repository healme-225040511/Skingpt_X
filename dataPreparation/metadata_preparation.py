#!/usr/bin/env python3
# save as: build_csv.py
import pathlib
import random
import csv
from collections import defaultdict

random.seed(42)

root       = pathlib.Path("/Volumes/T7/SkinGPT-X-Dataset/ISIC_2024_Resize224")   # 数据集根目录
out_csv    = root / "dataset.csv"
train_root = root / "train"
test_root  = root / "test"

# ---------- 1. 标签映射 ----------
label_dirs = sorted([d for d in train_root.iterdir() if d.is_dir()])
label2id   = {d.name: idx for idx, d in enumerate(label_dirs)}

# ---------- 2. 收集 ----------
data = defaultdict(list)   # split -> List[ (rel_path, label_id) ]

# 2.1 test 部分直接归到 split=test
for img in test_root.rglob("*.jpg"):
    label_name = img.parent.name
    rel_path   = img.relative_to(root).as_posix()
    data['test'].append((rel_path, label2id[label_name]))

# 2.2 train 部分再拆出 val
for label_dir in label_dirs:
    label_id = label2id[label_dir.name]
    imgs     = list(label_dir.glob("*.jpg"))
    random.shuffle(imgs)
    n_val    = int(len(imgs) * 0.2)
    for i, img in enumerate(imgs):
        rel_path = img.relative_to(root).as_posix()
        split    = 'val' if i < n_val else 'train'
        data[split].append((rel_path, label_id))

# ---------- 3. 写入 ----------
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for split in ['train', 'val', 'test']:
        for rel_path, label_id in data[split]:
            writer.writerow([rel_path, label_id, split])

print(f"✅ 已生成 {out_csv.resolve()}")
print("label 映射：", label2id)