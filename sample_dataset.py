#!/usr/bin/env python3
import os
import random

# ========== 参数 ==========
ROOT_DIR = r'./SkinGPT-X-Dataset/Dermnet/test'          # 换成你的根目录
EXT = {'.jpg', '.jpeg', '.png'}      # 支持的图片后缀（小写）
INDEX_FILE = 'temp_dataset.txt'      # 输出的索引文件
SEED = 42                            # 固定随机种子，可复现
# ==========================

random.seed(SEED)

# 1. 建索引：{子文件夹绝对路径: [图片绝对路径, ...]}
index = {}
for dirpath, _, filenames in os.walk(ROOT_DIR):
    pics = [os.path.join(dirpath, f) for f in filenames
            if os.path.splitext(f.lower())[1] in EXT]
    if pics:                      # 只保留有图的文件夹
        index[dirpath] = pics

# 2. 每个文件夹独立抽 10 %
sampled = []
for folder, pics in index.items():
    k = max(1, len(pics) // 10)   # 至少 1 张
    sampled.extend(random.sample(pics, k))

# 3. 写出索引
with open(INDEX_FILE, 'w', encoding='utf-8') as f:
    for p in sampled:
        f.write(p + '\n')

print(f'共抽取 {len(sampled)} 张图片，索引已保存到 {os.path.abspath(INDEX_FILE)}')