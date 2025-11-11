#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取差异样本路径到列表
"""


def read_samples_from_file(file_path):
    """
    从文件中读取样本路径到列表

    Args:
        file_path: 文件路径

    Returns:
        list: 样本路径列表
    """
    samples = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 跳过前3行（标题行）
    for line in lines[3:]:
        line = line.strip()[0]
        # 跳过空行
        if line:
            samples.append(line)

    return samples


def read_samples_simple(file_path):
    """
    更简单的方法读取样本路径到列表

    Args:
        file_path: 文件路径

    Returns:
        list: 样本路径列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # 读取所有行并过滤掉空行和标题
        samples = [line.strip() for line in f if line.strip() and not line.strip().startswith('仅') and '=' not in line]

    return samples


if __name__ == '__main__':
    # 文件路径
    file_path = 'samples_only_in_reasoning.txt'

    # 方法1: 使用第一个函数
    print("=" * 60)
    print("方法1: 读取样本路径")
    print("=" * 60)
    samples = read_samples_from_file(file_path)
    print(f"共读取 {len(samples)} 个样本")
    print("\n前5个样本:")
    for i, sample in enumerate(samples[:5], 1):
        print(f"{i}. {sample}")

    print("\n" + "=" * 60)
    print("方法2: 简化方法")
    print("=" * 60)
    samples_simple = read_samples_simple(file_path)
    print(f"共读取 {len(samples_simple)} 个样本")
    print("\n前5个样本:")
    for i, sample in enumerate(samples_simple[:5], 1):
        print(f"{i}. {sample}")

    print("\n" + "=" * 60)
    print("使用建议:")
    print("=" * 60)
    print("""
# 在你的代码中使用:
from read_samples import read_samples_from_file

samples = read_samples_from_file('samples_only_in_reasoning.txt')
print(f"共 {len(samples)} 个样本")

# 或者使用简化方法:
from read_samples import read_samples_simple

samples = read_samples_simple('samples_only_in_reasoning.txt')
    """)

