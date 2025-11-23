#!/usr/bin/env python3
"""
按 key 索引把 src.json 的值写到 dst.json 对应 key 下，
其余内容保持不变，结果输出为 merged.json
"""
import json
import argparse
from pathlib import Path


def merge_by_key(src_path: Path, dst_path: Path, out_path: Path = Path("merged.json")):
    # 1. 读取两个文件
    with src_path.open("r", encoding="utf-8") as f:
        src = json.load(f)
    with dst_path.open("r", encoding="utf-8") as f:
        dst = json.load(f)

    # 2. 仅更新 dst 中已存在的 key
    if not isinstance(src, dict) or not isinstance(dst, dict):
        raise ValueError("两个 JSON 都必须是 object（最外层为 dict）")

    for k in dst:
        if k in src:
            dst[k] = src[k]

    # 3. 写回新文件
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(dst, f, ensure_ascii=False, indent=2)
    print(f"已生成 {out_path.resolve()}")


if __name__ == "__main__":
    merge_by_key(Path('/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/Reasoning_output.json'),
                 Path('/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/Reasoning_output_副本.json'),
                 Path('/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/Reasoning_output_副本.json'))
