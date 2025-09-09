#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 agent JSON 输出提取 Diagnostic Assessment
失败 → 写 "not success"
生成 数据集名称_diagnostic_assessment.csv
"""
import json
import csv
import argparse
from pathlib import Path
import re

def extract_da(text: str) -> str:
    """
    用正则提取 ### 3. Diagnostic Assessment 与下一个 ### 之间的内容
    找不到 → 返回 "not success"
    """
    if not text:
        return "not success"
    m = re.search(r'- \*\*Primary Diagnosis\*\*:\s*(.*?)(?=\n)', text)
    return m.group(1).strip() if m else "not success"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="output/TreatmentRecommend_output.json",
                        help="Agent 输出 JSON 路径")
    parser.add_argument("--agent_name", type=str, default="",
                        help="Agent名称")
    args = parser.parse_args()

    json_file = Path(args.json).expanduser().resolve()
    agent_name = args.agent_name or json_file.stem.replace("_output", "")
    csv_file = json_file.with_name(f"{agent_name}_diagnostic_assessment.csv")

    if not json_file.exists():
        raise FileNotFoundError(json_file)

    data = json.loads(json_file.read_text(encoding="utf-8"))

    rows = []
    for img_name, content in data.items():
        text = content if isinstance(content, str) else str(content)
        da = extract_da(text)
        rows.append({"image_name": img_name, "DiagnosticAssessment": da})

    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "DiagnosticAssessment"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ 已提取 {len(rows)} 条记录 → {csv_file}")
    # 额外打印失败条数
    fail_cnt = sum(1 for r in rows if r["DiagnosticAssessment"] == "not success")
    if fail_cnt:
        print(f"⚠️ 其中 {fail_cnt} 条未匹配到格式，已标记为 'not success'")

if __name__ == "__main__":
    main()