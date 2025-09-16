#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 TreatmentRecommend_output.json 提取 PrimaryDiagnosis
生成 {agent}_diagnostic_assessment.csv
"""
import json
import csv
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="output/TreatmentRecommend_output.json",
                        help="JSON 文件路径")
    parser.add_argument("--agent_name", type=str, default="Treatment",
                        help="Agent 名称（决定输出 CSV 文件名）")
    args = parser.parse_args()

    json_file = Path(args.json).expanduser().resolve()
    csv_file  = json_file.with_name(f"{args.agent_name}_diagnostic_assessment.csv")

    if not json_file.exists():
        raise FileNotFoundError(json_file)

    data = json.loads(json_file.read_text(encoding="utf-8"))

    rows = []
    for img, body in data.items():
        # 直接取 PrimaryDiagnosis，缺失则标记 not success
        diag = body.get("PrimaryDiagnosis") or "not success"
        rows.append({"image_name": img, "DiagnosticAssessment": diag})

    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "DiagnosticAssessment"])
        writer.writeheader()
        writer.writerows(rows)

    fail = sum(1 for r in rows if r["DiagnosticAssessment"] == "not success")
    print(f"✅ 已提取 {len(rows)} 条 → {csv_file}")
    if fail:
        print(f"⚠️  {fail} 条缺失 PrimaryDiagnosis，已标为 'not success'")

if __name__ == "__main__":
    main()