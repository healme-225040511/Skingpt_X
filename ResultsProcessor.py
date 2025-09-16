#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All-in-one 诊断结果后处理工具
--------------------------------
prepare_csv()      -> 合并 4 个 agent CSV + groundtruth + JSON PrimaryDiagnosis
clean_nan()        -> 去掉空值行
extract_assessment() -> 从任意 agent JSON 提取 DiagnosticAssessment
extract_treatment()  -> 专用于 TreatmentRecommend_output.json 提取 PrimaryDiagnosis
highlight_excel()  -> 生成关键词高亮 Excel
--------------------------------
命令行示例（与老脚本完全兼容）：
# 1. 合并
python all_in_one_diagnosis_utils.py prepare_csv
# 2. 清洗
python all_in_one_diagnosis_utils.py clean_nan
# 3. 提取（通用）
python all_in_one_diagnosis_utils.py extract_assessment --json output/RAG_output.json --agent_name RAG
# 4. 提取（Treatment 专用）
python all_in_one_diagnosis_utils.py extract_treatment
# 5. 高亮
python all_in_one_diagnosis_utils.py highlight_excel
"""
from __future__ import annotations
import json
import csv
import re
import argparse
import pandas as pd
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font

# ---------- 路径默认值 ----------
OUTPUT_DIR = Path("/Users/macbook/Desktop/SkinGPT-X-EvaluationResults/DDI/output")
DATA_DIR   = Path("/Users/macbook/Desktop/SkinGPT-X-Dataset/DDI")

# ---------- 1. 合并 CSV + groundtruth + JSON ----------
def prepare_csv(
    output_dir: str | Path = OUTPUT_DIR,
    data_dir: str | Path = DATA_DIR,
) -> Path:
    output_dir, data_dir = Path(output_dir), Path(data_dir)

    # 1. 读取 4 个诊断 CSV
    file_map = {
        "RAG": output_dir / "RAG_diagnostic_assessment.csv",
        "SkinGPT": output_dir / "SkinGPT_diagnostic_assessment.csv",
        "WebSearch": output_dir / "WebSearch_diagnostic_assessment.csv",
        "Treatment": output_dir / "treatmentRecommend_diagnostic_assessment.csv",
    }
    dfs = {k: pd.read_csv(v) for k, v in file_map.items()}

    # 剔除失败
    for k in ["RAG", "SkinGPT", "WebSearch"]:
        df = dfs[k]
        df.drop(df[df["DiagnosticAssessment"].str.contains("not success", na=False)].index, inplace=True)
        df.rename(columns={"DiagnosticAssessment": f"{k}_diagnostic_assessment"}, inplace=True)

    dfs["Treatment"].rename(columns={"DiagnosticAssessment": "Treatment_diagnostic_assessment"}, inplace=True)

    # 2. 合并
    merged = dfs["RAG"].merge(dfs["SkinGPT"], on="image_name", how="outer")\
                       .merge(dfs["WebSearch"], on="image_name", how="outer")\
                       .merge(dfs["Treatment"], on="image_name", how="outer")

    # 3. groundtruth
    gt_df = pd.read_csv(data_dir / "ddi_metadata.csv")[["DDI_file", "disease"]]\
              .rename(columns={"DDI_file": "image_name", "disease": "groundtruth"})
    merged = merged.merge(gt_df, on="image_name", how="left")

    # 4. JSON PrimaryDiagnosis
    tx_json = output_dir / "TreatmentRecommend_output.json"
    if tx_json.exists():
        with tx_json.open(encoding="utf-8") as f:
            tx_dict = json.load(f)
        tx_df = pd.DataFrame([
            {"image_name": k, "PrimaryDiagnosis": v.get("PrimaryDiagnosis", "")}
            for k, v in tx_dict.items()
        ])
        merged = merged.merge(tx_df, on="image_name", how="left")

    out_csv = output_dir / "merged_diagnostic_assessment_with_groundtruth.csv"
    merged.to_csv(out_csv, index=False)
    print(f"[prepare_csv] 合并完成 -> {out_csv}")
    return out_csv

# ---------- 2. 清洗空值 ----------
def clean_nan(
    in_csv: str | Path | None = None,
    out_csv: str | Path | None = None,
) -> Path:
    if in_csv is None:
        in_csv = OUTPUT_DIR / "merged_diagnostic_assessment_with_groundtruth.csv"
    if out_csv is None:
        out_csv = OUTPUT_DIR / "merged_diagnostic_assessment_with_groundtruth_clean.csv"

    df = pd.read_csv(in_csv).replace("", pd.NA).dropna()
    df.to_csv(out_csv, index=False)
    print(f"[clean_nan] 清洗完成 -> {out_csv}，共删除 {pd.read_csv(in_csv).shape[0] - df.shape[0]} 行")
    return out_csv

# ---------- 3. 通用提取 DiagnosticAssessment ----------
def extract_assessment(
    json_file: str | Path,
    agent_name: str | None = None,
) -> Path:
    json_file = Path(json_file)
    agent_name = agent_name or json_file.stem.replace("_output", "")
    out_csv = json_file.with_name(f"{agent_name}_diagnostic_assessment.csv")

    with json_file.open(encoding="utf-8") as f:
        data = json.load(f)

    def _extract(text: str) -> str:
        if not text:
            return "not success"
        m = re.search(r"### 3\. Diagnostic Assessment\s*[\r\n]+(.*?)(?=###|\Z)", text, flags=re.S)
        return m.group(1).strip() if m else "not success"

    rows = [{"image_name": k, "DiagnosticAssessment": _extract(v if isinstance(v, str) else str(v))}
            for k, v in data.items()]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=["image_name", "DiagnosticAssessment"]).writerows(rows)

    fail = sum(r["DiagnosticAssessment"] == "not success" for r in rows)
    print(f"[extract_assessment] {agent_name} -> {out_csv}，{fail} 条失败")
    return out_csv

# ---------- 4. 专用提取 Treatment PrimaryDiagnosis ----------
def extract_treatment(
    json_file: str | Path = OUTPUT_DIR / "TreatmentRecommend_output.json",
) -> Path:
    json_file = Path(json_file)
    out_csv = json_file.with_name("Treatment_diagnostic_assessment.csv")

    with json_file.open(encoding="utf-8") as f:
        data = json.load(f)

    rows = [{"image_name": k, "DiagnosticAssessment": v.get("PrimaryDiagnosis") or "not success"}
            for k, v in data.items()]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=["image_name", "DiagnosticAssessment"]).writerows(rows)

    fail = sum(r["DiagnosticAssessment"] == "not success" for r in rows)
    print(f"[extract_treatment] -> {out_csv}，{fail} 条失败")
    return out_csv

# ---------- 5. 高亮 Excel ----------
def highlight_excel(
    csv_file: str | Path | None = None,
    excel_file: str | Path | None = None,
) -> Path:
    if csv_file is None:
        csv_file = OUTPUT_DIR / "merged_diagnostic_assessment_with_groundtruth_clean.csv"
    if excel_file is None:
        excel_file = OUTPUT_DIR / "output_highlighted.xlsx"

    df = pd.read_csv(csv_file)
    wb, ws = Workbook(), Workbook().active
    ws = wb.active
    ws.title = "Results"
    red = Font(color="FF0000")

    # 表头
    for col_idx, col_name in enumerate(df.columns, 1):
        ws.cell(row=1, column=col_idx, value=col_name)

    for row_idx, row in df.iterrows():
        groundtruth = str(row["groundtruth"]).strip().lower()
        keywords = [w.strip() for w in groundtruth.split("-") if w.strip()]
        pattern = re.compile(r"\b(" + "|".join(map(re.escape, keywords)) + r")\b", flags=re.I) if keywords else None

        # 固定列
        ws.cell(row=row_idx + 2, column=1, value=str(row["image_name"]))
        ws.cell(row=row_idx + 2, column=5, value=str(row["groundtruth"]))

        # 比对列
        for offset, col in enumerate(["RAG_diagnostic_assessment",
                                      "SkinGPT_diagnostic_assessment",
                                      "WebSearch_diagnostic_assessment",
                                      "PrimaryDiagnosis"], start=2):
            val = str(row[col])
            cell = ws.cell(row=row_idx + 2, column=offset, value=val)
            if pattern and pattern.search(val):
                cell.font = red

    wb.save(excel_file)
    print(f"[highlight_excel] 高亮完成 -> {excel_file}")
    return excel_file

# ---------- CLI ----------
def _cli():
    parser = argparse.ArgumentParser(description="All-in-one 诊断结果后处理工具")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("prepare_csv", help="合并 CSV+groundtruth+JSON")
    p1.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    p1.add_argument("--data_dir", type=str, default=DATA_DIR)

    p2 = sub.add_parser("clean_nan", help="去掉空值行")
    p2.add_argument("--in_csv", type=str, default=None)
    p2.add_argument("--out_csv", type=str, default=None)

    p3 = sub.add_parser("extract_assessment", help="从任意 agent JSON 提取 DiagnosticAssessment")
    p3.add_argument("--json", type=str, required=True)
    p3.add_argument("--agent_name", type=str, default=None)

    p4 = sub.add_parser("extract_treatment", help="专提 Treatment PrimaryDiagnosis")
    p4.add_argument("--json", type=str, default=OUTPUT_DIR / "TreatmentRecommend_output.json")

    p5 = sub.add_parser("highlight_excel", help="生成关键词高亮 Excel")
    p5.add_argument("--csv", dest="csv_file", type=str, default=None)
    p5.add_argument("--excel", dest="excel_file", type=str, default=None)

    args = parser.parse_args()

    if args.cmd == "prepare_csv":
        prepare_csv(args.output_dir, args.data_dir)
    elif args.cmd == "clean_nan":
        clean_nan(args.in_csv, args.out_csv)
    elif args.cmd == "extract_assessment":
        extract_assessment(args.json, args.agent_name)
    elif args.cmd == "extract_treatment":
        extract_treatment(args.json)
    elif args.cmd == "highlight_excel":
        highlight_excel(args.csv_file, args.excel_file)

if __name__ == "__main__":
    _cli()