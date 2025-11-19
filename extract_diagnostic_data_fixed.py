#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取诊断数据的脚本
从多个JSON文件中提取Diagnostic Assessment和PrimaryDiagnosis信息，并生成CSV表
"""

import json
import csv
import os
import re
from typing import Dict, List, Any

def load_json_file(file_path: str) -> Dict[str, Any]:
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return {}

def extract_diagnostic_assessment(content: str) -> str:
    """从文本内容中提取Diagnostic Assessment部分"""
    # 查找"### 3. Diagnostic Assessment"或"Diagnostic Assessment"部分
    patterns = [
        r'### 3\. Diagnostic Assessment.*?(?=###|\Z)',
        r'Diagnostic Assessment.*?(?=###|\Z)',
        r'### 3\. Diagnostic Assessment.*?(?=\n###|\Z)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(0).strip()
    
    # 如果没有找到完整部分，尝试查找Primary Diagnosis
    primary_diagnosis_pattern = r'Primary Diagnosis[:\s]*([^\n]+)'
    match = re.search(primary_diagnosis_pattern, content, re.IGNORECASE)
    if match:
        return f"Primary Diagnosis: {match.group(1).strip()}"
    
    return ""

def extract_primary_diagnosis(data: Dict[str, Any]) -> str:
    """从数据结构中提取PrimaryDiagnosis"""
    if isinstance(data, dict) and 'PrimaryDiagnosis' in data:
        return data['PrimaryDiagnosis']
    elif isinstance(data, str):
        # 如果是字符串，尝试提取Primary Diagnosis
        pattern = r'Primary Diagnosis[:\s]*([^\n]+)'
        match = re.search(pattern, data, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""

def create_label_from_filename(filename: str) -> str:
    """从文件名创建label（去掉.jpg后缀）"""
    return filename.replace('.jpg', '').replace('.png', '').replace('.jpeg', '')

def main():
    # 文件路径
    base_path = "/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test"
    
    files = {
        'RAG_output': os.path.join(base_path, 'RAG_output.json'),
        'SkinGPT_output': os.path.join(base_path, 'SkinGPT_output.json'),
        'WebSearch_output': os.path.join(base_path, 'WebSearch_output.json'),
        'CaseReview_output': os.path.join(base_path, 'CaseReview_output.json'),
        'Reasoning_output': os.path.join(base_path, 'Reasoning_output.json'),
        'TreatmentRecommend_output': os.path.join(base_path, 'TreatmentRecommend_output.json')
    }
    
    # 存储所有数据
    all_data = []
    
    # 获取所有图像文件名（从第一个文件获取）
    rag_data = load_json_file(files['SkinGPT_output'])
    image_names = list(rag_data.keys())
    
    print(f"找到 {len(image_names)} 个图像文件")
    
    # 定义列的顺序
    fieldnames = [
        'image_name', 'label',
        'RAG_output_Diagnostic_Assessment', 'SkinGPT_output_Diagnostic_Assessment', 'WebSearch_output_Diagnostic_Assessment',
        'CaseReview_output_PrimaryDiagnosis', 'Reasoning_output_PrimaryDiagnosis', 'TreatmentRecommend_output_PrimaryDiagnosis'
    ]
    
    # 为每个图像提取数据
    for image_name in image_names:
        row_data = {
            'image_name': image_name,
            'label': create_label_from_filename(image_name),
            'RAG_output_Diagnostic_Assessment': '',
            'SkinGPT_output_Diagnostic_Assessment': '',
            'WebSearch_output_Diagnostic_Assessment': '',
            'CaseReview_output_PrimaryDiagnosis': '',
            'Reasoning_output_PrimaryDiagnosis': '',
            'TreatmentRecommend_output_PrimaryDiagnosis': ''
        }
        
        # 提取RAG、SkinGPT、WebSearch的Diagnostic Assessment
        for file_type in ['RAG_output', 'SkinGPT_output', 'WebSearch_output']:
            file_data = load_json_file(files[file_type])
            if image_name in file_data:
                content = file_data[image_name]
                if isinstance(content, str):
                    diagnostic_assessment = extract_diagnostic_assessment(content)
                    row_data[f'{file_type}_Diagnostic_Assessment'] = diagnostic_assessment
                else:
                    row_data[f'{file_type}_Diagnostic_Assessment'] = str(content)
        
        # 提取CaseReview、Reasoning、TreatmentRecommend的PrimaryDiagnosis
        for file_type in ['CaseReview_output', 'Reasoning_output', 'TreatmentRecommend_output']:
            file_data = load_json_file(files[file_type])
            if image_name in file_data:
                content = file_data[image_name]
                primary_diagnosis = extract_primary_diagnosis(content)
                row_data[f'{file_type}_PrimaryDiagnosis'] = primary_diagnosis
        
        all_data.append(row_data)
    
    # 保存CSV文件
    output_file = "/Volumes/T7/Skingpt-X-EvaluationResults/diagnostic_data_extraction.csv"
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)
    
    print(f"数据提取完成！共处理 {len(all_data)} 个图像")
    print(f"CSV文件已保存到: {output_file}")
    
    # 显示前几行数据作为预览
    print("\n数据预览（前5行）:")
    for i, row in enumerate(all_data[:5]):
        print(f"行 {i+1}:")
        for key, value in row.items():
            # 截断长文本以便显示
            display_value = value[:100] + "..." if len(str(value)) > 100 else value
            print(f"  {key}: {display_value}")
        print()
    
    # 显示数据统计
    print(f"\n数据统计:")
    print(f"总行数: {len(all_data)}")
    print(f"总列数: {len(fieldnames)}")
    
    # 检查每列的非空值数量
    print("\n各列非空值统计:")
    for field in fieldnames:
        non_empty_count = sum(1 for row in all_data if row.get(field, '') != '')
        print(f"{field}: {non_empty_count}/{len(all_data)} 非空值")


import csv
import os


def splitMedgammaResults(input_file):
    """
    Split the CSV file into two separate files:
    1. filename_to_label.csv - maps filename to label
    2. filename_to_medgamma_pred.csv - maps filename to medgamma prediction
    """

    # Initialize lists to store data
    filename_label_data = []
    filename_pred_data = []

    # Read the original CSV file
    with open(input_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            print(row)
            # Extract the full path from image_path column
            full_path = row['image_path']

            # Split the path to get filename and label
            path_parts = full_path.split('/')
            filename = path_parts[-1]  # Last part is the filename
            label = path_parts[-2]  # Second-to-last part is the label

            # Get the prediction
            prediction = row[' medgamma_pred']

            # Store in respective lists
            filename_label_data.append({
                'filename': filename,
                'label': label
            })

            filename_pred_data.append({
                'filename': filename,
                'medgamma_pred': prediction
            })

    # Write File 1: filename_to_label.csv
    with open('/Volumes/T7/SkinGPT-X-EvaluationResults/filename_to_label.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filename_label_data)

    # Write File 2: filename_to_medgamma_pred.csv
    with open('/Volumes/T7/SkinGPT-X-EvaluationResults/filename_to_medgamma_pred.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'medgamma_pred']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filename_pred_data)

    print(f"Successfully created:")
    print(f"1. filename_to_label.csv with {len(filename_label_data)} entries")
    print(f"2. filename_to_medgamma_pred.csv with {len(filename_pred_data)} entries")


# Usage

if __name__ == "__main__":
    # extract()
    splitMedgammaResults('/Volumes/T7/SkinGPT-X-EvaluationResults/results_medgemma.csv')
