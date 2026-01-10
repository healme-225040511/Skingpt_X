import base64
import os
from typing import List
import pathlib
import markdown
from markdown.extensions import Extension
from markdown.treeprocessors import Treeprocessor
from llama_index.core.schema import ImageNode, TextNode
import re
import csv
from pathlib import Path
import aiofiles
from Constants import HAM10000_DISEASE_MAPPING_NAME,DERMNET_DISEASE_NAME, HAM10000_DISEASE_NAME
import pandas as pd, json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from Constants import Fitzpatrick17k_DISEASE_NAME
class CustomTreeProcessor(Treeprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodes = []

    def run(self, root):
        current_text = []

        for element in root:
            if element.tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                if current_text:
                    self.nodes.append(TextNode(text=''.join(current_text).strip()))
                    current_text = []

            elif element.tag == 'p':
                if element.text is not None:
                    text = re.sub(r'\\(.)', r'\1', element.text)
                    current_text.append(text)

                for child in element.iter():
                    if child.tag == 'img':
                        # src = child.get('src')
                        # self.nodes.append(ImageNode(text="this is an image", image_url=src))
                        pass
                    elif child.text is not None:
                        current_text.append(child.text)

            elif element.tag in ['ul', 'ol']:
                for li in element.findall('li'):
                    if li.text is not None:
                        list_item_text = re.sub(r'\\(.)', r'\1', li.text)
                        current_text.append(f"- {list_item_text}\n")  

            elif element.tag == 'img':
                # src = element.get('src')
                # self.nodes.append(ImageNode(text="this is an image", image_url=src))
                pass

        if current_text:
            self.nodes.append(TextNode(text=''.join(current_text).strip()))

        return root


class CustomExtension(Extension):
    def extendMarkdown(self, md):
        md.treeprocessors.register(CustomTreeProcessor(md), 'custom_tree_processor', 15)


def process_markdown(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        md_text = file.read()
    md = markdown.Markdown(extensions=[CustomExtension()])
    md.convert(md_text)
    return md.treeprocessors['custom_tree_processor'].nodes


def remove_json_markers(text):
    if text.startswith('```json'):
        text = text[len('```json'):]
    if text.endswith('```'):
        text = text[:-len('```')]
    return text.strip()


def unescape_markdown(text):
    """
    Convert escape characters in Markdown text to their actual meaning.
    For example, convert `\\n` to a newline, `\\t` to a tab, etc.
    """
    if isinstance(text, str):
        return text.encode().decode('unicode_escape')
    return text

def load_set(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return {line.rstrip("\n") for line in f if line.strip()}

def mark_done(path, DONE_LOG):
    """追加记录已处理路径"""
    with open(DONE_LOG, "a", encoding="utf-8") as f:
        f.write(path + "\n")
import re, json

def safe_load_json(text: str) -> dict:
    fallback = {
        "PrimaryDiagnosis": "Unable to parse model output",
        "ConfidenceLevel": "Low",
        "DifferentialDiagnoses": ["Parsing error"],
        "KeyFindings": "Model returned non-JSON or empty",
        "KnowledgeAndResearch": {}
    }
    text = text.strip()
    if not text:
        return fallback
    # 去掉 ```json / ``` 围栏
    text = re.sub(r'^```json\s*', '', text, flags=re.I)
    text = re.sub(r'\s*```$', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return fallback
import json
from pathlib import Path
from typing import Any, Dict, Union

class SafeLoadError(Exception):
    """自定义异常，方便调用方统一捕获"""
    pass


def safe_load_json_qwen(src: Union[str, Path]) -> Dict[str, Any]:
    """
    安全地把 JSON 读成 Python dict。
    支持两种输入：
      1. 文件路径（str 或 Path 对象）
      2. 已存在的 JSON 字符串（自动检测）
    失败时抛出 SafeLoadError，而不是默认的 json.JSONDecodeError。
    """
    try:
        # 如果是 Path 对象，先转字符串
        src_str = str(src).strip()

        # 简单启发式：如果以 { 或 [ 开头，就当成原始 JSON 字符串
        if src_str.startswith(("{", "[")):
            return json.loads(src_str)

        # 否则当成文件路径
        path = Path(src_str)
        if not path.exists():
            raise SafeLoadError(f"文件不存在: {path.resolve()}")

        with path.open(encoding="utf-8") as fh:
            return json.load(fh)

    except json.JSONDecodeError as e:
        raise SafeLoadError(f"JSON 解析失败: {e}") from e
    except OSError as e:
        raise SafeLoadError(f"文件读取失败: {e}") from e
def encode_image_to_base64(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"错误：找不到图片文件 '{image_path}'")

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_prob_vec(pre_csv_path, image_path: str):
    """
    传入本地图片路径，返回对应 n 维概率向量；找不到返回 None
    """
    # 1. 把本地路径转成 csv 里的 filename 格式
    #    例如 ./SkinGPT-X-Dataset/Dermnet/test/xxx/yyy.jpg -> xxx/yyy.jpg
    path = pathlib.Path(image_path).resolve()
    # 假设 csv 里存的都是“相对/xxx/yyy.jpg”形式，且目录层级固定
    # 这里简单取后两级，可按实际调整
    key = str(pathlib.Path(*path.parts[-2:])).replace("\\", "/")
    # 2. 读 csv 找行
    with open(pre_csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            feature_dim = len(row) - 3
            if row["filename"] == key:
                # 3. 提取 prob_cls0 ... prob_cls22
                vec = [float(row[f"prob_cls{i}"]) for i in range(feature_dim)]
                return vec
    return None
def build_prelimary_text(prob_vec: List[float], disease_name: List[str]) -> str:
    """
    prob_vec: 长度为 N 的 softmax 概率列表。
    disease_name: 长度为 N 的疾病名称列表，顺序与 prob_vec 严格对应。

    返回一段自然语言，告诉 LLM 目前所有诊断的排名及其概率。

    替代了 torch.topk 的功能，使用标准 Python 排序实现。
    """

    if len(prob_vec) != len(disease_name):
        return "[ERROR] 概率向量和疾病名称列表长度不匹配，无法生成报告。"

    # 1. 将概率和它们对应的索引打包成元组列表: [(prob_0, idx_0), (prob_1, idx_1), ...]
    #    使用 enumerate 生成 (索引, 概率) 对
    indexed_probs = list(enumerate(prob_vec))

    # 2. 按照概率值（元组的第二个元素 x[1]）降序排列
    #    reverse=True 表示降序
    #    新的列表 sorted_probs 格式为 [(idx, prob), (idx, prob), ...]
    sorted_probs_by_value = sorted(indexed_probs, key=lambda x: x[1], reverse=True)

    # 3. 格式化输出字符串
    # lines = [
    #     "### You should take account of the preliminary diagnosis and their possibility below and rethink of your diagnosis:"]
    lines = []
    # 遍历排序后的结果
    # for idx, p in sorted_probs_by_value:
    #     # idx 是疾病名称在 disease_name 列表中的原始索引
    #     lines.append(f"- {disease_name[idx]}: {p * 100:.1f}%")
    for idx, p in sorted_probs_by_value:
        # idx 是疾病名称在 disease_name 列表中的原始索引
        lines.append(f"{disease_name[idx]}")

    # return "\n".join(lines)
    return lines

def expand_disease_names(short_list=HAM10000_DISEASE_NAME):
    """
    将 HAM10000 简写列表转成完整医学描述列表
    :param short_list: 简写列表，例如 ["mel", "nv"]
    :return: 完整描述列表，例如 ["melanoma", "melanocytic nevi "]
    """
    return [HAM10000_DISEASE_MAPPING_NAME[abbr] for abbr in short_list]

def convert_disease_names(old_dict):
    new_dict = {HAM10000_DISEASE_MAPPING_NAME[k]: v for k, v in old_dict.items()}
    return new_dict

def extract_json_items(json_file_path, filename_list_path, output_file_path):
    """
    根据txt文件中的文件名列表，从json文件中提取对应的项，并保存到新的json文件。

    Args:
        json_file_path (str): 包含所有数据的JSON文件路径。
        filename_list_path (str): 包含要提取的文件名（键）列表的TXT文件路径。
        output_file_path (str): 提取结果要保存到的JSON文件路径。
    """

    # 1. 检查文件是否存在
    if not os.path.exists(json_file_path):
        print(f"❌ 错误：JSON 文件不存在于路径: {json_file_path}")
        return
    if not os.path.exists(filename_list_path):
        print(f"❌ 错误：TXT 文件不存在于路径: {filename_list_path}")
        return

    # 2. 读取要查找的文件名列表 (Keys)
    print(f"📚 正在读取文件名列表: {filename_list_path}...")
    filenames_to_find = set()
    try:
        with open(filename_list_path, 'r', encoding='utf-8') as f:
            # 读取每一行，去除首尾空白符（包括换行符）
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    filenames_to_find.add(stripped_line)
    except Exception as e:
        print(f"❌ 读取TXT文件时发生错误: {e}")
        return

    if not filenames_to_find:
        print("⚠️ 警告：TXT 文件中没有找到任何有效的文件名。")
        return

    # 3. 读取完整的 JSON 数据
    print(f"📖 正在读取 JSON 数据: {json_file_path}...")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except json.JSONDecodeError:
        print(f"❌ 错误：JSON 文件格式不正确。请检查文件: {json_file_path}")
        return
    except Exception as e:
        print(f"❌ 读取JSON文件时发生错误: {e}")
        return

    # 4. 提取对应的项
    extracted_data = {}
    missing_keys = []

    print("🔍 正在提取匹配的项...")
    for key in filenames_to_find:
        # 查找 JSON 数据中是否有这个键
        if key in full_data:
            extracted_data[key] = full_data[key]
        else:
            missing_keys.append(key)

    # 5. 保存提取的结果
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            # 使用 indent=4 使输出的 JSON 文件格式美观易读
            json.dump(extracted_data, f, ensure_ascii=False, indent=4)
        print(f"✅ 提取完成！成功将 {len(extracted_data)} 个项保存到: {output_file_path}")

        if missing_keys:
            print(f"ℹ️ 注意：在JSON中未能找到 {len(missing_keys)} 个键。部分示例：{missing_keys[:5]}")

    except Exception as e:
        print(f"❌ 保存输出文件时发生错误: {e}")


# ---------- 4. 示例 ----------
def get_pre_diagnosis(fname, pred_csv_path='/225040511/project/Panderm-EvaluationResults/Dermnet/', MAPPING=Fitzpatrick17k_DISEASE_NAME):
    # 1. 读取CSV
    pred_df = pd.read_csv(pred_csv_path)
    
    # 2. 找到对应 filename 的那一行数据
    target_row = pred_df[pred_df['filename'] == fname]
    
    if target_row.empty:
        return "Unknown", 0.0  # 如果没找到文件，返回默认值
    
    # 3. 获取预测的标签索引 (int)
    pred_idx = int(target_row['predicted_label'].values[0])
    # 4. 构造概率列的名字，例如 "probability_class_9"
    prob_column_name = f'probability_class_{pred_idx}'
    
    # 5. 获取该标签对应的概率值
    prob_value = target_row[prob_column_name].values[0]
    
    # 6. 获取映射后的疾病名称
    # 假设 MAPPING 是列表或字典
    try:
        pred_name = MAPPING[pred_idx]
    except (KeyError, IndexError):
        pred_name = "Unknown"
    return pred_name, prob_value
        # print(count, update_count)