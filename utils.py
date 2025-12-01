import base64
import os
from typing import List

import markdown
from markdown.extensions import Extension
from markdown.treeprocessors import Treeprocessor
from llama_index.core.schema import ImageNode, TextNode
import re
from pathlib import Path
import aiofiles

from Constants import HAM10000_DISEASE_MAPPING_NAME, HAM10000_DISEASE_NAME


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

def encode_image_to_base64(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"错误：找不到图片文件 '{image_path}'")

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


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
    lines = [
        "### You should take account of the preliminary diagnosis and their possibility below and rethink of your diagnosis:"]

    # 遍历排序后的结果
    for idx, p in sorted_probs_by_value:
        # idx 是疾病名称在 disease_name 列表中的原始索引
        lines.append(f"- {disease_name[idx]}: {p * 100:.1f}%")

    return "\n".join(lines)

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