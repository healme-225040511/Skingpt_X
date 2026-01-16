import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.image_utils import load_image  # 👈 新增：用于安全加载图像（支持路径 & URL）
from typing import Dict, List, Optional  # 👈 修改：补充 Optional
import json
import os

# 必须在任何 transformers 导入前执行
os.environ["HF_HOME"] = "/225040511/project/hf_cache"
# from PIL import Image  # 不再需要单独导入（transformers 内部已封装）

# =============== 【新增】轻量 VL 模型配置 ===============
# 👇 替换为你的本地 2B VL 模型路径（推荐！）
# 从 HF 下载命令：huggingface-cli download Qwen/Qwen2-VL-2B-Instruct --local-dir ./qwen2-vl-2b
VL_MODEL_NAME = "/225040511/project/hf_cache/Qwen-VL-8B-Instruct"
# 若想用 7B：VL_MODEL_NAME = "/225040511/project/hf_cache/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/latest"

vl_model = None      # 👈 新增：VL 模型全局缓存
vl_processor = None  # 👈 新增：VL tokenizer 全局缓存
# ====================================================


# =============== 【新增】VL 模型加载函数 ===============
def _load_qwen_vl_model():
    """初始化并缓存 Qwen-VL 模型和 Tokenizer（轻量版）"""
    global vl_model, vl_processor
    if vl_model is None or vl_processor is None:
        print(f"⏳ 首次加载轻量多模态模型: {os.path.basename(VL_MODEL_NAME)}...")
        try:
            vl_processor = AutoProcessor.from_pretrained(
                VL_MODEL_NAME,
                trust_remote_code=True,
                use_fast=False  # Qwen-VL 推荐关闭 fast tokenizer
            )

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            vl_model = AutoModelForImageTextToText.from_pretrained(
                VL_MODEL_NAME,
                torch_dtype='auto',
                device_map="auto",
                trust_remote_code=True,
            ).eval()
            print("✅ 轻量 VL 模型加载成功（2B/7B）。")
        except Exception as e:
            print(f"❌ VL 模型加载失败: {e}")
            raise
# ====================================================


import re
from pathlib import Path

def parse_skin_disease_path(image_path):
    p = Path(image_path)
    
    # 1. 获取目录部分（即大类）
    parent_dir = p.parent.name.strip()
    disease_class = parent_dir
    
    # 2. 从文件名提取子类：去掉扩展名，按 '-' 拆分
    stem = p.stem  # e.g., "Psoriasis-Guttate-96"
    
    # 尝试匹配：Psoriasis-Type-...
    parts = stem.split('-')
    while parts and re.fullmatch(r'\d+', parts[-1]):
        parts.pop()
    subclass = ' '.join(part.title() for part in parts if part)

    # 3. 生成描述
    # description = f"Disease Category: {disease_class}, SubCategory: {subclass}"
    description = f"{disease_class}"
    return description
# =============== 【补全】多模态推理函数 ===============
def local_generate_response_vl(
    temperature: float,
    max_tokens: int,
    prompt: str,
    image_path: str = None  # 兼容你的原接口
) -> str:
    """
    使用本地轻量 Qwen-VL 模型（2B/7B）生成响应，支持图像输入。
    
    注意：当 image_path 为空时，退化为纯文本推理（但不如纯文本模型高效）。
    """
    _load_qwen_vl_model()
    image = Image.open(image_path).convert("RGB")

    # 构造 messages：Qwen2-VL 支持混合 content
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    # 使用 processor 格式化输入
    text = vl_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = vl_processor(
        text=[text], images=[image], return_tensors="pt"
    ).to(vl_model.device)
    
    # 生成
    with torch.no_grad():
        generated_ids = vl_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            top_p=1,
            temperature=temperature
        )
    
    # 解码输出（去掉输入部分）
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    response = vl_processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    return response.strip()
