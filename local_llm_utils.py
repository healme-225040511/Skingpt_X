import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Qwen2VLForConditionalGeneration, AutoProcessor,AutoModelForImageTextToText
from Constants import DERMNET_DISEASE_NAME
from typing import Dict, List
import json
import os
# 必须在任何 transformers 导入前执行
os.environ["HF_HOME"] = "/225040511/project/hf_cache"
from PIL import Image  # 仅用于多模态模型 (如 Qwen-VL)

# 全局变量：用于缓存模型和分词器，避免每次调用都重新加载
# 请确保您的环境中安装了 'Qwen/Qwen-7B-Chat' 模型
# MODEL_NAME = "/225040511/project/hf_cache/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137"
MODEL_NAME = "/225040511/project/hf_cache/Qwen3-30B-A3B"
# MODEL_NAME = "/225040511/project/hf_cache/DeepSeek-R1-67B-chat"
max_mem = {i: "30GiB" for i in range(torch.cuda.device_count())}
max_mem["cpu"] = "50GiB"          # 溢出部分放内存，避免直接炸显存

model = None
tokenizer = None
def _load_qwen_model():
    """初始化并缓存 Qwen 模型和 Tokenizer"""
    global model, tokenizer
    if model is None or tokenizer is None:
        print(f"⏳ 首次加载本地模型: {MODEL_NAME}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            # bnb_config = BitsAndBytesConfig(
            #     load_in_8bit=True,
            # )
            # max_memory 同上
            # if tokenizer.chat_template is None:
            #     # Qwen-Chat 模板（与官方 repo 一致）
            #     tokenizer.chat_template = (
            #         "{% for message in messages %}"
            #         "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
            #         "{% endfor %}"
            #         "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
            #     )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                # torch_dtype=torch.float16,  # 建议使用 float16 减少显存占用
                local_files_only=True,
                device_map="auto",               # 会识别 X 张 32 GB
                # max_memory=max_mem,
                quantization_config=bnb_config,
                # load_in_4bit=True
            ).eval()
            print("✅ 模型加载成功。")
        except Exception as e:
            print(f"❌ 模型加载失败，请检查模型名称和环境配置（如 PyTorch、GPU 驱动）: {e}")
            raise


def local_generate_response(
        engine: str,
        temperature: float,
        max_tokens: int,
        system_role: str,
        user_input: str,
        image_path: str = None  # Qwen-7B-Chat 不支持图像，但为兼容接口保留
) -> str:
    """
    使用本地 Qwen 模型生成响应。

    Args:
        engine (str): 模型名称（本地加载时可能不使用）。
        temperature (float): 控制生成随机性。
        max_tokens (int): 最大生成 Token 数。
        system_role (str): 系统角色设定。
        user_input (str): 用户输入的提示词。
        image_path (str): 图像路径（仅适用于 Qwen-VL 等多模态模型）。

    Returns:
        str: 模型的 JSON 格式响应文本。
    """
    _load_qwen_model()

    # 将输入打包成 Qwen-Chat 的对话历史格式
    messages = [
        {"role": "user", "content": system_role + '\n' + user_input}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    # 2. 模型推理
    input_ids = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **input_ids,
            max_new_tokens=max_tokens,
            # do_sample=False,
            # temperature=temperature,
            # pad_token_id=tokenizer.eos_token_id,
            # 由于 Qwen 模型可能需要更多参数来稳定生成 JSON
            # top_p=0.8,
            # repetition_penalty=1.05,
        )
    # 3. 解析和清理输出
    response_text = tokenizer.decode(outputs[0][input_ids.input_ids.shape[-1]:], skip_special_tokens=True)
    # print(response_text)
    # 重要的后处理：清理输出，确保它是有效的 JSON
    # Qwen 输出可能包含额外的文本，需要提取第一个 JSON 块
    try:
        json_str = None
        m = re.search(r'```json\s*(\{.*?\})\s*```', response_text, flags=re.S)
        if m:
            json_str = m.group(1)
        else:
            # 2. 兜底：直接找最外层 {}
            json_str = re.sub(r'<think>.*?</think>', '', response_text, flags=re.S).strip()
        return json.loads(json_str)

    except Exception as e:
        print(f"⚠️ 无法解析模型输出为 JSON，返回原始输出。错误: {e}")

    return response_text


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

# 模型路径
MODEL_NAME_7B = "/225040511/project/hf_cache/DeepSeek-R1-7B-chat" 

# 显存配置（7B 模型 8-bit 量化仅占用约 8-9GB 显存，32GB 环境非常轻松）
max_mem = {i: "30GiB" for i in range(torch.cuda.device_count())}
max_mem["cpu"] = "50GiB"

model_7b = None
tokenizer_7b = None

def _load_deepseek_chat_model():
    """初始化并缓存 DeepSeek-Chat 模型"""
    global model_7b, tokenizer_7b
    if model_7b is None or tokenizer_7b is None:
        print(f"⏳ 首次加载 DeepSeek-Chat 模型: {MODEL_NAME_7B}...")
        try:
            tokenizer_7b = AutoTokenizer.from_pretrained(MODEL_NAME_7B, trust_remote_code=True)
            tokenizer_7b.pad_token = tokenizer_7b.eos_token 

            # 8-bit 量化足以保持精度并大幅节省显存
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )

            model_7b = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME_7B,
                local_files_only=True,
                device_map="auto", # 自动分配到 GPU
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).eval()
            print("✅ DeepSeek-Chat 加载成功。")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise

def local_generate_deepseek_chat_response(
        engine: str,
        temperature: float,
        max_tokens: int,
        system_role: str,
        user_input: str,
        image_path: str = None 
) -> dict:
    """
    使用本地 DeepSeek-Chat 模型生成标准 JSON 响应。
    """
    _load_deepseek_chat_model()

    # 1. 组装对话格式
    # DeepSeek-Chat 对 system_role 比较敏感，建议清晰隔离
    messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": user_input}
    ]

    # 使用模型自带的 chat_template
    input_text = tokenizer_7b.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    input_ids = tokenizer_7b([input_text], return_tensors="pt").to(model_7b.device)

    # 2. 模型生成
    with torch.no_grad():
        outputs = model_7b.generate(
            **input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature if temperature > 0 else 0.1,
            top_p=0.9,
            repetition_penalty=1.1, # 7B 模型容易出现重复，稍微加一点惩罚
            eos_token_id=tokenizer_7b.eos_token_id,
        )

    # 3. 解析输出文本
    # 只提取模型新生成的 token
    generated_text = tokenizer_7b.decode(outputs[0][input_ids.input_ids.shape[-1]:], skip_special_tokens=True)

    # 4. 健壮的 JSON 提取逻辑
    try:
        # 1. 寻找 JSON 块
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', generated_text, flags=re.S)
        if json_match:
            return json.loads(json_match.group(1))
        
        # 2. 寻找最外层的 {}
        json_match = re.search(r'(\{.*\})', generated_text, flags=re.S)
        if json_match:
            return json.loads(json_match.group(1))
            
        # 3. 如果实在找不到 {}，说明模型输出了纯文本，抛出异常进入下面的 except
        raise ValueError("No JSON found")

    except Exception as e:
        # 关键点：如果解析失败，直接返回包含原始文本的字典，外层函数会处理它
        return {
            "summary": generated_text, # 假设模型直接输出了总结内容
            "raw_output": generated_text,
            "status": "error_parsed_as_raw"
        }


if __name__ == '__main__':
    image_path='/225040511/project/Dataset/Dermnet/train/Eczema Photos/factitial-dermatitis-4.jpg'
    skin_disease = parse_skin_disease_path(image_path)
    print(skin_disease)
    prompt = f"""
    You are a frontline medical professional specializing in performing initial patient assessments based on dermatological images. You have been informed that the disease category for this image is: **{skin_disease}**.

    Your primary role is to organize preliminary medical observations from images and provide insights to support further diagnosis and agent collaboration.

    ---

    ### ⚠️ CRITICAL INSTRUCTIONS FOR KEY FINDINGS:

    When writing the "KeyFindings" section, you MUST:
    1. **Start with anatomical location and context** (e.g., "sun-exposed malar region", "chronic photodamaged skin").
    2. **Describe morphology in clinical terms**: use words like *ill-defined*, *infiltrative*, *indurated*, *violaceous*, *telangiectatic*, *ecchymotic*.
    3. **Highlight "red flags" or "warning signs"** — e.g., “resembling a persistent bruise”, “infiltrative borders”, “induration suggests deeper involvement”.
    4. **Mention differential diagnostic tension** — e.g., “While visually resembles X, the Y feature raises concern for Z.”
    5. **Explicitly rate severity** at the end: “Severity: [Normal / Mild / Moderate / Severe] — due to [brief reason].”
    6. **Avoid bullet points or lists** — write in fluent, narrative clinical prose.

    ---

    ### ✅ FORMAT YOUR RESPONSE STRICTLY AS JSON:

    {{
    "PrimaryDiagnosis": "[Re-evaluate based on expert agent's input and your own analysis]",
    "ConfidenceLevel": "[High/Medium/Low]",
    "DifferentialDiagnoses": ["List of top 5 differentials"],
    "KeyFindings": "[A single cohesive paragraph following the instructions above — DO NOT USE BULLET POINTS OR LISTS]",
    "\"ProbabilityDistribution\": [\"sorted list: {{ disease: '...', probability: 0.xx }}\"],Probability distribution over the full {DERMNET_DISEASE_NAME}  list, sorted descending." \
    "KnowledgeAndResearch": "[Fluent summary of relevant knowledge, including references if possible]"
    }}

    ---

    ### 💡 EXAMPLE OF EXPECTED KEY FINDINGS (DO NOT COPY VERBATIM):

    "The patient presents with a large, ill-defined, and infiltrative plaque on the sun-exposed skin of the head/neck (malar region). Key morphological features include a striking violaceous (purplish) and erythematous hue resembling a persistent bruise (ecchymosis), induration (hardening) of the tissue, and prominent telangiectasias. While the visual classifier leans towards severe Rosacea due to the redness and vascularity, the 'bruise-like' quality, infiltrative borders, and induration are critical 'red flags' for malignancy. The lesion is located on skin showing signs of chronic photodamage (dermatoheliosis). The severity is rated as Severe due to the high suspicion of an aggressive underlying pathology."

    ---

    Now analyze the provided image and generate your response in the exact JSON format above.
    """
    response = local_generate_response_vl(
                temperature=0.2,
                max_tokens=4096,
                prompt=prompt,
                image_path='/225040511/project/Dataset/Dermnet/test/Eczema Photos/factitial-dermatitis-7.jpg'
            )
    print(response)