import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Qwen2VLForConditionalGeneration, AutoProcessor,AutoModelForImageTextToText
from Constants import DERMNET_DISEASE_NAME
from typing import Dict, List
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("SKINGPT_PROJECT_ROOT", Path(__file__).resolve().parent))
DEFAULT_HF_HOME = PROJECT_ROOT.parent / "hf_cache"
HF_HOME_PATH = Path(
    os.environ.get("SKINGPT_HF_HOME", os.environ.get("HF_HOME", str(DEFAULT_HF_HOME)))
).expanduser().resolve()
BGE_MODEL_PATH = HF_HOME_PATH / "bge-small-en-v1.5" / "BAAI" / "bge-small-en-v1___5"
QWEN3_VL_MODEL_PATH = HF_HOME_PATH / "Qwen3-VL-30B"
QWEN_VL_MODEL_PATH = HF_HOME_PATH / "Qwen-VL-8B-Instruct"
MEDGEMMA_MODEL_PATH = HF_HOME_PATH / "medgemma-27b-it"

# # 必须在任何 transformers 导入前执行
os.environ["HF_HOME"] = str(HF_HOME_PATH)
from PIL import Image  # 仅用于多模态模型 (如 Qwen-VL)

# 全局变量：用于缓存模型和分词器，避免每次调用都重新加载
# 请确保您的环境中安装了 'Qwen/Qwen-7B-Chat' 模型
# MODEL_NAME = str(HF_HOME_PATH / "models--Qwen--Qwen3-32B" / "snapshots" / "9216db5781bf21249d130ec9da846c4624c16137")
MODEL_NAME = str(QWEN3_VL_MODEL_PATH)
# MODEL_NAME = str(HF_HOME_PATH / "DeepSeek-R1-67B-chat")
max_mem = {i: "30GiB" for i in range(torch.cuda.device_count())}
max_mem["cpu"] = "50GiB"          # 溢出部分放内存，避免直接炸显存

model = None
tokenizer = None
def _load_qwen_model():
    """初始化并缓存 Qwen-VL 模型和 Tokenizer轻量版"""
    global vl_model, vl_processor
    if vl_model is None or vl_processor is None:
        print(f"⏳ 首次加载轻量多模态模型: {os.path.basename(MODEL_NAME)}...")
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


def local_generate_response(
        temperature: float,
        max_tokens: int,
        system_role: str,
        user_input: str,
        image_path: str = None  # Qwen-7B-Chat 不支持图像，但为兼容接口保留
) -> str:
    """
    使用本地轻量 Qwen-VL 模型（2B/7B）生成响应，支持图像输入。
    
    注意：当 image_path 为空时，退化为纯文本推理（但不如纯文本模型高效）。
    """
    _load_qwen_model()
    image = Image.open(image_path).convert("RGB")

    # 构造 messages：Qwen2-VL 支持混合 content
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_input},
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


# import torch
# import re
# import json
# import os
# from transformers import (
#     AutoModelForCausalLM, 
#     AutoTokenizer, 
#     BitsAndBytesConfig,
#     AutoModelForImageTextToText
# )

# # 必须在任何 transformers 导入前执行环境路径设置
# os.environ["HF_HOME"] = str(HF_HOME_PATH)

# # ================= 配置区域 =================
# # 更换为 MedGemma-27B 的本地路径
# MODEL_NAME = str(HF_HOME_PATH / "medgemma-27b-it")

# # 全局变量缓存
# model = None
# tokenizer = None

# def _load_medgemma_model():
#     global model, tokenizer
#     if model is None or tokenizer is None:
#         print(f"⏳ 正在加载 MedGemma-27B...")
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,                    # 开启 4-bit 量化
#             bnb_4bit_quant_type="nf4",            # 使用 NF4 (Normal Float 4) 格式，适合训练和推理
#             bnb_4bit_use_double_quant=True,       # 二次量化，进一步节省约 0.4 bits/param
#             bnb_4bit_compute_dtype=torch.bfloat16 # 计算时使用的精度，MedGemma 建议使用 bf16
#         )

#         try:
#             model = AutoModelForImageTextToText.from_pretrained(
#                 MODEL_NAME,
#                 quantization_config=bnb_config,
#                 device_map="auto",
#                 trust_remote_code=True,
#             )
#             tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#         except Exception as e:
#             print(f"❌ 加载失败: {e}")
#             raise
# def local_generate_response(
#         temperature: float,
#         max_tokens: int,
#         system_role: str,
#         user_input: str,
#         image_path: str = None 
# ) -> str:
#     """
#     使用本地 MedGemma-27B 生成医疗诊断响应。
#     """
#     _load_medgemma_model()

#     # MedGemma-it (Instruction Tuned) 对 Prompt 格式敏感
#     # 注意：Gemma 手册建议将系统指令与用户输入合并，
#     # 因为 Gemma 原生模板对 'system' role 的支持因版本而异。
#     image = Image.open(image_path)

#     messages = [
#         {"role": "system", "content": system_role},
#         {"role": "user", 
#          "content": [
#              {"type": "image", "data": image},
#              {"type": "text", "text": user_input}
#              ]
#         }
#     ]


#     inputs = tokenizer.apply_chat_template(
#         messages,
#         add_generation_prompt=True,
#         tokenize=True,
#         return_dict=True,
#         return_tensors="pt",
#     ).to(model.device)

#     input_len = inputs["input_ids"].shape[-1]
    
#     with torch.inference_mode():
#         generation = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
#         generation = generation[0][input_len:]

#     response_text = tokenizer.decode(generation, skip_special_tokens=True)
#     print(response_text)

#     # 医疗模型有时会输出比较长的分析，这里精准提取 JSON 段落
#     try:
#         # 1. 尝试寻找 Markdown 代码块形式的 JSON
#         m = re.search(r'```json\s*(\{.*?\})\s*```', response_text, flags=re.S)
#         if m:
#             return json.loads(m.group(1))
        
#         # 2. 尝试寻找最外层花括号
#         # 使用更稳健的正向查找
#         start_idx = response_text.find('{')
#         end_idx = response_text.rfind('}')
#         if start_idx != -1 and end_idx != -1:
#             json_str = response_text[start_idx:end_idx+1]
#             return json.loads(json_str)

#         return response_text # 如果没解析出 JSON 则返回纯文本内容

#     except Exception as e:
#         print(f"⚠️ 解析 JSON 失败: {e}")
#         return response_text


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoProcessor
from transformers.image_utils import load_image  # 👈 新增：用于安全加载图像（支持路径 & URL）
from typing import Dict, List, Optional  # 👈 修改：补充 Optional
import json
import os

# 必须在任何 transformers 导入前执行
os.environ["HF_HOME"] = str(HF_HOME_PATH)
from PIL import Image  # 不再需要单独导入（transformers 内部已封装）

# =============== 【新增】轻量 VL 模型配置 ===============
# 👇 替换为你的本地 2B VL 模型路径（推荐！）
# 从 HF 下载命令：huggingface-cli download Qwen/Qwen2-VL-2B-Instruct --local-dir ./qwen2-vl-2b
VL_MODEL_NAME = str(QWEN_VL_MODEL_PATH)
# 若想用 7B：VL_MODEL_NAME = str(HF_HOME_PATH / "models--Qwen--Qwen2-VL-7B-Instruct" / "snapshots" / "latest")

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

MODEL_NAME_MEDGEMMA = str(MEDGEMMA_MODEL_PATH)

max_mem = {i: "30GiB" for i in range(torch.cuda.device_count())}
max_mem["cpu"] = "50GiB"

med_model = None
med_processor = None

def _load_medgemma_model():
    global med_model, med_processor
    if med_model is None or med_processor is None:
        print(f"⏳ 首次加载 MedGemma-27B-it: {MODEL_NAME_MEDGEMMA}...")
        try:
            med_processor = AutoProcessor.from_pretrained(
                MODEL_NAME_MEDGEMMA,
                trust_remote_code=True
            )
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            med_model = AutoModelForImageTextToText.from_pretrained(
                MODEL_NAME_MEDGEMMA,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
            ).eval()
            print("✅ MedGemma-27B-it 加载成功。")
        except Exception as e:
            print(f"❌ MedGemma 模型加载失败: {e}")
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
    """
    _load_medgemma_model()

    if image_path:
        image = Image.open(image_path).convert("RGB")
    else:
        image = None

    enhanced_prompt = (
        f"{system_role}\n\n"
        "You are reviewing and synthesizing several recent dermatology cases. Please:\n"
        "- Summarize commonalities and key differences across cases (anatomic site + morphological terminology);\n"
        "- Emphasize the most heuristic diagnostic clues and clinical 'red flags';\n"
        "- Highlight differential diagnostic pearls for common misinterpretations and suggest next-step investigations;\n"
        "- Actively integrate new observations into prototype representations to establish more robust global characteristics.\n\n"
        "User Input:\n"
        f"{user_input}\n\n"
        "Output ONLY in the following JSON format:\n"
        "{\"summary\": \"A single-paragraph, information-dense clinical narrative synthesizing cross-case insights.\"}\n"
    )

    if image is not None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": enhanced_prompt},
                ],
            }
        ]
        text = med_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = med_processor(text=[text], images=[image], return_tensors="pt").to(med_model.device)
    else:
        messages = [
            {"role": "user", "content": [{"type": "text", "text": enhanced_prompt}]}
        ]
        text = med_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = med_processor(text=[text], return_tensors="pt").to(med_model.device)

    with torch.no_grad():
        generated_ids = med_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=max(temperature, 0.1),
            top_p=0.9,
            repetition_penalty=1.1,
        )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    generated_text = med_processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    try:
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', generated_text, flags=re.S)
        if json_match:
            return json.loads(json_match.group(1))
        json_match = re.search(r'(\{.*\})', generated_text, flags=re.S)
        if json_match:
            return json.loads(json_match.group(1))
        raise ValueError("No JSON found")
    except Exception:
        return {
            "summary": generated_text,
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
