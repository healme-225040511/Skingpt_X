import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from typing import Dict, List
import json
import os
from PIL import Image  # 仅用于多模态模型 (如 Qwen-VL)

# 全局变量：用于缓存模型和分词器，避免每次调用都重新加载
# 请确保您的环境中安装了 'Qwen/Qwen-7B-Chat' 模型
MODEL_NAME = "Qwen/Qwen3-32B"
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
                llm_int8_enable_fp32_cpu_offload=True
            )
            if tokenizer.chat_template is None:
                # Qwen-Chat 模板（与官方 repo 一致）
                tokenizer.chat_template = (
                    "{% for message in messages %}"
                    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
                )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,  # 建议使用 float16 减少显存占用
                trust_remote_code=True,
                device_map="auto",               # 会识别 2 张 32 GB
                quantization_config=bnb_config
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

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 2. 模型推理
    input_ids = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,

            # 由于 Qwen 模型可能需要更多参数来稳定生成 JSON
            # top_p=0.8,
            # repetition_penalty=1.05,
        )

    # 3. 解析和清理输出
    response_text = tokenizer.decode(outputs[0][input_ids.input_ids.shape[-1]:], skip_special_tokens=True)

    # 重要的后处理：清理输出，确保它是有效的 JSON
    # Qwen 输出可能包含额外的文本，需要提取第一个 JSON 块
    try:
        # 查找 JSON 块的开始和结束位置
        start_index = response_text.find('{')
        end_index = response_text.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_str = response_text[start_index: end_index + 1]
            # 尝试解析 JSON 确保其有效性
            json.loads(json_str)
            return json_str
    except Exception as e:
        print(f"⚠️ 无法解析模型输出为 JSON，返回原始输出。错误: {e}")

    return response_text


# --- 示例多模态函数 (如果使用 Qwen-VL) ---
def local_generate_response_vl(
        # ... 参数与上面相同 ...
        image_path: str = None
) -> str:
    # 这个函数需要加载 Qwen-VL 模型，并且在输入中包含图像 token
    # 示例:
    # query = tokenizer.from_list_format([
    #     {'image': image_path},
    #     {'text': prompt},
    # ])
    # inputs = tokenizer(query, return_tensors='pt')
    # ...
    pass
# -----------------------------------------------