import csv
import pathlib
import tempfile

import openai
import base64
import asyncio
from typing import Dict, Optional
from google.genai import types

from google import genai
from openai import OpenAI

from Constants import ISIC_PRECSV_PATH
from prompt_template import get_domain_expert_prompt
from utils import encode_image_to_base64
safety_settings = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
]

class SkingptAgent:
    def __init__(self, model: str = "gpt-4o", domain: str = "SkinGPT", api_key: str = "", pre_csv_path: str = "./Dermnet_predprob.csv"):
        self.domain = domain
        self.model = model                       # 默认用 gpt-4o，也可传 gpt-4o-mini
        self.api_key = api_key
        self.pre_csv_path = pre_csv_path

    # ---------- 工具：图片 → base64 ----------
    @staticmethod
    def _encode_image(image_path: str) -> str:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    # ---------- 工具：图片 → 临时文件 ----------
    @staticmethod
    def _image_to_temp_file(image_path: str) -> str:
        """把任意本地图片转成一个临时 jpeg 文件，返回路径供 Gemini 使用"""
        from PIL import Image
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        Image.open(image_path).convert("RGB").save(tmp.name, "JPEG")
        return tmp.name

    # ---------- 主要接口 ----------
    async def analyze(self, query: str, image_path: str) -> Optional[str]:
        """
        基于视觉模型给出诊断/建议
        :param query: 临床问题字符串
        :param image_path: 本地图片路径
        :return: 模型返回文本
        """
        tmp_path = self._image_to_temp_file(image_path)
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        img_part = types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
        try:
            if self.api_key.startswith('sk-'):
                client = OpenAI(api_key=self.api_key, base_url="https://hiapi.online/v1")
                base64_image = encode_image_to_base64(image_path)

                image_mime_type = "image/jpeg"
                if image_path.lower().endswith(".png"):
                    image_mime_type = "image/png"
                elif image_path.lower().endswith(".gif"):
                    image_mime_type = "image/gif"
                elif image_path.lower().endswith(".webp"):
                    image_mime_type = "image/webp"
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{query}"
                            },
                            {
                                "type": "image_url",
                                "image_url": dict(url=f"data:{image_mime_type};base64,{base64_image}")
                            }
                        ]
                    }
                ]
                resp = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=self.model,
                    messages=messages,
                )
                return resp.choices[0].message.content
            client = genai.Client(api_key=self.api_key)
            # 官方 SDK 是同步的，扔到线程池
            resp = await asyncio.to_thread(
                client.models.generate_content,
                model=self.model,
                contents=[img_part, query],
                config=types.GenerateContentConfig(temperature=0.0, safety_settings=safety_settings)
            )
            return resp.text
        except Exception as e:
            print(f"[SkinGPT Vision] Error: {e}")
            return None
        finally:
            pathlib.Path(tmp_path).unlink(missing_ok=True)  # 清理临时文件
    def get_prob_vec(self, image_path: str):
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
        with open(self.pre_csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                feature_dim = len(row) - 3
                if row["filename"] == key:
                    # 3. 提取 prob_cls0 ... prob_cls22
                    if self.pre_csv_path == ISIC_PRECSV_PATH:
                        vec = [float(row[f"prob_cls"])]
                        return vec
                    else:
                        vec = [float(row[f"prob_cls{i}"]) for i in range(feature_dim)]
                        return vec
        return None

# ------------------- 快速自测 -------------------
if __name__ == "__main__":
    api_key = "sk-iCv69YeaJn8TXm9tk6ZUUAqftw51aB2yddvmstNNl7QjkIKB"
    agent = SkingptAgent(model="gemini-2.5-pro", api_key=api_key, pre_csv_path='/Volumes/T7/SkinGPT-X-EvaluationResults/PanDerm_Base_LP_result/ISIC/PanDerm_Base_LP_predprob.csv')
    image_file_path = "/Volumes/T7/SkinGPT-X-Dataset/Dermnet/test/Acne and Rosacea Photos/acne-closed-comedo-36.jpg"  # Replace with your actual image file path
    prob_vec = agent.get_prob_vec(image_path=image_file_path)
    query = get_domain_expert_prompt("SkinGPT", prob_vec)
    async def main():
        analysis_result = await agent.analyze(query, image_file_path)
        print(analysis_result)

    asyncio.run(main())