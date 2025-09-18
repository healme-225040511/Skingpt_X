import pathlib
import tempfile

import openai
import base64
import asyncio
from typing import Dict, Optional
from google.genai import types

from google import genai

from prompt_template import get_domain_expert_prompt

class SkinGPTOpenAIAgent:
    def __init__(self, model: str = "gpt-4o", domain: str = "SkinGPT", api_key: str = ""):
        self.domain = domain
        self.model = model                       # 默认用 gpt-4o，也可传 gpt-4o-mini
        self.api_key = api_key

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
            client = genai.Client(api_key=self.api_key)
            # 官方 SDK 是同步的，扔到线程池
            resp = await asyncio.to_thread(
                client.models.generate_content,
                model=self.model,
                contents=[img_part, query],
                config=types.GenerateContentConfig(temperature=0.0)
            )
            return resp.text
        except Exception as e:
            print(f"[SkinGPT Vision] Error: {e}")
            return None
        finally:
            pathlib.Path(tmp_path).unlink(missing_ok=True)  # 清理临时文件

# ------------------- 快速自测 -------------------
if __name__ == "__main__":
    agent = SkinGPTOpenAIAgent(model="gemini-2.5-pro", api_key="AIzaSyC-9og_9OsxvKZ0rBXeMGboXBrMOpG5-do")
    query = get_domain_expert_prompt("SkinGPT")
    image_file_path = "./SkinGPT-X-Dataset/Dermnet/test/Seborrheic Keratoses and other Benign Tumors/seborrheic-keratosis-irritated-28.jpg"  # Replace with your actual image file path

    async def main():
        analysis_result = await agent.analyze(query, image_file_path)
        print(analysis_result)

    asyncio.run(main())