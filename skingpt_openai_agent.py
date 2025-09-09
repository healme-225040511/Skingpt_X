import openai
import base64
import asyncio
from typing import Dict, Optional
from prompt_template import get_domain_expert_prompt

class SkinGPTOpenAIAgent:
    def __init__(self, model: str = "gpt-4o", domain: str = "SkinGPT", api_key: str = ""):
        self.domain = domain
        self.model = model                       # 默认用 gpt-4o，也可传 gpt-4o-mini
        openai.api_key = api_key

    # ---------- 工具：图片 → base64 ----------
    @staticmethod
    def _encode_image(image_path: str) -> str:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    # ---------- 主要接口 ----------
    async def analyze(self, query: str, image_path: str) -> Optional[str]:
        """
        基于视觉模型给出诊断/建议
        :param query: 临床问题字符串
        :param image_path: 本地图片路径
        :return: 模型返回文本
        """
        base64_image = self._encode_image(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                ],
            }
        ]

        try:
            # Use asyncio.to_thread to run the synchronous openai.chat.completions.create in a separate thread
            resp = await asyncio.to_thread(
                openai.chat.completions.create,
                model=self.model,
                messages=messages,
                temperature=0.2,  # 医疗场景低温度
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[SkinGPT Vision] Error: {e}")
            return None

# ------------------- 快速自测 -------------------
if __name__ == "__main__":
    agent = SkinGPTOpenAIAgent(model="gpt-4o", api_key="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")
    query = get_domain_expert_prompt("SkinGPT")
    image_file_path = "./data/images/1.png"

    async def main():
        analysis_result = await agent.analyze(query, image_file_path)
        print(analysis_result)

    asyncio.run(main())