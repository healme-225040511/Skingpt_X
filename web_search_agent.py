# web_search_agent.py
import asyncio, os, time, aiohttp, io, base64
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.google.gemini import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
from prompt_template import get_domain_expert_prompt
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import RatelimitException

class WebSearchAgent:
    def __init__(self, model: str, api_key: str, domain: str = "WebSearch"):
        self.model = model
        self.api_key = api_key
        self.domain = domain
        self.agent = Agent(
            model=Gemini(id=self.model, api_key=self.api_key, temperature=0.2),
            tools=[DuckDuckGoTools()],
            debug_mode=False
        )

    # ---------- 1. 搜索相似图片 ----------
    def search_similar_images(self, keyword: str, max_results: int = 5):
        """返回相似图像的标题+链接列表"""
        results = []
        try:
            with DDGS() as ddg:
                for item in ddg.images(keywords=keyword, max_results=max_results):
                    results.append({
                        "title": item.get("title", ""),
                        "image": item.get("image", ""),
                        "thumbnail": item.get("thumbnail", "")
                    })
        except Exception as e:
            print("[search_similar_images] error:", e)
        return results

    # ---------- 2. 下载并转 base64（避免临时文件） ----------
    async def url_to_base64(self, url: str) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        img = PILImage.open(io.BytesIO(data))
                        # 统一尺寸
                        img.thumbnail((500, 500))
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        return base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            print(f"[url_to_base64] skip {url}: {e}")
            return ""

    # ---------- 3. 主分析逻辑 ----------
    async def analyze(self, query: str, image_path: str):
        if not image_path or not os.path.exists(image_path):
            return "Please provide a valid image file path."

        try:
            # 上传图像
            user_img = PILImage.open(image_path)
            user_img.thumbnail((500, 500))
            temp_path = "temp_user.png"
            user_img.save(temp_path)
            user_agno = AgnoImage(filepath=temp_path)

            # 用文件名（无后缀）当关键词搜图
            keyword = os.path.splitext(os.path.basename(image_path))[0]
            sims = self.search_similar_images(keyword, max_results=5)

            # 并行下载相似图
            tasks = [self.url_to_base64(t["thumbnail"]) for t in sims]
            sim_b64s = await asyncio.gather(*tasks)
            sim_b64s = [b for b in sim_b64s if b]  # 去掉失败
            sim_agno = [AgnoImage(base64=b) for b in sim_b64s]

            # 构造提示
            ref_desc = "\n".join([f"{i+1}. {t['title']}" for i, t in enumerate(sims)])
            full_prompt = f"{query}\n\n参考相似图像（按相似度排序）：\n{ref_desc}\n\n请结合上述参考图像，对上传图像做出诊断预测。"

            # 发请求
            all_images = [user_agno] + sim_agno
            response = await asyncio.to_thread(self.agent.run, full_prompt, images=all_images)

            os.remove(temp_path)
            return response.content

        except RatelimitException:
            return "Rate limit exceeded. Please try again later."
        except Exception as e:
            return f"Analysis error: {e}"

# ------------------ 本地测试 ------------------
if __name__ == "__main__":
    model = "gemini-2.5-pro"
    api_key = "AIzaSyC-9og_9OsxvKZ0rBXeMGboXBrMOpG5-do"
    agent = WebSearchAgent(model=model, api_key=api_key)
    image_file_path = "/Volumes/T7/SkinGPT-X-Dataset/Dermnet/test/Acne and Rosacea Photos/acne-pustular-21.jpg"

    async def main():
        result = await agent.analyze(get_domain_expert_prompt("WebSearch"), image_file_path)
        print(result)

    asyncio.run(main())