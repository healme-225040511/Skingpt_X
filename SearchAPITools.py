"""
SearchApi 纯 requests 版（零依赖）
"""
import os
import json
import requests
from typing import Optional
from agno.tools import Toolkit
from pydantic import Field
from typing import Optional, List, Dict
BOCHA_API_KEY = 'sk-74829ed96e1c4d9793507d546527f5de'
SEARCH_API_KEY = 'JZ2yaBjJXaPFK8jZ4DkNadHP'
class SearchApiTool(Toolkit):
    api_key: str = Field(..., description="SearchApi 密钥")
    engine: str = Field("google", description="google | bing | yandex")

    def __init__(self, api_key: Optional[str] = None, engine: str = "google"):
        super().__init__()
        self.api_key = api_key or os.getenv("SEARCHAPI_API_KEY")
        if not self.api_key:
            raise ValueError("请提供 api_key 或环境变量 SEARCHAPI_API_KEY")
        self.engine = engine
        self.register(self.searchapi_search)

    def searchapi_search(self, query: str, max_results: int = 5) -> str:
        url = "https://www.searchapi.io/api/v1/search"
        params = {
            "api_key": self.api_key,
            "engine": self.engine,
            "q": query,
            "num": min(max_results, 20),
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = [
            {"title": r.get("title"), "href": r.get("link"), "body": r.get("snippet", "")}
            for r in data.get("organic_results", [])
        ]
        return json.dumps(results, ensure_ascii=False, indent=2)

class BochaSearchTool(Toolkit):
    api_key: str = Field(BOCHA_API_KEY, description="Bocha API 密钥")
    endpoint: str = Field("https://api.bochaai.com/v1/web-search", description="Bocha 接口地址")

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or os.getenv("BOCHA_API_KEY")
        self.endpoint = "https://api.bochaai.com/v1/web-search"
        if not self.api_key:
            raise ValueError("请提供 api_key 或环境变量 BOCHA_API_KEY")
        self.register(self.bocha_search)

    def bocha_search(self, query: str, max_results: int = 10) -> str:
        """
        调用 Bocha 搜索，返回统一格式的 JSON 字符串。
        """
        payload = json.dumps({
            "query": query,
            "summary": True,
            "count": min(max_results, 10),
        })
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.request('POST', self.endpoint, headers=headers, data=payload, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        # 把 bocha 返回字段映射成通用格式
        pages = raw.get("data", {}).get("webPages", {}).get("value", [])
        results = [
            {
                "title": p.get("name", ""),
                "href": p.get("url", ""),
                "body": p.get("snippet", ""),
            }
            for p in pages[:max_results]
        ]
        return json.dumps(results, ensure_ascii=False, indent=2)
# 临时测试
if __name__ == "__main__":
    tool = BochaSearchTool(api_key="sk-74829ed96e1c4d9793507d546527f5de")
    print(tool.bocha_search("why the sky is blue？", max_results=5))