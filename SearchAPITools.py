"""
SearchApi 纯 requests 版（零依赖）
"""
import os
import json
import requests
from typing import Optional
from agno.tools import Toolkit
from pydantic import Field

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

    def searchapi_search(self, query: str, max_results: int = 10) -> str:
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
# 临时测试
if __name__ == "__main__":
    tool = SearchApiTool(api_key="JZ2yaBjJXaPFK8jZ4DkNadHP", engine="google")
    print(tool.searchapi_search("milia diagnosis", max_results=3))