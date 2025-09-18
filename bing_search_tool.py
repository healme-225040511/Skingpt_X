"""
适配 agno 的 Bing Search v7 工具
"""
import os
import requests
from typing import Optional, List, Dict
from agno.tools import Toolkit
from pydantic import Field, ConfigDict

class BingSearchTool(Toolkit):
    """
    调用 Azure Bing Search v7，返回 JSON 字符串（与 DuckDuckGoTools 格式兼容）。
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: str = Field(..., description="Azure Bing Search v7 API Key")
    endpoint: str = Field(
        "https://api.bing.microsoft.com/v7.0/search",
        description="Bing Search API 端点"
    )
    count: int = Field(10, description="单次返回最大结果数")

    def __init__(self, api_key: Optional[str] = None, count: int = 10):
        super().__init__()
        self.api_key = api_key or os.getenv("BING_SEARCH_API_KEY")
        if not self.api_key:
            raise ValueError("请通过参数或环境变量 BING_SEARCH_API_KEY 提供 Bing API Key")
        self.count = count
        self.register(self.bing_search)

    def bing_search(self, query: str, max_results: Optional[int] = None) -> str:
        """
        与 DuckDuckGoTools 保持一致签名，返回 JSON 字符串。
        """
        count = min(max_results or self.count, 50)
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {"q": query, "count": count, "textDecorations": False, "safeSearch": "Moderate"}
        resp = requests.get(self.endpoint, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("webPages", {}).get("value", [])
        results = [
            {
                "title": item["name"],
                "href": item["url"],
                "body": item.get("snippet", ""),
            }
            for item in items
        ]
        import json
        return json.dumps(results, ensure_ascii=False, indent=2)