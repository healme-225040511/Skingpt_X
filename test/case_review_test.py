import json
from pathlib import Path

WEB_SEARCH_AGENT_OUTPUT_PATH = "/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/WebSearch_output.json"
RAG_AGENT_OUTPUT_PATH = "/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/RAG_output.json"
SKINGPT_AGENT_OUTPUT_PATH = "/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/SkinGPT_output.json"

def getAgentOutputs(json_path):
    """
        把形如 {"file1.jpg": "诊断文本...", "file2.png": "诊断文本...", ...}
        的 JSON 文件读进来，按文件名建立索引并返回 dict。

        参数
        ----
        json_path : str | pathlib.Path
            原始 JSON 文件路径。

        返回
        ----
        dict[str, Any]
            key -> 文件名（含后缀）
            value -> 该文件对应的整个 value（字符串或嵌套结构均可）。
        """
    json_path = Path(json_path)  # 统一成 Path，兼容各种系统
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)  # data 就是顶层 dict

    if not isinstance(data, dict):
        raise ValueError("JSON 顶层必须是对象（dict）结构！")

    # 直接返回，key 已经是文件名
    return data

if __name__ == "__main__":
    data = getAgentOutputs(WEB_SEARCH_AGENT_OUTPUT_PATH)
    print(data['perioral-dermatitis-109.jpg'])