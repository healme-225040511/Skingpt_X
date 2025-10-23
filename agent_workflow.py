import os
import json
import sys
import time
import hashlib
import asyncio
from html import unescape
from pathlib import Path

from tqdm import tqdm

# 假设以下模块已定义并可用
from rag_agent import RAGAgent
from web_search_agent import WebSearchAgent
from skingpt_openai_agent import SkinGPTOpenAIAgent
from reasoning_agent import ReasoningAgent
from case_review_agent import CaseReviewAgent
from treatment_recommend_agent import TreatmentRecommendAgent
from prompt_template import get_domain_expert_prompt

WEB_SEARCH_AGENT_OUTPUT_PATH = "/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/WebSearch_output.json"
RAG_AGENT_OUTPUT_PATH = "/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/RAG_output.json"
SKINGPT_AGENT_OUTPUT_PATH = "/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/SkinGPT_output.json"
REASONING_AGENT_OUTPUT_PATH = "/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/Reasoning_output.json"


async def analyze_image(all_agents, image_path, web_search_output, rag_output, skin_gpt_output, image_name,
                        output_folder, folder_name):
    tasks = []
    prob_vec = []
    if all_agents.get('SkinGPT'):
        prob_vec = all_agents['SkinGPT'].get_prob_vec(image_path=image_path)
    for domain, agent in all_agents.items():
        query = get_domain_expert_prompt(domain, prob_vec=prob_vec)
        tasks.append(async_analyze(agent, query, image_path))
    results = await asyncio.gather(*tasks)
    for domain, result in zip(all_agents.keys(), results):
        if domain == "WebSearch":
            web_search_output[image_name] = result
            save_output(web_search_output, "WebSearch", output_folder)
        elif domain == "RAG":
            rag_output[image_name] = result
            save_output(rag_output, "RAG", output_folder)
        elif domain == "SkinGPT":
            skin_gpt_output[image_name] = result
            save_output(skin_gpt_output, "SkinGPT", output_folder)


# 异步分析函数
async def async_analyze(agent, query, image_path):
    return await agent.analyze(query, image_path)


def save_output(output_data, agent_name, output_folder):
    """
    将输出数据保存为JSON文件
    参数:
        output_data: 要保存的数据字典
        agent_name: 代理名称，用于生成文件名
        output_folder: 输出文件夹路径
        folder_name: 文件夹名称(当前函数中未使用)
    """
    # 构建输出文件的完整路径
    output_file_path = os.path.join(output_folder, f"{agent_name}_output.json")
    # 如果代理名称是WebSearch、RAG或SkinGPT，对输出数据进行HTML转义处理
    if agent_name in ["WebSearch", "RAG", "SkinGPT"]:
        for key, value in output_data.items():
            output_data[key] = unescape(value)
    # 初始化现有数据字典
    existing_data = {}
    # 检查文件是否存在
    if os.path.isfile(output_file_path):
        # 如果文件存在，读取现有数据
        with open(output_file_path, "r", encoding="utf-8", errors='replace') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                # 如果JSON解析失败，清空数据并打印错误信息
                existing_data = {}
                print('open json file error')
                sys.exit(1)
    # 更新现有数据 with 新数据
    existing_data.update(output_data)
    with open(output_file_path, "w", encoding="utf-8", errors="replace") as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)


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


def getReasoningReport(json_path):
    json_path = Path(json_path)  # 统一成 Path，兼容各种系统
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)  # data 就是顶层 dict

    if not isinstance(data, dict):
        raise ValueError("JSON 顶层必须是对象（dict）结构！")

    # 直接返回，key 已经是文件名
    return data


def WorkFlow(
        all_agents,  # 所有智能体的集合
        treatment_recommend_agent=None,  # 治疗推荐智能体
        reasoning_agent=None,  # 推理智能体
        case_review_agent=None,  # 案例审查智能体
        output_folder="output/",  # 输出文件夹路径，默认为"output/"
        image_path="",  # 图片路径，默认为空字符串
        folder_name=None
):
    # 初始化 Agent 和输出字典
    image_name = image_path.split('/')[-2] + '/' + image_path.split('/')[-1]
    web_search_output = {}  # 存储网络搜索结果
    rag_output = {}  # 存储RAG（检索增强生成）结果
    skin_gpt_output = {}  # 存储SkinGPT模型输出
    reasoning_output = {}  # 存储推理结果
    case_review_output = {}  # 存储案例审查结果
    treatment_recommend_output = {}  # 存储治疗推荐结果
    # 异步执行图像分析任务
    asyncio.run(analyze_image(all_agents, image_path, web_search_output, rag_output, skin_gpt_output, image_name,
                              output_folder, folder_name))
    # analyze_image(all_agents, image_path, web_search_output, rag_output, skin_gpt_output, image_name,
    #                           output_folder, folder_name)
    # Generate report
    if reasoning_agent is not None:
        print("Generating report")
        report = reasoning_agent.generate_report({
            "WebSearch": web_search_output.get(image_name, "") if any(web_search_output.values()) else
            getAgentOutputs(WEB_SEARCH_AGENT_OUTPUT_PATH)[image_name],
            "RAG": rag_output.get(image_name, "") if any(rag_output.values()) else
            getAgentOutputs(RAG_AGENT_OUTPUT_PATH)[image_name],
            "SkinGPT": skin_gpt_output.get(image_name, "") if any(skin_gpt_output.values()) else
            getAgentOutputs(SKINGPT_AGENT_OUTPUT_PATH)[image_name]
        })
        reasoning_output[image_name] = report
        save_output(reasoning_output, "Reasoning", output_folder)
    if case_review_agent is not None:
        # Case review
        print("Case reviewing")
        report = report if reasoning_agent is not None else getReasoningReport(REASONING_AGENT_OUTPUT_PATH)[image_name]
        review_report = case_review_agent.review_case(report)
        # print(review_report)
        case_review_output[image_name] = review_report
        # Update the case in the database
        ## !!!!!
        ## May need to be optimized, because only good cases could be added
        case_review_agent._add_case_to_knowledge_graph(review_report)
        save_output(case_review_output, "CaseReview", output_folder)
    else:
        return
    if treatment_recommend_agent is not None:
        # Treatment recommendation
        print("Treatment recommending")
        try:
            treatment_recommend_result = treatment_recommend_agent.analyze(review_report)
            treatment_recommend = json.loads(treatment_recommend_result)
            treatment_recommend_output[image_name] = treatment_recommend
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Invalid JSON received: {treatment_recommend_result}")
            treatment_recommend_output[folder_name + '/' + image_name] = {"error": str(e), "raw_output": treatment_recommend_result}

        save_output(treatment_recommend_output, "TreatmentRecommend", output_folder)
    else:
        return
