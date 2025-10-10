import os
import json
import time
import hashlib
import asyncio
from html import unescape
from tqdm import tqdm

# 假设以下模块已定义并可用
from rag_agent import RAGAgent
from web_search_agent import WebSearchAgent
from skingpt_openai_agent import SkinGPTOpenAIAgent
from reasoning_agent import ReasoningAgent
from case_review_agent import CaseReviewAgent
from treatment_recommend_agent import TreatmentRecommendAgent
from prompt_template import get_domain_expert_prompt


async def analyze_image(all_agents, image_path, web_search_output, rag_output, skin_gpt_output, image_name, output_folder):
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
    output_file_path = os.path.join(output_folder, f"{agent_name}_output.json")
    if agent_name in ["WebSearch", "RAG", "SkinGPT"]:
        for key, value in output_data.items():
            output_data[key] = unescape(value)
    existing_data = {}
    if os.path.isfile(output_file_path):
        with open(output_file_path, "r", encoding="utf-8", errors='replace') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
    existing_data.update(output_data)
    with open(output_file_path, "w", encoding="utf-8", errors="replace") as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)


def WorkFlow(
        all_agents,                    # 所有智能体的集合
        treatment_recommend_agent=None,     # 治疗推荐智能体
        reasoning_agent = None,               # 推理智能体
        case_review_agent = None,             # 案例审查智能体
        output_folder="output/",       # 输出文件夹路径，默认为"output/"
        image_path="",                 # 图片路径，默认为空字符串

):
    # 初始化 Agent 和输出字典
    image_name = image_path.split("/")[-1]    # 从图片路径中提取图片名称
    web_search_output = {}      # 存储网络搜索结果
    rag_output = {}            # 存储RAG（检索增强生成）结果
    skin_gpt_output = {}       # 存储SkinGPT模型输出
    reasoning_output = {}      # 存储推理结果
    case_review_output = {}    # 存储案例审查结果
    treatment_recommend_output = {}  # 存储治疗推荐结果
    # 异步执行图像分析任务
    asyncio.run(analyze_image(all_agents, image_path, web_search_output, rag_output, skin_gpt_output, image_name, output_folder))
    # Generate report
    if reasoning_agent is not None:
        print("Generating report")
        report = reasoning_agent.generate_report({
            "WebSearch": web_search_output.get(image_name, ""),
            "RAG": rag_output.get(image_name, ""),
            "SkinGPT": skin_gpt_output.get(image_name, "")
        })
        reasoning_output[image_name] = report
        save_output(reasoning_output, "Reasoning", output_folder)
    else:
        return
    if case_review_agent is not None:
        # Case review
        print("Case reviewing")
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
            treatment_recommend_output[image_name] = {"error": str(e), "raw_output": treatment_recommend_result}

        save_output(treatment_recommend_output, "TreatmentRecommend", output_folder)
    else:
        return