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


async def analyze_image(all_agents, image_path, web_search_output, rag_output, skin_gpt_output, image_name):
    tasks = []
    for domain, agent in all_agents.items():
        query = get_domain_expert_prompt(domain)
        tasks.append(async_analyze(agent, query, image_path))
    results = await asyncio.gather(*tasks)
    for domain, result in zip(all_agents.keys(), results):
        if domain == "WebSearch":
            web_search_output[image_name] = result
        elif domain == "RAG":
            rag_output[image_name] = result
        elif domain == "SkinGPT":
            skin_gpt_output[image_name] = result
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
        with open(output_file_path, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
    existing_data.update(output_data)
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)


def WorkFlow(
        all_agents,
        treatment_recommend_agent,
        reasoning_agent,
        case_review_agent,
        output_folder="output/",
        image_path="",

):
    # 初始化 Agent
    image_name = image_path.split("/")[-1]
    web_search_output = {}
    rag_output = {}
    skin_gpt_output = {}
    reasoning_output = {}
    case_review_output = {}
    treatment_recommend_output = {}
    asyncio.run(analyze_image(all_agents, image_path, web_search_output, rag_output, skin_gpt_output, image_name))

    # Generate report
    print("Generating report")
    report = reasoning_agent.generate_report({
        "WebSearch": web_search_output.get(image_name, ""),
        "RAG": rag_output.get(image_name, ""),
        "SkinGPT": skin_gpt_output.get(image_name, "")
    })
    reasoning_output[image_name] = report

    # Case review
    print("Case reviewing")
    review_report = case_review_agent.review_case(report)
    case_review_output[image_name] = review_report
    # Update the case in the database
    ## !!!!!
    ## May need to be optimized, because only good cases could be added
    case_review_agent._add_case_to_knowledge_graph(review_report)
    # Treatment recommendation
    print("Treatment recommending")
    try:
        treatment_recommend_result = treatment_recommend_agent.analyze(review_report)
        treatment_recommend = json.loads(treatment_recommend_result)
        treatment_recommend_output[image_name] = treatment_recommend
        end_time = time.time()
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Invalid JSON received: {treatment_recommend_result}")
        treatment_recommend_output[image_name] = {"error": str(e), "raw_output": treatment_recommend_result}

    save_output(web_search_output, "WebSearch", output_folder)
    save_output(rag_output, "RAG", output_folder)
    save_output(skin_gpt_output, "SkinGPT", output_folder)
    save_output(reasoning_output, "Reasoning", output_folder)
    save_output(case_review_output, "CaseReview", output_folder)
    save_output(treatment_recommend_output, "TreatmentRecommend", output_folder)