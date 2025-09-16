import os
import json
import argparse
import time

from altair import DateTime

from rag_agent import RAGAgent
from web_search_agent import WebSearchAgent
from skingpt_agent import SkinGPTAgent
from skingpt_openai_agent import SkinGPTOpenAIAgent
from reasoning_agent import ReasoningAgent
from case_review_agent import CaseReviewAgent
from treatment_recommend_agent import TreatmentRecommendAgent
from prompt_template import *
from tqdm import tqdm
from html import unescape
import ssl
import asyncio

# 全局禁用 SSL 验证（影响整个 Python 进程）
ssl._create_default_https_context = ssl._create_unverified_context


# 修改 agent 的 analyze 方法为异步方法
async def async_analyze(agent, query, image_path):
    return await agent.analyze(query, image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gpt-4o-mini")
    parser.add_argument("--image_folder", type=str, default="data/images/",
                        help="Path to the folder containing images for analysis")
    parser.add_argument("--markdown_file_path", type=str, default="skin_handbook.md",
                        help="Path to the markdown file for reference")
    parser.add_argument("--output_folder", type=str, default="output/",
                        help="Path to the folder to save JSON output files")
    args = parser.parse_args()

    neo4j_url = "neo4j://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "Czty100165188"
    api_key = "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"

    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    # Initialize all agents
    all_agents = {}
    skingpt_agent = SkinGPTOpenAIAgent(model="gpt-4o", domain="SkinGPT", api_key=api_key)
    all_agents["SkinGPT"] = skingpt_agent
    rag_agent = RAGAgent(
        model=args.model_name,
        api_key=api_key,
        domain="RAG",
        markdown_file_path=args.markdown_file_path
    )
    all_agents["RAG"] = rag_agent
    web_search_agent = WebSearchAgent(
        model=args.model_name,
        api_key=api_key,
        domain="WebSearch"
    )
    all_agents["WebSearch"] = web_search_agent
    reasoning_agent = ReasoningAgent(model=args.model_name)
    case_review_agent = CaseReviewAgent(
        model=args.model_name,
        neo4j_uri=neo4j_url,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password
    )
    treatment_recommend_agent = TreatmentRecommendAgent(
        model=args.model_name, 
        api_key=api_key
    )

    # Process each image in the specified folder
    image_list = os.listdir(args.image_folder)
    for image_name in tqdm(image_list, desc="Processing images"):
        # Check if the file is an image by its extension
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            print(f"Skipping non-image file: {image_name}")
            continue  # Skip non-image files

        image_path = os.path.join(args.image_folder, image_name)

        # Initialize output dictionaries for the current image
        web_search_output = {}
        rag_output = {}
        skin_gpt_output = {}
        reasoning_output = {}
        case_review_output = {}
        treatment_recommend_output = {}

        # Perform analysis for each domain
        print("\nFirst round analyzing")


        async def analyze_image():
            tasks = []
            for domain, agent in all_agents.items():
                print('domain ' + domain)
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

        start_time = time.time()
        asyncio.run(analyze_image())

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
            print(f"Raw treatment recommendation result:" + (end_time - start_time).__str__())
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Invalid JSON received: {treatment_recommend_result}")
            treatment_recommend_output[image_name] = {"error": str(e), "raw_output": treatment_recommend_result}

        # Save results for the current image
        def save_output(output_data, agent_name):
            output_file_path = os.path.join(args.output_folder, f"{agent_name}_output.json")

            # Process Markdown-formatted outputs to convert escape characters
            if agent_name in ["WebSearch", "RAG", "SkinGPT"]:
                for key, value in output_data.items():
                    output_data[key] = unescape(value)  # Use html.unescape to handle escape characters

            # Save the output to a JSON file
            if os.path.exists(output_file_path):
                with open(output_file_path, "r", encoding="utf-8") as f:
                    try:
                        exsiting_data = json.load(f)
                    except json.JSONDecodeError:
                        exsiting_data = {}
            exsiting_data.update(output_data)
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(exsiting_data, f, indent=4,
                          ensure_ascii=False)  # ensure_ascii=False supports non-ASCII characters


        # Save outputs for each agent for the current image
        save_output(web_search_output, "WebSearch")
        save_output(rag_output, "RAG")
        save_output(skin_gpt_output, "SkinGPT")
        save_output(reasoning_output, "Reasoning")
        save_output(case_review_output, "CaseReview")
        save_output(treatment_recommend_output, "TreatmentRecommend")