import os
import json
import argparse
from rag_agent import RAGAgent
from web_search_agent import WebSearchAgent
from skingpt_agent import SkinGPTAgent
from reasoning_agent import ReasoningAgent
from case_review_agent import CaseReviewAgent
from treatment_recommend_agent import TreatmentRecommendAgent
from prompt_template import *
from tqdm import tqdm
from html import unescape
import ssl

# 全局禁用 SSL 验证（影响整个 Python 进程）
ssl._create_default_https_context = ssl._create_unverified_context
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gpt-4o-mini")
    parser.add_argument("--image_folder", type=str, default="data/images/", help="Path to the folder containing images for analysis")
    parser.add_argument("--markdown_file_path", type=str, default="skin_handbook.md", help="Path to the markdown file for reference")
    parser.add_argument("--output_folder", type=str, default="output/", help="Path to the folder to save JSON output files")
    args = parser.parse_args()

    neo4j_url = "neo4j://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "Czty100165188"
    api_key = "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"

    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    # Initialize all agents
    all_agents = {}
    skingpt_agent = SkinGPTAgent(model="llama3.2-vision", domain="SkinGPT")
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
    for domain, agent in all_agents.items():
        query = get_domain_expert_prompt(domain)
        analysis_result = agent.analyze(query, image_path)

        # Store results for the current image
        if domain == "WebSearch":
            web_search_output[image_name] = analysis_result
        elif domain == "RAG":
            rag_output[image_name] = analysis_result
        elif domain == "SkinGPT":
            skin_gpt_output[image_name] = analysis_result

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
    treatment_recommend = json.loads(treatment_recommend_agent.analyze(review_report))
    treatment_recommend_output[image_name] = treatment_recommend

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
            json.dump(exsiting_data, f, indent=4, ensure_ascii=False)  # ensure_ascii=False supports non-ASCII characters

    # Save outputs for each agent for the current image
    save_output(web_search_output, "WebSearch")
    save_output(rag_output, "RAG")
    save_output(skin_gpt_output, "SkinGPT")
    save_output(reasoning_output, "Reasoning")
    save_output(case_review_output, "CaseReview")
    save_output(treatment_recommend_output, "TreatmentRecommend")