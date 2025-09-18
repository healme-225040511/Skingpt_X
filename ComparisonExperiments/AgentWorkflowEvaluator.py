# AgentWorkflowEvaluator.py
import os
import json
from pathlib import Path
import ssl

from tqdm import tqdm

from case_review_agent import CaseReviewAgent
from rag_agent import RAGAgent
from reasoning_agent import ReasoningAgent
from skingpt_openai_agent import SkinGPTOpenAIAgent
from treatment_recommend_agent import TreatmentRecommendAgent
from web_search_agent import WebSearchAgent

ssl._create_default_https_context = ssl._create_unverified_context
from utils import load_set, mark_done
# ① 直接导入主流程函数
from agent_workflow import WorkFlow


def LoadLog(logFile):
    if os.path.isfile(logFile):
        with open(logFile, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def EvaluationOnDermnet(
        model_name: str = "gemini-2.5-pro",
        dataset_root: str = "./SkinGPT-X-Dataset/Dermnet/test",
        markdown_file_path: str = "./skin_handbook.md",
        output_root: str = "./SkinGPT-X-EvaluationResults/Dermnet/test",
        api_key: str = "AIzaSyC-9og_9OsxvKZ0rBXeMGboXBrMOpG5-do",
        openai_api_key: str = "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",
        neo4j_url: str = "neo4j://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "Czty100165188"
):
    """
    遍历数据集目录，每个疾病文件夹下的每张图片调用一次 process_images
    """
    disease_dirs = [d for d in Path(dataset_root).iterdir() if d.is_dir()]
    all_agents = {
        "SkinGPT": SkinGPTOpenAIAgent(model=model_name, domain="SkinGPT", api_key=api_key),
        "RAG": RAGAgent(model=model_name, api_key=api_key, domain="RAG", markdown_file_path=markdown_file_path),
        "WebSearch": WebSearchAgent(model=model_name, api_key=api_key, domain="WebSearch")
    }
    reasoning_agent = ReasoningAgent(model='gpt-4o-mini')
    case_review_agent = CaseReviewAgent(model='gpt-4o-mini', neo4j_uri=neo4j_url, neo4j_user=neo4j_user,
                                        neo4j_password=neo4j_password)
    treatment_recommend_agent = TreatmentRecommendAgent(model='gpt-4o-mini', api_key=openai_api_key)
    if not disease_dirs:
        print("⚠️  数据集根目录下没有找到任何疾病文件夹")
        return
    logFilePath = os.path.join(output_root, "processed.log")
    logFilePathList = load_set(logFilePath)
    for disease_dir in tqdm(disease_dirs, desc="Processing images"):
        print('⏳ 正在处理疾病：', disease_dir.name)
        diseasePath = os.path.join(dataset_root, disease_dir.name)
        imgList = os.listdir(diseasePath)

        for imgName in imgList:
            if os.path.join(diseasePath, imgName) in logFilePathList:
                print(f'已处理文件{os.path.join(diseasePath, imgName)}, 跳过')
                continue
            try:
                # ② 单张模式调用
                WorkFlow(
                    all_agents=all_agents,
                    reasoning_agent=reasoning_agent,
                    case_review_agent=case_review_agent,
                    treatment_recommend_agent=treatment_recommend_agent,
                    output_folder=output_root,
                    image_path=os.path.join(diseasePath, imgName),
                )
                mark_done(os.path.join(diseasePath, imgName), os.path.join(output_root, 'processed.log'))
            except Exception as e:
                print(f"[WARN] 处理失败，跳过 ：{e}")


if __name__ == "__main__":
    EvaluationOnDermnet()