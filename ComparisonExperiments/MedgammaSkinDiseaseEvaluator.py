# AgentWorkflowEvaluator.py
import csv
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

def MedgammaInference(imagePath):
    from transformers import pipeline
    from PIL import Image
    import torch

    pipe = pipeline(
        "image-text-to-text",
        model="google/medgemma-4b-it",
        torch_dtype=torch.bfloat16,
        device="cuda",
    )

    # Image attribution: Stillwaterising, CC0, via Wikimedia Commons
    image = Image.open(imagePath).convert("RGB")

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert radiologist."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What skin disease is this?"},
                {"type": "image", "image": image}
            ]
        }
    ]

    output = pipe(text=messages, max_new_tokens=200)
    return output[0]["generated_text"][-1]["content"]
def EvaluationOnDermnet(
        dataset_root: str = "./SkinGPT-X-Dataset/Dermnet/test",
        output_root: str = "./SkinGPT-X-EvaluationResults/Dermnet/test",
):
    """
    遍历数据集目录，每个疾病文件夹下的每张图片调用一次 process_images
    """
    disease_dirs = [d for d in Path(dataset_root).iterdir() if d.is_dir()]

    if not disease_dirs:
        print("⚠️  数据集根目录下没有找到任何疾病文件夹")
        return
    logFilePath = os.path.join(output_root, "processed.log")
    logFilePathList = load_set(logFilePath)
    csv_path = Path(output_root) / "results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    for disease_dir in tqdm(disease_dirs, desc="Processing images"):
        print('⏳ 正在处理疾病：', disease_dir.name)
        diseasePath = os.path.join(dataset_root, disease_dir.name)
        imgList = os.listdir(diseasePath)

        for imgName in imgList:
            if os.path.join(diseasePath, imgName) in logFilePathList:
                print(f'已处理文件{os.path.join(diseasePath, imgName)}, 跳过')
                continue
            # 如果 csv 不存在，写表头
            if not csv_path.exists():
                csv_path.write_text("image_path, medgamma_pred\n", encoding="utf-8")

                # --------------------------------------------------
                # 修改后的 try-except 块（替换你原来的 try-except）
                # --------------------------------------------------
                try:
                    answer = MedgammaInference(os.path.join(diseasePath, imgName))

                    # ① 原逻辑：标记已处理
                    mark_done(os.path.join(diseasePath, imgName),
                              os.path.join(output_root, 'processed.log'))

                    # ② 新逻辑：追加结果到 csv
                    with csv_path.open("a", newline='', encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([os.path.join(diseasePath, imgName), answer])

                except Exception as e:
                    print(f"[WARN] 处理失败，跳过 ：{e}")
                    # 失败也写一行，预测列为空
                    with csv_path.open("a", newline='', encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([os.path.join(diseasePath, imgName), ""])


if __name__ == "__main__":
    EvaluationOnDermnet(dataset_root='./autodl-tmp/SkinGPT-X-Dataset/Dermnet/test', output_root='./autodl-tmp/SkinGPT-X-Dataset/Dermnet')