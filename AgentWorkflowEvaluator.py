# AgentWorkflowEvaluator.py
import argparse
import csv
import os
import json
import time
import traceback
from pathlib import Path
import ssl
from typing import Any

import pandas as pd
from tensorflow.python.data.experimental.ops.testing import sleep
from tqdm import tqdm

from Constants import DERMNET_DATASET_ROOT, EVALUATION_ROOT
from case_review_agent import CaseReviewAgent
from rag_agent import RAGAgent
from read_mispred_ISIC import read_misclassified_filenames
from reasoning_agent import ReasoningAgent
from skingpt_agent import SkingptAgent
from treatment_recommend_agent import TreatmentRecommendAgent
from web_search_agent import WebSearchAgent

ssl._create_default_https_context = ssl._create_unverified_context
from utils import load_set, mark_done
# ① 直接导入主流程函数
from agent_workflow import WorkFlow
from read_samples import read_samples_from_file


def LoadLog(logFile):
    if os.path.isfile(logFile):
        with open(logFile, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def EvaluationOnDermnet(
        model_name: str = "gemini-2.5-pro",
        dataset_root: str = "/Volumes/T7/SkinGPT-X-Dataset/Dermnet/test",
        markdown_file_path: str = "/Volumes/T7/Skingpt_X/skin_handbook.md",
        output_root: str = "/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test",
        api_key: str = "AIzaSyC-9og_9OsxvKZ0rBXeMGboXBrMOpG5-do",
        openai_api_key: str = "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",
        neo4j_url: str = "neo4j://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "Czty100165188",
        is_single_agent: bool = False,
        agent_type: int = 0
):
    """
    遍历数据集目录，每个疾病文件夹下的每张图片调用一次 process_images
    """
    disease_dirs = [d for d in Path(dataset_root).iterdir() if d.is_dir()]
    all_agents = {}
    reasoning_agent = None
    case_review_agent = None
    selected_agent = 'SkinGPT'
    if is_single_agent:
        if int(agent_type) == 0:
            all_agents = {
                "WebSearch": WebSearchAgent(model=model_name, api_key=api_key, domain="WebSearch",
                                            searchapi_key='sk-74829ed96e1c4d9793507d546527f5de',
                                            temp_image_path=os.path.join(dataset_root, 'temp_resized_image.png'))
            }
            selected_agent = 'WebSearch'
        if int(agent_type) == 1:
            all_agents = {
                "SkinGPT": SkingptAgent(model="gemini-2.5-pro", api_key=api_key,
                                        pre_csv_path='/Volumes/T7/SkinGPT-X-EvaluationResults/PanDerm_Base_LP_result/ISIC/PanDerm_Base_LP_predprob.csv'),
            }
        elif int(agent_type) == 2:
            all_agents = {
                "RAG": RAGAgent(model=model_name, api_key=api_key, domain="RAG", markdown_file_path=markdown_file_path),
            }
            selected_agent = 'RAG'
    else:
        # all_agents = {
        #     "SkinGPT": SkinGPTOpenAIAgent(model="gemini-2.5-pro", api_key=api_key, pre_csv_path='/Volumes/T7/SkinGPT-X-EvaluationResults/PanDerm_Base_LP_result/Dermnet_predprob.csv'),
        #     "RAG": RAGAgent(model=model_name, api_key=api_key, domain="RAG", markdown_file_path=markdown_file_path),
        #     "WebSearch": WebSearchAgent(model=model_name, api_key=api_key, domain="WebSearch",
        #                                 searchapi_key='sk-74829ed96e1c4d9793507d546527f5de',
        #                                 temp_image_path=os.path.join(dataset_root, 'temp_resized_image.png'))
        # }
        # reasoning_agent = ReasoningAgent(model="gemini-2.5-flash", api_key=api_key)
        selected_agent = 'CaseReview'
        # selected_agent = 'CaseReview'
        case_review_agent = CaseReviewAgent(model="gemini-2.5-flash", neo4j_uri=neo4j_url, neo4j_user=neo4j_user,
                                            neo4j_password=neo4j_password, clear_mode=False, api_key=api_key)
        # treatment_recommend_agent = TreatmentRecommendAgent(model="gemini-2.5-flash", api_key=api_key, searchapi_key='sk-74829ed96e1c4d9793507d546527f5de')
    if not disease_dirs:
        print("⚠️  数据集根目录下没有找到任何疾病文件夹")
        return
    # logFilePath = os.path.join(output_root, "processed.log")
    # logFilePathList = load_set(logFilePath)

    with open(os.path.join(output_root, f'{selected_agent}_output.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取所有以 .jpg 结尾的键名
    processedFiles = [key for key in data.keys() if key.endswith('.jpg')]

    # samples = read_samples_from_file('samples_only_in_reasoning.txt')
    samples = ['Acne and Rosacea Photos/rosacea-25.jpg',
               'Acne and Rosacea Photos/rosacea-nose-16.jpg',
               'Acne and Rosacea Photos/rosacea-nose-65.jpg',
               'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions/actinic-keratosis-horn-13.jpg',
               'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions/basal-cell-carcinoma-face-22.jpg',
               'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions/basal-cell-carcinoma-face-30.jpg',
               'Atopic Dermatitis Photos/05PityriasisAlba.jpg',
               'Atopic Dermatitis Photos/12IMG006.jpg',
               'Atopic Dermatitis Photos/IchthosisIMG030-GP3.jpg',
               'Bullous Disease Photos/bullous-pemphigoid-43.jpg',
               'Exanthems and Drug Eruptions/roseola-infantum-2.jpg',
               'Tinea Ringworm Candidiasis and other Fungal Infections/tinea-foot-plantar-7.jpg',
               'Tinea Ringworm Candidiasis and other Fungal Infections/tinea-groin-40.jpg',
               'Vascular Tumors/cherry-angioma-16.jpg',
               'Vascular Tumors/cherry-angioma-17.jpg',
               'Vascular Tumors/hemangioma-infancy-3.jpg',
               'Vascular Tumors/hemangioma-infancy-39.jpg',
               'Vascular Tumors/hemangioma-infancy-4.jpg',
               'Vascular Tumors/hemangioma-infancy-6.jpg',
               'Vascular Tumors/hemangioma-infancy-60.jpg',
               'Vascular Tumors/kaposi-sarcoma-41.jpg',
               'Vascular Tumors/kaposi-sarcoma-42.jpg',
               'Vascular Tumors/kaposi-sarcoma-6.jpg',
               'Vascular Tumors/pyogenic-granuloma-112.jpg',
               'Vascular Tumors/pyogenic-granuloma-2.jpg',
               'Vascular Tumors/pyogenic-granuloma-20.jpg',
               'Vascular Tumors/pyogenic-granuloma-35.jpg',
               'Vascular Tumors/pyogenic-granuloma-84.jpg',
               'Vascular Tumors/pyogenic-granuloma-95.jpg',
               'Vascular Tumors/spider-angioma-13.jpg',
               'Vascular Tumors/vascular-anomaly-1.jpg',
               'Vascular Tumors/vascular-anomaly-3.jpg',
               'Vascular Tumors/vascular-anomaly-5.jpg',
               'Vascular Tumors/venous-lake-10.jpg',
               'Vascular Tumors/venous-lake-15.jpg',
               'Vascular Tumors/venous-lake-28.jpg']
    print(samples)
    print(f"共 {len(samples)} 个样本")
    for imgName in tqdm(samples, desc='samples not contain'):
        if imgName in processedFiles:
            print(f'已处理文件 {imgName}, 跳过')
            continue
        try:
            startTime = time.time()
            WorkFlow(
                all_agents=all_agents,
                # reasoning_agent=reasoning_agent,
                case_review_agent=case_review_agent,
                output_folder=output_root,
                image_path=os.path.join(DERMNET_DATASET_ROOT, imgName),
                folder_name=imgName.split('/')[0],
            )
            endTime = time.time()
            print(f'{imgName}  处理完成，耗时{endTime - startTime}s')
        except Exception as e:
            # 原句换成下面两行
            print(f"[WARN] 处理失败，跳过：{imgName}")
            traceback.print_exc()  # ← 打印完整报错堆栈

    # for disease_dir in disease_dirs:
    #     print('⏳ 正在处理疾病：', disease_dir.name)
    #     diseasePath = os.path.join(dataset_root, disease_dir.name)
    #     imgList = os.listdir(diseasePath)
    #
    #     for imgName in tqdm(imgList, desc='image name'):
    #         if (disease_dir.name + '/' + imgName ) in processedFiles:
    #             print(f'已处理文件 {disease_dir.name}/{imgName}, 跳过')
    #             continue
    #         try:
    #             startTime = time.time()
    #             WorkFlow(
    #                 all_agents=all_agents,
    #                 reasoning_agent=reasoning_agent,
    #                 output_folder=output_root,
    #                 image_path=os.path.join(diseasePath, imgName),
    #                 folder_name=disease_dir.name
    #             )
    #             mark_done(os.path.join(diseasePath, imgName),
    #                       os.path.join(output_root, 'processed.log'))
    #             endTime = time.time()
    #             print(f'{imgName}  处理完成，耗时{endTime - startTime}s')
    #         except Exception as e:
    #             # 原句换成下面两行
    #             print(f"[WARN] 处理失败，跳过：{imgName}")
    #             traceback.print_exc()      # ← 打印完整报错堆栈


def load_pending_list(txt_path: str) -> list[Any]:
    """
    读取包含待处理文件路径的 TXT 文件。
    假设 TXT 文件中，每行是一个完整的文件路径（例如：disease/img.jpg）。
    返回待处理文件名的列表。
    """
    path = Path(txt_path)
    if not txt_path or not path.exists():
        print('[INFO] 待处理文件列表 TXT 不存在，将处理所有文件。')
        return []

    try:
        # --- FIX: 使用标准 Python IO 代替 np.loadtxt，避免 "delimiter cannot be a newline" 错误 ---

        # 使用 Path.read_text() 读取整个文件内容，并使用 splitlines() 按行分割
        # 这种方法对文件路径列表更加健壮和高效
        content = path.read_text(encoding='utf-8')
        pending_list = content.splitlines()

        # 清理空字符串
        pending = [item.strip() for item in pending_list if item.strip()]

    except Exception as e:
        print(f"[ERROR] 读取文件 ({txt_path}) 失败: {e}")
        # 如果读取失败，返回空列表，避免程序中断
        return []

    print(f'[INFO] 从 TXT 文件加载到 {len(pending)} 张待处理图片')
    return pending
def Evaluation(
        model_name: str = "gemini-2.5-pro",
        dataset_root: str = "/Volumes/T7/SkinGPT-X-Dataset/Dermnet/test",
        markdown_file_path: str = "./skin_handbook.md",
        output_root: str = "/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test",
        api_key: str = "AIzaSyC-9og_9OsxvKZ0rBXeMGboXBrMOpG5-do",
        openai_api_key: str = "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",
        neo4j_url: str = "neo4j://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "Czty100165188",
        pre_predporb_csv_path: str = "",
        is_single_agent: bool = False,
        agent_type: int = 0
):
    """
    遍历数据集目录，每个疾病文件夹下的每张图片调用一次 process_images
    """
    disease_dirs = [d for d in Path(dataset_root).iterdir() if d.is_dir()]
    all_agents = {}
    reasoning_agent = None
    case_review_agent = None
    selected_agent = 'SkinGPT'
    if is_single_agent:
        if int(agent_type) == 0:
            all_agents = {
                "WebSearch": WebSearchAgent(model=model_name, api_key=api_key, domain="WebSearch",
                                            searchapi_key='sk-74829ed96e1c4d9793507d546527f5de',
                                            temp_image_path=os.path.join(dataset_root, 'temp_resized_image.png'))
            }
            selected_agent = 'WebSearch'
        if int(agent_type) == 1:
            all_agents = {
                "SkinGPT": SkingptAgent(model="gemini-2.5-pro", api_key=api_key,
                                        pre_csv_path=pre_predporb_csv_path),
            }
        elif int(agent_type) == 2:
            all_agents = {
                "RAG": RAGAgent(model=model_name, api_key=api_key, domain="RAG", markdown_file_path=markdown_file_path),
            }
            selected_agent = 'RAG'
    else:
        # all_agents = {
        #     "SkinGPT": SkinGPTOpenAIAgent(model="gemini-2.5-pro", api_key=api_key, pre_csv_path='/Volumes/T7/SkinGPT-X-EvaluationResults/PanDerm_Base_LP_result/Dermnet_predprob.csv'),
        #     "RAG": RAGAgent(model=model_name, api_key=api_key, domain="RAG", markdown_file_path=markdown_file_path),
        #     "WebSearch": WebSearchAgent(model=model_name, api_key=api_key, domain="WebSearch",
        #                                 searchapi_key='sk-74829ed96e1c4d9793507d546527f5de',
        #                                 temp_image_path=os.path.join(dataset_root, 'temp_resized_image.png'))
        # }
        reasoning_agent = ReasoningAgent(model="gemini-2.5-flash", api_key=api_key)
        selected_agent = 'Reasoning'
        # selected_agent = 'CaseReview'
        # case_review_agent = CaseReviewAgent(model="gemini-2.5-flash", neo4j_uri=neo4j_url, neo4j_user=neo4j_user,
        #                                     neo4j_password=neo4j_password, clear_mode=False, api_key=api_key)
        # treatment_recommend_agent = TreatmentRecommendAgent(model="gemini-2.5-flash", api_key=api_key, searchapi_key='sk-74829ed96e1c4d9793507d546527f5de')
    if not disease_dirs:
        print("⚠️  数据集根目录下没有找到任何疾病文件夹")
        return

    with open(os.path.join(output_root, f'{selected_agent}_output.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取所有以 .jpg 结尾的键名
    processedFiles = [key for key in data.keys() if key.endswith('.jpg')]
    for disease_dir in disease_dirs:
        print('⏳ 正在处理疾病：', disease_dir.name)
        diseasePath = os.path.join(dataset_root, disease_dir.name)
        imgList = os.listdir(diseasePath)

        for imgName in tqdm(imgList, desc='image name'):
            if (disease_dir.name + '/' + imgName) in processedFiles:
                print(f'已处理文件 {disease_dir.name}/{imgName}, 跳过')
                continue
            try:
                startTime = time.time()
                WorkFlow(
                    all_agents=all_agents,
                    reasoning_agent=reasoning_agent,
                    # case_review_agent=case_review_agent,
                    output_folder=output_root,
                    image_path=os.path.join(diseasePath, imgName),
                    folder_name=disease_dir.name
                )
                mark_done(os.path.join(diseasePath, imgName),
                          os.path.join(output_root, 'processed.log'))
                endTime = time.time()
                print(f'{imgName}  处理完成，耗时{endTime - startTime}s')
            except Exception as e:
                # 原句换成下面两行
                print(f"[WARN] 处理失败，跳过：{imgName}")
                traceback.print_exc()  # ← 打印完整报错堆栈

def Evaluation_on_txt(
        model_name: str = "gemini-2.5-pro",
        dataset_root: str = "/Volumes/T7/SkinGPT-X-Dataset/Dermnet/test",
        markdown_file_path: str = "./skin_handbook.md",
        output_root: str = "/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test",
        api_key: str = "AIzaSyC-9og_9OsxvKZ0rBXeMGboXBrMOpG5-do",
        openai_api_key: str = "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",
        neo4j_url: str = "neo4j://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "Czty100165188",
        pre_predporb_csv_path: str = "",
        is_single_agent: bool = False,
        agent_type: int = 0,
        pending_set_path: str = '',
        EVALUATION_ROOT = '/225040511/project/Evaluation_Results/fitzpatrick17k/Panderm/1.0/'
):
    """
    遍历数据集目录，每个疾病文件夹下的每张图片调用一次 process_images
    """
    all_agents = {}
    reasoning_agent = None
    case_review_agent = None
    selected_agent = 'SkinGPT'
    if is_single_agent:
        if int(agent_type) == 0:
            all_agents = {
                "WebSearch": WebSearchAgent(model=model_name, api_key=api_key, domain="WebSearch",
                                            searchapi_key='sk-74829ed96e1c4d9793507d546527f5de',
                                            temp_image_path=os.path.join(dataset_root, f'temp_resized_image_{pending_set_path.split("/")[-1].split(".")[0]}.png'))
            }
            selected_agent = 'WebSearch'
        if int(agent_type) == 1:
            all_agents = {
                "SkinGPT": SkingptAgent(model="gemini-2.5-pro", api_key=api_key,
                                        pre_csv_path='/225040511/project/Skingpt_X/Panderm_Dermnet_predprob.csv'),
            }
        elif int(agent_type) == 2:
            all_agents = {
                "RAG": RAGAgent(model="gemini-2.5-pro", api_key=api_key, domain="RAG", markdown_file_path=markdown_file_path),
            }
            selected_agent = 'RAG'
    else:
        # all_agents = {
        #     "SkinGPT": SkinGPTOpenAIAgent(model="gemini-2.5-pro", api_key=api_key, pre_csv_path='/Volumes/T7/SkinGPT-X-EvaluationResults/PanDerm_Base_LP_result/Dermnet_predprob.csv'),
        #     "RAG": RAGAgent(model=model_name, api_key=api_key, domain="RAG", markdown_file_path=markdown_file_path),
        #     "WebSearch": WebSearchAgent(model=model_name, api_key=api_key, domain="WebSearch",
        #                                 searchapi_key='sk-74829ed96e1c4d9793507d546527f5de',
        #                                 temp_image_path=os.path.join(dataset_root, 'temp_resized_image.png'))
        # }
        # reasoning_agent = ReasoningAgent(model="gemini-3-pro-preview", api_key=api_key)
        # selected_agent = 'Reasoning'
        
        selected_agent = 'CaseReview'
        train_feat_file = EVALUATION_ROOT + 'train_feats.npy' 
        train_json_file = EVALUATION_ROOT + 'train_files.json'
        test_feat_file = EVALUATION_ROOT + 'test_feats.npy' 
        test_json_file = EVALUATION_ROOT + 'test_files.json' 
        case_review_agent = CaseReviewAgent(
            model='Qwen/Qwen3-32B',
            neo4j_uri=neo4j_url,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            clear_mode=False, api_key='',
            train_feat_path=train_feat_file, 
            train_json_path=train_json_file,
            test_feat_path = test_feat_file,
            test_json_path = test_json_file
        )
        # treatment_recommend_agent = TreatmentRecommendAgent(model="gemini-2.5-flash", api_key=api_key, searchapi_key='sk-74829ed96e1c4d9793507d546527f5de')

    with open(os.path.join(output_root, f'{selected_agent}_output.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
    pending_set = load_pending_list(pending_set_path)
    # 提取所有以 .jpg 结尾的键名
    processedFiles = [key for key in data.keys() if key.endswith('.jpg')]
    for f_index in tqdm(range(len(pending_set)), desc='Pending'):
        img_path = os.path.join(dataset_root, pending_set[f_index])
        if pending_set[f_index] in processedFiles:
            print(f'{pending_set[f_index]} already processed')
            continue
        try:
            startTime = time.time()
            WorkFlow(
                all_agents=all_agents,
                # reasoning_agent=reasoning_agent,
                case_review_agent=case_review_agent,
                output_folder=output_root,
                image_path=img_path,
                folder_name=img_path.split('/')[-2],
            )
            endTime = time.time()
            print(f'{img_path}  处理完成，耗时{endTime - startTime}s')
        except Exception as e:
            # 原句换成下面两行
            print(f"[WARN] 处理失败，跳过：{img_path}, {e}")
            traceback.print_exc()  # ← 打印完整报错堆栈


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="/Volumes/T7/SkinGPT-X-Dataset/HAM10000/test")
    parser.add_argument("--output_root", default="/Volumes/T7/SkinGPT-X-EvaluationResults/HAM10000/test")
    parser.add_argument("--markdown_file_path", default="/225040511/project/Skingpt_X/skin_handbook.md")
    parser.add_argument("--api_key", default='sk-emhI8AjXfPpIpS1H1mgMm45AWGbMJzxHuJrNYb2WBzCIJgkG')
    parser.add_argument("--is_single_agent", default=False)
    parser.add_argument("--agent_type", default=0)
    parser.add_argument("--pre_predprob_csv_path", default=0)
    parser.add_argument("--pending_set_path", default='')
    args = parser.parse_args()
    Evaluation_on_txt(dataset_root=args.dataset_root, output_root=args.output_root,
               markdown_file_path=args.markdown_file_path, api_key=args.api_key,
               is_single_agent=args.is_single_agent, agent_type=args.agent_type,
               pre_predporb_csv_path=args.pre_predprob_csv_path, pending_set_path=args.pending_set_path)
