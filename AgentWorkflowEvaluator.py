# AgentWorkflowEvaluator.py
import argparse
import os
import json
import time
import traceback
from pathlib import Path
import ssl

from tensorflow.python.data.experimental.ops.testing import sleep
from tqdm import tqdm

from Constants import DERMNET_DATASET_ROOT
from case_review_agent import CaseReviewAgent
from rag_agent import RAGAgent
from read_mispred_ISIC import read_misclassified_filenames
from reasoning_agent import ReasoningAgent
from skingpt_openai_agent import SkinGPTOpenAIAgent
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
                "WebSearch": WebSearchAgent(model=model_name, api_key=api_key, domain="WebSearch", searchapi_key='sk-74829ed96e1c4d9793507d546527f5de', temp_image_path=os.path.join(dataset_root, 'temp_resized_image.png'))
            }
            selected_agent = 'WebSearch'
        if int(agent_type) == 1:
            all_agents = {
                "SkinGPT": SkinGPTOpenAIAgent(model="gemini-2.5-pro", api_key=api_key, pre_csv_path='/Volumes/T7/SkinGPT-X-EvaluationResults/PanDerm_Base_LP_result/ISIC/PanDerm_Base_LP_predprob.csv'),
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
            traceback.print_exc()      # ← 打印完整报错堆栈

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

def EvaluationOnISIC(
        model_name: str = "gemini-2.5-pro",
        dataset_root: str = "/Volumes/T7/SkinGPT-X-Dataset/Dermnet/test",
        markdown_file_path: str = "./skin_handbook.md",
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
                "WebSearch": WebSearchAgent(model=model_name, api_key=api_key, domain="WebSearch", searchapi_key='sk-74829ed96e1c4d9793507d546527f5de', temp_image_path=os.path.join(dataset_root, 'temp_resized_image.png'))
            }
            selected_agent = 'WebSearch'
        if int(agent_type) == 1:
            all_agents = {
                "SkinGPT": SkinGPTOpenAIAgent(model="gemini-2.5-pro", api_key=api_key, pre_csv_path='./PanDerm_ISIC_predprob.csv'),
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
    # logFilePath = os.path.join(output_root, "processed.log")
    # logFilePathList = load_set(logFilePath)

    with open(os.path.join(output_root, f'{selected_agent}_output.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取所有以 .jpg 结尾的键名
    processedFiles = [key for key in data.keys() if key.endswith('.jpg')]
    for disease_dir in disease_dirs:
        print('⏳ 正在处理疾病：', disease_dir.name)
        diseasePath = os.path.join(dataset_root, disease_dir.name)
        imgList = os.listdir(diseasePath)

        for imgName in tqdm(imgList, desc='image name'):
            if (disease_dir.name + '/' + imgName ) in processedFiles:
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
                traceback.print_exc()      # ← 打印完整报错堆栈
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="/Volumes/T7/SkinGPT-X-Dataset/Dermnet/test")
    parser.add_argument("--output_root", default="/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test")
    parser.add_argument("--markdown_file_path", default="./skin_handbook.md")
    parser.add_argument("--api_key", default='sk-emhI8AjXfPpIpS1H1mgMm45AWGbMJzxHuJrNYb2WBzCIJgkG')
    parser.add_argument("--is_single_agent", default=False)
    parser.add_argument("--agent_type", default=0)
    args = parser.parse_args()
    EvaluationOnISIC(dataset_root=args.dataset_root, output_root=args.output_root,
                        markdown_file_path=args.markdown_file_path, api_key=args.api_key,
                        is_single_agent=args.is_single_agent, agent_type=args.agent_type)