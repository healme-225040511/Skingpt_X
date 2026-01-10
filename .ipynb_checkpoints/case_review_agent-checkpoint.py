import os
import json
from pathlib import Path
from typing import Dict, List
from neo4j import GraphDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from api_utils import generate_response
from prompt_template import get_case_review_prompt
from fuzzywuzzy import fuzz
import re
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import numpy as np
from local_llm_utils import local_generate_response as generate_response
from utils import safe_load_json_qwen


class CaseReviewAgent:
    def __init__(self, model: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str, clear_mode: bool = True,
                 api_key=''):
        """
        Initialize the CaseReviewAgent with an API key and Neo4j connection details.

        Args:
            model (str): The name of the model to use.
            neo4j_uri (str): The URI for the Neo4j database.
            neo4j_user (str): The username for the Neo4j database.
            neo4j_password (str): The password for the Neo4j database.
        """
        self.model = model
        self.embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.api_key = api_key
        # Clear all Case nodes in the Neo4j database
        if clear_mode:
            self.clear_all_case_nodes()

    def clear_all_case_nodes(self):
        """
        Delete all `Case` nodes from the Neo4j database.
        """
        query = "MATCH (c:Case) DETACH DELETE c"
        with self.driver.session() as session:
            session.run(query)
        print("All `Case` nodes have been cleared from the Neo4j database.")

    def review_case(self, current_case: Dict, image_path: str = None) -> Dict:
        """
        Review the current case by comparing it with historical cases and best practices.

        Args:
            current_case (Dict): The current case data to review.

        Returns:
            Dict: The review report.
        """

        # Generate review report by comparing with historical cases
        review_report = self._generate_review_report(current_case, image_path)
        return review_report

    def _generate_review_report(self, current_case: Dict, image_path: str = None) -> Dict:
        """
        Generate a review report by comparing the current case with historical cases in the Neo4j knowledge graph.

        Args:
            current_case (Dict): The current case data.

        Returns:
            Dict: The review report.
        """
        # Check if the current diagnosis is consistent with historical cases
        similar_cases = self._find_similar_diagnoses(current_case)
        print(similar_cases)
        if similar_cases:
            reviewer, prompt = get_case_review_prompt(current_case, similar_cases)
            # Call OpenAI API to generate the report
            response = generate_response(
                engine=self.model,
                temperature=0.2,
                max_tokens=4096,
                system_role=reviewer,
                user_input=prompt,
                image_path=image_path
            )
            report = json.loads(response)
        else:
            report = current_case
        return report

    def _add_case_to_knowledge_graph(self, new_case: Dict):
        """
        将新病例写入 Neo4j；新增 critical_features 字段用于后续纯 CF 向量匹配。
        """
        def add_case(tx, data):
            # 若解析失败直接跳过
            if data.get("PrimaryDiagnosis") == 'Unable to parse model output':
                return

            # 统一把 CriticalFeatures 列表 → 拼接字符串（缺省空列表）
            cf_list = data.get("CriticalFeatures", [])
            cf_txt = " | ".join([c.strip().lower() for c in cf_list if c.strip()])

            tx.run("""
                CREATE (c:Case {
                    primary_diagnosis: $pd,
                    confidence_level: $cl,
                    differential_diagnoses: $dd,
                    key_findings: $kf,
                    critical_features: $cf,      
                    critical_features_text: $cft,
                    knowledge_and_research: $kr,
                    probability_distribution: $prob
                })
            """, pd=data.get("PrimaryDiagnosis", "N/A"),
                cl=data.get("ConfidenceLevel", "N/A"),
                dd=data.get("DifferentialDiagnoses", []),
                kf=data.get("KeyFindings", "N/A"),
                cf=cf_list,                   # 原列表保留，方便后续导出
                cft=cf_txt,                   # 拼接后字符串，用于向量编码
                kr=data.get("KnowledgeAndResearch", "N/A"),
                prob=json.dumps(data.get("ProbabilityDistribution", {})))

        with self.driver.session() as session:
            session.execute_write(add_case, new_case)
        print("Case added to the knowledge graph with CriticalFeatures.")

    def _find_similar_diagnoses(self, current_case: Dict) -> List[dict]:
        """
        1. 当前 case 的 Differential Diagnosis 列表 → 模糊匹配库里的 primary_diagnosis
        2. 匹配成功的 case 再用 CriticalFeatures 向量取 Top-5
        """
        # ----- 0. 解析当前 dd & CF -----
        dd_list = current_case.get("DifferentialDiagnoses", [])
        query_cf = current_case.get("CriticalFeatures", [])
        if not dd_list:
            return []

        query_vec = self.embedding_model._get_text_embedding(
            " | ".join([c.strip().lower() for c in query_cf])
        )

        # ----- 1. 拉全库（仅含 pd） -----
        with self.driver.session() as session:
            records = session.run("""
                MATCH (c:Case)
                RETURN c.primary_diagnosis   AS Primary_Diagnosis,
                    c.confidence_level    AS Confidence_Level,
                    c.key_findings        AS Key_Findings,
                    c.knowledge_and_research AS Knowledge_and_Research,
                    c.probability_distribution AS prob,
                    c.critical_features   AS cf
            """)
            all_cases = [dict(r) for r in records]
        if not all_cases:
            return []

        # ----- 2. 粗筛：pd 与当前任一 dd fuzzy≥80 则保留 -----
        def norm(dx: str) -> str:
            return re.sub(r"\s*\(.*?\)", "", dx.lower()).strip()

        current_dd_norm = {norm(dx) for dx in dd_list}
        cand_cases = []
        for case in all_cases:
            pd_norm = norm(case["Primary_Diagnosis"])
            if any(fuzz.ratio(pd_norm, dd) >= 50 for dd in current_dd_norm):
                cand_cases.append(case)

        if not cand_cases:
            return []

        # ----- 3. 精排：CriticalFeatures 向量 Top-5 -----
        vecs, nodes = [], []
        for case in cand_cases:
            # cf_list = case["cf"] or []
            # cf_txt = " | ".join([c.strip().lower() for c in cf_list])
            # if not cf_txt:        # 库缺 CF 直接跳过
            #     continue
            # vecs.append(self.embedding_model._get_text_embedding(cf_txt))
            # nodes.append(case)
            keyfindings_txt = case["Key_Findings"]
            vecs.append(self.embedding_model._get_text_embedding(keyfindings_txt))
            nodes.append(case)

        if not vecs:
            return []

        vecs = np.array(vecs)
        sims = cosine_similarity(np.array([query_vec]), vecs).flatten()
        top_idx = np.argsort(sims)[-3:][::-1]
        return [nodes[i] for i in top_idx]

    def close(self):
        """
        Close the Neo4j driver connection.
        """
        self.driver.close()


def extract_json_items(json_file_path, filename_list_path, output_file_path):
    """
    根据txt文件中的文件名列表，从json文件中提取对应的项，并保存到新的json文件。

    Args:
        json_file_path (str): 包含所有数据的JSON文件路径。
        filename_list_path (str): 包含要提取的文件名（键）列表的TXT文件路径。
        output_file_path (str): 提取结果要保存到的JSON文件路径。
    """

    # 1. 检查文件是否存在
    if not os.path.exists(json_file_path):
        print(f"❌ 错误：JSON 文件不存在于路径: {json_file_path}")
        return
    if not os.path.exists(filename_list_path):
        print(f"❌ 错误：TXT 文件不存在于路径: {filename_list_path}")
        return

    # 2. 读取要查找的文件名列表 (Keys)
    print(f"📚 正在读取文件名列表: {filename_list_path}...")
    filenames_to_find = set()
    try:
        with open(filename_list_path, 'r', encoding='utf-8') as f:
            # 读取每一行，去除首尾空白符（包括换行符）
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    filenames_to_find.add(stripped_line)
    except Exception as e:
        print(f"❌ 读取TXT文件时发生错误: {e}")
        return

    if not filenames_to_find:
        print("⚠️ 警告：TXT 文件中没有找到任何有效的文件名。")
        return

    # 3. 读取完整的 JSON 数据
    print(f"📖 正在读取 JSON 数据: {json_file_path}...")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except json.JSONDecodeError:
        print(f"❌ 错误：JSON 文件格式不正确。请检查文件: {json_file_path}")
        return
    except Exception as e:
        print(f"❌ 读取JSON文件时发生错误: {e}")
        return

    # 4. 提取对应的项
    extracted_data = {}
    missing_keys = []

    print("🔍 正在提取匹配的项...")
    for key in filenames_to_find:
        # 查找 JSON 数据中是否有这个键
        if key in full_data:
            extracted_data[key] = full_data[key]
        else:
            missing_keys.append(key)

    # 5. 保存提取的结果
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            # 使用 indent=4 使输出的 JSON 文件格式美观易读
            json.dump(extracted_data, f, ensure_ascii=False, indent=4)
        print(f"✅ 提取完成！成功将 {len(extracted_data)} 个项保存到: {output_file_path}")

        if missing_keys:
            print(f"ℹ️ 注意：在JSON中未能找到 {len(missing_keys)} 个键。部分示例：{missing_keys[:5]}")

    except Exception as e:
        print(f"❌ 保存输出文件时发生错误: {e}")


if __name__ == "__main__":
    # 初始化 CaseReviewAgent
    model = "Qwen/Qwen-7B-Chat"
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "Czty100165188"
    case_review_agent = CaseReviewAgent(
        model=model,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        clear_mode=False, api_key=''
    )

    current_case ={
    "PrimaryDiagnosis": "Further evaluation is required to determine the exact cause of the lesion.",
    "ConfidenceLevel": "High",
    "DifferentialDiagnoses": [
      "Actinic Keratosis",
      "Basal Cell Carcinoma",
      "Squamous Cell Carcinoma",
      "Lichen Planus",
      "Warts Molluscum and other Viral Infections"
    ],
    "KeyFindings": "The lesion is located on the sun-exposed skin of the head/neck (malar region). It is an ill-defined, infiltrative plaque with a violaceous and erythematous hue, resembling a persistent bruise (ecchymosis). The tissue shows induration (hardening) and prominent telangiectasias. The lesion is located on skin showing signs of chronic photodamage (dermatoheliosis).",
    "KnowledgeAndResearch": "The lesion's characteristics, including its location on sun-exposed skin, the violaceous and erythematous hue, induration, and telangiectasias, suggest a possible diagnosis of actinic keratosis, basal cell carcinoma, or squamous cell carcinoma. However, the 'bruise-like' quality and infiltrative borders are critical'red flags' for malignancy. Further evaluation and biopsy are recommended to confirm the diagnosis and rule out malignancy."
  }
    # extract_json_items('/225040511/project/SkinGPT-X-EvaluationResults/Dermnet/SkinGPT-X/SkinGPT_output.json', './test/correct_list.txt', './output/CaseReview_output_correct.json')
    with Path('/225040511/project/SkinGPT-X-EvaluationResults/Dermnet/test4/SkinGPT_output.json').open(encoding="utf-8") as f:
        history_cases = json.load(f)
#     print(history_cases)
    for key, value in history_cases.items():
        case_review_agent._add_case_to_knowledge_graph(value)
    # 审查当前病例
#     similar_cases = case_review_agent._find_similar_diagnoses(current_case)
#     print(similar_cases)
    # review_report = case_review_agent.review_case(current_case,
    #                                               image_path='')
    # print(json.dumps(review_report, indent=2))

    # case_review_agent.close()