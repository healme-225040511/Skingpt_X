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
from local_llm_utils import local_generate_deepseek_chat_response as generate_response_chat
from utils import safe_load_json_qwen
import uuid
import pandas as pd
from Constants import EVALUATION_ROOT, EVALUATION_ROOT_fitzpatrick17k
class CaseReviewAgent:
    def __init__(self, model: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str, clear_mode: bool = True,
                 api_key='', train_feat_path:str=None, train_json_path:str = None, test_feat_path:str=None, test_json_path:str=None, evolution_threshold:int=50):
        """
        Initialize the CaseReviewAgent with an API key and Neo4j connection details.

        Args:
            model (str): The name of the model to use.
            neo4j_uri (str): The URI for the Neo4j database.
            neo4j_user (str): The username for the Neo4j database.
            neo4j_password (str): The password for the Neo4j database.
        """
        self.model = model
        self.embedding_model = HuggingFaceEmbedding(model_name="/225040511/project/hf_cache/bge-small-en-v1.5/BAAI/bge-small-en-v1___5")
        self.true_label = ''
        self.evolution_threshold = evolution_threshold
        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.api_key = api_key
        # Clear all Case nodes in the Neo4j database
        if clear_mode:
            self.clear_all_case_nodes()
        # Initialize feature dataframes and combined map
        self.train_df = None
        self.train_arr = None
        self.test_df = None
        self.test_arr = None
        self.all_files_to_feats_map = {} # A combined map for faster feature lookup

        if train_feat_path and train_json_path:
            self._load_features(train_feat_path, train_json_path, 'train')
        if test_feat_path and test_json_path:
            self._load_features(test_feat_path, test_json_path, 'test')
      
        # Combine filenames and features from both train and test into one map
        if self.train_df is not None:
            self.all_files_to_feats_map.update(dict(zip(self.train_df['filename'], self.train_df['feat'])))
        if self.test_df is not None:
            self.all_files_to_feats_map.update(dict(zip(self.test_df['filename'], self.test_df['feat'])))
        
    def _load_features(self, feat_path: str, json_path: str, split_name: str):
        """
        Helper method to load features and filenames from .npy and .json files.
        """
        try:
            feats = np.load(Path(feat_path))  # (N, 1024)
            with open(json_path, 'r') as f:
                files = json.load(f)  # list[str]
            
            df = pd.DataFrame({'filename': files, 'feat': list(feats)})
            if split_name == 'train':
                self.train_df = df
                self.train_arr = feats
            elif split_name == 'test':
                self.test_df = df
                self.test_arr = feats
            print(f"Loaded {len(df)} {split_name} features from {feat_path} and {json_path}.")
        except FileNotFoundError:
            print(f"Warning: Feature file(s) not found for {split_name} split at {EVALUATION_ROOT_fitzpatrick17k + feat_path} or {EVALUATION_ROOT_fitzpatrick17k + json_path}.")
        except Exception as e:
            print(f"Error loading {split_name} features: {e}")
            
    def clear_all_case_nodes(self):
        """
        Delete all `Case` nodes from the Neo4j database.
        """
        query = "MATCH (c:Case) DETACH DELETE c"
        with self.driver.session() as session:
            session.run(query)
        print("All `Case` nodes have been cleared from the Neo4j database.")
    def distill_prototypes(self, disease_name: str):
        """
        自我演化逻辑：将该疾病下的 Top 10 高置信度病例提炼为一个 Prototype 节点
        """
        with self.driver.session() as session:
            # 修改 distill_prototypes 中的 SQL
            records = session.run("""
                MATCH (c:Case {primary_diagnosis: $name})
                RETURN c.key_findings as kf, c.knowledge_and_research as kr
                ORDER BY c.confidence DESC  // 改为按置信度排序
                LIMIT 10
            """, name=disease_name)
        if len(cases) < 3: return # 数据太少不演化

        # 调用 LLM 提炼
        prompt = f"""
            You are a medical knowledge architect. 
            Analyze the following {len(cases)} cases of {disease_name} and summarize a standard diagnostic prototype.
            
            CASES:
            {cases}
            
            OUTPUT FORMAT:
            Return ONLY a JSON object with a single key "summary". 
            The value should be a concise, professional paragraph summarizing the findings.
            Example: {{"summary": "Clinical features include..."}}
            """
            
        distilled_text = generate_response_chat(engine = self.model, system_role="Medical Knowledge Base Architect", user_input=prompt, max_tokens=4096, temperature=0.2)
         # --- 健壮性处理：确保我们拿到的是字符串 ---
        if isinstance(distilled_text, dict):
            # 如果解析成功，提取 summary 键；如果解析失败（返回了报错字典），提取 raw_output
            summary_text = distilled_text.get("summary") or distilled_text.get("raw_output")
        else:
            summary_text = str(distilled_text)

        # 如果 summary_text 还是为空，给个兜底
        if not summary_text:
            summary_text = f"Standard presentation for {disease_name}."
        with self.driver.session() as session:
            session.run("""
                MERGE (p:Prototype {disease: $name})
                SET p.summary = $summary, p.updated_at = timestamp()
            """, name=disease_name, summary=summary_text)
        print(f"Memory Evolved: Prototype created for {disease_name}")
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
        similar_cases = self._find_similar_diagnoses(current_case, image_path)
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
            return response
        else:
            report = current_case
        return report

    def _add_case_to_knowledge_graph(self, new_case: Dict, image_filename: str):
        """
        将新病例写入 Neo4j；新增 critical_features 字段用于后续纯 CF 向量匹配。
        """
        
        feature_vector_list = []
        if image_filename and self.train_df is not None:
            try:
                # Look up feature from the train_df
                # Assuming image_filename matches the format in train_files_clean.json
                feature_vector = self.train_df[self.train_df['filename'] == image_filename]['feat'].iloc[0]
                feature_vector_list = feature_vector.tolist() # Convert numpy array to list for Neo4j
            except IndexError:
                print(f"Warning: Feature vector not found for {image_filename} in training features. Case will be added without image feature_vector.")
            except Exception as e:
                print(f"Error retrieving feature for {image_filename}: {e}. Case will be added without image feature_vector.")
        else:
            print("Warning: image_filename not provided or training features not loaded. Case will be added without image feature_vector.")
        uid = str(uuid.uuid4())
        disease_name = new_case.get("PrimaryDiagnosis", "N/A")
        
        # 2. 定义合并后的事务（包含基础写入和 Pitfall 提取）
        def integrated_add_tx(tx, data, fv, uid):
            # 写入基础病例
            prob_dist = data.get("ProbabilityDistribution", [])
            conf = prob_dist[0].get("probability", 0) if prob_dist else 0
            
            tx.run("""
                CREATE (c:Case {
                    case_id: $id, primary_diagnosis: $pd, key_findings: $kf,
                    knowledge_and_research: $kr, feature_vector: $fv,
                    confidence: $conf, critical_features: $cf, created_at: timestamp() 
                })
            """, id=uid, pd=disease_name, kf=data.get("KeyFindings", "N/A"),
                kr=data.get("KnowledgeAndResearch", "N/A"),
                fv=fv, conf=conf, cf=data.get("CriticalFeatures", []))

            # 提取 Pitfall (演化逻辑 A: 避坑指南)
            hist_comp = data.get("HistoricalCaseComparison", {})
            inconsistent = hist_comp.get("InConsistentDiagnosis")
            if inconsistent and any(word in inconsistent.lower() for word in ["misdiagnosed", "inconsistent", "correction"]):
                tx.run("""
                    MERGE (p:Pitfall {description: $desc})
                    ON CREATE SET p.pitfall_id = apoc.create.uuid(), p.created_at = timestamp()
                    WITH p
                    MATCH (c:Case {case_id: $id})
                    MERGE (c)-[:HAS_LESSON]->(p)
                """, desc=inconsistent, id=uid)

        # 执行写入
        with self.driver.session() as session:
            session.execute_write(integrated_add_tx, new_case, feature_vector_list, uid)
            
            # 3. 检查阈值并触发演化逻辑 B: 记忆蒸馏
            if disease_name != "N/A" and disease_name != 'Unable to parse model output':
                count_result = session.run(
                    "MATCH (c:Case {primary_diagnosis: $pd}) RETURN count(c) AS cnt",
                    pd=disease_name
                ).single()
                
                current_count = count_result["cnt"]
                
                # 刚好达到阈值，或者在阈值倍数时触发（例如每 10, 20, 30... 条时）
                if current_count >= self.evolution_threshold and current_count % self.evolution_threshold == 0:
                    print(f"--- Automatic Evolution Triggered for {disease_name} (Total: {current_count} cases) ---")
                    # 这里调用你已经写好的 distill_prototypes
                    self.distill_prototypes(disease_name)
        # print(f"Case {uid} added to the knowledge graph with CriticalFeatures and (optionally) FeatureVector.")

    def _find_similar_diagnoses(self, current_case: Dict, image_path=None) -> List[dict]:
        """
        多样化检索逻辑（优化版）：
        检索 3 条同类病例（Positive Cases） + 2 条长得像的异类病例（Contrastive Cases）
        """
        # 1. 获取初步诊断的标签
        preliminary_diagnosis = current_case.get("PrimaryDiagnosis", "N/A")

        # 2. 获取当前图像特征向量
        current_case_feature_vector = None
        if image_path and self.all_files_to_feats_map:
            current_case_feature_vector = self.all_files_to_feats_map.get(image_path)
        
        if current_case_feature_vector is None:
            print(f"Error: Feature vector not found for {image_path}")
            return []

        # 3. 从 Neo4j 获取所有病例的 ID, 向量, 以及 标签
        with self.driver.session() as session:
            records = session.run("""
                MATCH (c:Case) 
                WHERE c.feature_vector IS NOT NULL AND size(c.feature_vector) > 0
                RETURN c.case_id AS id, c.feature_vector AS fv, c.primary_diagnosis AS label
            """)
            
            # 将数据转为容易处理的格式
            all_cases = []
            query_vec = np.array(current_case_feature_vector).reshape(1, -1)
            expected_dim = query_vec.shape[1]

            for r in records:
                fv = np.array(r['fv'])
                if fv.shape[0] == expected_dim:
                    all_cases.append({
                        'id': r['id'],
                        'fv': fv,
                        'label': r['label']
                    })

        if not all_cases:
            return []

        # 4. 计算所有相似度
        all_vectors = np.array([item['fv'] for item in all_cases])
        sims = cosine_similarity(query_vec, all_vectors).flatten()

        for i, item in enumerate(all_cases):
            item['score'] = sims[i]

        # 5. 执行多样化采样
        # 分为“同类”和“异类”
        positive_pool = [c for c in all_cases if c['label'] == preliminary_diagnosis]
        negative_pool = [c for c in all_cases if c['label'] != preliminary_diagnosis]

        # 按相似度从高到低排序
        positive_pool.sort(key=lambda x: x['score'], reverse=True)
        negative_pool.sort(key=lambda x: x['score'], reverse=True)

        # 挑选目标 ID：3个同类 + 2个异类
        target_ids = []
        
        # 选同类 Top 3
        target_ids.extend([c['id'] for c in positive_pool[:3]])
        
        # 选异类 Top 2 (最容易混淆的)
        target_ids.extend([c['id'] for c in negative_pool[:2]])

        # 如果同类不够（冷启动阶段），用异类补充；反之亦然
        if len(target_ids) < 5:
            remaining = [c['id'] for c in all_cases if c['id'] not in target_ids]
            remaining_sims = sorted([c for c in all_cases if c['id'] in remaining], 
                                    key=lambda x: x['score'], reverse=True)
            target_ids.extend([c['id'] for c in remaining_sims[:(5 - len(target_ids))]])

        # 6. 【核心检索】从数据库拉取详细信息，包括 Prototype 和 Pitfalls
        with self.driver.session() as session:
            final_records = session.run("""
                MATCH (c:Case)
                WHERE c.case_id IN $ids
                OPTIONAL MATCH (p:Prototype {disease: c.primary_diagnosis})
                OPTIONAL MATCH (c)-[:HAS_LESSON|GENERATED_LESSON]->(pit:Pitfall)
                RETURN c { .*, feature_vector: null } as case, 
                    p.summary as prototype, 
                    collect(DISTINCT pit.description) as pitfalls,
                    // 额外标记这个案例是“确认用”还是“对比用”
                    (c.primary_diagnosis = $current_diag) as is_positive_match
                ORDER BY is_positive_match DESC  // 让同类的排在前面
            """, ids=target_ids, current_diag=preliminary_diagnosis)
        
        return [dict(r) for r in final_records]
    
    def close(self):
        """
        Close the Neo4j driver connection.
        """
        self.driver.close()
    def delete_case_by_id(self, case_id: str) -> bool:
        """
        根据 case_id 删除单个 Case 节点
        :param case_id: 生成好的 UUID 字符串
        :return: 是否真删到节点
        """
        def _tx(tx, cid: str):
            # 1. 先探测是否存在
            exists = tx.run(
                "MATCH (c:Case {case_id: $cid}) RETURN count(c) AS cnt",
                cid=cid
            ).single()["cnt"]
            if exists == 0:
                return False

            # 2. 删除并返回已删标记
            tx.run(
                "MATCH (c:Case {case_id: $cid}) DELETE c",
                cid=cid
            )
            return True

        with self.driver.session() as session:
            return session.execute_write(_tx, case_id)


if __name__ == "__main__":
    
    model = "Qwen/Qwen-7B-Chat"
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "Czty100165188"
    train_feat_file = EVALUATION_ROOT_fitzpatrick17k + 'train_feats_clean.npy' 
    train_json_file = EVALUATION_ROOT_fitzpatrick17k + 'train_files_clean.json'
    test_feat_file = EVALUATION_ROOT_fitzpatrick17k + 'test_feats.npy' 
    test_json_file = EVALUATION_ROOT_fitzpatrick17k + 'test_files.json' 

    case_review_agent = CaseReviewAgent(
        model=model,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        clear_mode=False, # Set to True if you want to clear all cases before re-adding
        api_key='',
        train_feat_path=train_feat_file, 
        train_json_path=train_json_file,
        test_feat_path = test_feat_file,
        test_json_path = test_json_file
    )
    current_case = {
        "ImageRegion": "Nose, central and anterior positioning, sun-exposed area",
        "KeyFindings": "The lesion is localized to the nasal tip and ala, presenting as a markedly enlarged, bulbous, and irregularly contoured structure. The skin exhibits significant thickening, with a rough, warty, and nodular texture, and is hyperpigmented with a reddish-brown hue. The surface is uneven, with visible telangiectasias and multiple small papules. There are no signs of acute inflammation, ulceration, or pain, and no associated scaling or itching. The lesion is symmetrically distributed on the nasal prominence. Severity: Severe.",
        "CriticalFeatures": [
            "Severe nasal enlargement with nodular and warty texture",
            "Hyperpigmentation and telangiectasias"
        ],
        "PrimaryDiagnosis": "Rhinophyma",
        "KnowledgeAndResearch": "Rhinophyma is a subtype of acne rosacea characterized by progressive enlargement of the nasal sebaceous glands, leading to a bulbous, red, and nodular appearance. It predominantly affects middle-aged to elderly individuals, especially males, and is associated with chronic sun exposure and inflammatory skin changes. The condition is often linked to long-standing rosacea and may be exacerbated by environmental factors. Treatment options include laser therapy, electrosurgery, and topical or systemic anti-inflammatory agents, though management is primarily cosmetic. Evidence from dermatology literature supports its association with aging and chronic inflammation, with no direct link to systemic immunosuppression or infection."
    }
    # --- Loading history cases and adding them to the graph with features ---
    # history_cases_json_path = EVALUATION_ROOT_fitzpatrick17k + 'RAG_output_train.json'
    # if Path(train_json_file).exists():
    #     with open(train_json_file, 'r') as f:
    #         allowed_files = set(json.load(f)) # 转换为 set 提高查找速度
    #     print(f"Loaded filter list with {len(allowed_files)} allowed filenames.")
    # else:
    #     allowed_files = None
    #     print(f"Warning: Filter list {train_json_file} not found. Will process all cases.")

    # if Path(history_cases_json_path).exists():
    #     with Path(history_cases_json_path).open(encoding="utf-8") as f:
    #         history_cases = json.load(f)
        
    #     # Uncomment the line below if you want to clear all existing cases in Neo4j
    #     # before adding new ones with features.
    #     # case_review_agent.clear_all_case_nodes() 

    #     for key, value in history_cases.items():
    #         if allowed_files is not None and key.split('/')[-1] not in allowed_files:
    #             continue
    #         case_filename = key.split('/')[-1]
    #         if case_filename:
    #             case_review_agent._add_case_to_knowledge_graph(value, image_filename=case_filename)
    #         # else:
    #         #     print(f"Warning: Case {key} from JSON does not have a 'filename'. Skipping feature vector storage for this case.")
    # else:
    #     print(f"\nWarning: {history_cases_json_path} not found. No historical cases loaded to Neo4j.")

    # # # --- Reviewing a current case ---
    example_image_path =  "bdbc05b476c3076ad9ac3b06a3eaded1.jpg"
    
    print(f"\nAttempting to review case for image: {example_image_path}")
    if case_review_agent.all_files_to_feats_map and example_image_path in case_review_agent.all_files_to_feats_map:
        review_report = case_review_agent.review_case(current_case, image_path=example_image_path)
        print("\nReview Report:")
        print(json.dumps(review_report, indent=2))
    else:
        print(f"Error: Feature for current case image '{example_image_path}' not found in loaded features. Cannot proceed with review.")

    # case_review_agent.close()
    # extract_json_items('/225040511/project/SkinGPT-X-EvaluationResults/Dermnet/SkinGPT-X/SkinGPT_output.json', './test/correct_list.txt', './output/CaseReview_output_correct.json')
    # with Path('/225040511/project/SkinGPT-X-EvaluationResults/Dermnet/new_rag/RAG_output_8B_train.json').open(encoding="utf-8") as f:
    #     history_cases = json.load(f)
    # # print(history_cases)
    # for key, value in history_cases.items():
    #     case_review_agent._add_case_to_knowledge_graph(value)
    # 审查当前病例
    # print(current_case)
    # similar_cases = case_review_agent._find_similar_diagnoses(current_case)
    # for case in similar_cases:
    #     print("Primary_Diagnosis:" + case['Primary_Diagnosis'])
    #     print("Key_Findings:"+case['Key_Findings'])
    
    # # path = Path('/225040511/project/Skingpt_X/test/process_wronglist_Dermnet.txt')
    # # # --- FIX: 使用标准 Python IO 代替 np.loadtxt，避免 "delimiter cannot be a newline" 错误 ---

    # # # 使用 Path.read_text() 读取整个文件内容，并使用 splitlines() 按行分割
    # # # 这种方法对文件路径列表更加健壮和高效
    # # content = path.read_text(encoding='utf-8')
    # # pending_list = content.splitlines()

    # # 清理空字符串
    # # with Path('/225040511/project/SkinGPT-X-EvaluationResults/Dermnet/new_rag/reasoning/RAG_output.json').open(encoding="utf-8") as f:
    # #     reasoning_cases = json.load(f)
    # # pending = [item.strip() for item in pending_list if item.strip()]
    # # for image_path in pending:
    # #     similar_cases = case_review_agent._find_similar_diagnoses(current_case, image_path)
    #     # for case in similar_cases:
    #     #     print("Primary_Diagnosis:" + case['Primary_Diagnosis'])
    #     #     print("Key_Findings:"+case['Key_Findings'])
    
    # review_report = case_review_agent.review_case(current_case,
    #                                               image_path='Psoriasis pictures Lichen Planus and related diseases/psoriasis-palms-soles-185.jpg')
    # print(json.dumps(review_report, indent=2))
    # case_review_agent._add_case_to_knowledge_graph(current_case, image_filename='Psoriasis pictures Lichen Planus and related diseases/psoriasis-palms-soles-185.jpg')
    # case_review_agent.distill_prototypes("Urticaria Hives")

    # case_review_agent.close()
    # case_review_agent._backfill_case_id()