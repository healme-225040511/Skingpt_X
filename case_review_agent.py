import os
import json
from pathlib import Path
from typing import Dict, List
from neo4j import GraphDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from api_utils import generate_response
from prompt_template import get_case_review_prompt
from thefuzz import fuzz
import re
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import numpy as np
from local_llm_utils import local_generate_response as generate_response
from local_llm_utils import local_generate_deepseek_chat_response as generate_response_chat
from utils import safe_load_json_qwen
import uuid
import pandas as pd
from Constants import EVALUATION_ROOT, EVALUATION_ROOT_fitzpatrick17k, DERMNET_DISEASE_NAME, SUPERDEMNET_DISEASE_NAME

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
            self.clear_all_nodes()
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
        print(f"Combined feature map contains {len(self.all_files_to_feats_map)} entries.")
        
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
            
    def clear_all_nodes(self):
        """
        删除数据库中所有与本项目相关的节点：Case, Prototype, Pitfall
        """
        # 匹配多种标签并删除
        query = """
        MATCH (n) 
        WHERE n:Case OR n:Prototype OR n:Pitfall 
        DETACH DELETE n
        """
        with self.driver.session() as session:
            session.run(query)
        print("All Case, Prototype, and Pitfall nodes have been cleared from the Neo4j database.")
    def initialize_fixed_categories(self, disease_list: List[str]):
        with self.driver.session() as session:
            for disease in disease_list:
                session.run("""
                    MERGE (p:Prototype {disease: $name})
                    ON CREATE SET p.summary = 'Initial entry. Awaiting data for evolution.', p.updated_at = timestamp()
                """, name=disease)
        print(f"Initialized {len(disease_list)} disease prototypes.")

    def distill_prototypes(self, disease_name: str):
        """
        基于固定类别的演化逻辑：结合正确病例与报错教训，更新诊断标准。
        """
        with self.driver.session() as session:
            # 1. 获取该疾病现有的原型总结（用于参考更新，实现“不断更新”）
            old_res = session.run(
                "MATCH (p:Prototype {disease: $name}) RETURN p.summary as s", 
                name=disease_name
            ).single()
            old_summary = old_res["s"] if old_res else "No existing summary."

            # 2. 获取该疾病最新的 10 个正确病例 (Golden Cases)
            case_records = session.run("""
                MATCH (c:Case {true_label: $name, is_correct: true})
                RETURN c.key_findings as kf
                ORDER BY c.confidence DESC LIMIT 10
            """, name=disease_name)
            golden_cases = [r["kf"] for r in case_records]

            # 3. 获取与该疾病相关的“坑” (Pitfalls) —— 这是提高准确率的关键
            # 这里的逻辑是：搜集那些本来是这个病，但被看错了，或者别人被看成这个病的记录
            pitfall_records = session.run("""
                MATCH (c:Case)-[:HAS_LESSON]->(p:Pitfall)
                WHERE c.true_label = $name OR c.primary_diagnosis = $name
                RETURN DISTINCT p.description as desc LIMIT 5
            """, name=disease_name)
            pitfalls = [r["desc"] for r in pitfall_records]

        if len(golden_cases) < 3: return # 数据不足

        # 4. 构建“进化” Prompt
        prompt = f"""
        You are a Medical Knowledge Architect. You are updating the diagnostic standard for '{disease_name}'.
        
        [Current Standard]: 
        {old_summary}

        [New Positive Evidence (Correct Cases)]:
        {golden_cases}

        [Learned Lessons (Common Misdiagnoses/Pitfalls)]:
        {pitfalls}

        [Task]:
        Create an updated, more accurate diagnostic prototype. 
        Your summary MUST include:
        1. Core visual features (from correct cases).
        2. A "Warning Section": Based on the Pitfalls, explain how to distinguish '{disease_name}' from the other 22 diseases it is often confused with.
        
        [Output Format]:
        Return ONLY a JSON object: {{"summary": "The updated paragraph..."}}
        """

        # 调用 LLM 并更新 (逻辑同前)
        response = generate_response_chat(engine=self.model, system_role="Expert Dermatologist", user_input=prompt, temperature=0.2, max_tokens=4096)
        
        # 解析并写入 Neo4j
        new_summary = response.get("summary") if isinstance(response, dict) else str(response)
        
        with self.driver.session() as session:
            session.run("""
                MERGE (p:Prototype {disease: $name})
                SET p.summary = $summary, p.updated_at = timestamp()
            """, name=disease_name, summary=new_summary)
            
        print(f"--- Evolution Complete: Prototype for '{disease_name}' has been updated with new pitfalls. ---")
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
        print("similar_cases:\n"+similar_cases)
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
    SIMILARITY_THRESHOLD = 85 

    def bulk_load_exercise_data(self, json_data: Dict):
        """
        批量加载演练数据，自动从路径提取 Ground Truth。
        """
        print(f"Starting bulk load of {len(json_data)} cases...")
        
        for image_path, case_data in json_data.items():
            # 1. 自动提取 Ground Truth (路径的文件夹名字)
            # 例如 "Psoriasis pictures.../image.jpg" -> "Psoriasis pictures..."
            true_label = Path(image_path).parent.name 
            
            # 2. 判定对错
            predicted_label = case_data.get("PrimaryDiagnosis")
            # 建议使用 fuzz 匹配或者简单的字符串包含，因为文件夹名和模型输出可能略有差异
            # 这里先用简单的相等判定，如果准确率低可以改用 fuzz.ratio
            if not predicted_label:
                is_correct = False
                similarity_score = 0
            else:
                # 清洗字符串：转小写并去除空格
                pred_clean = str(predicted_label).lower().strip()
                true_clean = str(true_label).lower().strip()
                
                # 计算相似度得分 (0-100)
                # ratio: 比较整体相似度
                # partial_ratio: 比较是否存在包含关系（如 "Psoriasis" 在 "Psoriasis pictures" 中得分会很高）
                similarity_score = fuzz.token_set_ratio(pred_clean, true_clean)
                
                # 根据阈值判定是否正确
                is_correct = (similarity_score >= self.SIMILARITY_THRESHOLD)
            
            # 3. 提取文件名用于匹配特征向量
            filename = Path(image_path).name
            
            # 4. 结果记录与 Pitfall 分析
            if not is_correct:
                print(f"❌ Mismatch [{filename}]: Score={similarity_score}%")
                print(f"   Pred: '{predicted_label}' vs GT: '{true_label}'")
                pitfall_lesson = self._analyze_misdiagnosis(case_data, true_label)
            else:
                print(f"✅ Match [{filename}]: Score={similarity_score}%")
                pitfall_lesson = None
            
            # 5. 写入 Neo4j (传入得分以便后续分析)
            self._save_exercise_node(case_data, image_path, filename, true_label, is_correct, pitfall_lesson)
            
            # 6. 检查是否触发演化 (仅针对正确且信心高的病例)
            if is_correct:
                self._check_evolution_trigger(true_label)

    def _analyze_misdiagnosis(self, case_data: Dict, true_label: str) -> str:
        """
        调用 LLM 分析误诊原因，提取“坑” (Pitfall)
        """
        prompt = f"""
        You are a Dermatological Auditor. An AI model misidentified a case.
        [AI's Key Findings]: {case_data.get('KeyFindings')}
        [AI's Incorrect Diagnosis]: {case_data.get('PrimaryDiagnosis')}
        [The Actual Truth]: {true_label}
        
        Task: Analyze the 'Key Findings'. Why would someone confuse this with the incorrect diagnosis? 
        Identify the specific visual feature that acted as a 'trap'.
        
        Output: A single concise sentence starting with "Pitfall:".
        Example: "Pitfall: The annular erythema was mistaken for Tinea, but the lack of peripheral scaling actually points to Granuloma Annulare."
        """
        try:
            analysis = generate_response_chat(
                engine=self.model, 
                system_role="Diagnostic Quality Controller", 
                user_input=prompt,
                max_tokens=2000,
                temperature=0.3
            )
            # 处理返回结果格式
            if isinstance(analysis, dict):
                return analysis.get("summary") or analysis.get("raw_output", "Unknown trap.")
            return str(analysis)
        except Exception as e:
            return f"Pitfall: Diagnostic confusion between {case_data.get('PrimaryDiagnosis')} and {true_label}."

    def _save_exercise_node(self, data, full_path, filename, true_label, is_correct, pitfall_lesson):
        uid = str(uuid.uuid4())
        pitfall_uid = str(uuid.uuid4()) 
        
        # 初始化为空列表
        fv = []
        
        # --- 修复：多重路径匹配逻辑，找不到不崩溃 ---
        # 构造可能的 Key
        possible_keys = [
            '/train/' + full_path,
            '/test/' + full_path,
            full_path,
            filename
        ]
        
        for key in possible_keys:
            if key in self.all_files_to_feats_map:
                feat = self.all_files_to_feats_map[key]
                # 确保转为 list (如果是 numpy array)
                fv = feat.tolist() if hasattr(feat, 'tolist') else list(feat)
                print(f"✅ Found feature vector using key: {key}")
                break
        
        if not fv:
            print(f"⚠️ Warning: No feature vector found for {full_path}. Case will be saved with empty vector.")
        else:
            print(f"Saving case {uid} with feature vector length: {len(fv)}")

        def _tx(tx):
            # 创建 Case 节点
            tx.run("""
                CREATE (c:Case {
                    case_id: $uid,
                    image_path: $path,
                    primary_diagnosis: $pd,
                    true_label: $td,
                    is_correct: $is_correct,
                    key_findings: $kf,
                    critical_features: $cf,
                    knowledge: $kr,
                    feature_vector: $fv,
                    created_at: timestamp()
                })
            """, uid=uid, path=full_path, pd=data.get("PrimaryDiagnosis"), 
                td=true_label, is_correct=is_correct, kf=data.get("KeyFindings"),
                cf=data.get("CriticalFeatures", []), kr=data.get("KnowledgeAndResearch", ""), 
                fv=fv) # 这里的 fv 可能是 []
            
            if pitfall_lesson:
                tx.run("""
                    MATCH (c:Case {case_id: $uid})
                    MERGE (p:Pitfall {description: $desc})
                    ON CREATE SET p.pitfall_id = $pit_id, p.created_at = timestamp()
                    MERGE (c)-[:HAS_LESSON]->(p)
                """, uid=uid, desc=pitfall_lesson, pit_id=pitfall_uid)

        with self.driver.session() as session:
            session.execute_write(_tx)

    def _check_evolution_trigger(self, disease_name):
        """检查正确病例数是否达到演化阈值"""
        with self.driver.session() as session:
            res = session.run(
                "MATCH (c:Case {true_label: $name, is_correct: true}) RETURN count(c) as cnt",
                name=disease_name
            ).single()
            if res and res["cnt"] >= self.evolution_threshold and res["cnt"] % self.evolution_threshold == 0:
                print(f"--- Automatic Evolution Triggered for {disease_name} (Total correct: {res['cnt']}) ---")
                self.distill_prototypes(disease_name)


if __name__ == "__main__":
    EVALUATION_ROOT = '/225040511/project/Evaluation_Results/SuperDermnet/Panderm/'
    model = "Qwen/Qwen-7B-Chat"
    neo4j_uri = "bolt://100.91.219.86:7687"
    neo4j_user = "neo4j"
    neo4j_password = "Czty100165188"
    train_feat_file = EVALUATION_ROOT + 'train_feats.npy' 
    train_json_file = EVALUATION_ROOT + 'train_files.json'
    test_feat_file = EVALUATION_ROOT + 'test_feats.npy' 
    test_json_file = EVALUATION_ROOT + 'test_files.json' 

    case_review_agent = CaseReviewAgent(
        model=model,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        clear_mode=True, # Set to True if you want to clear all cases before re-adding
        api_key='',
        train_feat_path=train_feat_file, 
        train_json_path=train_json_file,
        test_feat_path = test_feat_file,
        test_json_path = test_json_file
    )

    # exercise_json_path = "/225040511/project/Evaluation_Results/SuperDermnet/SkinGPT-X/train/RAG_output_train.json"
    
    # with open(exercise_json_path, 'r') as f:
    #     exercise_data = json.load(f)
    
    # # 执行批量导入
    # # 注意：确保 clear_mode=True 如果你想重新构建知识库
    # case_review_agent.initialize_fixed_categories(SUPERDEMNET_DISEASE_NAME)
    # case_review_agent.bulk_load_exercise_data(exercise_data)
    
    # print("Knowledge Graph construction complete.")

    case_review_agent.clear_all_nodes()