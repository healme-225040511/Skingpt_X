

import os
import json
import re
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import fuzz # 确保安装了 thefuzz

# LlamaIndex 相关
from llama_index.core import Settings, QueryBundle
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.indices import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 工具函数
from local_llm_utils import local_generate_response as generate_response
from local_llm_utils import local_generate_deepseek_chat_response as generate_response_chat # 假设演化用chat接口效果更好
from utils import process_markdown

class CaseReviewAgent:
    def __init__(self, 
                 model: str, 
                 neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 lancedb_uri: str,markdown_path: str,
                 train_feat_path: str, train_json_path: str,
                 test_feat_path: str=None, test_json_path: str=None,
                 evolution_threshold: int = 5):
        
        self.model = model
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="/225040511/project/hf_cache/bge-small-en-v1.5/BAAI/bge-small-en-v1___5"
        )
        self.text_store = LanceDBVectorStore(uri=lancedb_uri, table_name="text_collection", mode="ro")
        self.vector_index = VectorStoreIndex.from_vector_store(self.text_store)
        self.static_retriever = self.vector_index.as_retriever()
        
        self.all_files_to_feats_map = {}
        self._load_reference_features(train_feat_path, train_json_path)
        if test_feat_path:
            self._load_reference_features(test_feat_path, test_json_path, split_name="Test")

        self.evolution_threshold = evolution_threshold

    # ================= 1. 节点存入与演化逻辑 (新) =================

    def save_case_and_evolve(self, image_path: str, case_data: Dict, true_label: str = None):
        """
        保存病例到 Neo4j，建立与 Prototype 的关联，并检查是否触发知识演化。
        """
        filename = Path(image_path).name
        feat = self._get_feat_by_path(image_path)
        fv = feat.tolist() if feat is not None else []
        
        uid = str(uuid.uuid4())
        # 如果没有 true_label，说明是预测阶段，用预测的 label
        target_label = true_label if true_label else case_data.get("PrimaryDiagnosis")
        
        # 判定是否正确 (逻辑同第一个代码片段)
        is_correct = False
        if true_label and case_data.get("PrimaryDiagnosis"):
            is_correct = (fuzz.token_set_ratio(case_data.get("PrimaryDiagnosis").lower(), true_label.lower()) >= 85)

        def _save_tx(tx):
            # MERGE Prototype 并 CREATE Case
            tx.run("""
                MERGE (p:Prototype {disease: $td})
                ON CREATE SET p.summary = 'Initial entry. Awaiting evolution.', p.updated_at = timestamp()
                CREATE (c:Case {
                    case_id: $uid,
                    primary_diagnosis: $pd,
                    true_label: $td,
                    is_correct: $is_correct,
                    key_findings: $kf,
                    feature_vector: $fv,
                    created_at: timestamp()
                })
                MERGE (c)-[:BELONGS_TO]->(p)
            """, uid=uid, pd=case_data.get("PrimaryDiagnosis"), td=target_label, 
                is_correct=is_correct, kf=case_data.get("KeyFindings"), fv=fv)

        with self.driver.session() as session:
            session.execute_write(_save_tx)
            
            # 检查是否触发演化 (仅针对正确的病例累计)
            if target_label:
                res = session.run("MATCH (c:Case {true_label: $name, is_correct: true}) RETURN count(c) as cnt", name=target_label).single()
                if res and res["cnt"] >= self.evolution_threshold and res["cnt"] % self.evolution_threshold == 0:
                    print(f"--- Triggering Evolution for {target_label} ---")
                    self.distill_prototypes(target_label)

    def distill_prototypes(self, disease_name: str):
        """
        知识蒸馏：从正确病例中总结出“动态原型” (Evolved Prototype)
        """
        with self.driver.session() as session:
            # 获取历史病例
            case_records = session.run("""
                MATCH (c:Case {true_label: $name, is_correct: true})
                RETURN c.key_findings as kf LIMIT 10
            """, name=disease_name)
            golden_cases = [r["kf"] for r in case_records]
            
            # 获取现有的 summary
            old_res = session.run("MATCH (p:Prototype {disease: $name}) RETURN p.summary as s", name=disease_name).single()
            old_summary = old_res["s"] if old_res else ""

        if len(golden_cases) < 3: return

        prompt = f"Update the diagnostic summary for {disease_name}.\nOld Standard: {old_summary}\nNew Cases: {golden_cases}\nOutput JSON: {{'summary': '...'}}"
        
        response = generate_response_chat(engine=self.model, system_role="Expert Dermatologist", user_input=prompt)
        new_summary = response.get("summary") if isinstance(response, dict) else str(response)

        with self.driver.session() as session:
            session.run("MATCH (p:Prototype {disease: $name}) SET p.summary = $s, p.updated_at = timestamp()", name=disease_name, s=new_summary)

    # ================= 2. 修改后的 Review 逻辑 (关联 Prototype) =================

    def review_case(self, 
                    vision_key_findings: str, 
                    panderm_top5: List[Dict[str, Any]], 
                    image_path: str) -> Dict:
        """
        综合诊断审查
        """
        # 1. 提取 Panderm Top-1 疾病名
        top1_disease = panderm_top5[0]['disease'] if panderm_top5 else "N/A"

        # 2. 检索历史相似病例
        current_feat = self._get_feat_by_path(image_path)
        hybrid_cases = self._find_hybrid_historical_cases(current_feat, vision_key_findings)

        # 3. 检索静态知识 (Handbook)
        expert_knowledge_context = ""
        for item in panderm_top5[:5]: # 取前5个
            knowledge = self._retrieve_static_knowledge(item['disease'])
            expert_knowledge_context += f"\n[Handbook: {item['disease']}]\n{knowledge}\n"

        # 4. 【核心新增】从 Neo4j 检索 Top-1 疾病对应的 Evolved Prototype
        evolved_prototype = "No evolved knowledge yet."
        with self.driver.session() as session:
            res = session.run("MATCH (p:Prototype {disease: $name}) RETURN p.summary as s", name=top1_disease).single()
            if res and res["s"]:
                evolved_prototype = res["s"]

        # 5. 构建增强版 Prompt
        prompt = self._build_comprehensive_prompt(
            vision_findings=vision_key_findings,
            top5=panderm_top5,
            similar_cases=hybrid_cases,
            expert_knowledge=expert_knowledge_context,
            evolved_prototype=evolved_prototype, # 传入动态原型
            top1_name=top1_disease
        )

        response_raw = generate_response(
            engine=self.model,
            temperature=0.1,
            max_tokens=4096,
            system_role="Senior Clinical Dermatologist & Auditor",
            user_input=prompt
        )
        
        parsed_res = self._parse_json_response(response_raw)
        return parsed_res, prompt

    # ================= 3. Prompt 构建优化 =================

    def _build_comprehensive_prompt(self, vision_findings, top5, similar_cases, expert_knowledge, evolved_prototype, top1_name):
        # 整理参考案例
        history_str = ""
        for c in similar_cases:
            history_str += f"- Past Case: {c['diagnosis']} | Score: {c['score']:.2f}\n  Findings: {c['findings']}\n"

        top5_str = "\n".join([f"- {i['disease']}: {i['probability']:.2%}" for i in top5])

        return f"""
        [Clinical Task]: Final Diagnosis Review.

        [1. Current Visual Findings (from Vision Agent)]:
        {vision_findings}

        [2. Model Prediction Probability (Panderm)]:
        {top5_str}

        [3. Precedent Cases (Physical & Semantic Retrieval)]:
        {history_str if history_str else "No direct precedents found."}

        [4. Medical Standard (Static Handbook Knowledge)]:
        {expert_knowledge}

        [Instructions]:
        - Evaluate Evidence 1 & 4 for 'Structural Consistency'.
        - Check Evidence 3: Does this patient's presentation match the descriptors of past confirmed cases? 
        - Reconcile Evidence 2: Does the model's confidence align with the visual facts? 
        
        [Output Format (JSON)]:
        {{
            "KeyFindings": "Final clinical descriptor summary.",
            "PrimaryDiagnosis": "The finalized name.",
            "Evidence": "Detailed derivation logic citing visual cues and historical matches."
        }}
        """

    # --- 以下辅助函数基本保持不变 ---

    def _load_reference_features(self, feat_path, json_path, split_name="Train"):
        if not feat_path or not os.path.exists(feat_path): return
        feats = np.load(feat_path)
        with open(json_path, 'r') as f:
            files = json.load(f)
        self.all_files_to_feats_map.update(dict(zip(files, feats)))
        print(f"✅ Loaded {len(files)} {split_name} features.")

    def _get_feat_by_path(self, path):
        # 兼容处理：有些路径可能是全路径，有些是文件名
        if path in self.all_files_to_feats_map:
            return self.all_files_to_feats_map[path]
        fname = Path(path).name
        return self.all_files_to_feats_map.get(fname)

    def _retrieve_static_knowledge(self, disease_name: str):
        nodes = self.static_retriever.retrieve(QueryBundle(f"Features of {disease_name}"))
        return "\n".join([n.node.text for n in nodes[:2]])

    def _find_hybrid_historical_cases(self, current_feat, findings_text, k=5):
        if current_feat is None: return []
        all_retrieved_cases = {}
        with self.driver.session() as session:
            records = list(session.run("MATCH (c:Case) WHERE c.feature_vector IS NOT NULL RETURN c"))
            if not records: return []
            query_feat = np.array(current_feat).reshape(1, -1)
            phys_scored = []
            for r in records:
                node = r['c']
                db_vec = np.array(node['feature_vector']).reshape(1, -1)
                if db_vec.shape == query_feat.shape:
                    score = cosine_similarity(query_feat, db_vec)[0][0]
                    phys_scored.append((score, node))
            phys_scored.sort(key=lambda x: x[0], reverse=True)
            for score, node in phys_scored[:k]:
                all_retrieved_cases[node['case_id']] = {
                    "score": score, "diagnosis": node['true_label'] or node['primary_diagnosis'],
                    "findings": node['key_findings'], "type": "History"
                }
        return list(all_retrieved_cases.values())

    def _parse_json_response(self, text):
        if isinstance(text, dict): return text
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            return json.loads(match.group()) if match else {"error": "parse failed"}
        except:
            return {"error": "parse failed"}

    def close(self):
        self.driver.close()
# 使用示例
if __name__ == "__main__":
    # 配置参数
    agent = CaseReviewAgent(
        model="Qwen/Qwen2-8B-Instruct", # 或者你本地的推理接口
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        lancedb_uri="./test_lancedb",
        train_feat_path="./train_feats.npy", # 确保有这个文件或 mock 它
        train_json_path="./train_files.json"
    )

    try:
        print("--- Step 1: 模拟历史知识积累 (预置 Prototype) ---")
        # 我们手动在 Neo4j 中插入一个银屑病的原型，设定其“金标准”
        with agent.driver.session() as session:
            session.run("""
                MERGE (p:Prototype {disease: 'Psoriasis'})
                SET p.summary = 'Classic Psoriasis presents with thick, silvery-white micaceous scales on well-demarcated erythematous plaques. Auspitz sign is positive. Commonly affects extensors. It NEVER presents with greasy yellow scales.'
            """)
        print("✅ Prototype for Psoriasis seeded.")

        print("\n--- Step 2: 模拟一个冲突案例 ---")
        # 视觉描述：故意描述成脂溢性皮炎的特征
        vision_findings = (
            "The image shows poorly defined erythematous patches covered with greasy, "
            "yellowish scales located in the seborrheic areas. "
            "There are NO silvery scales or micaceous plating."
        )

        # PanDerm 预测：由于某种原因，模型错误地把 Psoriasis 排在第一
        panderm_top5 = [
            {"disease": "Psoriasis", "probability": 0.82},
            {"disease": "Seborrheic Dermatitis", "probability": 0.12},
            {"disease": "Eczema", "probability": 0.04},
            {"disease": "Tinea Corporis", "probability": 0.02}
        ]

        # 模拟图片路径（需要在 all_files_to_feats_map 中有对应，或者你能获取 feat）
        img_path = "test_image_conflict.jpg" 
        
        # 如果没有真实特征向量，我们可以临时 mock 一个
        agent.all_files_to_feats_map[img_path] = np.random.rand(1024).astype('float32')

        print(f"Vision Findings: {vision_findings}")
        print(f"AI Top-1: {panderm_top5[0]['disease']} ({panderm_top5[0]['probability']:.2%})")

        print("\n--- Step 3: 执行审计审查 ---")
        # 执行 Review
        report, full_prompt = agent.review_case(vision_findings, panderm_top5, img_path)

        print("\n--- Final Report (JSON) ---")
        print(json.dumps(report, indent=4))

        # 逻辑检查
        if report.get("PrimaryDiagnosis") == "Seborrheic Dermatitis":
            print("\n🔥 Test Result: SUCCESS! Agent rejected the incorrect Top-1 and pivoted to the correct diagnosis based on Prototype.")
        else:
            print("\n⚠️ Test Result: Agent followed the AI probability. Audit logic may need strengthening.")

    finally:
        agent.close()
