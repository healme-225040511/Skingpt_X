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

# LlamaIndex 相关
from llama_index.core import Settings, QueryBundle
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.indices import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 工具函数
from local_llm_utils import local_generate_response as generate_response
from utils import process_markdown

class CaseReviewAgent:
    def __init__(self, 
                 model: str, 
                 neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 lancedb_uri: str, markdown_path: str,
                 train_feat_path: str, train_json_path: str,
                 test_feat_path: str=None, test_json_path: str=None,  # 👈 新增参数
                 evolution_threshold: int = 5):
        
        self.model = model
        
        # 1. 初始化 Neo4j (用于检索历史经验案例)
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # 2. 初始化静态知识库 (LlamaIndex + LanceDB)
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="/225040511/project/hf_cache/bge-small-en-v1.5/BAAI/bge-small-en-v1___5"
        )
        self.text_store = LanceDBVectorStore(uri=lancedb_uri, table_name="text_collection", mode="ro")
        self.vector_index = VectorStoreIndex.from_vector_store(self.text_store)
        self.static_retriever = self.vector_index.as_retriever()
        
        # 3. 加载特征映射表 (用于快速定位当前图像的向量)
        self.all_files_to_feats_map = {}
        self._load_reference_features(train_feat_path, train_json_path)
        self._load_reference_features(test_feat_path, test_json_path, split_name="Test")

        self.evolution_threshold = evolution_threshold

    def _load_reference_features(self, feat_path, json_path, split_name="Train"):
        try:
            if not os.path.exists(feat_path) or not os.path.exists(json_path):
                print(f"⚠️ Warning: {split_name} files not found at {feat_path}")
                return

            feats = np.load(feat_path)
            with open(json_path, 'r') as f:
                files = json.load(f)
            
            # 使用 update 确保不覆盖之前加载的数据（除非文件名冲突）
            new_data = dict(zip(files, feats))
            self.all_files_to_feats_map.update(new_data)
            
            print(f"✅ CaseReviewAgent: Loaded {len(files)} {split_name} features.")
        except Exception as e:
            print(f"❌ Error loading {split_name} features: {e}")

    # ================= 核心分析接口 =================


    def review_case(self, 
                    vision_key_findings: str, 
                    panderm_top5: List[Dict[str, Any]], 
                    image_path: str) -> Dict:
        """
        综合诊断审查（双路检索优化版）
        """
        # 1. 双路检索历史病例：物理相似(5个) + 语义相似(5个)
        print("image_path:", image_path)
        current_feat = self._get_feat_by_path(image_path)
        hybrid_cases = self._find_hybrid_historical_cases(current_feat, vision_key_findings)
        print(f"🔍 Retrieved {len(hybrid_cases)} hybrid similar cases from Neo4j.")
        # print(hybrid_cases)
        print("-----")
        # 2. 专家检索 (静态 Handbook)
        expert_knowledge_context = ""
        for item in panderm_top5[:5]:
            disease = item['disease']
            knowledge = self._retrieve_static_knowledge(disease)
            expert_knowledge_context += f"\n--- Handbook for {disease} ---\n{knowledge}\n"
        print("🔍 Retrieved expert knowledge from static handbook.")
        # print(expert_knowledge_context)
        print("-----")
        # 3. 构建 Prompt 并调用 LLM
        prompt = self._build_comprehensive_prompt(
            vision_findings=vision_key_findings,
            top5=panderm_top5,
            similar_cases=hybrid_cases,
            expert_knowledge=expert_knowledge_context
        )
        print("📝 Constructed comprehensive prompt for LLM.")
        print(prompt)
        response_raw = generate_response(
            engine=self.model,
            temperature=0.1,
            max_tokens=4096,
            system_role="Senior Clinical Dermatologist & Auditor",
            user_input=prompt
        )
        print("🤖 Received response from LLM.")
        print(response_raw)
        parsed_res = self._parse_json_response(response_raw)
        
        return parsed_res, prompt

    # ================= 核心双路检索逻辑 =================

    def _find_hybrid_historical_cases(self, current_feat, findings_text, k=5):
        """
        双路检索：
        1. 基于图像 Feature Vector 的物理相似度 (k个)
        2. 基于 KeyFindings 文本嵌入的语义相似度 (k个)
        """
        all_retrieved_cases = {} # 使用 dict 通过 case_id 去重
        # print("🔍 Starting hybrid retrieval of historical cases...")
        # print(current_feat)
        with self.driver.session() as session:
            # 获取库中所有带向量和描述的 Case
            records = list(session.run("""
                MATCH (c:Case) 
                WHERE c.feature_vector IS NOT NULL AND c.key_findings IS NOT NULL
                RETURN c
            """))
            
            if not records: return []

            # --- 第一路：物理特征检索 (Image Feature Vector) ---
            if current_feat is not None:
                query_feat = np.array(current_feat).reshape(1, -1)
                phys_scored = []
                for r in records:
                    node = r['c']
                    db_vec = np.array(node['feature_vector']).reshape(1, -1)
                    if db_vec.shape == query_feat.shape:
                        score = cosine_similarity(query_feat, db_vec)[0][0]
                        phys_scored.append((score, node))
                
                # 取前 k 个
                phys_scored.sort(key=lambda x: x[0], reverse=True)
                for score, node in phys_scored[:k]:
                    all_retrieved_cases[node['case_id']] = {
                        "type": "Physically Similar",
                        "score": score,
                        "diagnosis": node['true_label'] or node['primary_diagnosis'],
                        "findings": node['key_findings']
                    }
            print(f"🔍 Physically similar cases found: {len(all_retrieved_cases)}")
            # --- 第二路：语义特征检索 (Text Embedding of KeyFindings) ---
            # 1. 将当前的 KeyFindings 文本转为向量
            # query_text_vec = np.array(Settings.embed_model.get_text_embedding(findings_text)).reshape(1, -1)
            
            # sem_scored = []
            # for r in records:
            #     node = r['c']
            #     # 注意：这里需要历史病例也存了 key_findings 的 embedding
            #     # 如果没存，我们现场算（如果历史病例不多的话）或者从字段中提取
            #     # 假设你之前存入时也将文本转为了向量存入 c.findings_embedding 字段
            #     if 'findings_embedding' in node and node['findings_embedding']:
            #         db_text_vec = np.array(node['findings_embedding']).reshape(1, -1)
            #         score = cosine_similarity(query_text_vec, db_text_vec)[0][0]
            #         sem_scored.append((score, node))
            #     else:
            #         # 如果库里没存文本向量，可以退而求其次使用简单的文本模糊匹配分数
            #         score = fuzz.token_set_ratio(findings_text, node['key_findings']) / 100.0
            #         sem_scored.append((score, node))

            # # 取前 k 个
            # sem_scored.sort(key=lambda x: x[0], reverse=True)
            # for score, node in sem_scored[:k]:
            #     cid = node['case_id']
            #     if cid in all_retrieved_cases:
            #         all_retrieved_cases[cid]['type'] = "Hybrid Match (Phys+Sem)"
            #     else:
            #         all_retrieved_cases[cid] = {
            #             "type": "Clinically Similar (Textual)",
            #             "score": score,
            #             "diagnosis": node['true_label'] or node['primary_diagnosis'],
            #             "findings": node['key_findings']
            #         }
        return list(all_retrieved_cases.values())

    # ================= 修改后的 Prompt 构建 =================

    def _build_comprehensive_prompt(self, vision_findings, top5, similar_cases, expert_knowledge):
        # 分门别类地向 LLM 展示参考案例
        history_str = ""
        for c in similar_cases:
            history_str += (f"- [{c['type']}] Diagnosis: {c['diagnosis']} (Score: {c['score']:.2f})\n"
                            f"  Past Findings: {c['findings']}\n")

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

    def _get_feat_by_path(self, path):
        # 尝试完整路径、文件名等匹配逻辑
        return self.all_files_to_feats_map.get(path)

    def _retrieve_static_knowledge(self, disease_name: str):
        """从 LlamaIndex 检索静态手册知识"""
        nodes = self.static_retriever.retrieve(QueryBundle(f"Diagnostic criteria and visual features of {disease_name}"))
        return "\n".join([n.node.text for n in nodes[:3]])

    def _find_similar_historical_cases(self, current_feat, top_k=3):
        """从 Neo4j 寻找历史病例"""
        if current_feat is None: return []
        
        with self.driver.session() as session:
            # 此处逻辑：拉取库中所有带向量的 Case，计算余弦相似度
            records = session.run("MATCH (c:Case) WHERE c.feature_vector IS NOT NULL RETURN c")
            
            scored_cases = []
            query_vec = np.array(current_feat).reshape(1, -1)
            
            for r in records:
                node = r['c']
                db_vec = np.array(node['feature_vector']).reshape(1, -1)
                if db_vec.shape == query_vec.shape:
                    score = cosine_similarity(query_vec, db_vec)[0][0]
                    scored_cases.append({
                        "score": score,
                        "diagnosis": node['true_label'] or node['primary_diagnosis'],
                        "findings": node['key_findings']
                    })
            
            scored_cases.sort(key=lambda x: x['score'], reverse=True)
            return scored_cases[:top_k]


    def _parse_json_response(self, text):
        # 1. 如果 text 已经是字典了，直接返回，不需要解析
        if isinstance(text, dict):
            return text

        # 2. 如果 text 不是字符串，强转成字符串（或者报错）
        if not isinstance(text, str):
            print(f"⚠️ Warning: Input is not a string, but {type(text)}")
            text = str(text)

        try:
            # 3. 兼容处理带 Markdown 代码块的情况
            # 先清理掉可能存在的 ```json 和 ```
            clean_text = text.strip()
            if clean_text.startswith("```"):
                # 提取代码块内部的内容
                match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', clean_text, re.DOTALL)
                if match:
                    clean_text = match.group(1)

            # 4. 尝试标准解析
            try:
                return json.loads(clean_text)
            except json.JSONDecodeError:
                # 5. 如果标准解析失败，再尝试提取中间的 { }
                match = re.search(r'\{.*\}', clean_text, re.DOTALL)
                if match:
                    # 有些 LLM 会输出单引号的 "JSON"，这不符合标准，尝试替换
                    json_str = match.group().replace("'", '"')
                    return json.loads(json_str)
                else:
                    raise ValueError("No JSON object found")

        except Exception as e:
            print(f"❌ JSON Parse Error: {e}")
            # 如果彻底失败，把原始文本存入 Evidence
            return {
                "KeyFindings": "Parse failed", 
                "PrimaryDiagnosis": "Error", 
                "Evidence": str(text)
            }
        
    def close(self):
        self.driver.close()

# 使用示例
if __name__ == "__main__":
    # 配置参数
    agent = CaseReviewAgent(
        model="Qwen3-VL-30B",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        lancedb_uri="/path/to/lancedb",
        markdown_path="./skin_handbook.md",
        train_feat_path="./train_feats.npy",
        train_json_path="./train_files.json"
    )

    # 模拟输入
    vision_findings = "Erythematous plaque with silvery scale, well-defined borders on the extensor surface..."
    panderm_top5 = [
        {"disease": "Psoriasis", "probability": 0.85},
        {"disease": "Eczema", "probability": 0.10},
        {"disease": "Lichen Planus", "probability": 0.05}
    ]
    img_path = "sample_image_01.jpg"

    # 执行 Review
    report = agent.review_case(vision_findings, panderm_top5, img_path)
    print(json.dumps(report, indent=2))