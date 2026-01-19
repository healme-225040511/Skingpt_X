import os
import json
import numpy as np
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from neo4j import GraphDatabase
from vision_agent import VisionAgent 
# 引入 LLM 接口用于知识蒸馏 (请根据你的 utils 文件名调整)
from local_llm_utils import local_generate_deepseek_chat_response as generate_response_chat

# --- 配置路径 ---
EVAL_DIR = "/225040511/project/Evaluation_Results/Dermnet/SkinGPT-X/"
TRAIN_FEATS_PATH = EVAL_DIR + "train_feats.npy"
TRAIN_JSON_PATH = EVAL_DIR + "train_files.json" 
IMAGE_BASE_DIR = "/225040511/Dataset/Dermnet/"

class KnowledgeBaseBuilder:
    def __init__(self, neo4j_uri, user, password, part_txt_path, evolution_threshold=20):
        # ... 原有的初始化代码 ...
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(user, password), liveness_check_timeout=30)
        self.vision_agent = VisionAgent()
        self.embed_model = HuggingFaceEmbedding(model_name="/225040511/project/hf_cache/bge-small-en-v1.5/BAAI/bge-small-en-v1___5")
        
        self.all_train_feats = np.load(TRAIN_FEATS_PATH)
        with open(TRAIN_JSON_PATH, 'r') as f:
            all_files_list = json.load(f)
            self.path_to_idx = {path: i for i, path in enumerate(all_files_list)}

        with open(part_txt_path, 'r') as f:
            self.my_files = [line.strip() for line in f.readlines() if line.strip()]
        
        self.cache_path = os.path.join(EVAL_DIR, f"cache_{Path(part_txt_path).stem}.json")
        self.vision_results_cache = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r') as f:
                self.vision_results_cache = json.load(f)

        # 新增：演化阈值和模型名称
        self.evolution_threshold = evolution_threshold
        self.llm_model = "Qwen/Qwen-7B-Chat" # 或其他你使用的模型
    def _case_exists(self, image_path):
        """检查数据库中是否已存在该病例"""
        query = "MATCH (c:Case {image_path: $path}) RETURN count(c) > 0 AS exists"
        with self.driver.session() as session:
            result = session.run(query, path=image_path).single()
            return result["exists"]
    def run(self):
        print(f"🏃 Processing {len(self.my_files)} images...")
        
        for file_rel_path in tqdm(self.my_files, desc="Processing"):
            if file_rel_path not in self.path_to_idx: continue
            global_idx = self.path_to_idx[file_rel_path]
            true_label = Path(file_rel_path).parent.name


            if self._case_exists(file_rel_path):
                # 虽然病例存在，但仍需检查该疾病的 Prototype 是否需要演化更新
                continue 
            # 1. 获取视觉描述 (从缓存或 VisionAgent)
            if file_rel_path in self.vision_results_cache:
                key_findings = self.vision_results_cache[file_rel_path]["key_findings"]
            else:
                full_image_path = IMAGE_BASE_DIR + file_rel_path
                if not os.path.exists(full_image_path): continue
                try:
                    res = self.vision_agent.analyze(full_image_path)
                    key_findings = res.get("key_findings", "")
                    if not key_findings: continue
                    self.vision_results_cache[file_rel_path] = {"key_findings": key_findings, "label": true_label}
                except Exception as e:
                    print(f"❌ Vision Error: {e}"); continue

            # 2. 写入数据库并建立 Prototype 关联
            try:
                findings_embedding = self.embed_model.get_text_embedding(key_findings)
                self._save_to_neo4j(
                    case_id=f"train_{global_idx}",
                    image_path=file_rel_path,
                    true_label=true_label,
                    key_findings=key_findings,
                    feature_vector=self.all_train_feats[global_idx].tolist(),
                    findings_embedding=findings_embedding
                )

                # 3. 检查是否触发知识演化 (Evolution)
                self._check_and_evolve(true_label)

            except Exception as e:
                print(f"❌ DB Error: {e}")

            if len(self.vision_results_cache) % 10 == 0:
                self._save_cache_to_disk()

        self._save_cache_to_disk()
        print("✅ Part Complete.")

    def _save_to_neo4j(self, case_id, image_path, true_label, key_findings, feature_vector, findings_embedding):
        """修改后的写入逻辑：同时维护 Case 和 Prototype 及其关系"""
        query = """
        MERGE (p:Prototype {disease: $true_label})
        ON CREATE SET p.summary = 'Initial knowledge state.', p.updated_at = timestamp()
        
        MERGE (c:Case {image_path: $image_path})
        SET c.case_id = $case_id,
            c.true_label = $true_label,
            c.primary_diagnosis = $true_label,
            c.key_findings = $key_findings,
            c.feature_vector = $feature_vector,
            c.findings_embedding = $findings_embedding,
            c.is_correct = true,
            c.created_at = timestamp()
            
        MERGE (c)-[:BELONGS_TO]->(p)
        """
        with self.driver.session() as session:
            session.run(query, 
                image_path=image_path, case_id=case_id, true_label=true_label,
                key_findings=key_findings, feature_vector=feature_vector, 
                findings_embedding=findings_embedding)

    def _check_and_evolve(self, disease_name):
        """检查病例数是否达标，达标则蒸馏知识"""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (c:Case {true_label: $name}) RETURN count(c) as cnt", 
                name=disease_name
            ).single()
            
            count = result["cnt"]
            if count >= self.evolution_threshold and count % self.evolution_threshold == 0:
                print(f"--- 🧬 Evolving Prototype for {disease_name} (Total cases: {count}) ---")
                self.distill_prototypes(disease_name)

    def distill_prototypes(self, disease_name):
        """调用 LLM 总结该疾病的历史特征"""
        with self.driver.session() as session:
            # 1. 获取该疾病最新的 10 个病例描述
            records = session.run("""
                MATCH (c:Case {true_label: $name})
                RETURN c.key_findings as kf
                ORDER BY c.created_at DESC LIMIT 10
            """, name=disease_name)
            cases_text = [r["kf"] for r in records]
            
            # 2. 获取旧的总结
            old_summary_res = session.run(
                "MATCH (p:Prototype {disease: $name}) RETURN p.summary as s", 
                name=disease_name
            ).single()
            old_summary = old_summary_res["s"] if old_summary_res else "None"

        if not cases_text: return

        # 3. 构建 Prompt
        prompt = f"""
        You are a Medical Knowledge Architect. Update the diagnostic standard (Prototype) for '{disease_name}'.
        
        [Current Standard]: {old_summary}
        [Recent Verified Cases]: {json.dumps(cases_text, indent=2)}
        
        Task: Based on these real cases, refine the core visual features of {disease_name}. 
        Focus on consistent patterns (color, shape, scale type, location).
        
        Output format: Return ONLY a JSON object: {{"summary": "Refined paragraph here..."}}
        """
        
        try:
            response = generate_response_chat(
                engine=self.llm_model, 
                system_role="Expert Dermatologist", 
                user_input=prompt,
                max_tokens=4096,
                temperature=0.2
            )
            new_summary = response.get("summary", "") if isinstance(response, dict) else str(response)
            
            # 4. 更新回 Neo4j
            with self.driver.session() as session:
                session.run("""
                    MATCH (p:Prototype {disease: $name})
                    SET p.summary = $summary, p.updated_at = timestamp()
                """, name=disease_name, summary=new_summary)
        except Exception as e:
            print(f"❌ Evolution Error for {disease_name}: {e}")

    def _save_cache_to_disk(self):
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.vision_results_cache, f, indent=2, ensure_ascii=False)

    def close(self):
        self.driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt", type=str, required=True, help="Path to the split TXT file")
    args = parser.parse_args()

    builder = KnowledgeBaseBuilder(
        neo4j_uri="bolt://127.0.0.1:7687",
        user="neo4j",
        password="Czty100165188",
        part_txt_path=args.txt,
    )
    builder.run()
    builder.close()