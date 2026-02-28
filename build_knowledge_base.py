import os
import json
import numpy as np
import torch
import argparse
import re
from pathlib import Path
from tqdm import tqdm
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from neo4j import GraphDatabase
from vision_agent import VisionAgent 
# 引入 LLM 接口用于知识蒸馏
from local_llm_utils import local_generate_deepseek_chat_response as generate_response_chat

class KnowledgeBaseBuilder:
    def __init__(self, neo4j_uri, user, password, eval_dir, image_base_dir, 
                 part_txt_path=None, evolution_threshold=20, use_sub_label=True):
        # 初始化数据库驱动
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(user, password), liveness_check_timeout=30)
        self.vision_agent = VisionAgent()
        self.embed_model = HuggingFaceEmbedding(model_name="/225040511/project/hf_cache/bge-small-en-v1.5/BAAI/bge-small-en-v1___5")
        
        self.eval_dir = eval_dir
        self.image_base_dir = image_base_dir
        self.use_sub_label = use_sub_label  # 控制开关

        # 构建内部路径
        train_feats_path = os.path.join(self.eval_dir, "train_feats.npy")
        train_json_path = os.path.join(self.eval_dir, "train_files.json")

        # 加载特征和路径索引
        if os.path.exists(train_feats_path):
            self.all_train_feats = np.load(train_feats_path)
        if os.path.exists(train_json_path):
            with open(train_json_path, 'r') as f:
                all_files_list = json.load(f)
                self.path_to_idx = {path: i for i, path in enumerate(all_files_list)}

        # 如果提供了 txt 路径，则加载待处理文件列表
        self.my_files = []
        if part_txt_path:
            with open(part_txt_path, 'r') as f:
                self.my_files = [line.strip() for line in f.readlines() if line.strip()]
            self.cache_path = os.path.join(self.eval_dir, f"cache_{Path(part_txt_path).stem}.json")
            self.vision_results_cache = {}
            if os.path.exists(self.cache_path):
                with open(self.cache_path, 'r') as f:
                    self.vision_results_cache = json.load(f)

        # 演化配置
        self.evolution_threshold = evolution_threshold
        self.llm_model = "Qwen/Qwen-7B-Chat"

    def _extract_sub_label(self, image_path):
        """如果开启了开关，则从文件名提取子标签，否则返回 None"""
        if not self.use_sub_label:
            return None
            
        filename = Path(image_path).stem
        temp_name = filename.replace('-', ' ')
        temp_name = re.sub(r'\d+', '', temp_name)
        sub_label = " ".join(temp_name.split()).strip()
        return sub_label

    def update_sub_labels_only(self):
        if not self.use_sub_label:
            print("⚠️ use_sub_label is False. Skipping update.")
            return

        print("🔍 Fetching all cases from database to update sub_label...")
        fetch_query = "MATCH (c:Case) RETURN c.image_path AS path"
        with self.driver.session() as session:
            result = session.run(fetch_query)
            all_paths = [record["path"] for record in result]

        if not all_paths:
            print("No cases found in database.")
            return

        print(f"🔄 Processing {len(all_paths)} records...")
        update_query = """
        UNWIND $data AS item
        MATCH (c:Case {image_path: item.path})
        SET c.sub_label = item.sub_label
        """
        batch_size = 100
        for i in tqdm(range(0, len(all_paths), batch_size)):
            batch = all_paths[i : i + batch_size]
            payload = [{"path": p, "sub_label": self._extract_sub_label(p)} for p in batch]
            with self.driver.session() as session:
                session.run(update_query, data=payload)
        print(f"✅ Successfully updated sub_label for {len(all_paths)} cases.")

    def _case_exists(self, image_path):
        query = "MATCH (c:Case {image_path: $path}) RETURN count(c) > 0 AS exists"
        with self.driver.session() as session:
            result = session.run(query, path=image_path).single()
            return result["exists"]

    def _save_to_neo4j(self, case_id, image_path, true_label, key_findings, feature_vector, findings_embedding):
        sub_label = self._extract_sub_label(image_path)
        
        # Cypher 逻辑：如果 sub_label 为 null，Neo4j 会自动忽略或移除该属性
        query = """
        MERGE (p:Prototype {disease: $true_label})
        ON CREATE SET p.summary = 'Initial knowledge state.', p.updated_at = timestamp()
        
        MERGE (c:Case {image_path: $image_path})
        SET c.case_id = $case_id,
            c.true_label = $true_label,
            c.sub_label = $sub_label,
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
                sub_label=sub_label, key_findings=key_findings, 
                feature_vector=feature_vector, findings_embedding=findings_embedding)

    def _check_and_evolve(self, disease_name):
        with self.driver.session() as session:
            result = session.run(
                "MATCH (c:Case {true_label: $name}) RETURN count(c) as cnt", 
                name=disease_name
            ).single()
            count = result["cnt"]
            if count >= self.evolution_threshold and count % self.evolution_threshold == 0:
                print(f"\n--- 🧬 Evolving Prototype for {disease_name} (Total cases: {count}) ---")
                self.distill_prototypes(disease_name)

    def distill_prototypes(self, disease_name):
        with self.driver.session() as session:
            records = session.run("""
                MATCH (c:Case {true_label: $name})
                RETURN c.key_findings as kf
                ORDER BY c.created_at DESC LIMIT 10
            """, name=disease_name)
            cases_text = [r["kf"] for r in records]
            
            old_summary_res = session.run(
                "MATCH (p:Prototype {disease: $name}) RETURN p.summary as s", 
                name=disease_name
            ).single()
            old_summary = old_summary_res["s"] if old_summary_res else "None"

        if not cases_text: return

        prompt = f"""
        You are a Medical Knowledge Architect. Update the diagnostic standard (Prototype) for '{disease_name}'.
        [Current Standard]: {old_summary}
        [Recent Verified Cases]: {json.dumps(cases_text, indent=2)}
        Task: Refine the core visual features of {disease_name}. Focus on consistent patterns.
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
            with self.driver.session() as session:
                session.run("""
                    MATCH (p:Prototype {disease: $name})
                    SET p.summary = $summary, p.updated_at = timestamp()
                """, name=disease_name, summary=new_summary)
        except Exception as e:
            print(f"❌ Evolution Error for {disease_name}: {e}")

    def run(self):
        if not self.my_files:
            print("No files to process. Did you provide --txt?")
            return

        print(f"🏃 Processing {len(self.my_files)} images...")
        for file_rel_path in tqdm(self.my_files, desc="Processing"):
            if not hasattr(self, 'path_to_idx') or file_rel_path not in self.path_to_idx: 
                print(f"⚠️ Skipping {file_rel_path}: not found in path_to_idx.");
                continue
            global_idx = self.path_to_idx[file_rel_path]
            true_label = Path(file_rel_path).parent.name

            # if self._case_exists(file_rel_path):
            #     continue 
            # 1. 获取视觉描述
            if file_rel_path in self.vision_results_cache:
                print(f"Using cached analysis for: {file_rel_path}")
                key_findings = self.vision_results_cache[file_rel_path]["key_findings"]
            else:
                full_image_path = self.image_base_dir + file_rel_path
                print(f"Analyzing image: {full_image_path}")
                if not os.path.exists(full_image_path):
                    print(f"❌ Image not found: {full_image_path}"); 
                    continue
                try:
                    res = self.vision_agent.analyze(full_image_path)
                    key_findings = res.get("key_findings", "")
                    if not key_findings: continue
                    self.vision_results_cache[file_rel_path] = {"key_findings": key_findings, "label": true_label}
                    print(f"Analyzing image done: {full_image_path}")
                except Exception as e:
                    print(f"❌ Vision Error for {file_rel_path}: {e}"); continue

            # 2. 写入数据库
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
                self._check_and_evolve(true_label)
            except Exception as e:
                print(f"❌ DB Error: {e}")

            if len(self.vision_results_cache) % 1 == 0:
                self._save_cache_to_disk()

        self._save_cache_to_disk()
        print("✅ Batch processing complete.")

    def _save_cache_to_disk(self):
        if hasattr(self, 'cache_path'):
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.vision_results_cache, f, indent=2, ensure_ascii=False)

    def close(self):
        self.driver.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt", type=str, help="Path to the split TXT file for new images")
    parser.add_argument("--eval_dir", type=str, required=True, help="Directory containing train_feats.npy and train_files.json")
    parser.add_argument("--image_dir", type=str, required=True, help="Base directory where images are stored")
    parser.add_argument("--update_existing", action="store_true", help="Only add sub_label to existing database records")
    # 新增参数：使用 action="store_true" 默认为 False，只有命令行出现该 flag 时才为 True
    parser.add_argument("--use_sub_label", type=bool, help="Whether to extract sub_label from filename")
    args = parser.parse_args()

    builder = KnowledgeBaseBuilder(
        neo4j_uri="bolt://100.91.219.86:7687",
        user="neo4j",
        password="Czty100165188", 
        eval_dir=args.eval_dir,
        image_base_dir=args.image_dir,
        part_txt_path=args.txt,
        use_sub_label=args.use_sub_label,
        evolution_threshold=10
    )

    if args.update_existing:
        builder.update_sub_labels_only()
    else:
        builder.run()

    builder.close()