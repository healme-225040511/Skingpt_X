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

# --- 配置路径 ---
EVAL_DIR = "/225040511/project/Evaluation_Results/Dermnet/SkinGPT-X/"
TRAIN_FEATS_PATH = EVAL_DIR + "train_feats.npy"
TRAIN_JSON_PATH = EVAL_DIR + "train_files.json" # 全量 JSON，用于建立路径到索引的映射
IMAGE_BASE_DIR = "/225040511/Dataset/Dermnet/"

class KnowledgeBaseBuilder:
    def __init__(self, neo4j_uri, user, password, part_txt_path):
        
        # 2. 初始化模型和驱动
        self.driver = GraphDatabase.driver(
            neo4j_uri, 
            auth=(user, password),
            liveness_check_timeout=30 # 解决 SessionExpired 问题
        )
        self.vision_agent = VisionAgent()
        self.embed_model = HuggingFaceEmbedding(
            model_name="/225040511/project/hf_cache/bge-small-en-v1.5/BAAI/bge-small-en-v1___5"
        )
        
        # 3. 加载全量元数据（用于获取正确的 feature_vector 索引）
        self.all_train_feats = np.load(TRAIN_FEATS_PATH)
        with open(TRAIN_JSON_PATH, 'r') as f:
            all_files_list = json.load(f)
            # 建立 路径 -> 全局索引 的映射字典
            self.path_to_idx = {path: i for i, path in enumerate(all_files_list)}

        # 4. 加载当前分片（TXT 文件）
        with open(part_txt_path, 'r') as f:
            # 假设 TXT 每行一个路径
            self.my_files = [line.strip() for line in f.readlines() if line.strip()]
        
        # 5. 设置当前分片的缓存路径（防止多个进程写同一个 JSON 冲突）
        self.cache_path = os.path.join(EVAL_DIR, f"cache_{Path(part_txt_path).stem}.json")
        self.vision_results_cache = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r') as f:
                self.vision_results_cache = json.load(f)
            print(f"📦 Loaded existing cache with {len(self.vision_results_cache)} samples from {self.cache_path}")

    def run(self):
        print(f"🏃 Processing {len(self.my_files)} images...")
        
        for file_rel_path in tqdm(self.my_files, desc="Processing"):
            # 0. 获取该文件在全量数据中的原始索引
            if file_rel_path not in self.path_to_idx:
                print(f"⚠️ Warning: {file_rel_path} not found in master JSON. Skipping.")
                continue
            global_idx = self.path_to_idx[file_rel_path]

            # 1. 检查缓存
            if file_rel_path in self.vision_results_cache:
                key_findings = self.vision_results_cache[file_rel_path]["key_findings"]
            else:
                # 2. 调用 Vision Agent
                full_image_path = IMAGE_BASE_DIR + file_rel_path
                if not os.path.exists(full_image_path): continue
                
                try:
                    res = self.vision_agent.analyze(full_image_path)
                    key_findings = res.get("key_findings", "")
                    if not key_findings: continue
                    
                    # 存入缓存
                    self.vision_results_cache[file_rel_path] = {
                        "key_findings": key_findings,
                        "label": Path(file_rel_path).parent.name
                    }
                    
                    # 每 10 张图存一次盘
                    if len(self.vision_results_cache) % 10 == 0:
                        self._save_cache_to_disk()
                except Exception as e:
                    print(f"❌ Vision Error for {file_rel_path}: {e}")
                    continue

            # 3. 计算文本嵌入
            try:
                findings_embedding = self.embed_model.get_text_embedding(key_findings)
                # 4. 写入 Neo4j
                # self._save_to_neo4j(
                #     case_id=f"train_{global_idx}",
                #     image_path=file_rel_path,
                #     true_label=Path(file_rel_path).parent.name,
                #     key_findings=key_findings,
                #     feature_vector=self.all_train_feats[global_idx].tolist(),
                #     findings_embedding=findings_embedding
                # )
            except Exception as e:
                print(f"❌ DB/Embed Error for {file_rel_path}: {e}")

        self._save_cache_to_disk()
        print("✅ Part Complete.")

    def _save_cache_to_disk(self):
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.vision_results_cache, f, indent=2, ensure_ascii=False)

    def _save_to_neo4j(self, case_id, image_path, true_label, key_findings, feature_vector, findings_embedding):
        query = """
        MERGE (c:Case {image_path: $image_path})
        SET c.case_id = $case_id,
            c.true_label = $true_label,
            c.primary_diagnosis = $true_label,
            c.key_findings = $key_findings,
            c.feature_vector = $feature_vector,
            c.findings_embedding = $findings_embedding,
            c.source = 'training_set',
            c.is_correct = true,
            c.created_at = timestamp()
        """
        with self.driver.session() as session:
            session.run(query, 
                image_path=image_path, case_id=case_id, true_label=true_label,
                key_findings=key_findings, feature_vector=feature_vector, 
                findings_embedding=findings_embedding)

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