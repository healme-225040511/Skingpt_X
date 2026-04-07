import os
import json
import numpy as np
import argparse
import re
from pathlib import Path
from tqdm import tqdm
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from neo4j import GraphDatabase
from vision_agent import VisionAgent
from local_llm_utils import local_generate_deepseek_chat_response as generate_response_chat

class KnowledgeBaseBuilder:
    def __init__(
        self,
        neo4j_uri,
        user,
        password,
        eval_dir,
        image_base_dir,
        part_txt_path=None,
        evolution_threshold=20,
        use_sub_label=True,
        distill_recent_k=10,     # 本次蒸馏使用最近K个case
    ):
        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(user, password),
            liveness_check_timeout=30
        )
        self.vision_agent = VisionAgent()
        self.embed_model = HuggingFaceEmbedding(
            model_name="/225040511/project/hf_cache/bge-small-en-v1.5/BAAI/bge-small-en-v1___5"
        )

        self.eval_dir = eval_dir
        self.image_base_dir = image_base_dir
        self.use_sub_label = use_sub_label
        self.distill_recent_k = distill_recent_k

        train_feats_path = os.path.join(self.eval_dir, "train_feats.npy")
        train_json_path = os.path.join(self.eval_dir, "train_files.json")

        if os.path.exists(train_feats_path):
            self.all_train_feats = np.load(train_feats_path)
        else:
            self.all_train_feats = None

        if os.path.exists(train_json_path):
            with open(train_json_path, "r") as f:
                all_files_list = json.load(f)
                self.path_to_idx = {path: i for i, path in enumerate(all_files_list)}
        else:
            self.path_to_idx = {}

        self.my_files = []
        self.vision_results_cache = {}
        if part_txt_path:
            with open(part_txt_path, "r") as f:
                self.my_files = [line.strip() for line in f.readlines() if line.strip()]
            self.cache_path = os.path.join(self.eval_dir, f"cache_train_split.json")
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "r") as f:
                    self.vision_results_cache = json.load(f)

        self.evolution_threshold = evolution_threshold
        self.llm_model = "Qwen/Qwen-7B-Chat"

    def _extract_sub_label(self, image_path):
        if not self.use_sub_label:
            return None
        filename = Path(image_path).stem
        temp_name = filename.replace("-", " ")
        temp_name = re.sub(r"\d+", "", temp_name)
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

        f_emb = findings_embedding.tolist() if hasattr(findings_embedding, "tolist") else findings_embedding
        f_vec = feature_vector.tolist() if hasattr(feature_vector, "tolist") else feature_vector

        query = """
        // 1) Prototype
        MERGE (p:Prototype {disease: $true_label})
        ON CREATE SET
            p.summary = 'Initial knowledge state.',
            p.current_embedding = $findings_embedding,
            p.version_count = 0,
            p.updated_at = timestamp()

        // 2) 初始版本v0（仅创建一次）
        WITH p
        FOREACH (_ IN CASE WHEN p.version_count = 0 AND NOT (p)-[:HAS_SNAPSHOT]->() THEN [1] ELSE [] END |
            CREATE (p)-[:HAS_SNAPSHOT]->(:PrototypeVersion {
                version_idx: 0,
                summary: p.summary,
                embedding: p.current_embedding,
                delta_from_previous: 0.0,
                timestamp: timestamp(),
                used_case_count: 0,
                used_case_ids: []
            })
        )

        // 3) Case
        WITH p
        MERGE (c:Case {image_path: $image_path})
        SET c.case_id = $case_id,
            c.true_label = $true_label,
            c.sub_label = $sub_label,
            c.primary_diagnosis = $true_label,
            c.key_findings = $key_findings,
            c.feature_vector = $f_vec,
            c.findings_embedding = $f_emb,
            c.is_correct = true,
            c.created_at = timestamp()

        MERGE (c)-[:BELONGS_TO]->(p)
        """
        with self.driver.session() as session:
            session.run(
                query,
                image_path=image_path,
                case_id=case_id,
                true_label=true_label,
                sub_label=sub_label,
                key_findings=key_findings,
                f_vec=f_vec,
                f_emb=f_emb,
                findings_embedding=f_emb
            )

    def _check_and_evolve(self, disease_name):
        with self.driver.session() as session:
            result = session.run(
                "MATCH (c:Case {true_label: $name}) RETURN count(c) as cnt",
                name=disease_name
            ).single()
            count = result["cnt"] or 0
            if count >= self.evolution_threshold and count % self.evolution_threshold == 0:
                print(f"\n--- 🧬 Evolving Prototype for {disease_name} (Total cases: {count}) ---")
                self.distill_prototypes(disease_name)

    def cosine_similarity(self, v1, v2):
        if v1 is None or v2 is None:
            return 1.0
        v1, v2 = np.array(v1), np.array(v2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def distill_prototypes(self, disease_name):
        # 1) 拉取 Prototype + 最近K个Case（包含标识，后续用于建立 USED_CASE 关系）
        with self.driver.session() as session:
            res = session.run(f"""
                MATCH (p:Prototype {{disease: $name}})
                MATCH (c:Case {{true_label: $name}})
                WITH p, c ORDER BY c.created_at DESC LIMIT $k
                RETURN
                    p.summary as current_sum,
                    p.current_embedding as prev_emb,
                    p.version_count as v_idx,
                    collect({{
                        case_id: c.case_id,
                        image_path: c.image_path,
                        created_at: c.created_at,
                        key_findings: c.key_findings,
                        sub_label: c.sub_label
                    }}) as cases
            """, name=disease_name, k=int(self.distill_recent_k)).single()

        if not res or not res["cases"]:
            return

        old_summary = res["current_sum"] or ""
        prev_emb = res["prev_emb"]
        current_v_idx = res["v_idx"] or 0
        used_cases = res["cases"] or []
        used_cases = [x for x in used_cases if x and x.get("case_id")]

        if not used_cases:
            return

        cases_text_only = [x.get("key_findings", "") for x in used_cases if x.get("key_findings")]
        used_case_ids = [x["case_id"] for x in used_cases if x.get("case_id")]

        # 2) LLM 蒸馏
        prompt = f"""
You are a Medical Knowledge Architect. Evolve the diagnostic standard (Prototype) for '{disease_name}' by synthesizing insights across multiple recent cases.

[Current Standard]
{old_summary}

[Recent Verified Cases • key_findings only]
{json.dumps(cases_text_only, indent=2, ensure_ascii=False)}

Objective:
- After reviewing these cases, integrate cross-case shared features and critical differences.
- Enrich the feature inventory with precise dermatology lexicon (location + morphology).
- Highlight red flags/risk indicators and common pitfalls in differential diagnosis.
- Provide more insightful guidance for diagnosis and next-step workup (labs/imaging/biopsy).
- Produce a single cohesive clinical narrative that subsumes new observations into the prototype.

Constraints:
- Be concise but information-dense; avoid lists; write fluent clinical prose.
- Emphasize patterns that generalize across cases while noting differentiators.

Output format:
Return ONLY a JSON object: {{"summary": "Refined, cross-case, insight-rich paragraph here..."}}
""".strip()

        try:
            response = generate_response_chat(
                engine=self.llm_model,
                system_role="Expert Dermatologist",
                user_input=prompt,
                max_tokens=4096,
                temperature=0.2
            )
            new_summary = response.get("summary", "") if isinstance(response, dict) else str(response)
            new_summary = (new_summary or "").strip()
            if not new_summary:
                print("❌ Evolution Error: empty summary from LLM")
                return

            # 3) 计算 delta（相对于上一个版本的 embedding）
            new_embedding = self.embed_model.get_text_embedding(new_summary)

            delta_score = 0.0
            if prev_emb:
                similarity = self.cosine_similarity(new_embedding, prev_emb)
                delta_score = 1.0 - similarity

            # 4) 写入：创建新版本节点 + 建 USED_CASE 关系
            new_v_idx = int(current_v_idx) + 1

            with self.driver.session() as session:
                session.run("""
                    MATCH (p:Prototype {disease: $name})
                    SET p.summary = $summary,
                        p.current_embedding = $emb,
                        p.version_count = $new_idx,
                        p.updated_at = timestamp()

                    CREATE (p)-[:HAS_SNAPSHOT]->(pv:PrototypeVersion {
                        version_idx: $new_idx,
                        summary: $summary,
                        embedding: $emb,
                        delta_from_previous: $delta,
                        timestamp: timestamp(),
                        used_case_ids: $used_case_ids,
                        used_case_count: size($used_case_ids)
                    })

                    WITH pv
                    UNWIND $used_case_ids AS cid
                    MATCH (c:Case {case_id: cid})
                    MERGE (pv)-[:USED_CASE]->(c)
                """,
                name=disease_name,
                summary=new_summary,
                emb=new_embedding,
                new_idx=new_v_idx,
                delta=float(delta_score),
                used_case_ids=used_case_ids)

            print(f"✨ v{new_v_idx} 记录成功 | 变化改变量: {delta_score:.4f} | used_cases={len(used_case_ids)}")

        except Exception as e:
            print(f"❌ Evolution Error: {e}")

    def run(self):
        if not self.my_files:
            print("No files to process. Did you provide --txt?")
            return

        print(f"🏃 Processing {len(self.my_files)} images...")
        for file_rel_path in tqdm(self.my_files, desc="Processing"):
            if file_rel_path not in self.path_to_idx:
                print(f"⚠️ Skipping {file_rel_path}: not found in path_to_idx.")
                continue

            global_idx = self.path_to_idx[file_rel_path]
            true_label = Path(file_rel_path).parent.name

            # 1) 视觉描述
            if file_rel_path in self.vision_results_cache:
                key_findings = self.vision_results_cache[file_rel_path]["key_findings"]
            else:
                full_image_path = self.image_base_dir + file_rel_path
                if not os.path.exists(full_image_path):
                    print(f"❌ Image not found: {full_image_path}")
                    continue
                try:
                    res = self.vision_agent.analyze(full_image_path)
                    key_findings = res.get("key_findings", "")
                    if not key_findings:
                        continue
                    self.vision_results_cache[file_rel_path] = {"key_findings": key_findings, "label": true_label}
                except Exception as e:
                    print(f"❌ Vision Error for {file_rel_path}: {e}")
                    continue

            # 2) 若已存在则跳过
            if self._case_exists(file_rel_path):
                print(f"⏩ 跳过已存在 case: {file_rel_path}")
                continue

            # 3) 写入 DB
            try:
                findings_embedding = self.embed_model.get_text_embedding(key_findings)

                if self.all_train_feats is None:
                    raise RuntimeError("train_feats.npy not loaded; self.all_train_feats is None")
                # sub_label = self._extract_sub_label(file_rel_path)
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

        #     if len(self.vision_results_cache) % 1 == 0:
        #         self._save_cache_to_disk()

        # self._save_cache_to_disk()
        print("✅ Batch processing complete.")

    def _save_cache_to_disk(self):
        if hasattr(self, "cache_path"):
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.vision_results_cache, f, indent=2, ensure_ascii=False)

    def close(self):
        self.driver.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt", type=str, help="Path to the split TXT file for new images")
    parser.add_argument("--eval_dir", type=str, required=True, help="Directory containing train_feats.npy and train_files.json")
    parser.add_argument("--image_dir", type=str, required=True, help="Base directory where images are stored")
    parser.add_argument("--update_existing", action="store_true", help="Only add sub_label to existing database records")

    # 修复：bool flag 正确写法（默认 False，出现即 True）
    parser.add_argument("--use_sub_label", action="store_true", help="Extract sub_label from filename (flag)")

    # 可选：控制蒸馏窗口大小
    parser.add_argument("--distill_recent_k", type=int, default=10, help="Use last K cases when distilling")

    args = parser.parse_args()

    builder = KnowledgeBaseBuilder(
        neo4j_uri="bolt://100.91.219.110:7687",
        user="neo4j",
        password="Czty100165188",
        eval_dir=args.eval_dir,
        image_base_dir=args.image_dir,
        part_txt_path=args.txt,
        use_sub_label=args.use_sub_label,
        evolution_threshold=10,
        distill_recent_k=args.distill_recent_k
    )

    if args.update_existing:
        builder.update_sub_labels_only()
    else:
        builder.run()

    builder.close()
