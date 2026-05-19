import os
import json
import re
import uuid
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import fuzz
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# LlamaIndex 相关
from llama_index.core import Settings, QueryBundle
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.indices import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from Constants import RDD_DISEASE_NAME, DDI_DISEASE_NAME
# 工具函数
from local_llm_utils import BGE_MODEL_PATH
from local_llm_utils import local_generate_response as generate_response
from local_llm_utils import local_generate_deepseek_chat_response as generate_response_chat
from utils import process_markdown

class CaseReviewAgent:
    def __init__(self, 
                 model: str, 
                 neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 lancedb_uri: str, markdown_path: str,
                 train_feat_path: str, train_json_path: str,
                 test_feat_path: str=None, test_json_path: str=None,
                 evolution_threshold: int = 5):
        
        self.model = model
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=str(BGE_MODEL_PATH)
        )
        self.text_store = LanceDBVectorStore(uri=lancedb_uri, table_name="text_collection", mode="ro")
        self.vector_index = VectorStoreIndex.from_vector_store(self.text_store)
        self.static_retriever = self.vector_index.as_retriever()
        
        self.all_files_to_feats_map = {}
        self._load_reference_features(train_feat_path, train_json_path)
        if test_feat_path:
            self._load_reference_features(test_feat_path, test_json_path, split_name="Test")

        self.evolution_threshold = evolution_threshold

    # ================= 辅助函数 =================

    def _extract_sub_label(self, path: str) -> str:
        """从文件名提取子类标签 (例如: acne-cystic-10.jpg -> acne cystic)"""
        filename = Path(path).stem
        name = filename.replace('-', ' ')
        name = re.sub(r'\d+', '', name)
        return " ".join(name.split()).strip()

    def _load_reference_features(self, feat_path, json_path, split_name="Train"):
        if not feat_path or not os.path.exists(feat_path): return
        feats = np.load(feat_path)
        with open(json_path, 'r') as f:
            files = json.load(f)
        self.all_files_to_feats_map.update(dict(zip(files, feats)))
        print(f"✅ Loaded {len(files)} {split_name} features.")

    def _get_feat_by_path(self, path):
        if path in self.all_files_to_feats_map:
            return self.all_files_to_feats_map[path]
        return self.all_files_to_feats_map.get(path)

    
    # ================= 2. 主审查逻辑 =================
    def review_sub_class(self, 
                        vision_key_findings: str, 
                        main_top5: List[Dict[str, Any]],  
                        sub_top5: List[Dict[str, Any]],   
                        image_path: str, full_image_path: str) -> Dict:
        # 新增：参数校验
        if not main_top5:
            return {"error": "main_top5 cannot be empty"}, ""
        if not sub_top5:
            return {"error": "sub_top5 cannot be empty"}, ""

        # 1. 检索历史病例（含子类标签）
        current_feat = self._get_feat_by_path(image_path)
        print(f"Current feature shape: {current_feat.shape if current_feat is not None else 'N/A'}")
        historical_cases = self._find_sub_historical_cases(current_feat, sub_top5)

        # 2. 检索大类知识（静态手册 + 动态原型）
        top1_main = main_top5[0]['disease']
        static_knowledge = self._retrieve_static_knowledge(top1_main)
        evolved_prototype = "No evolved knowledge yet."
        with self.driver.session() as session:
            res = session.run("MATCH (p:Prototype {disease: $name}) RETURN p.summary as s", name=top1_main).single()
            if res and res["s"]:
                evolved_prototype = res["s"]

        # 3. 调用独立的子类Prompt构建函数
        prompt = self._build_subclass_prompt(
            vision_findings=vision_key_findings,
            main_top5=main_top5,
            sub_top5=sub_top5,
            historical_cases=historical_cases,
            static_knowledge=static_knowledge
        )
        print("=== Sub-class Prompt ===")
        print(prompt)
        print(image_path)
        # 4. 调用LLM生成子类诊断结果（新增异常捕获）
        try:
            response_raw = generate_response(
                temperature=0.1,
                max_tokens=4096,
                system_role="Dermatology Sub-category Specialist",
                user_input=prompt,
                image_path=full_image_path
            )
            parsed_res = self._parse_json_response(response_raw)
            # 新增：校验输出格式
            required_keys = ["PrimaryDiagnosis", "SubDiagnosis", "Reasoning"]
            if not all(key in parsed_res for key in required_keys):
                parsed_res["error"] = "Missing required keys in response"
            return parsed_res, prompt
        except Exception as e:
            return {"error": f"LLM inference failed: {str(e)}"}, prompt
    def _find_historical_cases_in(self, current_feat, top_predictions, total_k=6):
        """
        修改后的检索逻辑：
        1. 按照 Top 3 的 sub_label 从数据库中粗筛候选集。
        2. 计算所有候选病例与当前特征的相似度。
        3. 统一排序，选出相似度最高的 Top 6 个病例。
        
        :param current_feat: 当前图像的特征向量
        :param top_predictions: AI预测的子分类列表 (sorted by prob)
        :param total_k: 最终选取的总病例数 (默认 6)
        """
        if current_feat is None or not top_predictions:
            return []

        # 1. 提取预测概率最高的 Top 3 子分类名称
        target_sub_classes = [p['disease'] for p in top_predictions[:3]]
        query_feat = np.array(current_feat).reshape(1, -1)
        
        # 存放所有候选病例的相似度计算结果
        all_candidates = []

        with self.driver.session() as session:
            # 2. 粗筛：拉取属于这 3 个子分类的所有病例
            cypher = """
            MATCH (c:Case) 
            WHERE c.true_label IN $classes AND c.feature_vector IS NOT NULL 
            RETURN c
            """
            records = list(session.run(cypher, classes=target_sub_classes))
            
            if not records:
                return []

            # 3. 对粗筛出的所有记录进行相似度计算
            for r in records:
                node = r['c']
                db_vec = np.array(node['feature_vector']).reshape(1, -1)
                
                # 确保向量维度匹配
                if db_vec.shape == query_feat.shape:
                    score = float(cosine_similarity(query_feat, db_vec)[0][0])
                    
                    # 将结果存入大列表，不分桶
                    all_candidates.append({
                        "score": score,
                        "diagnosis": node.get('true_label') or node.get('primary_diagnosis', 'N/A'),
                        "sub_diagnosis": node.get('sub_label', 'N/A'),
                        "findings": node.get('key_findings', 'No description available.')
                    })

        # 4. 全局排序：按照相似度分数降序排列
        all_candidates.sort(key=lambda x: x['score'], reverse=True)

        # 5. 提取相似度最高的 Top k 个病例
        # 对 score 进行格式化处理
        retrieved_results = []
        for item in all_candidates[:total_k]:
            item["score"] = round(item["score"], 5)
            retrieved_results.append(item)

        return retrieved_results
    def _find_sub_historical_cases(self, current_feat, top_sub_predictions, total_k=6):
        """
        修改后的检索逻辑：
        1. 按照 Top 3 的 sub_label 从数据库中粗筛候选集。
        2. 计算所有候选病例与当前特征的相似度。
        3. 统一排序，选出相似度最高的 Top 6 个病例。
        
        :param current_feat: 当前图像的特征向量
        :param top_sub_predictions: AI预测的子分类列表 (sorted by prob)
        :param total_k: 最终选取的总病例数 (默认 6)
        """
        if current_feat is None or not top_sub_predictions:
            return []

        # 1. 提取预测概率最高的 Top 3 子分类名称
        target_sub_classes = [p['disease'] for p in top_sub_predictions[:3]]
        query_feat = np.array(current_feat).reshape(1, -1)
        
        # 存放所有候选病例的相似度计算结果
        all_candidates = []

        with self.driver.session() as session:
            # 2. 粗筛：拉取属于这 3 个子分类的所有病例
            cypher = """
            MATCH (c:Case) 
            WHERE c.sub_label IN $classes AND c.feature_vector IS NOT NULL 
            RETURN c
            """
            records = list(session.run(cypher, classes=target_sub_classes))
            
            if not records:
                return []

            # 3. 对粗筛出的所有记录进行相似度计算
            for r in records:
                node = r['c']
                db_vec = np.array(node['feature_vector']).reshape(1, -1)
                
                # 确保向量维度匹配
                if db_vec.shape == query_feat.shape:
                    score = float(cosine_similarity(query_feat, db_vec)[0][0])
                    
                    # 将结果存入大列表，不分桶
                    all_candidates.append({
                        "score": score,
                        "diagnosis": node.get('true_label') or node.get('primary_diagnosis', 'N/A'),
                        "sub_diagnosis": node.get('sub_label', 'N/A'),
                        "findings": node.get('key_findings', 'No description available.')
                    })

        # 4. 全局排序：按照相似度分数降序排列
        all_candidates.sort(key=lambda x: x['score'], reverse=True)

        # 5. 提取相似度最高的 Top k 个病例
        # 对 score 进行格式化处理
        retrieved_results = []
        for item in all_candidates[:total_k]:
            item["score"] = round(item["score"], 5)
            retrieved_results.append(item)

        return retrieved_results
    def _get_prototype_summary(self, disease_name: str) -> str:
        """
        根据疾病名称从 Neo4j 检索原型总结
        """
        try:
            with self.driver.session() as session:
                # 使用你提供的查询逻辑
                res = session.run(
                    "MATCH (p:Prototype {disease: $name}) RETURN p.summary as s", 
                    name=disease_name
                ).single()
                
                if res and res["s"]:
                    return res["s"]
        except Exception as e:
            print(f"Error querying Neo4j for {disease_name}: {e}")
        return ""
    def review_case(self, 
                    vision_key_findings: str, 
                    panderm_top5: List[Dict[str, Any]], 
                    image_path: str, full_image_path: str) -> Dict:
        """
        综合诊断审查
        """
        # 1. 提取 Panderm Top-1 疾病名
        top1_disease = panderm_top5[0]['disease'] if panderm_top5 else "N/A"

        # 2. 检索历史相似病例
        current_feat = self._get_feat_by_path(image_path)
        hybrid_cases = self._find_historical_cases_in(current_feat, top_predictions=panderm_top5)

        # 3. 检索静态知识 (Handbook)
        expert_knowledge_context = ""
        prototypes_info = []
        for item in panderm_top5[:5]: # 取前5个
            disease_name = item['disease']

            knowledge = self._retrieve_static_knowledge(item['disease'])
            expert_knowledge_context += f"\n[Handbook: {item['disease']}]\n{knowledge}\n"
            prototype_summary = self._get_prototype_summary(disease_name)        
            if prototype_summary:    
                prototypes_info.append({            
                    "disease": disease_name,            
                    "summary": prototype_summary,            
                    "prob": item['probability']            
                })
        # 5. 构建增强版 Prompt
        print(hybrid_cases)
        prompt = self._build_comprehensive_prompt_withoutmemory(
            vision_findings=vision_key_findings,
            top5=panderm_top5,
            similar_cases=hybrid_cases,
            expert_knowledge=expert_knowledge_context,
            prototypes=prototypes_info
        )
        full_prompt = prompt
        print("=== Comprehensive Prompt ===")
        print(full_prompt)
        response_raw = generate_response(
            temperature=0.1,
            max_tokens=8181,
            system_role="Senior Clinical Dermatologist & Visual Auditor",
            user_input=full_prompt,
            image_path=full_image_path  # <--- 关键修改：传入图片路径
        )
        parsed_res = self._parse_json_response(response_raw)
        return parsed_res, prompt

    # ================= 3. Prompt 构建优化 =================
    def _build_comprehensive_prompt_without_pre(self, vision_findings, top5, similar_cases, expert_knowledge, prototypes):
        """
        重构说明：
        1. 移除了 top5 概率分布和置信度提示（Confidence Gap）。
        2. 核心逻辑转为“特征匹配”与“排他性诊断”。
        3. candidate_diseases 仅作为选项列表提供，不包含分数值。
        """
        
        # 1. 整理参考信息
        prototypes_str = ""    
        for p in prototypes:        
            prototypes_str += f"- [{p['disease']} Standard]: {p['summary']}\n"
        
        history_str = ""
        for c in similar_cases:
            # 只保留既往病例的诊断结果和表现，移除相似度分数，避免分数产生干扰
            history_str += f"- Past Confirmed Case [{c['diagnosis']}]:\n  Findings: {c['findings']}\n"

        # 选项列表（纯文字）
        # candidates_list = ", ".join(top5)

        # 2. 返回增强版 Prompt
        return f"""
        [Clinical Task]: Evidence-Based Dermatological Diagnosis.
        You are acting as a Senior Dermatologist. Your goal is to provide a final diagnosis by cross-referencing visual observations with medical standards and historical precedents, without relying on any pre-calculated model probabilities.

        [1. Current Visual Findings (from Vision Agent)]:
        {vision_findings}

        [2. Diagnostic Prototypes (Standard Reference)]:
        {prototypes_str if prototypes_str else "No standard prototypes available."}

        [3. Precedent Cases (Similar Confirmed Cases)]:
        {history_str if history_str else "No direct precedents found."}

        [4. Medical Standard (Static Handbook Knowledge)]:
        {expert_knowledge}

        [Reasoning Instructions]:
        1. **Primary Observation Analysis**: 
        - Carefully evaluate the visual findings in [1]. Identify the "primary lesion" (e.g., color, border, symmetry) and "secondary features" (e.g., scale, crust, blue-white veil).
        
        2. **Pattern Matching (Prototypes)**:
        - Compare [1] against [2]. Which standard disease profile does the current lesion most closely align with? 
        - Note: Atypical presentations (e.g., a Blue Nevus without a typical network) should still be considered if the color and circumscription match the standard.

        3. **Differential Exclusion (Rule-Out)**:
        - Use [4] to actively rule out candidates. 
        - For example: If the lesion lacks any pigment network, dots, or globules, 'Melanocytic Nevi' is less likely unless it's a specific variant. If the surface is smooth/waxy, 'Actinic Keratoses (AKIEC)' can likely be ruled out.

        4. **Historical Consistency**:
        - Review [3]. Do the morphological features of this patient mirror confirmed historical cases of a specific disease? Use these precedents to confirm or challenge your hypothesis.

        5. **Final Decision**:
        - Synthesize all evidence. You must select the most probable diagnosis from the following list: {DDI_DISEASE_NAME}.

        [Output Format (JSON)]:
        {{
            "KeyFindings": "A refined summary of critical morphological features found in the image.",
            "DifferentialDiagnosis": "Briefly mention 1-2 diseases you ruled out and why.",
            "PrimaryDiagnosis": "The single most likely diagnosis from the provided list.",
            "Evidence": "A logical derivation: 'Matches X standard due to Y features; Z ruled out due to lack of A'."
        }}
        """
    def _build_comprehensive_prompt_withoutmemory(self, vision_findings, top5, similar_cases, expert_knowledge, prototypes):
                # 1. 将名称从 Prototypes 修改为 Guidelines，并优化格式
                # 1. 结构化数据准备
        guidelines_str = ""    
        for p in prototypes:        
            guidelines_str += f"- [Guideline for {p['disease']}]: {p['summary']}\n"
        
        history_str = ""
        for c in similar_cases:
            # 强化历史病例的描述，包含诊断和核心视觉特征
            history_str += f"- Past Confirmed Case: {c['diagnosis']} (Similarity: {c['score']:.2f})\n  Findings in that case: {c['findings']}\n"

        top5_str = "\n".join([f"- {i['disease']}: {i['probability']:.2%}" for i in top5])

        # 2. 综合推理 Prompt
        return f"""
        [Clinical Task]: Final Multi-Modal Diagnosis Audit.
        Role: Senior Dermatologist. 
        Context: You are reconciling raw visual data [1], clinical standards [2, 5], statistical AI predictions [3], and empirical evidence from past cases [4].

        [1. Current Visual Findings (from Vision Agent)]:
        {vision_findings}

        [2. Diagnostic Guidelines (Standard Criteria)]:
        No specific guidelines available.

        [3. Model Prediction Probabilities (Panderm)]:
        {top5_str}

        [4. Precedent Cases (Similar Confirmed History)]:
        {history_str if history_str else "No direct precedents found."}

        [5. Medical Standard (General Handbook Knowledge)]:
        {expert_knowledge}

        [Comprehensive Reasoning Instructions]:
        
        STEP 1: Visual Validation (Audit [1])
        - Independently verify the visual description in [1]. If you see features (e.g. 'telangiectasia' or 'blue-white veil') not mentioned in [1], add them to your reasoning.

        STEP 2: Guideline & Encyclopedia Cross-Check (Sync [1] with [2] & [5])
        - Compare the current findings against the [Diagnostic Guidelines] for the top candidates. 
        - Does the lesion meet the "must-have" criteria for the high-probability diseases?

        STEP 3: Empirical Comparison (Sync [1] with [4])
        - Analyze the [Precedent Cases]. If a Past Case of 'Disease X' looks very similar to the Current Case [1], it provides strong empirical support for that diagnosis, even if it contradicts the statistical model [3].

        STEP 4: Conflict Resolution & Synthesis
        - **If [1] & [2] align with [4]**: This is a high-confidence diagnosis.
        - **If [3] is high, but [1] & [2] strongly suggest another diagnosis**: Prioritize the Guidelines [2]. Acknowledge that the visual model [3] might be reacting to non-diagnostic noise.
        - **If the lesion is atypical**: Use [4] to see if such an atypical presentation has been confirmed as a specific disease before.

        STEP 5: Final Conclusion
        - Select the most likely diagnosis exclusively from: {", ".join([i['disease'] for i in top5])}.

        [Output Format (JSON)]:
        {{
            "VisualFindings": "Refined description after auditing [1].",
            "PrimaryDiagnosis": "Exact name from the top5 list.",
            "Evidence": "A logical synthesis of why this diagnosis was chosen, citing specific guideline matches and historical case similarities."
        }}
        """
    def _build_comprehensive_prompt(self, vision_findings, top5, similar_cases, expert_knowledge, prototypes):
                # 1. 将名称从 Prototypes 修改为 Guidelines，并优化格式
                # 1. 结构化数据准备
        guidelines_str = ""    
        for p in prototypes:        
            guidelines_str += f"- [Guideline for {p['disease']}]: {p['summary']}\n"
        
        history_str = ""
        for c in similar_cases:
            # 强化历史病例的描述，包含诊断和核心视觉特征
            history_str += f"- Past Confirmed Case: {c['diagnosis']} (Similarity: {c['score']:.2f})\n  Findings in that case: {c['findings']}\n"

        top5_str = "\n".join([f"- {i['disease']}: {i['probability']:.2%}" for i in top5])

        # 2. 综合推理 Prompt
        return f"""
        [Clinical Task]: Final Multi-Modal Diagnosis Audit.
        Role: Senior Dermatologist. 
        Context: You are reconciling raw visual data [1], clinical standards [2, 5], statistical AI predictions [3], and empirical evidence from past cases [4].

        [1. Current Visual Findings (from Vision Agent)]:
        {vision_findings}

        [2. Diagnostic Guidelines (Standard Criteria)]:
        {guidelines_str if guidelines_str else "No specific guidelines available."}

        [3. Model Prediction Probabilities (Panderm)]:
        {top5_str}

        [4. Precedent Cases (Similar Confirmed History)]:
        {history_str if history_str else "No direct precedents found."}

        [5. Medical Standard (General Handbook Knowledge)]:
        {expert_knowledge}

        [Comprehensive Reasoning Instructions]:
        
        STEP 1: Visual Validation (Audit [1])
        - Independently verify the visual description in [1]. If you see features (e.g. 'telangiectasia' or 'blue-white veil') not mentioned in [1], add them to your reasoning.

        STEP 2: Guideline & Encyclopedia Cross-Check (Sync [1] with [2] & [5])
        - Compare the current findings against the [Diagnostic Guidelines] for the top candidates. 
        - Does the lesion meet the "must-have" criteria for the high-probability diseases?

        STEP 3: Empirical Comparison (Sync [1] with [4])
        - Analyze the [Precedent Cases]. If a Past Case of 'Disease X' looks very similar to the Current Case [1], it provides strong empirical support for that diagnosis, even if it contradicts the statistical model [3].

        STEP 4: Conflict Resolution & Synthesis
        - **If [1] & [2] align with [4]**: This is a high-confidence diagnosis.
        - **If [3] is high, but [1] & [2] strongly suggest another diagnosis**: Prioritize the Guidelines [2]. Acknowledge that the visual model [3] might be reacting to non-diagnostic noise.
        - **If the lesion is atypical**: Use [4] to see if such an atypical presentation has been confirmed as a specific disease before.

        STEP 5: Final Conclusion
        - Select the most likely diagnosis exclusively from: {", ".join([i['disease'] for i in top5])}.

        [Output Format (JSON)]:
        {{
            "VisualFindings": "Refined description after auditing [1].",
            "GuidelineCompliance": "Assessment of how well the case fits the [Guidelines] in [2].",
            "PrimaryDiagnosis": "Exact name from the top5 list.",
            "Evidence": "A logical synthesis of why this diagnosis was chosen, citing specific guideline matches and historical case similarities."
        }}
        """

    def _build_subclass_prompt_withoutmemory(self, vision_findings, main_top5, sub_top5, historical_cases, static_knowledge):
        # 1. 格式化历史病例（突出子类标签和特征）
        history_str = ""
        for c in historical_cases:
            # 新增：过滤无意义的子类标签
            sub_label = c['sub_diagnosis'] if c['sub_diagnosis'] not in ["N/A", ""] else "Unlabeled"
            history_str += f"""
            [Past Case {c['score']:.2f}]
            - Main Category: {c['diagnosis']}
            - Sub Category: {sub_label}
            - Key Findings: {c['findings']}
            """

        # 2. 格式化预测概率（新增概率排序）
        main_str = "\n".join([f"- {i['disease']} ({i['probability']:.1%})" for i in main_top5[:3]])
        sub_str = "\n".join([f"- {i['disease']} ({i['probability']:.1%})" for i in sorted(sub_top5[:5], key=lambda x: x['probability'], reverse=True)])  # 按概率排序

        # 3. 构造最终Prompt（修复动态原型显示问题）
        return f"""
        # Clinical Task: Dermnet Sub-category Classification
        You are a dermatology expert specializing in fine-grained sub-category diagnosis.
        Your goal is to determine the precise sub-type of skin lesion based on visual findings and medical knowledge.

        ## 1. Current Patient Presentation
        {vision_findings}

        ## 2. AI Prediction Candidates
        ### Main Categories (Top 3):
        {main_str}
        ### Sub-category Candidates (Top 5, Sorted by Probability):
        {sub_str}

        ## 3. Historical Reference Cases
        {history_str if history_str else "No similar cases found in database."}

        ## 4. Knowledge Base
        ### Static Handbook Knowledge:
        {static_knowledge}... 

        ## Diagnostic Requirements:
        1. **Sub-type Specificity**: Focus on subtle visual cues (scale type, color gradient, distribution) that distinguish sub-categories.
        2. **Category Consistency**: The selected sub-category must logically belong to one of the main categories.
        3. **Evidence Alignment**: Cite specific findings from Section 1 that match historical cases in Section 3.

        ## Output Format (JSON ONLY):
        {{
            "KeyFindings": "Concise summary of subtype-defining visual features",
            "PrimaryDiagnosis": "Selected main category",
            "SubDiagnosis": "Precise sub-category name",
            "Confidence": "High/Medium/Low",
            "Reasoning": "Step-by-step comparison with historical cases and knowledge"
        }}
        """

    def _build_subclass_prompt(self, vision_findings, main_top5, sub_top5, historical_cases, static_knowledge):
        history_str = ""
        for c in historical_cases:
            sub_label = c['sub_diagnosis'] if c['sub_diagnosis'] not in ["N/A", ""] else "Unlabeled"
            history_str += f"""
            [Past Case {c['score']:.2f}]
            - Main Category: {c['diagnosis']}
            - Sub Category: {sub_label}
            - Key Findings: {c['findings']}
            """
        main_str = "\n".join([f"- {i['disease']} ({i['probability']:.1%})" for i in main_top5[:3]])
        sub_str = "\n".join([f"- {i['disease']} ({i['probability']:.1%})" for i in sorted(sub_top5[:5], key=lambda x: x['probability'], reverse=True)])
        guidelines_str = ""
        for item in sub_top5[:5]:
            summary = self._get_prototype_summary(item['disease'])
            if summary:
                guidelines_str += f"- [Guideline for {item['disease']}]: {summary}\n"
        return f"""
        # Clinical Task: Dermnet Sub-category Classification
        You are a dermatology expert specializing in fine-grained sub-category diagnosis.
        Your goal is to determine the precise sub-type of skin lesion based on visual findings and medical knowledge.

        ## 1. Current Patient Presentation
        {vision_findings}

        ## 2. AI Prediction Candidates
        ### Main Categories (Top 3):
        {main_str}
        ### Sub-category Candidates (Top 5, Sorted by Probability):
        {sub_str}

        ## 3. Diagnostic Guidelines (Standard Criteria)
        {guidelines_str if guidelines_str else "No specific guidelines available."}

        ## 4. Historical Reference Cases
        {history_str if history_str else "No similar cases found in database."}

        ## 5. Knowledge Base
        ### Static Handbook Knowledge:
        {static_knowledge}... 

        ## Diagnostic Requirements:
        1. **Sub-type Specificity**: Focus on subtle visual cues (scale type, color gradient, distribution) that distinguish sub-categories.
        2. **Guideline Cross-Check**: Compare Section 1 against Section 3 and Section 5 to confirm criteria match.
        3. **Category Consistency**: Ensure the selected sub-category logically belongs to one of the main categories in Section 2.
        4. **Evidence Alignment**: Cite specific findings from Section 1 that match historical cases in Section 4.

        ## Output Format (JSON ONLY):
        {{
            "KeyFindings": "Concise summary of subtype-defining visual features",
            "GuidelineCompliance": "Assessment of how the case fits Section 3 guidelines",
            "SubDiagnosis": "Precise sub-category name",
            "Confidence": "High/Medium/Low",
            "Reasoning": "Step-by-step synthesis using guidelines, knowledge, and historical cases"
        }}
        """

    # ================= 辅助函数 =================

    def _retrieve_static_knowledge(self, disease_name: str):
        try:
            nodes = self.static_retriever.retrieve(QueryBundle(f"Features of {disease_name}"))
            return "\n".join([n.node.text for n in nodes[:5]])
        except: return "N/A"

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
            print(query_feat.shape, len(phys_scored), phys_scored[0][0] if phys_scored else 'N/A')
            for score, node in phys_scored[0:k]:
                all_retrieved_cases[node['case_id']] = {
                    "score": score, 
                    "diagnosis": node['true_label'] or node['primary_diagnosis'],
                    "sub_diagnosis": node.get('sub_label', 'N/A'),  # 子类标签
                    "findings": node['key_findings']
                }
        return list(all_retrieved_cases.values())

    def _parse_json_response(self, text):
        if isinstance(text, dict): return text
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            return json.loads(match.group()) if match else {"error": "parse failed"}
        except: return {"error": "parse failed"}

    def close(self):
        self.driver.close()
