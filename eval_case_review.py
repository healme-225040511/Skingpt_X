import argparse
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import re
# 导入 Agent 类
from vision_agent import VisionAgent
from case_review_rag_agent import CaseReviewAgent 
from Constants import DERMNET_DISEASE_NAME, DERMNET_SUB_DISEASE_NAME, HAM10000_DISEASE_NAME_FULL, Fitzpatrick17k_DISEASE_NAME, RDD_DISEASE_NAME, PAD_DISEASE_NAME, DDI_DISEASE_NAME, BINARY_DISEASE_NAME, BCN_DISEASE_NAME

class EvaluationWorkflow:
    def __init__(self, 
                main_csv_path,  # 大类CSV路径（原 panderm_csv_path）
                sub_csv_path=None,  # 新增：子类CSV路径
                disease_names=None, 
                sub_disease_names=None,
                mode="main", 
                gpu_id=0, 
                neo4j_uri="bolt://100.88.67.17:7687", 
                neo4j_user="neo4j", 
                neo4j_password="Czty100165188",
                eval_dir="",
                image_dir=""):
        self.eval_dir = eval_dir
        print(eval_dir)
        self.image_dir = image_dir
        self.main_csv_path = os.path.join(self.eval_dir, "panderm_test_predictions.csv")
        self.gpu_id = gpu_id
        self.mode = mode
        self.sub_disease_names = sub_disease_names

        # 1. 初始化Vision Agent
        self.vision_agent = VisionAgent() 
        self.disease_names = disease_names 

        # 2. 加载大类CSV
        self.main_df = pd.read_csv(self.main_csv_path)
        self.main_df.set_index('filename', inplace=True)
        self.main_prob_cols = [col for col in self.main_df.columns if col.startswith('probability_class_')]
        print(f"[GPU {gpu_id}] 加载大类CSV: {len(self.main_df)}条数据, {len(self.main_prob_cols)}个大类概率列")

        # 3. 加载子类CSV（仅子类模式）
        self.sub_df = None
        self.sub_prob_cols = []
        if mode == "sub" and sub_csv_path:
            self.sub_df = pd.read_csv(sub_csv_path)
            self.sub_df.set_index('filename', inplace=True)
            # 自动检测子类概率列（支持 'subclass_prob_0' 或 'probability_subclass_0'）
            self.sub_prob_cols = [col for col in self.sub_df.columns if col.startswith(('probability_class_'))]
            self.sub_prob_cols.sort(key=lambda x: int(re.search(r'\d+', x).group()))  # 按数字排序
            print(f"[GPU {gpu_id}] 加载子类CSV: {len(self.sub_df)}条数据, {len(self.sub_prob_cols)}个子类概率列")
            # 校验文件名是否重叠（确保能关联）
            common_files = set(self.main_df.index) & set(self.sub_df.index)
            if not common_files:
                raise ValueError("大类CSV和子类CSV无共同filename，无法关联！")
            else:
                print(f"[GPU {gpu_id}] 大类/子类CSV共有 {len(common_files)} 个重叠文件")


        # 3. 初始化 Case Review Agent
        self.review_agent = CaseReviewAgent(
            model="Qwen2-VL-8B", 
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            lancedb_uri="/225040511/project/Skingpt_X/lancedb",
            markdown_path="/225040511/project/Skingpt_X/skin_handbook.md",
            train_feat_path=self.eval_dir + "train_feats.npy", 
            train_json_path=self.eval_dir + "train_files.json",
            test_feat_path=self.eval_dir + "test_feats.npy",
            test_json_path=self.eval_dir + "test_files.json",
        )

        # 4. 输出路径（按模式区分）
        self.vision_cache_path = os.path.join(self.eval_dir, f"test_vision_findings_gpu{gpu_id}.json")
        self.final_output_path = os.path.join(self.eval_dir, f"final_results_gpu{gpu_id}_{mode}.json")
        self.prompt_output_path = os.path.join(self.eval_dir, f"final_prompts_gpu{gpu_id}_{mode}.json")

        # 加载缓存
        self.vision_findings_cache = {}
        if os.path.exists(self.vision_cache_path):
            with open(self.vision_cache_path, 'r', encoding='utf-8') as f:
                self.vision_findings_cache = json.load(f)
        
        self.final_results = {}
        if os.path.exists(self.final_output_path):
            with open(self.final_output_path, 'r', encoding='utf-8') as f:
                self.final_results = json.load(f)
        
        self.final_prompts = {}
        if os.path.exists(self.prompt_output_path):
            with open(self.prompt_output_path, 'r', encoding='utf-8') as f:
                self.final_prompts = json.load(f)

   
    def _get_sub_top5(self, filename):
        """从子类CSV中获取Top5预测"""
        if self.mode != "sub" or self.sub_df is None or filename not in self.sub_df.index:
            return []
        row = self.sub_df.loc[filename]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]  # 处理重复行
        try:
            probs = row[self.sub_prob_cols].values.astype(float)
            top5_indices = np.argsort(probs)[-5:][::-1] if len(probs)>=5 else np.argsort(probs)[::-1]
            return [{"disease": self.sub_disease_names[idx], "probability": float(probs[idx])} for idx in top5_indices]
        except KeyError as e:
            print(f"⚠️ 子类列缺失: {e}")
            return []

    def _get_main_top5(self, filename):
        """从大类CSV中获取Top5预测（原 _get_top5_from_row 简化版）"""
        if filename not in self.main_df.index:
            return []
        row = self.main_df.loc[filename]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        probs = row[self.main_prob_cols].values.astype(float)
        top5_indices = np.argsort(probs)[-5:][::-1] if len(probs)>=5 else np.argsort(probs)[::-1]
        return [{"disease": self.disease_names[idx], "probability": float(probs[idx])} for idx in top5_indices]
    def _save_vision_cache(self):
        with open(self.vision_cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.vision_findings_cache, f, indent=2, ensure_ascii=False)

    def _save_results(self):
        with open(self.final_output_path, 'w', encoding='utf-8') as f:
            json.dump(self.final_results, f, indent=2, ensure_ascii=False)
        with open(self.prompt_output_path, 'w', encoding='utf-8') as f:
            json.dump(self.final_prompts, f, indent=2, ensure_ascii=False)

    def run_from_txt(self, txt_path):
        if not os.path.exists(txt_path):
            print(f"❌ TXT file not found: {txt_path}")
            return

        with open(txt_path, 'r', encoding='utf-8') as f:
            target_paths = [line.strip() for line in f if line.strip()]
        print(f"[GPU {self.gpu_id}][Mode: {self.mode}] 🚀 Starting analysis for {len(target_paths)} cases...")

        for file_rel_path in tqdm(target_paths, desc=f"GPU {self.gpu_id} | Mode: {self.mode}"):
            if file_rel_path in self.final_results:
                continue
            try:
                # 1. 校验文件是否存在于大类CSV
                if file_rel_path not in self.main_df.index:
                    print(f"⚠️ {file_rel_path} 不在大类CSV中，跳过")
                    continue
                full_image_path = self.image_dir + file_rel_path
                if not os.path.exists(full_image_path):
                    print(f"❌ Image not found: {full_image_path}")
                    continue

                # 1. 获取视觉描述
                if file_rel_path in self.vision_findings_cache:
                    key_findings = self.vision_findings_cache[file_rel_path]
                else:
                    vision_res = self.vision_agent.analyze(full_image_path)
                    key_findings = vision_res.get("key_findings", "")
                    self.vision_findings_cache[file_rel_path] = key_findings
                    self._save_vision_cache()

                main_top5 = self._get_main_top5(file_rel_path)
                if not main_top5:
                    print(f"⚠️ {file_rel_path} 无法获取大类预测，跳过")
                    continue

                if self.mode == "main":
                    # 大类模式（不变）
                    current_top5 = self._get_main_top5(file_rel_path)
                    print(f"[GPU {self.gpu_id}] Panderm Top-5 for {file_rel_path}: {current_top5}")
                    review_report, prompt_text = self.review_agent.review_case(
                        vision_key_findings=key_findings,
                        panderm_top5=current_top5,
                        image_path=file_rel_path,
                        full_image_path=full_image_path
                    )
                    print(f"[GPU {self.gpu_id}] Prompt for {file_rel_path}: {prompt_text}")
                    print(f"[GPU {self.gpu_id}] Review Report for {file_rel_path}: {review_report}")
                elif self.mode == "sub":
                    # 子类模式：从子类CSV获取sub_top5
                    sub_top5 = self._get_sub_top5(file_rel_path)
                    if not sub_top5:
                        print(f"⚠️ {file_rel_path} 无法获取子类预测")
                        continue
                    else:
                        review_report, prompt_text = self.review_agent.review_sub_class(
                            vision_key_findings=key_findings,
                            main_top5=main_top5,
                            sub_top5=sub_top5,
                            image_path=file_rel_path
                        )
                else:
                    raise ValueError(f"Invalid mode: {self.mode}. Use 'main' or 'sub'.")

                # 3. 记录结果
                gt_data = self.main_df.loc[file_rel_path, 'true_label']
                final_gt = gt_data.iloc[0] if isinstance(gt_data, pd.Series) else gt_data
                final_gt = int(final_gt.item()) if hasattr(final_gt, 'item') else int(final_gt)

                self.final_results[file_rel_path] = {
                    "ground_truth": final_gt,
                    "vision_findings": key_findings,
                    "main_top5": self._get_main_top5(file_rel_path),
                    "sub_top5": self._get_sub_top5(file_rel_path) if self.mode == "sub" else None,
                    "final_decision": review_report
                }
                self.final_prompts[file_rel_path] = prompt_text

                if len(self.final_results) % 1 == 0:
                    self._save_results()

            except Exception as e:
                print(f"❌ [GPU {self.gpu_id}][Mode: {self.mode}] Error processing {file_rel_path}: {e}")

        self._save_results()
        print(f"✅ [GPU {self.gpu_id}][Mode: {self.mode}] Task complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SkinGPT-X Multi-GPU Case Reviewer (Main/Sub Mode)")
    parser.add_argument("--task_file", type=str, required=True, help="任务TXT文件路径")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--mode", type=str, default="main", choices=["main", "sub"], help="模式: main/sub")
    parser.add_argument("--main_csv", type=str, required=True, help="大类预测CSV路径（必填）")  # 原 panderm_csv_path
    parser.add_argument("--sub_csv", type=str, help="子类预测CSV路径（sub模式必填）")  # 新增
    parser.add_argument("--neo4j_uri", type=str, default="bolt://100.91.219.47:7687", help="Neo4j URI")
    parser.add_argument("--neo4j_user", type=str, default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j_password", type=str, default="Czty100165188", help="Neo4j password")
    parser.add_argument("--eval_dir", type=str, default='', help="N")
    parser.add_argument("--image_dir", type=str, default='', help="N")
    args = parser.parse_args()
        # 校验参数
    if args.mode == "sub" and not args.sub_csv:
        parser.error("--sub_csv 必须在 --mode sub 时提供！")
    # 初始化工作流（子类模式需传入子类名称列表）
    workflow = EvaluationWorkflow(
        main_csv_path=args.main_csv,  # 传入大类CSV路径
        sub_csv_path=args.sub_csv,    # 传入子类CSV路径
        disease_names=Fitzpatrick17k_DISEASE_NAME,
        sub_disease_names=BCN_DISEASE_NAME if args.mode == "sub" else None,
        mode=args.mode,
        gpu_id=args.gpu_id,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        eval_dir=args.eval_dir,
        image_dir=args.image_dir
    )
    
    workflow.run_from_txt(args.task_file)