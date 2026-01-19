import argparse
import os
import json
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# 导入你定义的 Agent 类
from vision_agent import VisionAgent
from case_review_rag_agent import CaseReviewAgent 
from Constants import DERMNET_DISEASE_NAME

# --- 基础配置路径 ---
EVAL_DIR = "/225040511/project/Evaluation_Results/Dermnet/SkinGPT-X/"
PANDERM_CSV_PATH = os.path.join(EVAL_DIR, "panderm_test_predictions.csv")
IMAGE_BASE_DIR = "/Dermnet/V001"

class EvaluationWorkflow:
    def __init__(self, panderm_csv_path, disease_names, gpu_id=0, neo4j_uri:str="bolt://100.88.67.17:7687", neo4j_user:str="neo4j", neo4j_password:str="Czty100165188"):
        self.gpu_id = gpu_id
        
        # 1. 初始化 Vision Agent，并传入 GPU ID
        # 注意：你需要确保 VisionAgent 的 __init__ 接收 gpu_id 并设置 device_map
        self.vision_agent = VisionAgent() 
        self.disease_names = disease_names 
        
        # 2. 加载预测 CSV
        self.pred_df = pd.read_csv(panderm_csv_path)
        self.pred_df.set_index('filename', inplace=True)
        self.prob_cols = [f'probability_class_{i}' for i in range(len(disease_names))]

        # 3. 初始化 Case Review Agent
        self.review_agent = CaseReviewAgent(
            model="Qwen2-VL-8B", 
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password, # 👈 请确保密码正确
            lancedb_uri="/225040511/project/Skingpt_X/lancedb",
            markdown_path="/225040511/project/Skingpt_X/skin_handbook.md",
            train_feat_path=EVAL_DIR + "train_feats.npy", 
            train_json_path=EVAL_DIR + "train_files.json",
            test_feat_path=EVAL_DIR + "test_feats.npy",
            test_json_path=EVAL_DIR + "test_files.json",
        )

        # 4. 【核心修改】定义基于 GPU ID 的隔离输出路径
        # 这样不同 GPU 运行的结果会保存到不同文件，最后再合并
        self.vision_cache_path = os.path.join(EVAL_DIR, f"test_vision_findings_gpu{gpu_id}.json")
        self.final_output_path = os.path.join(EVAL_DIR, f"final_results_gpu{gpu_id}.json")
        self.prompt_output_path = os.path.join(EVAL_DIR, f"final_prompts_gpu{gpu_id}.json")

        # 加载本地缓存
        self.vision_findings_cache = {}
        if os.path.exists(self.vision_cache_path):
            with open(self.vision_cache_path, 'r', encoding='utf-8') as f:
                self.vision_findings_cache = json.load(f)
            print(f"[GPU {gpu_id}] 📦 Loaded cached vision findings.")
        

        self.final_results = {}
        if os.path.exists(self.final_output_path):
            with open(self.final_output_path, 'r', encoding='utf-8') as f:
                self.final_results = json.load(f)
            print(f"[GPU {gpu_id}] 📦 Resuming from {len(self.final_results)} completed cases.")
        self.final_prompts = {}
        if os.path.exists(self.prompt_output_path):
            with open(self.prompt_output_path, 'r', encoding='utf-8') as f:
                self.final_prompts = json.load(f)


    def _get_top5_from_row(self, filename):
        if filename not in self.pred_df.index:
            return None
        row = self.pred_df.loc[filename]
        if isinstance(row, pd.DataFrame): row = row.iloc[0]
        probs = row[self.prob_cols].values.astype(float)
        top5_indices = np.argsort(probs)[-5:][::-1]
        return [{"disease": self.disease_names[idx], "probability": float(probs[idx])} for idx in top5_indices]

    def _save_vision_cache(self):
        """保存到当前 GPU 专用的缓存文件"""
        with open(self.vision_cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.vision_findings_cache, f, indent=2, ensure_ascii=False)

    def _save_results(self):
        """保存到当前 GPU 专用的结果文件"""
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
        print(f"[GPU {self.gpu_id}] 🚀 Starting analysis for {len(target_paths)} cases...")
        
        for file_rel_path in tqdm(target_paths, desc=f"GPU {self.gpu_id}"):
            if file_rel_path in self.final_results:
                continue

            try:
                if file_rel_path not in self.pred_df.index: continue
                full_image_path = IMAGE_BASE_DIR+file_rel_path
                if not os.path.exists(full_image_path): continue

                # 1. 获取 Vision Findings
                if file_rel_path in self.vision_findings_cache:
                    key_findings = self.vision_findings_cache[file_rel_path]
                else:
                    vision_res = self.vision_agent.analyze(full_image_path)
                    key_findings = vision_res.get("key_findings", "")
                    self.vision_findings_cache[file_rel_path] = key_findings
                    print(f"[GPU {self.gpu_id}] 🖼️ Processed vision for {file_rel_path}")
                    self._save_vision_cache()
                # 2. 提取 Top-5 并执行 Case Review (取消注释)
                current_top5 = self._get_top5_from_row(file_rel_path)
                print(f"[GPU {self.gpu_id}] Panderm Top-5 for {file_rel_path}: {current_top5}")
                review_report, prompt_text = self.review_agent.review_case(
                    vision_key_findings=key_findings,
                    panderm_top5=current_top5,
                    image_path=file_rel_path
                )
                print(f"[GPU {self.gpu_id}] Prompt for {file_rel_path}: {prompt_text}")
                print(f"[GPU {self.gpu_id}] Review Report for {file_rel_path}: {review_report}")
                # 3. 记录结果
                gt_data = self.pred_df.loc[file_rel_path, 'true_label']
                
                # 如果返回的是 Series (意味着有重复行)，取第一个值
                if isinstance(gt_data, pd.Series):
                    final_gt = gt_data.iloc[0]
                else:
                    final_gt = gt_data
                
                # 确保转为标准的 Python int
                if hasattr(final_gt, 'item'): 
                    final_gt = final_gt.item()
                
                self.final_results[file_rel_path] = {
                    "ground_truth": int(final_gt), # 👈 现在这里安全了
                    "vision_findings": key_findings,
                    "panderm_top5": current_top5,
                    "final_decision": review_report
                }
                # --- 修改结束 ---

                # 新增：记录 Prompt
                self.final_prompts[file_rel_path] = prompt_text

                print(f"[GPU {self.gpu_id}] ✅ Completed case for {file_rel_path}")
                if len(self.final_results) % 1 == 0:
                    self._save_results()

            except Exception as e:
                print(f"❌ GPU {self.gpu_id} Error processing {file_rel_path}: {e}")

        self._save_results()
        print(f"✅ GPU {self.gpu_id} task complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SkinGPT-X Multi-GPU Case Reviewer")
    parser.add_argument("--task_file", type=str, required=True, help="Path to the .txt file")
    parser.add_argument("--gpu_id", type=int, default=0, help="Target GPU ID")
    parser.add_argument("--neo4j_uri", type=str, default=0, help="Target GPU ID")
    parser.add_argument("--neo4j_user", type=str, default=0, help="Target GPU ID")
    parser.add_argument("--neo4j_password", type=str, default=0, help="Target GPU ID")

    args = parser.parse_args()

    # 初始化 Workflow
    workflow = EvaluationWorkflow(
        panderm_csv_path=PANDERM_CSV_PATH, 
        disease_names=DERMNET_DISEASE_NAME[:-1],
        gpu_id=args.gpu_id,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
    )
    
    workflow.run_from_txt(args.task_file)