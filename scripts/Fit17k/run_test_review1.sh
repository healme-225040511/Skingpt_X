#!/bin/bash

# --- 路径配置 ---
TXT_DIR="/225040511/project/Skingpt_X/process_list/fitzpatrick17k/test"
SCRIPT_PATH='/225040511/project/Skingpt_X/eval_case_review.py'
LOG_DIR="./case_revew"
MAIN_CSV="/225040511/project/Evaluation_Results/fitzpatrick17k/SkinGPT-X/panderm_test_predictions.csv"
IMAGE_BASE_DIR="/225040511/Dataset/fitzpatrick17k/dataset"

# --- 数据库配置 ---
NEO4J_URI="bolt://100.91.219.94:7687"
NEO4J_PASS="Czty100165188"

# 进入工作目录
mkdir -p $LOG_DIR

echo "🚀 Starting Parallel Main Mode Evaluation on 4 GPUs..."


nohup python $SCRIPT_PATH --task_file $TXT_DIR/test_split_1.txt --image_dir $IMAGE_BASE_DIR --gpu_id 0 --neo4j_uri $NEO4J_URI --neo4j_user neo4j --neo4j_password $NEO4J_PASS --mode main --eval_dir /225040511/project/Evaluation_Results/fitzpatrick17k/SkinGPT-X/ --main_csv $MAIN_CSV > $LOG_DIR/processor0.log 2>&1 &
# python $SCRIPT_PATH --task_file $TXT_DIR/failed_cases1.txt --image_dir $IMAGE_BASE_DIR --gpu_id 0 --neo4j_uri $NEO4J_URI --neo4j_user neo4j --neo4j_password $NEO4J_PASS --mode main --eval_dir /225040511/project/Evaluation_Results/fitzpatrick17k/SkinGPT-X/ --main_csv $MAIN_CSV 