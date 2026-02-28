#!/bin/bash

# --- 路径配置 ---
TXT_DIR="/225040511/project/Skingpt_X/process_list/Dermnet/test"
SCRIPT_PATH='/225040511/project/Skingpt_X/eval_case_review.py'
LOG_DIR="/225040511/project/Skingpt_X/scripts/logs_4gpu"
MAIN_CSV="/225040511/project/Evaluation_Results/Dermnet/SkinGPT-X/panderm_test_predictions.csv"

# --- 数据库配置 ---
NEO4J_URI="bolt://100.91.178.230:7687"
NEO4J_PASS="Czty100165188"

# 进入工作目录
cd /225040511/project/Skingpt_X
mkdir -p $LOG_DIR

echo "🚀 Starting Parallel Main Mode Evaluation on 4 GPUs..."

nohup bash -c " \
python $SCRIPT_PATH --task_file $TXT_DIR/process_list5.txt --gpu_id 4 --neo4j_uri bolt://100.91.178.230:7687 --neo4j_user neo4j --neo4j_password Czty100165188 --mode main --eval_dir /225040511/project/Evaluation_Results/Dermnet/SkinGPT-X/ --main_csv $MAIN_CSV > $LOG_DIR/gpu4.log 2>&1 && \
python $SCRIPT_PATH --task_file $TXT_DIR/process_list6.txt --gpu_id 5 --neo4j_uri bolt://100.91.178.230:7687 --neo4j_user neo4j --neo4j_password Czty100165188 --mode main --eval_dir /225040511/project/Evaluation_Results/Dermnet/SkinGPT-X/ --main_csv $MAIN_CSV > $LOG_DIR/gpu5.log 2>&1 && \
python $SCRIPT_PATH --task_file $TXT_DIR/process_list7.txt --gpu_id 6 --neo4j_uri bolt://100.91.178.230:7687 --neo4j_user neo4j --neo4j_password Czty100165188 --mode main --eval_dir /225040511/project/Evaluation_Results/Dermnet/SkinGPT-X/ --main_csv $MAIN_CSV > $LOG_DIR/gpu6.log 2>&1 && \
python $SCRIPT_PATH --task_file $TXT_DIR/process_list8.txt --gpu_id 7 --neo4j_uri bolt://100.91.178.230:7687 --neo4j_user neo4j --neo4j_password Czty100165188 --mode main --eval_dir /225040511/project/Evaluation_Results/Dermnet/SkinGPT-X/ --main_csv $MAIN_CSV > $LOG_DIR/gpu7.log 2>&1 \
" > $LOG_DIR/total_sequence.log 2>&1 &
# # 进程 2: GPU 1, 处理 part2.txt
# python $SCRIPT_PATH --task_file $TXT_DIR/process_list4.txt --gpu_id 3
# python $SCRIPT_PATH --task_file $TXT_DIR/process_list5.txt --gpu_id 4
echo '✅ All tasks finished.'
 
# # 进程 3: GPU 2, 处理 part3.txt
# CUDA_VISIBLE_DEVICES=2 nohup python $SCRIPT_PATH --task_file $TXT_DIR/process_list3.txt --gpu_id 2 > $LOG_DIR/gpu7.log 2>&1 &
# echo "✅ Started Part 3 on GPU 2. Log: $LOG_DIR/gpu7.log"

# # 进程 4: GPU 3, 处理 part4.txt
# CUDA_VISIBLE_DEVICES=3 nohup python $SCRIPT_PATH --task_file $TXT_DIR/process_list4.txt --gpu_id 3 > $LOG_DIR/gpu8.log 2>&1 &
# echo "✅ Started Part 4 on GPU 3. Log: $LOG_DIR/gpu8.log"
