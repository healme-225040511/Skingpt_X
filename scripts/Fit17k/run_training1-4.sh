#!/bin/bash

# --- 基础配置 ---

# Python 脚本的绝对路径
SCRIPT_PATH="/225040511/project/Skingpt_X/build_knowledge_base_evolve.py"

# TXT 分片文件所在的目录
TXT_DIR="/225040511/project/Skingpt_X/process_list/fitzpatrick17k/train"

EVAL_DIR='/225040511/project/Evaluation_Results/fitzpatrick17k/SkinGPT-X'

IMAGE_DIR='/225040511/Dataset/fitzpatrick17k/dataset'
# 日志保存目录
LOG_DIR="./training_logs"
mkdir -p $LOG_DIR

echo "🚀 Starting 4 parallel processes for Knowledge Base construction..."

# --- 启动 4 个进程 ---

# Part 1
nohup python "$SCRIPT_PATH" --txt "$TXT_DIR/train_split_1.txt" --eval_dir "$EVAL_DIR" --image_dir "$IMAGE_DIR" > "$LOG_DIR/gpu1.log" 2>&1 &
wait $!
echo "✅ Finished Part 1. Log: $LOG_DIR/gpu7.log"

# Part 2
nohup python "$SCRIPT_PATH" --txt "$TXT_DIR/train_split_2.txt" --eval_dir "$EVAL_DIR" --image_dir "$IMAGE_DIR" > "$LOG_DIR/gpu2.log" 2>&1 &
wait $!
echo "✅ Finished Part 2. Log: $LOG_DIR/gpu8.log"

# Part 3
nohup python "$SCRIPT_PATH" --txt "$TXT_DIR/train_split_3.txt" --eval_dir "$EVAL_DIR" --image_dir "$IMAGE_DIR" > "$LOG_DIR/gpu3.log" 2>&1 &
wait $!
echo "✅ Finished Part 3. Log: $LOG_DIR/gpu3.log"

# Part 4
nohup python "$SCRIPT_PATH" --txt "$TXT_DIR/train_split_4.txt" --eval_dir "$EVAL_DIR" --image_dir "$IMAGE_DIR" > "$LOG_DIR/gpu4.log" 2>&1 &
wait $!
echo "✅ Finished Part 4. Log: $LOG_DIR/gpu4.log"

echo "--------------------------------------------------------"
echo "All processes are running in background."
echo "Use 'tail -f $LOG_DIR/gpu0.log' to check progress."
echo "Use 'nvidia-smi' to monitor GPU usage."

# nohup bash /225040511/project/Skingpt_X/scripts/Fit17k/run_training1-4.sh > ./training_logs/master.log 2>&1 &
# nohup bash /225040511/project/Skingpt_X/scripts/Fit17k/run_training5-8.sh > ./training_logs/master2.log 2>&1 &