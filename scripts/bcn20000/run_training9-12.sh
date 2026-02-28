#!/bin/bash

# --- 基础配置 ---

# Python 脚本的绝对路径
SCRIPT_PATH="/225040511/project/Skingpt_X/build_knowledge_base.py"

# TXT 分片文件所在的目录
TXT_DIR="/225040511/project/Skingpt_X/process_list/bcn20000/train"

EVAL_DIR='/225040511/project/Evaluation_Results/bcn20000/SkinGPT-X'

IMAGE_DIR='/225040511/Dataset/bcn20000/dataset'
# 日志保存目录
LOG_DIR="./training_logs"
mkdir -p $LOG_DIR

echo "🚀 Starting 4 parallel processes for Knowledge Base construction..."

# --- 启动 4 个进程 ---

# 进程 2: GPU 1, 处理 part2.txt
nohup python $SCRIPT_PATH --txt $TXT_DIR/train_split_9.txt --eval_dir $EVAL_DIR --image_dir $IMAGE_DIR --use_sub_label FALSE > $LOG_DIR/gpu9.log 2>&1 &
nohup python $SCRIPT_PATH --txt $TXT_DIR/train_split_10.txt --eval_dir $EVAL_DIR --image_dir $IMAGE_DIR --use_sub_label FALSE > $LOG_DIR/gpu10.log 2>&1 &
nohup python $SCRIPT_PATH --txt $TXT_DIR/train_split_11.txt --eval_dir $EVAL_DIR --image_dir $IMAGE_DIR --use_sub_label FALSE > $LOG_DIR/gpu11.log 2>&1 &
nohup python $SCRIPT_PATH --txt $TXT_DIR/train_split_12.txt --eval_dir $EVAL_DIR --image_dir $IMAGE_DIR --use_sub_label FALSE > $LOG_DIR/gpu12.log 2>&1 &

echo "✅ Started Part 2 on GPU 1. Log: $LOG_DIR/gpu2.log"

echo "--------------------------------------------------------"
echo "All processes are running in background."
echo "Use 'tail -f $LOG_DIR/gpu0.log' to check progress."
echo "Use 'nvidia-smi' to monitor GPU usage."