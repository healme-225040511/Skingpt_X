#!/bin/bash

# --- 基础配置 ---

# Python 脚本的绝对路径
SCRIPT_PATH="/225040511/project/Skingpt_X/build_knowledge_base.py"

# TXT 分片文件所在的目录
TXT_DIR="/225040511/project/Skingpt_X/process_list/DDI2/train"

EVAL_DIR='/225040511/project/Evaluation_Results/DDI2/SkinGPT-X'

IMAGE_DIR='/225040511/Dataset/DDI2/dataset'
# 日志保存目录
LOG_DIR="./training_logs"
mkdir -p $LOG_DIR

echo "🚀 Starting 4 parallel processes for Knowledge Base construction..."

# --- 启动 4 个进程 ---

# 进程 1: GPU 0, 处理 part1.txt
nohup python $SCRIPT_PATH --txt $TXT_DIR/train_split_1.txt --eval_dir $EVAL_DIR --image_dir $IMAGE_DIR --use_sub_label FALSE > $LOG_DIR/gpu1.log 2>&1 &
nohup python $SCRIPT_PATH --txt $TXT_DIR/train_split_2.txt --eval_dir $EVAL_DIR --image_dir $IMAGE_DIR --use_sub_label FALSE > $LOG_DIR/gpu2.log 2>&1 &
nohup python $SCRIPT_PATH --txt $TXT_DIR/train_split_3.txt --eval_dir $EVAL_DIR --image_dir $IMAGE_DIR --use_sub_label FALSE > $LOG_DIR/gpu3.log 2>&1 &
nohup python $SCRIPT_PATH --txt $TXT_DIR/train_split_4.txt --eval_dir $EVAL_DIR --image_dir $IMAGE_DIR --use_sub_label FALSE > $LOG_DIR/gpu4.log 2>&1 &

echo "✅ Started Part 1 on GPU 0. Log: $LOG_DIR/gpu1.log"


echo "--------------------------------------------------------"
echo "All processes are running in background."
echo "Use 'tail -f $LOG_DIR/gpu0.log' to check progress."
echo "Use 'nvidia-smi' to monitor GPU usage."