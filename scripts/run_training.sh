#!/bin/bash

# --- 基础配置 ---

# Python 脚本的绝对路径
SCRIPT_PATH="/225040511/project/Skingpt_X/build_knowledge_base.py"

# TXT 分片文件所在的目录
TXT_DIR="/225040511/project/Skingpt_X/process_list/Dermnet/train"

# 日志保存目录
LOG_DIR="./training_logs"
mkdir -p $LOG_DIR

echo "🚀 Starting 4 parallel processes for Knowledge Base construction..."

# --- 启动 4 个进程 ---

# # 进程 1: GPU 0, 处理 part1.txt
# CUDA_VISIBLE_DEVICES=0 nohup python $SCRIPT_PATH --txt $TXT_DIR/process_list5.txt > $LOG_DIR/gpu1.log 2>&1 &
# echo "✅ Started Part 1 on GPU 0. Log: $LOG_DIR/gpu1.log"

# # 进程 2: GPU 1, 处理 part2.txt
# CUDA_VISIBLE_DEVICES=1 nohup python $SCRIPT_PATH --txt $TXT_DIR/process_list6.txt > $LOG_DIR/gpu2.log 2>&1 &
# echo "✅ Started Part 2 on GPU 1. Log: $LOG_DIR/gpu2.log"

# # 进程 3: GPU 2, 处理 part3.txt
# CUDA_VISIBLE_DEVICES=2 nohup python $SCRIPT_PATH --txt $TXT_DIR/process_list7.txt > $LOG_DIR/gpu3.log 2>&1 &
# echo "✅ Started Part 3 on GPU 2. Log: $LOG_DIR/gpu3.log"

# # 进程 4: GPU 3, 处理 part4.txt
# CUDA_VISIBLE_DEVICES=3 nohup python $SCRIPT_PATH --txt $TXT_DIR/process_list8.txt > $LOG_DIR/gpu4.log 2>&1 &
# echo "✅ Started Part 4 on GPU 3. Log: $LOG_DIR/gpu4.log"

# 进程 1: GPU 0, 处理 part1.txt
CUDA_VISIBLE_DEVICES=0  python $SCRIPT_PATH --txt $TXT_DIR/process_list5.txt > $LOG_DIR/gpu5.log 2>&1 &
echo "✅ Started Part 5 on GPU 0. Log: $LOG_DIR/gpu5.log"

# 进程 2: GPU 1, 处理 part2.txt
CUDA_VISIBLE_DEVICES=1 nohup python $SCRIPT_PATH --txt $TXT_DIR/process_list6.txt > $LOG_DIR/gpu6.log 2>&1 &
echo "✅ Started Part 6 on GPU 1. Log: $LOG_DIR/gpu6.log"

# 进程 3: GPU 2, 处理 part3.txt
CUDA_VISIBLE_DEVICES=2 nohup python $SCRIPT_PATH --txt $TXT_DIR/process_list7.txt > $LOG_DIR/gpu7.log 2>&1 &
echo "✅ Started Part 7 on GPU 2. Log: $LOG_DIR/gpu7.log"

# 进程 4: GPU 3, 处理 part4.txt
CUDA_VISIBLE_DEVICES=3 nohup python $SCRIPT_PATH --txt $TXT_DIR/process_list8.txt > $LOG_DIR/gpu8.log 2>&1 &
echo "✅ Started Part 8 on GPU 3. Log: $LOG_DIR/gpu8.log"

echo "--------------------------------------------------------"
echo "All processes are running in background."
echo "Use 'tail -f $LOG_DIR/gpu0.log' to check progress."
echo "Use 'nvidia-smi' to monitor GPU usage."