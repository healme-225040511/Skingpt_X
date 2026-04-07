SCRIPT_PATH="/225040511/project/Skingpt_X/build_knowledge_base_evolve.py"

# TXT 分片文件所在的目录
TXT_DIR="/225040511/project/Skingpt_X/process_list/Dermnet/train"

EVAL_DIR='/225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X'

IMAGE_DIR='/225040511/Dataset/Dermnet'
# 日志保存目录
LOG_DIR="./training_logs"
mkdir -p $LOG_DIR

echo "🚀 Starting 4 parallel processes for Knowledge Base construction..."

# --- 启动 4 个进程 ---

# # Part 1
# nohup python "$SCRIPT_PATH" --txt "$TXT_DIR/process_list5.txt" --eval_dir "$EVAL_DIR" --image_dir "$IMAGE_DIR" --use_sub_label > "$LOG_DIR/gpu5.log" 2>&1 &
# wait $!
# echo "✅ Finished Part 5. Log: $LOG_DIR/gpu1.log"

# # Part 2
# nohup python "$SCRIPT_PATH" --txt "$TXT_DIR/process_list6.txt" --eval_dir "$EVAL_DIR" --image_dir "$IMAGE_DIR" --use_sub_label > "$LOG_DIR/gpu6.log" 2>&1 &
# wait $!
# echo "✅ Finished Part 6. Log: $LOG_DIR/gpu2.log"

# Part 3
nohup python "$SCRIPT_PATH" --txt "$TXT_DIR/process_list7.txt" --eval_dir "$EVAL_DIR" --image_dir "$IMAGE_DIR" --use_sub_label > "$LOG_DIR/gpu7.log" 2>&1 &
wait $!
echo "✅ Finished Part 7. Log: $LOG_DIR/gpu3.log"

# # Part 4
# nohup python "$SCRIPT_PATH" --txt "$TXT_DIR/process_list8.txt" --eval_dir "$EVAL_DIR" --image_dir "$IMAGE_DIR" --use_sub_label > "$LOG_DIR/gpu8.log" 2>&1 &
# wait $!
# echo "✅ Finished Part 8. Log: $LOG_DIR/gpu4.log"

echo "--------------------------------------------------------"
echo "All processes are running in background."
echo "Use 'tail -f $LOG_DIR/gpu0.log' to check progress."
echo "Use 'nvidia-smi' to monitor GPU usage."
