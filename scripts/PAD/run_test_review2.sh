
# --- 路径配置 ---
TXT_DIR="/225040511/project/Skingpt_X/process_list/PAD/test"
SCRIPT_PATH='/225040511/project/Skingpt_X/eval_case_review.py'
LOG_DIR="./case_revew"
MAIN_CSV="/225040511/project/Evaluation_Results/PAD-UFES-20/SkinGPT-X/panderm_test_predictions.csv"
IMAGE_BASE_DIR="/225040511/Dataset/PAD-UFES-20/dataset"
EVAL_DIR='/225040511/project/Evaluation_Results/PAD-UFES-20/SkinGPT-X/'

# --- 数据库配置 ---
NEO4J_URI="bolt://100.91.219.77:7687"
NEO4J_PASS="Czty100165188"

# 进入工作目录
mkdir -p $LOG_DIR

echo "🚀 Starting Parallel Main Mode Evaluation on 4 GPUs..."


CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python $SCRIPT_PATH --task_file $TXT_DIR/test_split_2.txt --image_dir $IMAGE_BASE_DIR --gpu_id 1 --neo4j_uri $NEO4J_URI --neo4j_user neo4j --neo4j_password $NEO4J_PASS --mode main --eval_dir $EVAL_DIR --main_csv $MAIN_CSV > $LOG_DIR/processor1.log 2>&1 &
# python $SCRIPT_PATH --task_file $TXT_DIR/failed_cases1.txt --image_dir $IMAGE_BASE_DIR --gpu_id 0 --neo4j_uri $NEO4J_URI --neo4j_user neo4j --neo4j_password $NEO4J_PASS --mode main --eval_dir /225040511/project/Evaluation_Results/fitzpatrick17k/SkinGPT-X/ --main_csv $MAIN_CSV 