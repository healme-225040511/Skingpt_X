
# --- 路径配置 ---
TXT_DIR="/225040511/project/Skingpt_X/process_list/ISIC_2024/test"
SCRIPT_PATH='/225040511/project/Skingpt_X/eval_case_review.py'
LOG_DIR="./case_revew"
MAIN_CSV="/225040511/project/Evaluation_Results/ISIC_2024/SkinGPT-X/panderm_test_predictions.csv"
IMAGE_BASE_DIR="/225040511/Dataset/ISIC_2024"
EVAL_DIR='/225040511/project/Evaluation_Results/ISIC_2024/SkinGPT-X/'

# --- 数据库配置 ---
NEO4J_URI="bolt://100.91.219.81:7687"
NEO4J_PASS="Czty100165188"

# 进入工作目录
mkdir -p $LOG_DIR

echo "🚀 Starting Parallel Main Mode Evaluation on 4 GPUs..."


CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python $SCRIPT_PATH --task_file $TXT_DIR/test_split_1.txt --image_dir $IMAGE_BASE_DIR --gpu_id 0 --neo4j_uri $NEO4J_URI --neo4j_user neo4j --neo4j_password $NEO4J_PASS --mode main --eval_dir $EVAL_DIR --main_csv $MAIN_CSV > $LOG_DIR/processor0.log 2>&1 &
# python $SCRIPT_PATH --task_file $TXT_DIR/test_split_1.txt --image_dir $IMAGE_BASE_DIR --gpu_id 0 --neo4j_uri $NEO4J_URI --neo4j_user neo4j --neo4j_password $NEO4J_PASS --mode main --eval_dir $EVAL_DIR --main_csv $MAIN_CSV 