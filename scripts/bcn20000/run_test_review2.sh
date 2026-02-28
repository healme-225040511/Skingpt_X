
# --- 路径配置 ---
TXT_DIR="/225040511/project/Skingpt_X/process_list/bcn20000/test"
SCRIPT_PATH='/225040511/project/Skingpt_X/eval_case_review.py'
LOG_DIR="./case_revew"
MAIN_CSV="/225040511/project/Evaluation_Results/bcn20000/SkinGPT-X/panderm_test_predictions.csv"
IMAGE_BASE_DIR="/225040511/Dataset/bcn20000/dataset"
EVAL_DIR='/225040511/project/Evaluation_Results/bcn20000/SkinGPT-X/'

# --- 数据库配置 ---
NEO4J_URI="bolt://100.91.219.87:7687"
NEO4J_PASS="Czty100165188"

# 进入工作目录
mkdir -p $LOG_DIR

echo "🚀 Starting Parallel Main Mode Evaluation on 4 GPUs..."
nohup python $SCRIPT_PATH --task_file $TXT_DIR/test_split_5.txt --image_dir $IMAGE_BASE_DIR --gpu_id 4 --neo4j_uri $NEO4J_URI --neo4j_user neo4j --neo4j_password $NEO4J_PASS --mode main --eval_dir $EVAL_DIR --main_csv $MAIN_CSV > $LOG_DIR/processor4.log 2>&1 &
nohup python $SCRIPT_PATH --task_file $TXT_DIR/test_split_6.txt --image_dir $IMAGE_BASE_DIR --gpu_id 5 --neo4j_uri $NEO4J_URI --neo4j_user neo4j --neo4j_password $NEO4J_PASS --mode main --eval_dir $EVAL_DIR --main_csv $MAIN_CSV > $LOG_DIR/processor5.log 2>&1 &
nohup python $SCRIPT_PATH --task_file $TXT_DIR/test_split_7.txt --image_dir $IMAGE_BASE_DIR --gpu_id 6 --neo4j_uri $NEO4J_URI --neo4j_user neo4j --neo4j_password $NEO4J_PASS --mode main --eval_dir $EVAL_DIR --main_csv $MAIN_CSV > $LOG_DIR/processor6.log 2>&1 &
nohup python $SCRIPT_PATH --task_file $TXT_DIR/test_split_8.txt --image_dir $IMAGE_BASE_DIR --gpu_id 7 --neo4j_uri $NEO4J_URI --neo4j_user neo4j --neo4j_password $NEO4J_PASS --mode main --eval_dir $EVAL_DIR --main_csv $MAIN_CSV > $LOG_DIR/processor7.log 2>&1 &

# python $SCRIPT_PATH --task_file $TXT_DIR/failed_cases1.txt --image_dir $IMAGE_BASE_DIR --gpu_id 0 --neo4j_uri $NEO4J_URI --neo4j_user neo4j --neo4j_password $NEO4J_PASS --mode main --eval_dir /225040511/project/Evaluation_Results/fitzpatrick17k/SkinGPT-X/ --main_csv $MAIN_CSV 