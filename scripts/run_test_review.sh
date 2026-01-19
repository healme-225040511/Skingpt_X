TXT_DIR="/225040511/project/Skingpt_X/process_list/Dermnet/test"
SCRIPT_PATH='/225040511/project/Skingpt_X/eval_case_review.py'
LOG_DIR="/225040511/project/Skingpt_X/scripts/logs_4gpu"
cd /225040511/project/Skingpt_X

# CUDA_VISIBLE_DEVICES=0 nohup python $SCRIPT_PATH --task_file $TXT_DIR/process_list1.txt --gpu_id 0 > $LOG_DIR/gpu0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python $SCRIPT_PATH --task_file $TXT_DIR/process_list2.txt --gpu_id 1 > $LOG_DIR/gpu1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python $SCRIPT_PATH --task_file $TXT_DIR/process_list3.txt --gpu_id 2 > $LOG_DIR/gpu2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python $SCRIPT_PATH --task_file $TXT_DIR/process_list4.txt --gpu_id 3 > $LOG_DIR/gpu3.log 2>&1 &
nohup bash -c " \
python $SCRIPT_PATH --task_file $TXT_DIR/process_list2.txt --gpu_id 1 --neo4j_uri bolt://100.91.178.230:7687 --neo4j_user neo4j --neo4j_password Czty100165188 > $LOG_DIR/gpu1.log 2>&1 && \
python $SCRIPT_PATH --task_file $TXT_DIR/process_list3.txt --gpu_id 2 --neo4j_uri bolt://100.91.178.230:7687 --neo4j_user neo4j --neo4j_password Czty100165188 > $LOG_DIR/gpu2.log 2>&1 && \
python $SCRIPT_PATH --task_file $TXT_DIR/process_list4.txt --gpu_id 3 --neo4j_uri bolt://100.91.178.230:7687 --neo4j_user neo4j --neo4j_password Czty100165188 > $LOG_DIR/gpu3.log 2>&1 && \
python $SCRIPT_PATH --task_file $TXT_DIR/process_list2.txt --gpu_id 0 --neo4j_uri bolt://100.91.178.230:7687 --neo4j_user neo4j --neo4j_password Czty100165188 > $LOG_DIR/gpu0.log 2>&1 \
" > $LOG_DIR/total_sequence1.log 2>&1 &
# echo "✅ Started Part 1 on GPU 0. Log: $LOG_DIR/gpu5.log"
# python $SCRIPT_PATH --task_file $TXT_DIR/process_list2.txt --gpu_id 1 --neo4j_uri bolt://100.91.178.230:7687 --neo4j_user neo4j --neo4j_password Czty100165188
# python $SCRIPT_PATH --task_file $TXT_DIR/process_list3.txt --gpu_id 2 --neo4j_uri bolt://100.91.178.230:7687 --neo4j_user neo4j --neo4j_password Czty100165188
# python $SCRIPT_PATH --task_file $TXT_DIR/process_list4.txt --gpu_id 3 --neo4j_uri bolt://100.91.178.230:7687 --neo4j_user neo4j --neo4j_password Czty100165188
# # 进程 2: GPU 1, 处理 part2.txt
# python $SCRIPT_PATH --task_file $TXT_DIR/process_list2.txt --gpu_id 1
# python $SCRIPT_PATH --task_file $TXT_DIR/process_list3.txt --gpu_id 2
echo '✅ All tasks finished.'
 
# # 进程 3: GPU 2, 处理 part3.txt
# CUDA_VISIBLE_DEVICES=2 nohup python $SCRIPT_PATH --task_file $TXT_DIR/process_list3.txt --gpu_id 2 > $LOG_DIR/gpu7.log 2>&1 &
# echo "✅ Started Part 3 on GPU 2. Log: $LOG_DIR/gpu7.log"

# # 进程 4: GPU 3, 处理 part4.txt
# CUDA_VISIBLE_DEVICES=3 nohup python $SCRIPT_PATH --task_file $TXT_DIR/process_list4.txt --gpu_id 3 > $LOG_DIR/gpu8.log 2>&1 &
# echo "✅ Started Part 4 on GPU 3. Log: $LOG_DIR/gpu8.log"
