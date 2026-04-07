TXT_DIR="/225040511/project/Skingpt_X/process_list/Dermnet/test"
SCRIPT_PATH='/225040511/project/Skingpt_X/eval_case_review.py'
LOG_DIR="/225040511/project/Skingpt_X/scripts/Dermnet_Sub"
cd /225040511/project/Skingpt_X

# # GPU 0
# nohup bash -c " 
# python $SCRIPT_PATH \
#   --task_file /225040511/project/Skingpt_X/process_list/Dermnet/test/process_list1.txt \
#   --gpu_id 0 \
#   --main_csv /225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X/panderm_test_predictions.csv \
#   --sub_csv /225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X/processed_panderm_subtest_predictions.csv \
#   --mode sub \
#   --neo4j_uri bolt://100.90.48.20:7687 \
#   --neo4j_user neo4j \
#   --neo4j_password Czty100165188 \
#   --eval_dir /225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X/ \
#   --image_dir /225040511/Dataset/Dermnet\

# " > $LOG_DIR/total_sequence0.log 2>&1 &

# # GPU 1
# nohup bash -c " 
# python $SCRIPT_PATH \
#   --task_file /225040511/project/Skingpt_X/process_list/Dermnet/test/process_list2.txt \
#   --gpu_id 1 \
#   --main_csv /225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X/panderm_test_predictions.csv \
#   --sub_csv /225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X/processed_panderm_subtest_predictions.csv \
#   --mode sub \
#   --neo4j_uri bolt://100.90.48.20:7687 \
#   --neo4j_user neo4j \
#   --neo4j_password Czty100165188 \
#   --eval_dir /225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X/ \
#   --image_dir /225040511/Dataset/Dermnet\

# " > $LOG_DIR/total_sequence1.log 2>&1 &

# # GPU 2
# nohup bash -c " 
# python $SCRIPT_PATH \
#   --task_file /225040511/project/Skingpt_X/process_list/Dermnet/test/process_list3.txt \
#   --gpu_id 2 \
#   --main_csv /225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X/panderm_test_predictions.csv \
#   --sub_csv /225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X/processed_panderm_subtest_predictions.csv \
#   --mode sub \
#   --neo4j_uri bolt://100.90.48.20:7687 \
#   --neo4j_user neo4j \
#   --neo4j_password Czty100165188 \
#   --eval_dir /225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X/ \
#   --image_dir /225040511/Dataset/Dermnet\

# " > $LOG_DIR/total_sequence2.log 2>&1 &

# # GPU 3
# nohup bash -c " 
# python $SCRIPT_PATH \
#   --task_file /225040511/project/Skingpt_X/process_list/Dermnet/test/process_list4.txt \
#   --gpu_id 3 \
#   --main_csv /225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X/panderm_test_predictions.csv \
#   --sub_csv /225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X/processed_panderm_subtest_predictions.csv \
#   --mode sub \
#   --neo4j_uri bolt://100.90.48.20:7687 \
#   --neo4j_user neo4j \
#   --neo4j_password Czty100165188 \
#   --eval_dir /225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X/ \
#   --image_dir /225040511/Dataset/Dermnet\

# " > $LOG_DIR/total_sequence3.log 2>&1 &

# GPU 4
nohup bash -c " 
python $SCRIPT_PATH \
  --task_file /225040511/project/Skingpt_X/process_list/Dermnet/test/process_list5.txt \
  --gpu_id 4 \
  --main_csv /225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X/panderm_test_predictions.csv \
  --sub_csv /225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X/processed_panderm_subtest_predictions.csv \
  --mode sub \
  --neo4j_uri bolt://100.90.48.20:7687 \
  --neo4j_user neo4j \
  --neo4j_password Czty100165188 \
  --eval_dir /225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X/ \
  --image_dir /225040511/Dataset/Dermnet\

" > $LOG_DIR/total_sequence4.log 2>&1 &

# GPU 5
nohup bash -c " 
python $SCRIPT_PATH \
  --task_file /225040511/project/Skingpt_X/process_list/Dermnet/test/process_list6.txt \
  --gpu_id 5 \
  --main_csv /225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X/panderm_test_predictions.csv \
  --sub_csv /225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X/processed_panderm_subtest_predictions.csv \
  --mode sub \
  --neo4j_uri bolt://100.90.48.20:7687 \
  --neo4j_user neo4j \
  --neo4j_password Czty100165188 \
  --eval_dir /225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X/ \
  --image_dir /225040511/Dataset/Dermnet\

" > $LOG_DIR/total_sequence5.log 2>&1 &

# GPU 6
nohup bash -c " 
python $SCRIPT_PATH \
  --task_file /225040511/project/Skingpt_X/process_list/Dermnet/test/process_list7.txt \
  --gpu_id 6 \
  --main_csv /225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X/panderm_test_predictions.csv \
  --sub_csv /225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X/processed_panderm_subtest_predictions.csv \
  --mode sub \
  --neo4j_uri bolt://100.90.48.20:7687 \
  --neo4j_user neo4j \
  --neo4j_password Czty100165188 \
  --eval_dir /225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X/ \
  --image_dir /225040511/Dataset/Dermnet\

" > $LOG_DIR/total_sequence6.log 2>&1 &

# GPU 7
nohup bash -c " 
python $SCRIPT_PATH \
  --task_file /225040511/project/Skingpt_X/process_list/Dermnet/test/process_list8.txt \
  --gpu_id 7 \
  --main_csv /225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X/panderm_test_predictions.csv \
  --sub_csv /225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X/processed_panderm_subtest_predictions.csv \
  --mode sub \
  --neo4j_uri bolt://100.90.48.20:7687 \
  --neo4j_user neo4j \
  --neo4j_password Czty100165188 \
  --eval_dir /225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X/ \
  --image_dir /225040511/Dataset/Dermnet\

" > $LOG_DIR/total_sequence7.log 2>&1 &
