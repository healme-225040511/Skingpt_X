TXT_DIR="/225040511/project/Skingpt_X/process_list/Dermnet/test"
SCRIPT_PATH='/225040511/project/Skingpt_X/eval_case_review.py'
LOG_DIR="/225040511/project/Skingpt_X/scripts/Dermnet_Sub"
cd /225040511/project/Skingpt_X

nohup bash -c " \
python $SCRIPT_PATH \
  --task_file /225040511/project/Skingpt_X/process_list/Dermnet/test_paths1.txt \
  --gpu_id 0 \
  --main_csv /225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X/panderm_test_predictions.csv \
  --sub_csv /225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X/processed_panderm_subtest_predictions.csv \
  --mode sub \
  --neo4j_uri bolt://100.89.15.55:7687 \
  --neo4j_user neo4j \
  --neo4j_password Czty100165188 \
  --eval_dir /225040511/project/Evaluation_Results/Dermnet_SUB/SkinGPT-X/ \

" > $LOG_DIR/total_sequence.log 2>&1 &