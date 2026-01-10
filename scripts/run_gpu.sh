WORK_DIR="/225040511/project/Skingpt_X"
 
python $WORK_DIR/AgentWorkflowEvaluator.py \
      --dataset_root /Dermnet_image/V001/test \
      --is_single_agent True \
      --agent_type 2 \
      --api_key AIzaSyDClRNJkcDgHv2wA90v6TODPvBlu8umIWU \
      --output_root /225040511/project/SkinGPT-X-EvaluationResults/Dermnet/new_rag/reasoning \
      --pending_set_path $WORK_DIR/test/process_wronglist_Dermnet.txt