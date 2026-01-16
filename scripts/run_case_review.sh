#!/usr/bin/env bash
# run_4gpu.sh 用法：./run_4gpu.sh [start|stop|tail]

LOG_DIR="./logs_4gpu"
mkdir -p "$LOG_DIR"
WORK_DIR="/225040511/project/Skingpt_X"

case "$1" in
  start)
    echo "🚀 [第 2 阶段] 启动前 4 个任务  ..."
    

    CUDA_VISIBLE_DEVICES=0 nohup python $WORK_DIR/AgentWorkflowEvaluator.py \
      --dataset_root /SuperDermnet/V001/test \
      --output_root /225040511/project/Evaluation_Results/SuperDermnet/SkinGPT-X/test/tmp1 \
      --pending_set_path /225040511/project/Skingpt_X/process_list/SuperDermnet/test_wrong/process_list1.txt > "$LOG_DIR/gpu5.log" 2>&1 &

    CUDA_VISIBLE_DEVICES=1 nohup python $WORK_DIR/AgentWorkflowEvaluator.py \
      --dataset_root /SuperDermnet/V001/test \
      --output_root /225040511/project/Evaluation_Results/SuperDermnet/SkinGPT-X/test/tmp2 \
      --pending_set_path /225040511/project/Skingpt_X/process_list/SuperDermnet/test_wrong/process_list2.txt > "$LOG_DIR/gpu6.log" 2>&1 &

    CUDA_VISIBLE_DEVICES=2 nohup python $WORK_DIR/AgentWorkflowEvaluator.py \
      --dataset_root /SuperDermnet/V001/test \
      --output_root /225040511/project/Evaluation_Results/SuperDermnet/SkinGPT-X/test/tmp3 \
      --pending_set_path /225040511/project/Skingpt_X/process_list/SuperDermnet/test_wrong/process_list3.txt > "$LOG_DIR/gpu7.log" 2>&1 &

    CUDA_VISIBLE_DEVICES=3 nohup python $WORK_DIR/AgentWorkflowEvaluator.py \
      --dataset_root /SuperDermnet/V001/test \
      --output_root /225040511/project/Evaluation_Results/SuperDermnet/SkinGPT-X/test/tmp4 \
      --pending_set_path /225040511/project/Skingpt_X/process_list/SuperDermnet/test_wrong/process_list4.txt > "$LOG_DIR/gpu8.log" 2>&1 &

    # CUDA_VISIBLE_DEVICES=4 nohup python $WORK_DIR/AgentWorkflowEvaluator.py \
    #   --dataset_root /fitzpatrick17k/V002/dataset/test \
    #   --output_root /225040511/project/Evaluation_Results/fitzpatrick17k/test5 \
    #   --pending_set_path /225040511/project/Skingpt_X/process_list/fitzpatrick17k/test/process_list5.txt > "$LOG_DIR/gpu9.log" 2>&1 &

    # CUDA_VISIBLE_DEVICES=5 nohup python $WORK_DIR/AgentWorkflowEvaluator.py \
    #   --dataset_root /fitzpatrick17k/V002/dataset/test \
    #   --output_root /225040511/project/Evaluation_Results/fitzpatrick17k/test6 \
    #   --pending_set_path /225040511/project/Skingpt_X/process_list/fitzpatrick17k/test/process_list6.txt > "$LOG_DIR/gpu10.log" 2>&1 &

    # CUDA_VISIBLE_DEVICES=6 nohup python $WORK_DIR/AgentWorkflowEvaluator.py \
    #   --dataset_root /fitzpatrick17k/V002/dataset/test \
    #   --output_root /225040511/project/Evaluation_Results/fitzpatrick17k/test7 \
    #   --pending_set_path /225040511/project/Skingpt_X/process_list/fitzpatrick17k/test/process_list7.txt > "$LOG_DIR/gpu11.log" 2>&1 &

    # CUDA_VISIBLE_DEVICES=7 nohup python $WORK_DIR/AgentWorkflowEvaluator.py \
    #   --dataset_root /fitzpatrick17k/V002/dataset/test \
    #   --output_root /225040511/project/Evaluation_Results/fitzpatrick17k/test8 \
    #   --pending_set_path /225040511/project/Skingpt_X/process_list/fitzpatrick17k/test/process_list8.txt > "$LOG_DIR/gpu12.log" 2>&1 &

    echo "✅ 所有阶段任务已提交。日志查看：$LOG_DIR"
    ;;

  stop)
    echo "🛑 停止所有任务 ..."
    pkill -f "AgentWorkflowEvaluator.py"
    echo "✅ 已 kill"
    ;;

  tail)
    echo "📒 实时合并日志（Ctrl+C 退出）"
    tail -f "$LOG_DIR"/gpu*.log
    ;;

  *)
    echo "用法：$0 {start|stop|tail}"
    exit 1
    ;;
esac