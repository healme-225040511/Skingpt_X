# run_agentflow.py
import subprocess
import sys

def run_agentflow(
    model_name="gpt-4o-mini",
    image_folder="/Volumes/T7/SkinGPT-X-Dataset/Dermnet/test/Acne and Rosacea Photos",
    markdown_file_path="skin_handbook.md",
    output_folder="/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/output/Acne and Rosacea Photos"
):
    """
    封装 agent_workflow.py 的调用，参数与 CLI 完全一致。
    如需自定义，直接改默认参数或外部传参即可。
    """
    cmd = [
        sys.executable,           # 当前 Python 解释器
        "agent_workflow.py",
        "--model_name", model_name,
        "--image_folder", image_folder,
        "--markdown_file_path", markdown_file_path,
        "--output_folder", output_folder
    ]
    print(">>> Running command:\n", " ".join(cmd))
    subprocess.check_call(cmd)    # 实时输出到终端，出错会抛异常

if __name__ == "__main__":
    run_agentflow()               # 使用默认参数