import os, json, subprocess, sys
from pathlib import Path
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def run_all_diseases(
    model_name="gemini-2.5-pro",
    dataset_root="./SkinGPT-X-Dataset/Dermnet/test",
    markdown_file_path="./skin_handbook.md",
    output_root="./SkinGPT-X-EvaluationResults/Dermnet/test"):

    DISEASE_DIRS = os.listdir(dataset_root)
    for disease in DISEASE_DIRS:
        in_dir = Path(dataset_root) / disease
        if not in_dir.exists():
            print(f"⚠️  跳过不存在目录：{in_dir}")
            continue

        jpgs = [p for p in in_dir.rglob("*.jpg")]
        if not jpgs:
            print(f"⚠️  目录下无 jpg：{in_dir}")
            continue

        print(f"\n>>> 开始处理：{disease}  （共 {len(jpgs)} 张）")
        for jpg in jpgs:
            cmd = [
                sys.executable,
                "agent_workflow.py",
                "--model_name", model_name,
                "--image_folder", str(jpg.parent),
                "--markdown_file_path", markdown_file_path,
                "--output_folder", str(output_root)
            ]
            try:
                subprocess.check_call(cmd)
            except subprocess.CalledProcessError as e:
                print(f"[WARN] 子进程失败，跳过 {jpg}  ：{e}")
            except Exception as e:
                print(f"[WARN] 其他异常，跳过 {jpg}  ：{e}")

if __name__ == "__main__":
    run_all_diseases()