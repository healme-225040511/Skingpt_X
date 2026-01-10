import os
from pathlib import Path
import numpy as np
from typing import List

# --- 1. 配置参数 ---

# !!! 必填：请修改此变量为包含图片的文件夹的路径 !!!
# 示例：IMAGE_DIR = "C:/Users/YourName/Desktop/MyImageFolder"
# 示例：IMAGE_DIR = "./photos" (如果文件夹在当前脚本目录下)
DATASET_ROOT = "/Volumes/T7/SkinGPT-X-Dataset/HAM10000"
IMAGE_DIR = "ISIC2018_Task3_Test_Input"

# 输出 TXT 文件的目标文件夹
OUTPUT_DIR = "image_split_output"

# 均分的份数
NUM_PARTS = 4

# 支持的图片文件扩展名
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']


def split_files_into_parts(image_dir: str, output_dir: str, num_parts: int, extensions: List[str]):
    """
    扫描指定目录下的图片文件，并将其路径均分为指定份数，然后写入单独的 TXT 文件。
    """
    image_path = Path(image_dir)
    output_path = Path(output_dir)

    # 检查输入文件夹是否存在
    if not image_path.is_dir():
        print(f"错误：找不到图片文件夹 '{image_dir}'。请将 IMAGE_DIR 变量设置为正确的路径。")
        return

    # 创建输出文件夹
    output_path.mkdir(exist_ok=True)
    print(f"输出文件将保存到: {output_dir}")

    # --- 2. 查找所有图片文件路径 ---
    print(f"正在扫描文件夹: {image_dir}...")
    image_files = []

    # 遍历文件夹中的所有文件（仅顶级目录）
    for item in image_path.iterdir():
        # 检查是否为文件且扩展名是否在列表中
        if item.is_file() and item.suffix.lower() in extensions:
            # 记录文件的绝对路径
            image_files.append(str(item.resolve()))

    if not image_files:
        print(f"警告：在文件夹 '{image_dir}' 中未找到任何符合扩展名要求的图片文件。请检查路径和扩展名。")
        return

    # --- 3. 均分文件列表 ---
    total_files = len(image_files)
    print(f"共找到 {total_files} 个图片文件。将均分为 {num_parts} 份。")

    # 使用 numpy 的 array_split 确保尽可能均匀分配
    file_chunks = np.array_split(np.array(image_files), num_parts)

    # --- 4. 生成 TXT 文件 ---
    for i, chunk in enumerate(file_chunks):
        part_number = i + 1
        output_filename = output_path / f"part_{part_number}.txt"

        # 将文件路径列表写入 TXT 文件，每行一个路径
        with open(output_filename, 'w', encoding='utf-8') as f:
            # 使用换行符连接路径
            f.write('\n'.join(chunk))

        print(f"成功创建文件: {output_filename.name} (包含 {len(chunk)} 个文件路径)")

    print(f"\n任务完成。所有 {num_parts} 个文件已生成。")


if __name__ == "__main__":
    # 如果用户忘记设置路径，给出提醒
    if IMAGE_DIR == "your_image_folder_path_here":
        print("--- 提醒 ---")
        print("您需要修改脚本顶部的 IMAGE_DIR 变量，将其设置为您图片文件夹的实际路径。")

    # 运行主逻辑
    split_files_into_parts(os.path.join(DATASET_ROOT, IMAGE_DIR), os.path.join(DATASET_ROOT, OUTPUT_DIR), NUM_PARTS, IMAGE_EXTENSIONS)