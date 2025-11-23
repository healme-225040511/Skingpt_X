import json
from pathlib import Path

root_dir   = Path(r'/Volumes/T7/SkinGPT-X-Dataset/Dermnet/test')          # 改成你的实际根目录
json_file  = Path(r'/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/Reasoning_output.json')     # 你的 JSON 文件
missing_list_file = Path(r'missing_files.txt')  # 可选：把缺失列表落地成文件

# 1. 支持的图片后缀（大小写不敏感）
IMG_SUFFIXES = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

# 2. 遍历磁盘，收集所有图片的“相对路径 + 文件名”
disk_files = {
    str(p.relative_to(root_dir)).replace('\\', '/')   # 统一用 /
    for p in root_dir.rglob('*')
    if p.suffix.lower() in IMG_SUFFIXES
}

# 3. 读 JSON 拿到已有 key
with json_file.open(encoding='utf-8') as f:
    json_keys = set(json.load(f).keys())

# 4. 求缺失
# print(disk_files)
missing = disk_files - json_keys

# 5. 输出
print(f'磁盘上共 {len(disk_files)} 张图片，JSON 中缺失 {len(missing)} 张：\n')
for name in sorted(missing):
    print(name)

# 6. 可选：把缺失列表写到文件
missing_list_file.write_text('\n'.join(sorted(missing)), encoding='utf-8')
print(f'\n缺失文件名已保存到 {missing_list_file.resolve()}')


# import json
# from pathlib import Path
#
# # 定义根目录和JSON文件路径
# root_dir = Path(r'/Volumes/T7/SkinGPT-X-Dataset/Dermnet/test')  # 修改为你的实际根目录
# json_file = Path(r'/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/WebSearch_merged_data.json')  # 修改为你的JSON文件路径
# missing_list_file = Path(r'missing_files.txt')  # 可选：保存缺失文件列表
#
# # 支持的图片后缀（大小写不敏感）
# IMG_SUFFIXES = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
#
# # 遍历磁盘，收集所有图片的“相对路径 + 文件名”
# disk_files = {
#     str(p.relative_to(root_dir)).replace('\\', '/')  # 统一用 /
#     for p in root_dir.rglob('*')
#     if p.suffix.lower() in IMG_SUFFIXES
# }
#
# # 读取JSON文件中的键
# with json_file.open(encoding='utf-8') as f:
#     json_keys = set(json.load(f).keys())
#
# # 计算JSON中有但磁盘中缺失的文件
# missing_from_disk = json_keys - disk_files
#
# # 输出结果
# print(f'JSON中共有 {len(json_keys)} 个键，磁盘中缺失 {len(missing_from_disk)} 个键：\n')
# for name in sorted(missing_from_disk):
#     print(name)
#
# # 可选：将缺失的键保存到文件
# if missing_list_file:
#     missing_list_file.write_text('\n'.join(sorted(missing_from_disk)), encoding='utf-8')
#     print(f'\n缺失的键已保存到 {missing_list_file.resolve()}')