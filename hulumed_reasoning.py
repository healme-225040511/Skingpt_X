#!/usr/bin/env python3
"""
批量为 temp_dataset.txt 里的每张图生成医学报告，结果统一写入 JSON
用法：python infer_report.py
"""
import os, torch, tqdm, json
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# ----------- 可改参数 -----------
TXT_PATH      = 'temp_dataset.txt'     # 之前抽样的索引
MODEL_ID      = "ZJU-AI4H/Hulu-Med-7B"         # 换成你本地或 HuggingFace 的模型名
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE         = torch.bfloat16 if DEVICE=='cuda' else torch.float32
MAX_NEW_TOKENS= 1024
JSON_OUT      = 'reports.json'         # ← 新增：统一保存的 JSON
# --------------------------------

processor = AutoProcessor.from_pretrained(MODEL_ID)
model     = AutoModelForVision2Seq.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)

def generate_report(image_path:str) -> str:
    """单张图推理，返回生成的报告字符串"""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": {"image_path": image_path}},
                {"type": "text",  "text":  "Generate a medical report for this image."},
            ]
        }
    ]

    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(DEVICE)

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    report = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        use_think=False
    )[0].strip()
    return report

def main():
    # 1. 读取待处理列表
    with open(TXT_PATH, 'r', encoding='utf-8') as f:
        img_list = [line.strip() for line in f if line.strip()]
    print(f'共 {len(img_list)} 张图待处理')

    # 2. 断点续跑：若 JSON 已存在，先读出来
    reports = {}
    if os.path.exists(JSON_OUT):
        reports = json.load(open(JSON_OUT, 'r', encoding='utf-8'))
        print(f'已存在 JSON，已包含 {len(reports)} 条记录，继续追加…')

    # 3. 遍历推理
    for img_path in tqdm.tqdm(img_list, desc='Infer'):
        if not os.path.isfile(img_path):
            tqdm.tqdm.write(f'跳过缺失文件: {img_path}')
            continue
        if img_path in reports:          # 已生成直接跳过
            tqdm.tqdm.write(f'已存在报告，跳过: {img_path}')
            continue
        try:
            report = generate_report(img_path)
            reports[img_path] = report   # 更新内存
            # 实时落盘，防止中途崩溃
            json.dump(reports, open(JSON_OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        except Exception as e:
            tqdm.tqdm.write(f'处理失败 {img_path}: {e}')

    print(f'全部完成！共 {len(reports)} 条报告已保存到 {JSON_OUT}')

if __name__ == '__main__':
    main()