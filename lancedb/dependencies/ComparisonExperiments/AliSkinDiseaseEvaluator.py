#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量调用阿里云 DetectSkinDisease 接口
"""
import asyncio
import csv
import os
from pathlib import Path
from typing import List

import aiofiles
from aiohttp import ClientSession
from alibabacloud_credentials.client import Client as CredentialClient
from alibabacloud_tea_openapi.client import Client as OpenApiClient
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_tea_util import models as util_models
from tqdm.asyncio import tqdm_asyncio

# 阿里云接口配置
API_VERSION = '2020-03-20'
ACTION = 'DetectSkinDisease'
ENDPOINT = 'imageprocess.cn-shanghai.aliyuncs.com'
PROTOCOL = 'HTTPS'
METHOD = 'POST'

CONCURRENCY = 1       # 并发量
RESULT_CSV = 'result.csv'  # 输出文件
IMAGE_ROOT = Path('/Volumes/T7/SkinGPT-X-Dataset/Dermnet/test')

# 常见图片后缀
IMG_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


# ---------- 阿里云客户端 ----------
def create_client() -> OpenApiClient:
    config = open_api_models.Config(
        access_key_id='LTAI5tSXgLZifrqqSZCUuh9v',
        access_key_secret='BRdnyaHh2Ucz0JzFyHVZHQCjveWV2m'
    )
    config.endpoint = ENDPOINT
    return OpenApiClient(config)


def build_params() -> open_api_models.Params:
    return open_api_models.Params(
        action=ACTION,
        version=API_VERSION,
        protocol=PROTOCOL,
        method=METHOD,
        auth_type='AK',
        style='RPC',
        pathname='/',
        req_body_type='formData',
        body_type='json'
    )


CLIENT = create_client()
PARAMS = build_params()


# ---------- 单张图片推理 ----------
async def predict_one(image_path: Path) -> dict:
    """异步调用接口，返回 dict"""
    # 先把本地文件上传到可公网访问的临时 URL
    # 为简化 demo，这里直接使用「本地文件绝对路径」作为 file:// URL
    # 阿里云接口支持 file://  scheme（需确保运行环境能访问）
    file_url = image_path.absolute().as_uri()

    body = {
        'Url': file_url,
        'OrgId': '0001',
        'OrgName': 'demo'
    }
    request = open_api_models.OpenApiRequest(body=body)
    runtime = util_models.RuntimeOptions()

    try:
        resp = await CLIENT.call_api_async(PARAMS, request, runtime)
        # resp 是 Map，真正的 JSON 在 resp['body']
        return {
            'image': str(image_path),
            'error': '',
            'result': resp.get('body', {})
        }
    except Exception as e:
        return {
            'image': str(image_path),
            'error': str(e),
            'result': {}
        }


# ---------- 扫描所有图片 ----------
def scan_images(root: Path) -> List[Path]:
    return [p for p in root.rglob('*') if p.suffix.lower() in IMG_SUFFIXES]


# ---------- 主流程 ----------
async def main():
    images = scan_images(IMAGE_ROOT)
    if not images:
        print('未找到任何图片，请检查目录:', IMAGE_ROOT.absolute())
        return

    # 读取已有结果，实现断点续跑
    done = set()
    if Path(RESULT_CSV).exists():
        with open(RESULT_CSV, newline='', encoding='utf-8') as f:
            done = {row['image'] for row in csv.DictReader(f)}

    todo = [img for img in images if str(img) not in done]
    print(f'共 {len(images)} 张图，已处理 {len(done)} 张，待处理 {len(todo)} 张')

    # 并发执行
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def sem_predict(img: Path):
        async with semaphore:
            return await predict_one(img)

    results = []
    for coro in tqdm_asyncio.as_completed([sem_predict(img) for img in todo], total=len(todo)):
        results.append(await coro)
        print(results)

    # 写入 CSV（追加模式）
    fieldnames = ['image', 'error', 'result']
    with open(RESULT_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not done:  # 首次写表头
            writer.writeheader()
        writer.writerows(results)

    print('全部完成！结果见:', RESULT_CSV)


# ---------- 入口 ----------
if __name__ == '__main__':
    asyncio.run(main())