from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_imageprocess20200320.client import Client
from alibabacloud_imageprocess20200320 import models as img_models
import base64, json

# 1. 初始化客户端（华东 2 上海地域）
config = open_api_models.Config(
    access_key_id='LTAI5tSXgLZifrqqSZCUuh9v',
    access_key_secret='BRdnyaHh2Ucz0JzFyHVZHQCjveWV2m',
    endpoint='imageprocess.cn-shanghai.aliyuncs.com'
)
client = Client(config)

# 2. 读取本地图片并转 base64
with open(r'/Volumes/T7/SkinGPT-X-Dataset/Dermnet/test/Acne and Rosacea Photos/acne-cystic-122.jpg', 'rb') as f:
    b64_data = base64.b64encode(f.read()).decode()

# 3. 构造请求
req = img_models.DetectSkinDiseaseRequest(
    image=b64_data,      # 本地文件
    org_id='0001',            # 机构代码，可任意
    org_name='demo'           # 机构名称，可任意
)

# 4. 发送请求
resp = client.detect_skin_disease(req)
print(json.dumps(resp.body, ensure_ascii=False, indent=2))