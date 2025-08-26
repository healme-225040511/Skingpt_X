from openai import OpenAI
import requests

# 1. 测试网络连接
try:
    response = requests.get("https://api.openai.com", timeout=5,
                            proxies={"https": "http://192.168.2.204:7897"})
    print("网络连接测试:", "成功" if response.status_code < 400 else f"失败，状态码: {response.status_code}")
except Exception as e:
    print(f"网络连接测试失败: {e}")

# 2. 测试OpenAI连接
try:
    # 初始化客户端（新版写法）
    client = OpenAI(
        api_key="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"
        # 如需代理配置（中国大陆用户需要）
        # http_client=httpx.Client(
        #    proxies="http://127.0.0.1:1080"  # 替换为您的代理地址
        # )
    )

    # 新版API调用方式
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        timeout=10  # 超时设置
    )

    print("OpenAI API测试成功:", response.choices[0].message.content)

except Exception as e:
    print(f"其他错误: {e}")