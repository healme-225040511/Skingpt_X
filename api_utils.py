# from openai import OpenAI
# import time
#
# client = OpenAI(api_key="")
# def generate_response(engine, temperature, max_tokens, frequency_penalty, presence_penalty, stop, system_role, user_input):
#     response = client.chat.completions.create(
#                     model=engine, # engine is the name of the deployment
#                     temperature=temperature,
#                     max_tokens=max_tokens,
#                     top_p=1, # top_p的意思是选择概率质量值之和达到top_p的概率分布采样结果
#                     frequency_penalty=frequency_penalty,
#                     presence_penalty=presence_penalty,
#                     stop=stop,
#                     messages=[
#                         {"role": "system", "content": system_role},
#                         {"role": "user", "content": user_input}
#                     ],
#                     response_format={"type": "json_object"}
#                 )
#     return response.choices[0].message.content

# api_utils.py  ——  HuggingFace 免费模型，无需 API Key
import json, requests, time

HF_MODEL = "microsoft/DialoGPT-medium"  # 也可换成其他 HuggingFace 聊天模型
HF_URL   = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

def generate_response(engine, temperature, max_tokens,
                      frequency_penalty, presence_penalty, stop,
                      system_role, user_input):
    """
    完全兼容原函数签名，返回 JSON 字符串
    """
    prompt = f"{system_role}\n\nUser: {user_input}\nAssistant:"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1.0,
            "return_full_text": False,
            "stop_sequences": stop or []
        }
    }

    # 如果模型冷启动，最多重试 30 秒
    for _ in range(30):
        resp = requests.post(HF_URL, json=payload, timeout=60)
        if resp.status_code == 503:
            time.sleep(2)           # 等待模型加载
            continue
        resp.raise_for_status()
        answer = resp.json()[0]["generated_text"].strip()
        return json.dumps({"response": answer}, ensure_ascii=False)

    return json.dumps({"response": "模型加载超时，请稍后再试"}, ensure_ascii=False)