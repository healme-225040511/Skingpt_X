from agno.models.google import Gemini
from google import genai
from google.genai import types
from openai import OpenAI
import time

safety_settings = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
]

# ... 其他现有导入 ...

# 新增导入
from PIL import Image
from google import genai
from google.genai import types
import os  # 用于检查文件是否存在



def generate_response(engine, temperature, max_tokens, frequency_penalty, presence_penalty, stop, system_role,
                      user_input, image_path: str = None, api_key: str = None):
    """
    生成多模态内容的响应。

    新增 Args:
        image_path (str): 图片文件的本地路径，如果为 None 则只发送文本。
    """
    client = genai.Client(api_key=api_key)

    # 1. 初始化内容列表
    contents = []

    # # 2. 处理图片：加载图片文件并添加到内容列表
    # if image_path:
    #     if not os.path.exists(image_path):
    #         raise FileNotFoundError(f"图片文件未找到: {image_path}")
    #
    #     try:
    #         # 使用 PIL 加载图片。Gemini API 可以直接接受 PIL.Image 对象
    #         img = Image.open(image_path)
    #         contents.append(img)
    #         print(f"✅ 图片 {os.path.basename(image_path)} 已加载。")
    #     except Exception as e:
    #         print(f"加载图片时发生错误: {e}")
    #         # 如果加载失败，可以继续只发送文本
    #         pass

    # 3. 处理文本：将 System Role 和 User Input 合并后添加到内容列表
    full_prompt_text = system_role + user_input
    contents.append(full_prompt_text)

    if not contents:
        return "错误：没有内容（文本或图片）发送给模型。"

    response = client.models.generate_content(
        model=engine,
        # 将图片对象和文本字符串作为列表传递
        contents=contents,
        config=types.GenerateContentConfig(
            safety_settings=safety_settings,
            temperature=temperature,
            stop_sequences=stop,
            top_p=1,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        ),
    )
    return response.text

# --- 示例调用 ---
# 假设你有一个图片文件名为 'my_chart.png'
# response_text = generate_response(
#     engine='gemini-2.5-flash',
#     temperature=0.7,
#     max_tokens=2048,
#     frequency_penalty=0.0,
#     presence_penalty=0.0,
#     stop=None,
#     system_role="你是一位数据分析专家。请根据提供的图表进行详细解读。",
#     user_input="请分析这张图表，总结主要趋势。",
#     image_path='path/to/my_chart.png', # <--- 图片路径
#     api_key='YOUR_API_KEY'
# )
# print(response_text)