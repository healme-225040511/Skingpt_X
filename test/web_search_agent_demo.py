import base64
from openai import OpenAI
import os
import sys


client = OpenAI(api_key="sk-", base_url="https://hiapi.online/v1")





def encode_image_to_base64(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"错误：找不到图片文件 '{image_path}'")

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')



image_path = "C:/Users/12111/Pictures/tset.png"

try:
    base64_image = encode_image_to_base64(image_path)

    image_mime_type = "image/jpeg"
    if image_path.lower().endswith(".png"):
        image_mime_type = "image/png"
    elif image_path.lower().endswith(".gif"):
        image_mime_type = "image/gif"
    elif image_path.lower().endswith(".webp"):
        image_mime_type = "image/webp"


    completion_stream = client.chat.completions.create(
        model="gemini-2.5-flash-image-preview",
        stream=True,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "美化一下."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_mime_type};base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    )


    print("AI的回应: ", end="")
    for chunk in completion_stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="")
            sys.stdout.flush()

    print()

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"发生了一个错误: {e}")