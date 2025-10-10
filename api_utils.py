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


def generate_response(engine, temperature, max_tokens, frequency_penalty, presence_penalty, stop, system_role, user_input, api_key):
    if api_key.startswith("sk-"):
        client = OpenAI(api_key=api_key, base_url="https://hiapi.online/v1")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{system_role}"
                    },
                    {
                        "type": "text",
                        "text": f"{user_input}"
                    }
                ]
            }
        ]
        resp = client.chat.completions.create(model=engine, messages=messages)
        return resp.choices[0].message.content
    else:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
                        model=engine, # engine is the name of the deployment
                        contents=[system_role+user_input],
                        config=types.GenerateContentConfig(
                            safety_settings=safety_settings,  # 放到 config 里
                            temperature=temperature,  # 可选
                            stop_sequences=stop,  # 可选
                            top_p=1,
                            frequency_penalty=frequency_penalty,
                            presence_penalty=presence_penalty,
                        ),
                    )
    return response.text