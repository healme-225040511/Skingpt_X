from openai import OpenAI
import time

client = OpenAI(api_key="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")
def generate_response(engine, temperature, max_tokens, frequency_penalty, presence_penalty, stop, system_role, user_input):
    response = client.chat.completions.create(
                    model=engine, # engine is the name of the deployment
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1, # top_p的意思是选择概率质量值之和达到top_p的概率分布采样结果
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                    messages=[  
                        {"role": "system", "content": system_role},
                        {"role": "user", "content": user_input}
                    ],
                    response_format={"type": "json_object"}
                )
    return response.choices[0].message.content