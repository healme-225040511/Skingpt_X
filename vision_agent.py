import os
import json
import torch
from pathlib import Path
from PIL import Image
from local_llm_utils import local_generate_response_vl # 假设你之前的函数名是这个，或者根据上文修改
import re
from prompt_template import get_visual_findings_prompt
class VisionAgent:
    def __init__(self, model_path=None):
        """
        纯视觉分析 Agent，仅依赖 Qwen-VL-30B 模型。
        """
        # 如果需要，可以在这里设置环境变量
        os.environ["HF_HOME"] = "/225040511/project/hf_cache"
        self.model_path = model_path

    def analyze(self, image_path):
        """
        核心分析函数：不再依赖检索，直接调用 Qwen-30B 提取视觉特征。
        """
        if not os.path.exists(image_path):
            return {"error": f"Image not found at {image_path}"}

        # 1. 获取专用 Prompt
        prompt = get_visual_findings_prompt()

        print(f"🚀 Analyzing image: {os.path.basename(image_path)} using Qwen-30B...")

        try:
            # 2. 调用 Qwen-30B 本地 API (参考之前写的函数)
            # 注意：这里调用的是你刚才定义的那个函数名
            response = local_generate_response_vl(
                temperature=0.1,  # 视觉描述建议低随机性
                max_tokens=2048,
                prompt=prompt,
                image_path=image_path
            )

            # 3. 解析 JSON
            # 寻找 JSON 块，防止模型输出额外的解释文本
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)
                    
            return result

        except json.JSONDecodeError:
            print(f"⚠️ JSON parsing failed. Raw response: {response}")
            return {"error": "json_parse_error", "raw_response": response}
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            return {"error": str(e)}

# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 初始化（由于模型是在函数内部加载的，这里只需实例化）
    agent = VisionAgent()

    test_image = "/225040511/Dataset/HAM10000/test/ISIC_0034525.jpg"
    
    # 直接执行视觉分析
    result = agent.analyze(test_image)

    # 打印结果
    print("\n--- Visual Key Findings ---")
    print(json.dumps(result, indent=2))