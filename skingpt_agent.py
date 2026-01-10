import csv
import pathlib
import tempfile
import re
import openai
import base64
import asyncio
from typing import Dict, Optional
from google.genai import types
import json
from google import genai
from openai import OpenAI
from local_llm_utils import local_generate_response_vl
from Constants import ISIC_PRECSV_PATH
from prompt_template import get_domain_expert_prompt
from utils import encode_image_to_base64
from local_llm_utils import parse_skin_disease_path
from utils import build_prelimary_text
from Constants import DERMNET_DISEASE_NAME
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

class SkingptAgent:
    def __init__(self, model: str = "gpt-4o", domain: str = "SkinGPT", api_key: str = "", pre_csv_path: str = "./Dermnet_predprob.csv"):
        self.domain = domain
        self.model = model                       # 默认用 gpt-4o，也可传 gpt-4o-mini
        self.api_key = api_key
        self.pre_csv_path = pre_csv_path

    # ---------- 工具：图片 → base64 ----------
    @staticmethod
    def _encode_image(image_path: str) -> str:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    # ---------- 工具：图片 → 临时文件 ----------
    @staticmethod
    def _image_to_temp_file(image_path: str) -> str:
        """把任意本地图片转成一个临时 jpeg 文件，返回路径供 Gemini 使用"""
        from PIL import Image
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        Image.open(image_path).convert("RGB").save(tmp.name, "JPEG")
        return tmp.name

    # ---------- 主要接口 ----------
    async def analyze(self, query: str, image_path: str) -> Optional[str]:
        """
        基于视觉模型给出诊断/建议
        :param query: 临床问题字符串
        :param image_path: 本地图片路径
        :return: 模型返回文本
        """
        tmp_path = self._image_to_temp_file(image_path)
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        img_part = types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
        try:
           
            response = local_generate_response_vl(
                        temperature=0.01,
                        max_tokens=4096,
                        prompt=query,
                        image_path=image_path
                    )
            pattern = r'```json\s*(.*?)```'
            match = re.search(pattern, response, flags=re.S)   # re.S 让 . 匹配换行
            json_str = match.group(1).strip() if match else response
            return json.loads(json_str)
        except Exception as e:
            print(f"[SkinGPT Vision] Error: {e}")
            return None
        finally:
            pathlib.Path(tmp_path).unlink(missing_ok=True)  # 清理临时文件
    def get_prob_vec(self, image_path: str):
        """
        传入本地图片路径，返回对应 n 维概率向量；找不到返回 None
        """
        # 1. 把本地路径转成 csv 里的 filename 格式
        #    例如 ./SkinGPT-X-Dataset/Dermnet/test/xxx/yyy.jpg -> xxx/yyy.jpg
        path = pathlib.Path(image_path).resolve()
        # 假设 csv 里存的都是“相对/xxx/yyy.jpg”形式，且目录层级固定
        # 这里简单取后两级，可按实际调整
        key = str(pathlib.Path(*path.parts[-2:])).replace("\\", "/")
        # 2. 读 csv 找行
        with open(self.pre_csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                feature_dim = len(row) - 3
                if row["filename"] == key:
                    # 3. 提取 prob_cls0 ... prob_cls22
                    if self.pre_csv_path == ISIC_PRECSV_PATH:
                        vec = [float(row[f"prob_cls"])]
                        return vec
                    else:
                        vec = [float(row[f"prob_cls{i}"]) for i in range(feature_dim)]
                        return vec
        return None

# ------------------- 快速自测 -------------------
if __name__ == "__main__":
    api_key = "sk-iCv69YeaJn8TXm9tk6ZUUAqftw51aB2yddvmstNNl7QjkIKB"
    agent = SkingptAgent(model="gemini-2.5-pro", api_key=api_key, pre_csv_path='./Panderm_Dermnet_predprob.csv')
    image_file_path = "/Dermnet_image/V001/test/Light Diseases and Disorders of Pigmentation/phototoxic-reactions-17.jpg"  # Replace with your actual image file path
    pre_analysis = build_prelimary_text(agent.get_prob_vec(image_file_path),DERMNET_DISEASE_NAME[:-1])
    print(pre_analysis)
    # query = f"""
    #         You are a frontline medical professional specializing in performing initial patient assessments based on dermatological images.  Now you have know that the correct diagnosis of this image is {skin_disease}

    #         Your primary role is to organize preliminary medical observations from images and provide insights to support further diagnosis and agent collaboration.
    #         {pre_analysis}
    #         ---

    #         ### ⚠️ CRITICAL INSTRUCTIONS FOR KEY FINDINGS:

    #         When writing the "KeyFindings" section, you MUST:
    #         Start with anatomical location and context …
    #         Describe morphology in clinical terms …
    #         Highlight "red flags" or warning signs …
    #         Explicitly rate severity at the end …
    #         Avoid bullet points or lists …
    #         Additionally, immediately after the paragraph, append a separate sentence that begins exactly with:
    #         "Critical diagnostic differences:"
    #         and then list 1–2 ultra-short phrases that would help distinguish this condition from the most likely differential diagnoses (e.g., "silvery scale on extension, Auspitz-positive" or "spares nasolabial fold, no mucosal involvement").
    #         Do not use line breaks or bullets inside this sentence.
    #         ---

    #         ### ✅ FORMAT YOUR RESPONSE STRICTLY AS JSON:

    #         {{
    #             "DifferentialDiagnoses": [top-10 possible broad disease categories. You should select in {DERMNET_DISEASE_NAME}]
    #             "KeyFindings": "[single cohesive paragraph + Critical diagnostic differences: ...]",
    #             "CriticalFeatures": ["phrase1", "phrase2"],
    #             "KnowledgeAndResearch": "..."
    #         }}

        
    #         ---

    #         Now analyze the provided image and generate your response in the exact JSON format above.
    #         """
    # skin_disease = parse_skin_disease_path(image_file_path)
    # print(skin_disease)
    query = f"""
        You are a frontline medical professional specializing in performing initial patient assessments based on dermatological images. Now you have know that the correct diagnosis of this image is Light Diseases and Disorders of Pigmentation

        Your primary role is to organize preliminary medical observations from images and provide insights to support further diagnosis and agent collaboration.
        ---

        ### ⚠️ CRITICAL INSTRUCTIONS:

        ### 1. ImageRegion
        - Identify the affected anatomical region and positioning of the lesion or area of interest.
        - Note any contextual details about the region (e.g., sun-exposed area, friction-prone area).
        ### 2. KeyFindings
        - Systematically list primary observations focusing on the lesion(s) or skin abnormalities.
        - Describe the lesion(s) in detail, including:
            - Location, size, shape, and distribution.
            - Color variations, textures, borders, and any unique features.
            - Associated symptoms (e.g., itching, pain, scaling).
        - Rate severity: Normal / Mild / Moderate / Severe.
        ### 3. PrimaryDiagnosis
        - Provide a primary diagnosis with a confidence level (e.g., High/Medium/Low) based on observed evidence and supported by information retrieved from the knowledge base.
        - List differential diagnoses in order of likelihood, considering similar skin conditions, and back each with evidence from the knowledge base.
        - Support each diagnosis with observed evidence from the patient's imaging and related studies accessed through the knowledge base.
        - Highlight any critical or urgent findings that require immediate attention.
        ---

        ### ✅ FORMAT YOUR RESPONSE STRICTLY AS JSON:

        {{
            "ImageRegion": "..."
            "KeyFindings": "",
            "PrimaryDiagnosis": ["Diagnosis1", "Diagnosis2"],
        }}

    
        ---

        Now analyze the provided image and generate your response in the exact JSON format above. You should control you output in 5000 words
        """
    async def main():
        analysis_result = await agent.analyze(query, image_file_path)
        print(analysis_result)

    asyncio.run(main())