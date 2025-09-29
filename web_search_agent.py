import pathlib
import uuid

from PIL import Image as PILImage
import os
import asyncio
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
import logging
from SearchAPITools import BochaSearchTool
from prompt_template import *
from agno.models.google.gemini import Gemini

class WebSearchAgent:
    def __init__(self, model: str, api_key: str, domain: str = "WebSearch",
                 searchapi_key: str = None):
        """
        Initialize the Medical Image Analyzer with an API key.

        Args:
            model (str): The name of the model to use.
            api_key (str): The API key for accessing external services.
            domain (str): The domain of the agent (e.g., "WebSearch").
        """
        self.model = model
        self.api_key = api_key
        self.domain = domain
        # 替换工具
        self.agent = Agent(
            model=Gemini(id=self.model, api_key=self.api_key),
            tools=[BochaSearchTool(api_key=searchapi_key)],  # ←这里
            debug_mode=False
        )

    async def analyze(self, query, image_path):
        """
        Process an image file and get its analysis.

        Args:
            image_path (str): The path to the image file.
            query (str): The query to run on the image.

        Returns:
            str: The analysis result or an error message.
        """
        if image_path is not None and os.path.exists(image_path):
            try:
                # Load and resize the image
                image = PILImage.open(image_path)
                width, height = image.size
                aspect_ratio = width / height
                new_width = 500
                new_height = int(new_width / aspect_ratio)
                resized_image = image.resize((new_width, new_height))
                temp_path = "temp_resized_image.png"
                resized_image.save(temp_path)

                # Run analysis
                agno_image = AgnoImage(filepath=temp_path)  # Adjust if constructor differs

                # Use asyncio.to_thread to run the synchronous method in a separate thread
                response = await asyncio.to_thread(self.agent.run, query, images=[agno_image])
                os.remove(temp_path)
                return response.content
            except Exception as e:
                return f"Analysis error: {e}"
        else:
            return "Please provide a valid image file path."

if __name__ == "__main__":
    model = "gemini-2.5-pro"
    api_key = "AIzaSyC-9og_9OsxvKZ0rBXeMGboXBrMOpG5-do"
    query = get_domain_expert_prompt("WebSearch")
    agent = WebSearchAgent(model=model, api_key=api_key, domain="WebSearch", searchapi_key='sk-74829ed96e1c4d9793507d546527f5de')
    image_file_path = "/Volumes/T7/SkinGPT-X-Dataset/Dermnet/test/Urticaria Hives/PUPPP-18.jpg"  # Replace with your actual image file path

    async def main():
        analysis_result = await agent.analyze(query, image_file_path)
        print(analysis_result)

    asyncio.run(main())