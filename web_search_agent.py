from PIL import Image as PILImage
import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
from prompt_template import *


class WebSearchAgent:
    def __init__(self, model, api_key, domain):
        """
        Initialize the Medical Image Analyzer with an API key.

        Args:
            model (str): The name of the model to use.
            api_key (str): The API key for accessing external services.
            domain (str): The domain of the agent (e.g., "WebSearch").
        """
        self.model = model
        self.api_key = api_key
        self.agent = Agent(
            model=OpenAIChat(id=self.model, api_key=self.api_key),
            tools=[DuckDuckGoTools()],
            debug_mode=False
        )
        self.domain = domain

    def analyze(self, query, image_path):
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
                response = self.agent.run(query, images=[agno_image])
                os.remove(temp_path)
                return response.content
            except Exception as e:
                return f"Analysis error: {e}"
        else:
            return "Please provide a valid image file path."


if __name__ == "__main__":
    model = "gpt-4o-mini"
    api_key = "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"
    query = get_domain_expert_prompt("WebSearch")
    agent = WebSearchAgent(model=model, api_key=api_key, domain="WebSearch")
    image_file_path = "/Users/macbook/Desktop/research project/Skingpt_X/data/images/1.png"  # Replace with your actual image file path
    analysis_result = agent.analyze(query, image_file_path)
    print(analysis_result)