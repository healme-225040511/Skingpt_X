import requests
import json
from typing import Dict, Optional
from prompt_template import get_domain_expert_prompt
from ollama import chat
from ollama import ChatResponse
from typing import Dict, Optional

class SkinGPTAgent:
    def __init__(self, model: str, domain: str):
        """
        Initialize the SkinGPTAgent.

        Args:
            model (str): The name of the model to use.
            domain (str): The domain of the agent.
        """
        self.domain = domain
        self.model = model

    def analyze(self, query: str, image_path: str) -> Optional[Dict]:
        """
        Generate a diagnosis and treatment plan based on an image and clinical context.

        Args:
            query (str): The query to run on the image
            image_path (str): The path to the image file.


        Returns:
            Optional[Dict]: A dictionary containing the diagnosis and treatment plan, or None if the request fails.
        """
        # Prepare the messages for the chat
        messages = [
            {
                "role": "user",
                "content": query,
                "image": [image_path]
            }
        ]
        try:
            # Call the Ollama chat function
            response: ChatResponse = chat(model=self.model, messages=messages)
            return response.message.content
        except Exception as e:
            print(f"Error generating diagnosis and treatment plan: {e}")
            return None


if __name__ == "__main__":
    agent = SkinGPTAgent(model="llama3.2-vision", domain="SkinGPT")
    image_file_path = "/Users/macbook/Desktop/research project/Skingpt_X/data/images/1.png"  # Replace with your actual image file path
    query = get_domain_expert_prompt("SkinGPT")
    analysis_result = agent.analyze(query, image_file_path)
    print(analysis_result)