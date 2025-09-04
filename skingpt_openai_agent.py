import openai
from typing import Dict, Optional
from prompt_template import get_domain_expert_prompt

class SkinGPTOpenAIAgent:
    def __init__(self, model: str, domain: str, api_key: str):
        """
        Initialize the SkinGPTAgent.

        Args:
            model (str): The name of the model to use.
            domain (str): The domain of the agent.
            api_key (str): The OpenAI API key.
        """
        self.domain = domain
        self.model = model
        openai.api_key = api_key

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
                "content": f"{query} [Image: {image_path}]"
            }
        ]
        try:
            # Call the OpenAI chat function
            print(messages)
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating diagnosis and treatment plan: {e}")
            return None

if __name__ == "__main__":
    # Replace with your actual OpenAI API key
    api_key = "your_openai_api_key_here"
    agent = SkinGPTOpenAIAgent(model="gpt-4", domain="SkinGPT", api_key=api_key)
    image_file_path = "/Users/macbook/Desktop/research project/Skingpt_X/data/images/1.png"  # Replace with your actual image file path
    query = get_domain_expert_prompt("SkinGPT")
    analysis_result = agent.analyze(query, image_file_path)
    print(analysis_result)