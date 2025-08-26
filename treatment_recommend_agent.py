from PIL import Image as PILImage
import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from prompt_template import *
import json
from utils import remove_json_markers

class TreatmentRecommendAgent:
    def __init__(self, model, api_key):
        """
        Initialize the Treatment Recommend Agent with an API key.
        
        Args:
            model (str): The name of the model to use.
            api_key (str): The API key for accessing external services.
        """
        self.model = model
        self.api_key = api_key
        self.agent = Agent(
            model=OpenAIChat(id=self.model, api_key=self.api_key),
            tools=[DuckDuckGoTools()],
            debug_mode=False,
            structured_outputs=True
        )
    
    def analyze(self, current_case):
        """
        Get the treatment recommendations based on the query.
        
        Args:
            current_case (dict): The current case data.
            
        Returns:
            str: The analysis result or an error message.
        """
        # Run analysis
        query = get_treatment_recommend_prompt(current_case)
        response = self.agent.run(query)
        response = remove_json_markers(response.content)
        return response


if __name__ == "__main__":
    model = "gpt-4o-mini"
    api_key = ""  # Replace with your actual API key
    current_case = {
        "PrimaryDiagnosis": "Squamous Cell Carcinoma (SCC)",
        "ConfidenceLevel": "High",
        "DifferentialDiagnoses": [
            "Basal Cell Carcinoma (BCC)",
            "Melanoma",
            "Actinic Keratosis"
        ],
        "KeyFindings": "The lesion is located on the lateral aspect of the right cheek, near the eye, measuring approximately 1.5 cm in diameter. It has an irregular, triangular shape with color variations ranging from dark brown to black, and a shiny, possibly ulcerated surface. The borders are irregular, with some areas appearing well-defined. Symptoms may include discomfort, and the severity is rated as moderate, indicating a potential for malignancy.",
        "KnowledgeAndResearch": "According to the American Academy of Dermatology (AAD), excisional biopsy is the standard protocol for suspected malignant lesions, particularly melanoma and SCC (AAD Skin Cancer Guidelines: https://www.aad.org). Recent literature highlights advancements in immunotherapy for squamous cell carcinoma, which can significantly impact survival outcomes (Current Progress and Future Directions of Immunotherapy in Head and Neck Squamous Cell Carcinoma: https://pubmed.ncbi.nlm.nih.gov/40048196/). The National Comprehensive Cancer Network (NCCN) provides comprehensive guidelines for management, emphasizing early detection and intervention (NCCN Guidelines for Squamous Cell Skin Cancer: https://www.nccn.org/guidelines/guidelines-detail?category=1&id=1465).",
        "HistoricalCaseComparison": {
            "SimilarKeyFindings": "Both historical cases (Case 1 and Case 2) have identical key findings that strongly support the diagnosis of SCC. The presence of an irregular, potentially ulcerated lesion with color variations and discomfort aligns with typical presentations of SCC. The treatment recommendations are consistent across these cases, emphasizing immediate referral for biopsy and potential surgical intervention.",
            "ConsistentDiagnosis": "The consistent primary diagnosis of SCC in historical cases reinforces the current diagnosis. The treatment recommendations align with established guidelines, indicating that surgical excision is a standard approach. The emphasis on immunotherapy in recent literature is also consistent with the evolving treatment landscape for SCC."
        },
        "IdentifiedInconsistencies": []
    }
    agent = TreatmentRecommendAgent(model=model, api_key=api_key)
    analysis_result = agent.analyze(current_case)
    print(analysis_result)