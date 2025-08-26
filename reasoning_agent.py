import os
import json
import openai
from typing import Dict, List, Optional
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import ImageNode, TextNode
import lancedb
from api_utils import generate_response
from prompt_template import get_synthesized_report_prompt
from llama_index.core.storage import StorageContext
from llama_index.core.indices import VectorStoreIndex


class ReasoningAgent:
    def __init__(self, model):
        """
        Initialize the ReasoningAgent with an API key and optional historical cases file path.
        
        Args:
            model (str): The name of the model to use.
            api_key (str): The API key for accessing OpenAI services.
        """
        self.model = model

    def generate_report(self, current_case: Dict) -> Dict:
        """
        Generate a comprehensive diagnostic report for the current case using OpenAI.
        
        Args:
            current_case (Dict): The current case data.
        
        Returns:
            Dict: The generated report.
        """
        # Prepare the prompt for OpenAI
        synthesizer, prompt = self._build_prompt(current_case)

        # Call OpenAI API to generate the report
        response = generate_response(
            engine=self.model, 
            temperature=0.5, 
            max_tokens=2500, 
            frequency_penalty=0, 
            presence_penalty=0, 
            stop=None, 
            system_role=synthesizer, 
            user_input=prompt
        )
        report = json.loads(response)

        return report

    def _build_prompt(self, analyses) -> str:
        """
        Build a prompt for OpenAI based on the current case.
        
        Args:
            analyses (str): The total analyses from different agents.
        
        Returns:
            synthesizer (str): The synthesizer role in the conversation.
            prompt (str): The prompt for OpenAI.
        """
        synthesizer, prompt = get_synthesized_report_prompt(analyses)
        return synthesizer, prompt


if __name__ == "__main__":
    # Build the agent
    model = "gpt-4o-mini"
    reasoning_agent = ReasoningAgent(model=model)

    current_case = {
        "rag_diagnosis": """
            ### 1. Image Region
            - **Anatomical Region**: The lesion is located on the lower abdominal area.
            - **Contextual Details**: This area is typically not sun-exposed and is less likely to experience friction compared to other parts, such as the inner thighs or underarms.

            ### 2. Key Findings
            - **Primary Observations**:
            - **Location**: Lower abdominal area.
            - **Size**: Approximately 5 mm, as indicated by the ruler.
            - **Shape**: Oval/elongated.
            - **Color Variations**: Dark brown/black.
            - **Textures**: Smooth surface.
            - **Borders**: Well-defined.
            - **Associated Symptoms**: None observed or mentioned (e.g., no itching, pain, or scaling).
            - **Severity**: Mild (if asymptomatic and stable over time).

            ### 3. Diagnostic Assessment
            - **Primary Diagnosis**: Melanocytic Nevus (Confidence Level: High).
            - **Evidence**: The characteristics (size, shape, color, and well-defined borders) are consistent with common nevi.

            - **Differential Diagnoses**:
            - **Melanoma** (Medium): Considered due to similarities in color and size; requires dermatoscopic evaluation.
            - **Seborrheic Keratosis** (Medium): Characterized by a similar appearance but typically has a more irregular surface and is often lighter.
            - **Basal Cell Carcinoma** (Low): Unlikely due to the lack of nodularity and scaling usually associated with BCC.

            - **Critical Findings**:
            - None observed that require immediate attention; however, if the lesion changes in color, size, or symptoms arise, further investigation is warranted.

            ### 4. Research Context
            - **Diagnostic Criteria**:
            - Melanocytic nevi diagnosis primarily relies on clinical assessment involving characteristics like size, shape, and color boundaries.
            - Further evaluation through dermoscopy is recommended (e.g., checking for the pigment network correlating with malignancy).

            - **Evidence-Based Insights**:
            - Melanocytic nevi are common, benign growths often presenting in adults. Monitoring is critical for signs of change indicative of melanoma.

            - **Relevant Medical Resources**:
            - **Key Studies**:
                - Morton CA et al. Br J Dermatol 2014;170:245-60. [Guidelines for melanoma recognition](https://www.bad.org.uk).
                - Cancer Research UK. Melanoma skin cancer. [Information on melanoma](http://www.cancerresearchuk.org/about-cancer/type/melanoma/).

            - **Guidelines**:
            - Consider using dermatoscopy for clearer assessment of lesions as per recent protocols.

            ### References
            - National guidelines and clinical studies provide a framework for assessing skin lesions and managing suspected cases of melanoma. The integration of clinical history and dermatoscopic findings are crucial in distinguishing between benign lesions and malignancy.
        """,
        "web_search_diagnosis": """
            ### 1. Image Region
            - **Anatomical Region:** The lesion is located on the abdominal area, specifically in a central position between the ribs and the lower abdomen.
            - **Context:** The area appears to be a sun-exposed region due to the general location, though there is no noticeable sun damage or signs of friction-related irritation.

            ### 2. Key Findings
            - **Primary Observations:**
            - **Location:** Central abdomen.
            - **Size:** Approximately 5 mm (based on scale).
            - **Shape:** Oval.
            - **Distribution:** Isolated lesion.
            - **Color Variations:** The lesion is dark brown to black.
            - **Textures:** Appears smooth without scaling.
            - **Borders:** Well-defined edges.
            - **Unique Features:** No notable irregularities in shape or color asymmetry.
            - **Associated Symptoms:** No itching, pain, or any other associated symptoms are noted.
            - **Severity:** Mild.

            ### 3. Diagnostic Assessment
            - **Primary Diagnosis:** Benign nevi (mole) with high confidence.
            - **Differential Diagnoses:**
            1. **Seborrheic Keratosis:** Common benign skin growth; typically appears raised and varies in color.
                - **Evidence:** Comparatively irregular borders and a warty texture would not fit here.
            2. **Melanoma:** Rarely presents as an isolated small dark lesion but must be ruled out, especially if there are changes.
                - **Evidence:** Lack of asymmetry or irregular borders reduces likelihood.
            3. **Lentigo:** Benign flat brown lesions; does not fit the profile due to lack of prominence.
                - **Evidence:** Shape and borders are more characteristic of a nevus.
            - **Urgent Findings:** None; the lesion appears stable and benign.

            ### 4. Research Context
            Searching for recent literature and guidelines related to benign nevi and differential diagnoses.

            #### Relevant Literature and Guidelines
            - **Recent Medical Literature:**
            - Investigating the characteristics and management of skin lesions.
            - **Guidelines:**
            - Dermatoscopic criteria for assessing moles.

            I'll conduct searches for recent medical literature and guidelines now.### 4. Research Context (Continued)

            #### Relevant Literature
            - **Studies on Benign Nevi:**
            - **[The 2023 WHO updates on skin tumors: advances since the 2018 edition](https://pmc.ncbi.nlm.nih.gov/articles/PMC11460152/)**
                - Focuses on the classification of melanocytic tumors, including benign nevi.
            - **[Common Benign Melanocytic and Non-Melanocytic Skin Tumors - PubMed](https://pubmed.ncbi.nlm.nih.gov/36657431/)**
                - Discusses various benign skin tumors and correlates their characteristics with factors like sun exposure.
            - **[MPATH-Dx version 2.0 schema for melanocytic lesions](https://www.sciencedirect.com/science/article/pii/S0738081X24001767)**
                - Outlines the latest diagnostic criteria for melanocytic lesions, aiding in distinguishing benign from malignant.

            #### Relevant Guidelines and Resources
            - **[Clinical Guidelines - American Academy of Dermatology](https://www.aad.org/practicecenter/quality/clinical-guidelines)**
            - Guidelines regarding the diagnosis and management of skin conditions, essential for clinical practice.
            - **[Melanocytic Nevi - StatPearls](https://www.ncbi.nlm.nih.gov/books/NBK470451/)**
            - A comprehensive overview of pigmented lesions, including benign nevi and their characteristics.

            #### Key References
            1. **The 2023 WHO updates on skin tumors** - Important for understanding updates on diagnostic criteria.
            2. **Common Benign Melanocytic and Non-Melanocytic Skin Tumors - PubMed** - Relevant for current understanding of benign skin tumors.
            3. **MPATH-Dx version 2.0 schema for melanocytic lesions** - Crucial for accurate diagnosis and classification of melanocytic lesions.

            This research supports the primary diagnosis of benign nevi and guides any further assessment or monitoring necessary for patient care.
        """,
    }

    # generate report
    report = reasoning_agent.generate_report(current_case)
    print(json.dumps(report, indent=2))