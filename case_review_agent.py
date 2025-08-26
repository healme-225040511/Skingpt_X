import os
import json
from typing import Dict, List
from neo4j import GraphDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from api_utils import generate_response
from prompt_template import get_case_review_prompt
from fuzzywuzzy import fuzz  
import re  
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import numpy as np


class CaseReviewAgent:
    def __init__(self, model: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """
        Initialize the CaseReviewAgent with an API key and Neo4j connection details.
        
        Args:
            model (str): The name of the model to use.
            neo4j_uri (str): The URI for the Neo4j database.
            neo4j_user (str): The username for the Neo4j database.
            neo4j_password (str): The password for the Neo4j database.
        """
        self.model = model
        self.embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

        # Clear all Case nodes in the Neo4j database
        self.clear_all_case_nodes()

    def clear_all_case_nodes(self):
        """
        Delete all `Case` nodes from the Neo4j database.
        """
        query = "MATCH (c:Case) DETACH DELETE c"
        with self.driver.session() as session:
            session.run(query)
        print("All `Case` nodes have been cleared from the Neo4j database.")

    def review_case(self, current_case: Dict) -> Dict:
        """
        Review the current case by comparing it with historical cases and best practices.
        
        Args:
            current_case (Dict): The current case data to review.
        
        Returns:
            Dict: The review report.
        """

        # Generate review report by comparing with historical cases
        review_report = self._generate_review_report(current_case)

        return review_report

    def _generate_review_report(self, current_case: Dict) -> Dict:
        """
        Generate a review report by comparing the current case with historical cases in the Neo4j knowledge graph.
        
        Args:
            current_case (Dict): The current case data.
        
        Returns:
            Dict: The review report.
        """
        # Check if the current diagnosis is consistent with historical cases
        similar_cases = self._find_similar_diagnoses(current_case.get("KeyFindings", ""), current_case.get("PrimaryDiagnosis", ""))
        if similar_cases:
            reviewer, prompt = get_case_review_prompt(current_case, similar_cases)
            # Call OpenAI API to generate the report
            response = generate_response(
                engine=self.model, 
                temperature=0.5, 
                max_tokens=2500, 
                frequency_penalty=0, 
                presence_penalty=0, 
                stop=None, 
                system_role=reviewer, 
                user_input=prompt
            )
            report = json.loads(response)
        else:
            report = current_case
        return report

    def _add_case_to_knowledge_graph(self, new_case: Dict):
        """
        Add a new case to the Neo4j knowledge graph as a single node with all fields.
        
        Args:
            new_case (Dict): The new case data.
        """
        def add_case(tx, new_case_data):
            # Extract information from the new case
            primary_diagnosis = new_case_data.get("PrimaryDiagnosis", "N/A")
            confidence_level = new_case_data.get("ConfidenceLevel", "N/A")
            differential_diagnoses = new_case_data.get("DifferentialDiagnoses", [])
            key_findings = new_case_data.get("KeyFindings", "N/A")
            knowledge_and_research = new_case_data.get("KnowledgeAndResearch", "N/A")

            # Create a Case node with all fields
            tx.run("""
                CREATE (c:Case {
                    primary_diagnosis: $primary_diagnosis,
                    confidence_level: $confidence_level,
                    differential_diagnoses: $differential_diagnoses,
                    key_findings: $key_findings,
                    knowledge_and_research: $knowledge_and_research
                })
                """,
                primary_diagnosis=primary_diagnosis,
                confidence_level=confidence_level,
                differential_diagnoses=differential_diagnoses,
                key_findings=key_findings,
                knowledge_and_research=knowledge_and_research
            )

        with self.driver.session() as session:
            session.execute_write(add_case, new_case)
        print(f"Case added to the knowledge graph.")

    def _find_similar_diagnoses(self, key_findings: str, primary_diagnosis: str) -> List[dict]:
        """
        Find similar cases based on Key Findings and Primary Diagnosis using entity alignment, similarity calculation, and fuzzy string matching.
        
        Args:
            key_findings (str): The key findings to search for.
            primary_diagnosis (str): The primary diagnosis to match.
        
        Returns:
            List[dict]: A list of similar cases, each containing all relevant fields.
                    Returns an empty list if no data is found in Neo4j.
        """
        def get_all_cases(tx):
            # Retrieve all cases with their complete fields
            result = tx.run("""
                            MATCH (c:Case)
                            RETURN c.primary_diagnosis AS primary_diagnosis,
                                c.confidence_level AS confidence_level,
                                c.differential_diagnoses AS differential_diagnoses,
                                c.key_findings AS key_findings,
                                c.knowledge_and_research AS knowledge_and_research
                            """)
            return [{
                "Primary Diagnosis": record["primary_diagnosis"],
                "Confidence Level": record["confidence_level"],
                "Differential Diagnoses": record["differential_diagnoses"],
                "Key Findings": record["key_findings"],
                "Knowledge and Research": record["knowledge_and_research"]
            } for record in result]

        with self.driver.session() as session:
            all_cases = session.execute_read(get_all_cases)

        # If no cases are found in Neo4j, return an empty list
        if not all_cases:
            return []

        # Step 1: Filter cases by Primary Diagnosis using fuzzy matching and Confidence Level
        def normalize_diagnosis(diagnosis: str) -> str:
            """
            Normalize the diagnosis string by:
            - Converting to lowercase
            - Removing extra spaces
            - Removing common abbreviations or suffixes (e.g., " (ICD-10)")
            """
            diagnosis = diagnosis.lower().strip()  # Convert to lowercase and remove leading/trailing spaces
            diagnosis = re.sub(r"\s*\(.*\)", "", diagnosis)  # Remove abbreviations in parentheses
            diagnosis = re.sub(r"\s+", " ", diagnosis)  # Replace multiple spaces with a single space
            return diagnosis

        normalized_primary_diagnosis = normalize_diagnosis(primary_diagnosis)
        similar_cases_diagnosis = []
        for case in all_cases:
            case_diagnosis = normalize_diagnosis(case["Primary Diagnosis"])
            case_confidence_level = case["Confidence Level"]
            # Use fuzzy matching to handle minor differences and check Confidence Level
            if fuzz.ratio(normalized_primary_diagnosis, case_diagnosis) >= 85 and case_confidence_level == "High":
                similar_cases_diagnosis.append(case)
                # Limit to 3 cases
                if len(similar_cases_diagnosis) >= 3:
                    break

        # Step 2: Compute similarity based on Key Findings
        # Compute embedding for the current key findings
        current_embedding = self.embedding_model._get_text_embedding(key_findings)

        # Compute embeddings for historical key findings one by one
        historical_embeddings = []
        for case in all_cases:
            case_key_findings = case["Key Findings"]
            case_embedding = self.embedding_model._get_text_embedding(case_key_findings)
            historical_embeddings.append(case_embedding)

        # Convert historical embeddings to a numpy array
        historical_embeddings = np.array(historical_embeddings)

        # Calculate cosine similarity between current key findings and historical key findings
        current_embedding = np.array(current_embedding).reshape(1, -1)
        similarities = cosine_similarity(current_embedding, historical_embeddings).flatten()

        # Step 3: Filter cases by similarity threshold and get the top 2 cases
        # Only consider cases with similarity > 90
        high_similarity_indices = [i for i, sim in enumerate(similarities) if sim > 0.8]
        if high_similarity_indices:
            # Sort by similarity and get the top 2 cases
            top_indices = sorted(high_similarity_indices, key=lambda i: similarities[i], reverse=True)[:2]
            similar_cases_findings = [all_cases[i] for i in top_indices]
        else:
            similar_cases_findings = []

        # Combine results from both methods
        return similar_cases_diagnosis + similar_cases_findings
    
    def close(self):
        """
        Close the Neo4j driver connection.
        """
        self.driver.close()


if __name__ == "__main__":
    # 初始化 CaseReviewAgent
    model = "gpt-4o-mini"
    neo4j_uri = "bolt://localhost:7687"  
    neo4j_user = "neo4j"  
    neo4j_password = "" 
    case_review_agent = CaseReviewAgent(
        model=model, 
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password
    )

    current_case = {
        "PrimaryDiagnosis": "Squamous Cell Carcinoma (SCC)",
        "ConfidenceLevel": "High",
        "DifferentialDiagnoses": [
            "Basal Cell Carcinoma (BCC)",
            "Melanoma",
            "Actinic Keratosis"
        ],
        "KeyFindings": "The lesion is located on the lateral aspect of the right cheek, near the eye, measuring approximately 1.5 cm in diameter. It has an irregular, triangular shape with color variations ranging from dark brown to black, and a shiny, possibly ulcerated surface. The borders are irregular, with some areas appearing well-defined. Symptoms may include discomfort, and the severity is rated as moderate, indicating a potential for malignancy.",
        "KnowledgeAndResearch": "According to the American Academy of Dermatology (AAD), excisional biopsy is the standard protocol for suspected malignant lesions, particularly melanoma and SCC (AAD Skin Cancer Guidelines: https://www.aad.org). Recent literature highlights advancements in immunotherapy for squamous cell carcinoma, which can significantly impact survival outcomes (Current Progress and Future Directions of Immunotherapy in Head and Neck Squamous Cell Carcinoma: https://pubmed.ncbi.nlm.nih.gov/40048196/). The National Comprehensive Cancer Network (NCCN) provides comprehensive guidelines for management, emphasizing early detection and intervention (NCCN Guidelines for Squamous Cell Skin Cancer: https://www.nccn.org/guidelines/guidelines-detail?category=1&id=1465)."
    }

    # 审查当前病例
    review_report = case_review_agent.review_case(current_case)
    print(json.dumps(review_report, indent=2))

    case_review_agent.close()