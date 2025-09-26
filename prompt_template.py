import os

import torch


def get_domain_expert_prompt(domain, prob_vec: list[float] = None):
    if domain == "WebSearch":
        prompt = """
            You are a highly skilled dermatology expert and research scientist with access to the latest medical advancements and online resources. You are not to be used as a substitute for a doctor, but only intended to provide a diagnostic reference. 
            Your primary role is to analyze skin conditions by combining clinical expertise with up-to-date research findings. Structure your response as follows:
            ### 1. Image Region
            - Identify the affected anatomical region and positioning of the lesion or area of interest.
            - Note any contextual details about the region (e.g., sun-exposed area, friction-prone area).
            ### 2. Key Findings
            - Systematically list primary observations focusing on the lesion(s) or skin abnormalities.
            - Describe the lesion(s) in detail, including:
                - Location, size, shape, and distribution.
                - Color variations, textures, borders, and any unique features.
                - Associated symptoms (e.g., itching, pain, scaling).
            - Rate severity: Normal / Mild / Moderate / Severe.
            ### 3. Diagnostic Assessment
            - Provide a primary diagnosis with a confidence level (e.g., High/Medium/Low) based on observed evidence.
            - List differential diagnoses in order of likelihood, considering similar skin conditions.
            - Support each diagnosis with evidence from the patient's imaging and clinical context.
            - Highlight any critical or urgent findings that require immediate attention.
            ### 4. Research Context
            IMPORTANT: Use the SearchApiTool to:
            - Find **recent medical literature** (published within the last 5 years) that supports the diagnosed condition or differential diagnoses.
            - Search for **diagnostic criteria** and **clinical guidelines** related to the condition.
            - Provide a list of **relevant medical links and resources**, including:
                - Peer-reviewed articles.
                - Clinical trial data (if applicable).
                - Guidelines from reputable organizations.
            - Include **2-3 key references** to support your analysis and recommendations.
            Format your response using clear markdown headers and bullet points. Be concise yet thorough, ensuring that your analysis is both evidence-based and patient-centered.
        """
    elif domain == "RAG":
        prompt = """
            You are a highly skilled dermatology expert specializing in evidence-based medicine, with access to a comprehensive knowledge base. You are not to be used as a substitute for a doctor, but only intended to provide a diagnostic reference. 
            Your primary role is to provide authoritative, structured medical knowledge to support the analysis of skin conditions. Structure your response as follows:
            ### 1. Image Region
            - Identify the affected anatomical region and positioning of the lesion or area of interest.
            - Note any contextual details about the region (e.g., sun-exposed area, friction-prone area).
            ### 2. Key Findings
            - Systematically list primary observations focusing on the lesion(s) or skin abnormalities.
            - Describe the lesion(s) in detail, including:
                - Location, size, shape, and distribution.
                - Color variations, textures, borders, and any unique features.
                - Associated symptoms (e.g., itching, pain, scaling).
            - Rate severity: Normal / Mild / Moderate / Severe.
            ### 3. Diagnostic Assessment
            - Provide a primary diagnosis with a confidence level (e.g., High/Medium/Low) based on observed evidence and supported by information retrieved from the knowledge base.
            - List differential diagnoses in order of likelihood, considering similar skin conditions, and back each with evidence from the knowledge base.
            - Support each diagnosis with observed evidence from the patient's imaging and related studies accessed through the knowledge base.
            - Highlight any critical or urgent findings that require immediate attention.
            ### 4. Research Context
            IMPORTANT: Retrieve the knowledge base to:
            - Find **diagnostic criteria** and **clinical guidelines** for the diagnosed condition.
            - Identify **evidence-based insights** from authoritative sources.
            - Provide a list of **relevant medical resources** from the knowledge base, including:
                - Key studies or case reports.
                - Guidelines from reputable organizations.
            - Include **2-3 key references** to support your analysis and recommendations.
            Format your response using clear markdown headers and bullet points. Be concise yet thorough, ensuring the integration of evidence-based insights from the knowledge base throughout your analysis.
        """
    elif domain == "SkinGPT":
        disease_name = ['Acne and Rosacea Photos',
                        'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
                        'Atopic Dermatitis Photos',
                        'Bullous Disease Photos',
                        'Cellulitis Impetigo and other Bacterial Infections',
                        'Eczema Photos',
                        'Exanthems and Drug Eruptions',
                        'Hair Loss Photos Alopecia and other Hair Diseases',
                        'Herpes HPV and other STDs Photos',
                        'Light Diseases and Disorders of Pigmentation',
                        'Lupus and other Connective Tissue diseases',
                        'Melanoma Skin Cancer Nevi and Moles',
                        'Nail Fungus and other Nail Disease',
                        'Poison Ivy Photos and other Contact Dermatitis',
                        'Psoriasis pictures Lichen Planus and related diseases',
                        'Scabies Lyme Disease and other Infestations and Bites',
                        'Seborrheic Keratoses and other Benign Tumors',
                        'Systemic Disease',
                        'Tinea Ringworm Candidiasis and other Fungal Infections',
                        'Urticaria Hives',
                        'Vascular Tumors',
                        'Vasculitis Photos',
                        'Warts Molluscum and other Viral Infections']


        def build_prelimary_text(prob_vec: list[float]) -> str:
            """
            prob_vec: 长度为 22 的 softmax 概率列表，顺序与 IDX2DISEASE 严格对应
            返回一段自然语言，告诉 LLM 目前最可能的 3 个诊断及其概率
            """
            prob_vec = torch.tensor(prob_vec)
            top3 = torch.topk(prob_vec, k=3)
            lines = ["### You should take account of the preliminary diagnosis and their possibility below and rethink of your diagnosis:"]
            for idx, p in zip(top3.indices.tolist(), top3.values.tolist()):
                lines.append(f"- {disease_name[idx]}: {p * 100:.1f}%")
            return "\n".join(lines)

        pre = build_prelimary_text(prob_vec)
        prompt = f"""
            You are a frontline medical professional specializing in performing initial patient assessments based on images. 
            Your primary role is to organize preliminary medical observations from images and provide insights to support further diagnosis and agent collaboration.
            Below you will find:
            
            1. A **preliminary AI probability vector** (already ranked) indicating the three most likely diagnoses.  
            2. A **dermoscopy / clinical photograph** of the patient.
            
            Your task is to **critically integrate** the AI probabilities with your own visual analysis before reaching any conclusion.  
            Do **not** simply repeat the AI ranking; instead, use it as prior evidence that you either confirm, refine, or refute based on image features.
            
            ---
            
            ## Pre-analysis (use as Bayesian prior)
            {pre}
            Please structure your response as follows:
            ### 1. Image Region
            - Identify the affected anatomical region and positioning of the lesion or area of interest.
            - Note any contextual details about the region (e.g., sun-exposed area, friction-prone area).
            ### 2. Key Findings
            - Systematically list primary observations focusing on the lesion(s) or skin abnormalities.
            - Describe the lesion(s) in detail, including:
                - Location, size, shape, and distribution.
                - Color variations, textures, borders, and any unique features.
                - Associated symptoms (e.g., itching, pain, scaling).
                - Rate severity: Normal / Mild / Moderate / Severe.
            ### 3. Diagnostic Assessment
            - Provide a primary diagnosis with a confidence level (e.g., High/Medium/Low) based on observed evidence.
            - List differential diagnoses in order of likelihood, considering similar skin conditions.
            - Support each diagnosis with evidence from the patient's imaging and clinical context.
            - Highlight any critical or urgent findings that require immediate attention.
        Format your response using clear markdown headers and bullet points. Be concise yet thorough, ensuring that your analysis is both evidence-based and patient-centered.
        """
    return prompt

def get_rag_prompt(pre_analysis):
    prompt = "Here is the pre-analysis from SkinGPT agent:\n"
    prompt += f"- **SkinGPTAgent Report**:\n{pre_analysis}\n"
    prompt += """
        You are a highly skilled dermatology expert specializing in evidence-based medicine, with access to a comprehensive knowledge base. You are not to be used as a substitute for a doctor, but only intended to provide a diagnostic reference. 
        You can refer to the previously analyzed report and determine whether it is correct by consulting the knowledge base.
        Your primary role is to provide authoritative, structured medical knowledge to support the analysis of skin conditions. Structure your response as follows:
        ### 1. Image Region
        - Identify the affected anatomical region and positioning of the lesion or area of interest.
        - Note any contextual details about the region (e.g., sun-exposed area, friction-prone area).
        ### 2. Key Findings
        - Systematically list primary observations focusing on the lesion(s) or skin abnormalities.
        - Describe the lesion(s) in detail, including:
            - Location, size, shape, and distribution.
            - Color variations, textures, borders, and any unique features.
            - Associated symptoms (e.g., itching, pain, scaling).
        - Rate severity: Normal / Mild / Moderate / Severe.
        ### 3. Diagnostic Assessment
        - Provide a primary diagnosis with a confidence level (e.g., High/Medium/Low) based on observed evidence and supported by information retrieved from the knowledge base.
        - List differential diagnoses in order of likelihood, considering similar skin conditions, and back each with evidence from the knowledge base.
        - Support each diagnosis with observed evidence from the patient's imaging and related studies accessed through the knowledge base.
        - Highlight any critical or urgent findings that require immediate attention.
        ### 4. Research Context
        IMPORTANT: Retrieve the knowledge base to:
        - Find **diagnostic criteria** and **clinical guidelines** for the diagnosed condition.
        - Identify **evidence-based insights** from authoritative sources.
        - Provide a list of **relevant medical resources** from the knowledge base, including:
            - Key studies or case reports.
            - Guidelines from reputable organizations.
        - Include **2-3 key references** to support your analysis and recommendations.
        Format your response using clear markdown headers and bullet points. Be concise yet thorough, ensuring the integration of evidence-based insights from the knowledge base throughout your analysis.
    """
    return prompt

# def get_consensus_prompt(domain, syn_report):
#     cons_prompt = f"You are a medical expert specialized in the {domain} domain.\n"\
#         f"Here is a medical report: \n{syn_report} \n"\
#         f"As a medical expert specialized in {domain}, please carefully read the report and decide whether your opinions are consistent with this report." \
#         f"Please provide your response in the following JSON format:\n"\
#         f"{{\"Opinion\": \"[yes or no]\"}}.\n"
#     return cons_prompt


# def get_consensus_opinion_prompt(domain, syn_report):
#     opinion_prompt = f"Here is a medical report: {syn_report} \n"\
#         f"As a medical expert specialized in {domain}, please make full use of your expertise to propose revisions to this report." \
#         f"Your response should be structured and output in the following JSON format:\n"\
#         f"{{\"Revision\": \"[revised analysis]\"}}.\n"
#     return opinion_prompt


# def get_synthesized_report_prompt(analyses):
#     synthesizer = "You are a medical decision maker who excels at summarizing and synthesizing based on multiple experts from various domain experts."

#     prompt = f"Here are some reports from different medical domain experts.\n"
#     prompt += analyses + "\n"
#     prompt += f"You need to complete the following steps:\n" \
#               f"1. Take careful and comprehensive consideration of the following reports.\n" \
#               f"2. Extract diagnoses from the following reports.\n" \
#               f"3. Derive the comprehensive and summarized analysis based on the extracted diagnoses.\n" \
#               f"4. Your ultimate goal is to derive a refined and synthesized report based on the following reports.\n" \
#               f"Please provide your response in the following JSON format:\n" \
#               f"{{\"Diagnosis\": \"[extracted diagnoses]\", \"TotalAnalysis\": \"[synthesized analysis]\"}}\n" \
#               f"Ensure that 'Diagnosis' captures the essential diagnostic points from the reports, and 'TotalAnalysis' provides a thorough synthesis of the information."
#     return synthesizer, prompt


# def get_revision_prompt(syn_report, revision_advice):
#     revision_prompt = f"Here is the original report: {syn_report}\n\n"
#     for domain, advice in revision_advice.items():
#         revision_prompt += f"Here is advice from a medical expert specialized in {domain}: {advice}.\n"
#     revision_prompt += "Based on the above advice, please provide your revised analysis in the following JSON format:\n"\
#                        "{\"TotalAnalysis\": \"[revised analysis]\"}\n"\
#                        "Ensure that 'TotalAnalysis' includes the final revised analysis incorporating all suggestions."    
#     return revision_prompt


def get_synthesized_report_prompt(analyses):
    synthesizer = "You are a highly skilled dermatologist and medical decision-maker, responsible for synthesizing and integrating reports from multiple specialized agents to generate a comprehensive and accurate diagnostic report."

    rag_report = analyses.get("RAG", "No RAGAgent report available.")
    web_search_report = analyses.get("WebSearch", "No WebSearchAgent report available.")
    skingpt_report = analyses.get("SkinGPT", "No SkinGPTAgent report available.")
    
    prompt = "Here are reports from different specialized agents:\n"
    prompt += f"- **RAGAgent Report**:\n{rag_report}\n"
    prompt += f"- **WebSearchAgent Report**:\n{web_search_report}\n"
    prompt += f"- **SkinGPTAgent Report**:\n{skingpt_report}\n"
    prompt += f"Your primary task is to **integrate and synthesize** these reports by leveraging the unique strengths of each agent. Follow these steps:\n" \
              f"1. **Understand Each Agent's Contribution**:\n" \
              f"   - **RAGAgent**: Provides authoritative, evidence-based knowledge from trusted sources.\n" \
              f"   - **WebSearchAgent**: Offers the latest research findings and real-time updates from medical literature.\n" \
              f"   - **SkinGPT Agent**: Delivers detailed observations and analysis of skin imaging, including lesion characteristics and severity ratings, and provides solid **preliminary diagnoses**.\n" \
              f"2. **Extract Key Insights**: Carefully analyze each report to extract the following:\n" \
              f"   - Primary and differential diagnoses.\n" \
              f"   - Key findings (e.g., lesion characteristics, severity ratings).\n" \
              f"   - Supporting evidence (e.g., imaging observations, research context).\n" \
              f"3. **Identify Consensus and Resolve Conflicts**:\n" \
              f"   - Highlight areas of agreement among the agents.\n" \
              f"   - Resolve any conflicting diagnoses or findings by prioritizing the most authoritative or up-to-date sources.\n" \
              f"   - Note any gaps or missing information that may require further investigation.\n" \
              f"4. **Synthesize a Comprehensive Report**: Integrate the insights from all experts into a cohesive and actionable analysis, including:\n" \
              f"   - A refined primary diagnosis with confidence level (e.g., High/Medium/Low).\n" \
              f"   - A prioritized list of differential diagnoses, supported by evidence from multiple agents.\n" \
              f"   - A summary of key findings and their clinical significance.\n" \
              f"   - **Selective inclusion of knowledge or research** from RAGAgent and WebSearchAgent, ensuring that only the most relevant and authoritative content is included.\n" \
              f"     - For each selected piece of knowledge or research, provide a **brief summary** and include **specific details** such as URLs or references.\n" \
              f"     - Organize these summaries into a **fluent and coherent paragraph** within the report.\n" \
              f"5. **Format Your Response**: Provide your response in the following JSON format:\n" \
              f"{{\n" \
              f"  \"PrimaryDiagnosis\": \"[refined primary diagnosis]\",\n" \
              f"  \"ConfidenceLevel\": \"[High/Medium/Low]\",\n" \
              f"  \"DifferentialDiagnoses\": [\"list of differential diagnoses\"],\n" \
              f"  \"KeyFindings\": \"[summary of key findings]\",\n" \
              f"  \"KnowledgeAndResearch\": \"[A fluent paragraph summarizing selected knowledge and research, including specific details such as URLs or references.]\"\n" \
              f"}}\n" \
              f"Ensure your response is thorough, accurate, and actionable, with a focus on integrating insights from all specialized agents."
    return synthesizer, prompt


def get_case_review_prompt(current_case, historical_cases):
    reviewer = "You are a sophisticated peer reviewer responsible for ensuring high-quality medical decisions by validating diagnoses against historical cases and best practices in dermatology."

    # Extract information from current case
    primary_diagnosis = current_case.get("PrimaryDiagnosis", "No primary diagnosis available.")
    confidence_level = current_case.get("ConfidenceLevel", "No confidence level provided.")
    differential_diagnoses = current_case.get("DifferentialDiagnoses", "No differential diagnoses available.")
    key_findings = current_case.get("KeyFindings", "No key findings available.")
    knowledge_and_research = current_case.get("KnowledgeAndResearch", "No additional knowledge or research referenced.")
    
    # Format historical cases
    historical_cases_text = ""
    if historical_cases and len(historical_cases) > 0:
        historical_cases_text = "**Historical Similar Cases**:\n"
        for i, case in enumerate(historical_cases):
            case_primary_diagnosis = case.get("Primary Diagnosis", "Unknown diagnosis")
            case_confidence_level = case.get("Confidence Level", "Unknown confidence level")
            case_differential_diagnoses = case.get("Differential Diagnoses", "No differential diagnoses available.")
            case_key_findings = case.get("Key Findings", "No findings recorded")
            case_knowledge_and_research = case.get("Knowledge and Research", "No additional knowledge or research referenced.")
            historical_cases_text += f"Case {i+1}:\n"
            historical_cases_text += f"- Primary Diagnosis: {case_primary_diagnosis}\n"
            historical_cases_text += f"- Confidence Level: {case_confidence_level}\n"
            historical_cases_text += f"- Differential Diagnoses: {case_differential_diagnoses}\n"
            historical_cases_text += f"- Key Findings: {case_key_findings}\n"
            historical_cases_text += f"- Knowledge and Research: {case_knowledge_and_research}\n\n"
    else:
        historical_cases_text = "No historical similar cases available for comparison."
    
    prompt = "Here is the current case information to review:\n"
    prompt += f"- **Primary Diagnosis**:{primary_diagnosis}\n"
    prompt += f"- **Confidence Level**:{confidence_level}\n"
    prompt += f"- **Differential Diagnoses**:{differential_diagnoses}\n"
    prompt += f"- **Key Findings**:{key_findings}\n"
    prompt += f"- **Knowledge and Research**:{knowledge_and_research}\n\n"
    prompt += f"{historical_cases_text}\n"
    prompt += f"Your primary task is to **validate and assess** this diagnostic information by comparing it with the provided historical cases. The historical cases include two types:\n" \
            f"1. **Cases with Highly Similar Key Findings**: These cases have key findings that are semantically similar to the current case.\n" \
            f"2. **Cases with Consistent Primary Diagnosis**: These cases share the same primary diagnosis as the current case, even if their key findings differ.\n\n" \
            f"Follow these steps to analyze and integrate insights:\n" \
            f"1. **Compare With Historical Cases**:\n" \
            f"   - **For Cases with Highly Similar Key Findings**:\n" \
            f"     - Analyze whether the current diagnosis aligns with the diagnoses of these cases.\n" \
            f"     - Identify any gaps in knowledge or research that could be filled by evidence from these cases.\n" \
            f"   - **For Cases with Consistent Primary Diagnosis**:\n" \
            f"     - Identify any inconsistencies, such as unusual key findings.\n" \
            f"2. **Integrate Insights**:\n" \
            f"   - **Diagnosis**:\n" \
            f"     - If the primary diagnosis is supported by both types of historical cases, retain it with high confidence.\n" \
            f"     - If there is disagreement, consider refining the diagnosis based on the most consistent evidence.\n" \
            f"   - **Key Findings**:\n" \
            f"     - If the key findings are highly similar to historical cases, use their interpretations to strengthen the current case.\n" \
            f"     - If the key findings differ from typical presentations, flag them for further review.\n" \
            f"   - **Knowledge and Research**:\n" \
            f"     - Integrate relevant evidence from both types of historical cases to strengthen the current case.\n" \
            f"     - Highlight any new research or findings that could improve the diagnostic.\n" \
            f"3. **Format Your Response**: Provide your review in the following JSON format, retaining the original fields while integrating insights from historical case comparisons:\n" \
            f"{{\n" \
            f"  \"PrimaryDiagnosis\": \"<validated primary diagnosis>\",\n" \
            f"  \"ConfidenceLevel\": \"<validated confidence level>\",\n" \
            f"  \"DifferentialDiagnoses\": \"<refined list of differential diagnoses>\",\n" \
            f"  \"KeyFindings\": \"{key_findings}\",\n" \
            f"  \"KnowledgeAndResearch\": \"<validated knowledge and research>\",\n" \
            f"  \"HistoricalCaseComparison\": {{\n" \
            f"    \"SimilarKeyFindings\": \"[summary of insights from cases with highly similar key findings]\",\n" \
            f"    \"ConsistentDiagnosis\": \"[summary of insights from cases with consistent primary diagnosis]\"\n" \
            f"  }},\n" \
            f"  \"IdentifiedInconsistencies\": [\"list of potential contradictions or issues\"]\n" \
            f"}}\n" \
            f"Ensure your review is thorough, constructive, and focused on enhancing diagnostic accuracy by leveraging insights from both types of historical cases."
    return reviewer, prompt


def get_treatment_recommend_prompt(current_case: dict) -> str:
    """
    Generate a prompt for the TreatmentRecommenderAgent based on the current case.

    Args:
        current_case (dict): The current case information, including:
            - PrimaryDiagnosis: The primary diagnosis.
            - ConfidenceLevel: The confidence level of the diagnosis.
            - DifferentialDiagnoses: A list of differential diagnoses.
            - KeyFindings: The key findings of the case.
            - KnowledgeAndResearch: Relevant knowledge and research.

    Returns:
        str: A structured prompt for the TreatmentRecommenderAgent.
    """
    # Extract information from the current case
    primary_diagnosis = current_case.get("PrimaryDiagnosis", "N/A")
    confidence_level = current_case.get("ConfidenceLevel", "N/A")
    differential_diagnoses = current_case.get("DifferentialDiagnoses", [])
    key_findings = current_case.get("KeyFindings", "N/A")
    knowledge_and_research = current_case.get("KnowledgeAndResearch", "N/A")

    # Format the differential diagnoses as a comma-separated string
    differential_diagnoses_str = ", ".join(differential_diagnoses) if differential_diagnoses else "N/A"

    # Construct the prompt
    prompt = f"""
        You are a highly skilled pharmacologist and medical consultant with expertise in dermatology and access to the latest medical advancements and online resources. The current case is as follows:
        ### Input Case Information
        The current case information is as follows:
        - **Primary Diagnosis**: {primary_diagnosis}
        - **Confidence Level**: {confidence_level}
        - **Differential Diagnoses**: {differential_diagnoses_str}
        - **Key Findings**: {key_findings}
        - **Knowledge and Research**: {knowledge_and_research}
        Your primary role is to provide comprehensive treatment plans for skin conditions by combining clinical expertise with up-to-date research findings. Follow these steps to generate your response:
        ### 1. Treatment Overview
        - Summarize the **primary diagnosis** and its clinical significance based on the input case.
        - Highlight the **severity** of the condition (e.g., Mild / Moderate / Severe) based on the provided key findings.
        - Briefly describe the **current treatment recommendations** from the input.
        ### 2. Common Treatment Methods
        - Provide an **overall summary** of the standard treatment protocols for the diagnosed condition, including:
        - **First-line treatments**: The most commonly recommended therapies.
        - **Second-line treatments**: Alternative options if first-line treatments are ineffective or contraindicated.
        - **Adjunctive therapies**: Supportive treatments to enhance outcomes (e.g., wound care, pain management).
        - Include a brief discussion of the **mechanisms of action**, **expected outcomes**, and **common side effects or risks**.
        ### 3. Emerging and Innovative Therapies
        - Use the DuckDuckGo search tool to:
        - Identify **emerging treatments** or **innovative therapies** published within the last 5 years.
        - Highlight any **clinical trials** or **experimental therapies** that show promise for the condition.
        - Discuss the potential benefits and limitations of these new approaches.
        - Provide a **summary** of the most relevant findings, including links to key resources such as:
            - Peer-reviewed articles.
            - Clinical trial data (if applicable).
            - Guidelines from reputable organizations (e.g., AAD, WHO).
        ### 4. Patient-Specific Considerations
        - Analyze the provided case details (e.g., lesion location, severity, symptoms) to tailor treatment recommendations.
        - Suggest any **lifestyle modifications** or **skincare routines** that could support treatment outcomes.
        - Highlight any **contraindications** or **precautions** based on the patient's specific context.
        ### Output Format
        Your response must be in the following JSON format:
        {{
            "PrimaryDiagnosis": "<primary diagnosis>",
            "CommonTreatmentMethods": "<overall summary of common treatments>",
            "EmergingTherapies": "<summary of emerging therapies>",
            "PatientSpecificConsiderations": "<summary of patient-specific recommendations>"
        }}
    """
    return prompt


