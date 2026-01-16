import os

from Constants import DERMNET_DISEASE_NAME, ISIC_DISEASE_NAME, HAM10000_DISEASE_NAME, Fitzpatrick17k_DISEASE_NAME
from utils import build_prelimary_text, expand_disease_names, convert_disease_names

def get_visual_findings_prompt():
        """
        专门为皮肤病临床特征提取定制的 Prompt。
        要求模型模仿高年级皮肤科医生的观察逻辑。
        """
        return (
            "You are a senior clinical dermatologist. Your task is to provide a highly professional, "
            "comprehensive, and detailed 'Key Findings' report for the provided skin lesion image.\n\n"
            "Please perform a systematic visual analysis covering the following dimensions:\n"
            "1. Primary Morphology: Specify if it is a macule, papule, plaque, nodule, vesicle, or bulla.\n"
            "2. Color & Pigmentation: Describe the shades (erythematous, brown, black, blue-gray, white) "
            "and the uniformity of distribution.\n"
            "3. Borders & Margin: Analyze if the borders are regular, irregular, notched, well-defined, or fading into normal skin.\n"
            "4. Surface & Texture: Look for scaling, crusting, ulceration, atrophy, lichenification, or telangiectasia (visible blood vessels).\n"
            "5. Global Structure: Evaluate symmetry (bilateral, radial, or asymmetric) and internal structural patterns (e.g., pigment network, globules, or regression areas).\n\n"
            "Strictly output the result in this JSON format:\n"
            "{\n"
            "  \"key_findings\": \"[Insert here a comprehensive, paragraph-style clinical description. "
            "Use precise medical terminology. Be as descriptive as possible about the lesion's "
            "visual characteristics as if you are documenting for a medical record.]\"\n"
            "}"
        )
def get_domain_expert_prompt(domain, prob_vec: list[float] = None, disease_name_mapping: list[str] = DERMNET_DISEASE_NAME):
    if domain == "WebSearch":
        prompt = f"""
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
        prompt = f"""
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
        disease_name = disease_name_mapping[:-1]
        pre = build_prelimary_text(prob_vec, disease_name)
        prompt = f"""
            You are a frontline medical professional specializing in performing initial patient assessments based on images. 
            Your primary role is to organize preliminary medical observations from images and provide insights to support further diagnosis and agent collaboration.
            Below you will find:
            
            1. A **preliminary AI probability vector** (already ranked) indicating the three most likely diagnoses.  
            2. A **dermoscopy / clinical photograph** of the patient.
            This image has one of the following diseases: {", ".join(disease_name)}
            Your task is to **critically integrate** the AI probabilities with your own visual analysis before reaching any conclusion.  
            Do **not** simply repeat the AI ranking; instead, use it as prior evidence that you either confirm, refine, or refute based on image features.
            
            ---
            
            ## Pre-analysis (use as a key knowledge prior)
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
            - Provide a primary diagnosis with a confidence probability on each disease of the disease list {disease_name} based on observed evidence.
            - Support top 5 diagnosis with evidence from the patient's imaging and clinical context. For other diseases, provide the confidence.
            - Highlight any critical or urgent findings that require immediate attention.
        Format your response using clear markdown headers and bullet points. Be concise yet thorough, ensuring that your analysis is both evidence-based and patient-centered. Control your output length to be within 5000 words.
        """
    return prompt

def get_rag_prompt_with_true_label(pre_analysis, retrieved_knowledge):
    prompt = "You are a highly skilled dermatology expert specializing in evidence-based medicine, with access to a comprehensive knowledge base."
    prompt += f"\nNow you have already known the primary diagnosis is {pre_analysis}\n"
    prompt += f"Context from medical handbook:\n{retrieved_knowledge}\n"
    prompt += """
       You are a highly skilled dermatology expert specializing in evidence-based medicine, with access to a comprehensive knowledge base. You don't have to give any primary diagnosis about this sample, but need to show the key medical observations.
        ### ⚠️ CRITICAL INSTRUCTIONS FOR KEY FINDINGS:
        - Identify the affected anatomical region and positioning of the lesion or area of interest.
        - Note any contextual details about the region (e.g., sun-exposed area, friction-prone area).
        - Systematically list primary observations focusing on the lesion(s) or skin abnormalities.
        - Describe the lesion(s) in detail, including:
            - Location, size, shape, and distribution.
            - Color variations, textures, borders, and any unique features.
            - Associated symptoms (e.g., itching, pain, scaling).
        - Rate severity: Normal / Mild / Moderate / Severe.
       You must return **only** the following JSON object (no markdown code block, no extra text):d as a substitu
        {
            "ImageRegion": " ①anatomical location ②positioning ③context (sun-exposed/friction/etc)>",
            "KeyFindings": " A single cohesive paragraph following the instructions above",
            "CriticalFeatures": ["urgent feature 1", "feature 2"] or ["None"]
            "PrimaryDiagnosis": "Provide a primary diagnosis based on observed evidence"
            "KnowledgeAndResearch": [Fluent summary of relevant knowledge, including references if possible]
        }
    """
    return prompt
def get_rag_prompt(pred_name, prob_value, retrieved_knowledge, MAPPINGSET):
    # mapping_options = ", ".join(MAPPINGSET)
    prompt = "You are a highly skilled dermatology expert specializing in evidence-based medicine, with access to a comprehensive knowledge base."
    prompt += f"\n[Prior Knowledge] Panderm model suggests: {pred_name} with {prob_value} probability.\n"
    prompt += f"Context from medical handbook:\n{retrieved_knowledge}\n"
    # prompt += f"You MUST choose the 'PrimaryDiagnosis' ONLY from the following list:\n{mapping_options}\n"
    prompt += """
        ### ⚠️ CRITICAL INSTRUCTIONS FOR KEY FINDINGS:
        - Identify the affected anatomical region and positioning of the lesion or area of interest.
        - Note any contextual details about the region (e.g., sun-exposed area, friction-prone area).
        - Systematically list primary observations focusing on the lesion(s) or skin abnormalities.
        - Describe the lesion(s) in detail, including:
            - Location, size, shape, and distribution.
            - Color variations, textures, borders, and any unique features.
            - Associated symptoms (e.g., itching, pain, scaling).
        - Rate severity: Normal / Mild / Moderate / Severe.
       You must return **only** the following JSON object (no markdown code block, no extra text):d as a substitu
        {
            "ImageRegion": " ①anatomical location ②positioning ③context (sun-exposed/friction/etc)>",
            "KeyFindings": " A single cohesive paragraph following the instructions above",
            "CriticalFeatures": ["urgent feature 1", "feature 2"] or ["None"]
            "PrimaryDiagnosis": "STRICTLY ONE FROM THE VALID DIAGNOSIS SET",
            "KnowledgeAndResearch": [Fluent summary of relevant knowledge, including references if possible]
        }
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


def get_synthesized_report_prompt(analyses, prob_vec: list[float] = None):
    synthesizer = "You are a highly skilled dermatologist and medical decision-maker, responsible for synthesizing and integrating reports from multiple specialized agents to generate a comprehensive and accurate diagnostic report."

    rag_report = analyses.get("RAG", "No RAGAgent report available.")
    web_search_report = analyses.get("WebSearch", "No WebSearchAgent report available.")
    skingpt_report = analyses.get("SkinGPT", "No SkinGPTAgent report available.")

    # skingpt_report = build_prelimary_text(prob_vec, expand_disease_names(HAM10000_DISEASE_NAME))
    print(skingpt_report)
    prompt = "Here are reports from different specialized agents:\n"
    prompt += f"- **RAGAgent Report**:\n{rag_report}\n"
    prompt += f"- **WebSearchAgent Report**:\n{web_search_report}\n"
    prompt += f"- **SkinGPTAgent Report**:\n{skingpt_report}\n"
    prompt += f"Your primary task is to **integrate and synthesize** these reports by leveraging the unique strengths of each agent.  Follow these steps:\n" \
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
              f"   - Probability distribution over the full {expand_disease_names(HAM10000_DISEASE_NAME)} list, sorted descending.\n" \
              f"   - A summary of key findings and their clinical significance.\n" \
              f"   - **Selective inclusion of knowledge or research** from RAGAgent and WebSearchAgent, ensuring that only the most relevant and authoritative content is included.\n" \
              f"     - For each selected piece of knowledge or research, provide a **brief summary** and include **specific details** such as URLs or references.\n" \
              f"     - Organize these summaries into a **fluent and coherent paragraph** within the report.\n" \
              f"5. **Format Your Response**: Provide your response in the following JSON format:\n" \
              f"{{\n" \
              f"  \"PrimaryDiagnosis\": \"[You should take account of the preliminary diagnosis of the expert agent and their possibility below, rethink of your Primary Diagnosis]\",\n" \
              f"  \"ConfidenceLevel\": \"[High/Medium/Low]\",\n" \
              f"  \"DifferentialDiagnoses\": [\"list of differential diagnoses\"],\n" \
              "\"ProbabilityDistribution\": [\"sorted list: { disease: '...', probability: 0.xx }\"],\n" \
              f"  \"KeyFindings\": \"[summary of key findings]\",\n" \
              f"  \"KnowledgeAndResearch\": \"[A fluent paragraph summarizing selected knowledge and research, including specific details such as URLs or references.]\"\n" \
              f"}}\n" \
              f"Ensure your response is thorough, accurate, and actionable, with a focus on integrating insights from all specialized agents."
    return synthesizer, prompt


def get_case_review_prompt(current_case, historical_cases):
    """
    重构后的 Prompt：引入专家辩论机制
    """
    
    # 格式化历史病例，强调“原型”和“坑”
    formatted_history = ""
    for i, hc in enumerate(historical_cases):
        formatted_history += f"--- Historical Case #{i+1} ---\n"
        formatted_history += f"Diagnosis: {hc['case']['primary_diagnosis']}\n"
        formatted_history += f"Key Findings: {hc['case']['key_findings']}\n"
        if hc.get('prototype'):
            formatted_history += f"Expert Standard (Prototype): {hc['prototype']}\n"
        if hc.get('pitfalls'):
            formatted_history += f"Known Pitfalls: {hc['pitfalls']}\n"
        formatted_history += "\n"

    system_role = "You are a Senior Dermatological Consultant specializing in differential diagnosis."
    
    user_prompt = f"""
[Current Case for Review]
- Stated Primary Diagnosis: {current_case.get('PrimaryDiagnosis')}
- Clinical Findings: {current_case.get('KeyFindings')}
- Critical Features: {current_case.get('CriticalFeatures')}

[Reference Knowledge Base]
Below are {len(historical_cases)} historical cases that are VISUALLY similar to the current case, including expert-distilled prototypes:
{formatted_history}

[Task Instruction]
You must perform a 'Critical Differential Diagnosis'. Do not simply agree with the current diagnosis. 
Follow these steps:
1. Compare: Does the current case align better with the 'Expert Prototype' of the stated diagnosis, or does it share more critical features with a DIFFERENT diagnosis from the historical cases?
2. Contrast: Identify any 'Red Flags' (e.g., the current case has telangiectasias, but the Expert Prototype for this disease says it should have scaling).
3. Validate or Correct: If the evidence from historical cases strongly suggests an alternative diagnosis, you MUST correct it.

[Output Format]
Return ONLY a JSON object:
{{
    "OriginalDiagnosis": "{current_case.get('PrimaryDiagnosis')}",
    "RevisedDiagnosis": "The correct diagnosis (may be the same)",
    "ConfidenceScore": 0-1.0,
    "Reasoning": "Why did you keep or change the diagnosis? Point out specific feature conflicts.",
    "KeyFindings": "Updated clinical findings if necessary"
}}
"""
    return system_role, user_prompt


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
    image_region = current_case.get("ConfidenceLevel", "N/A")
    differential_diagnoses = current_case.get("DifferentialDiagnoses", [])
    key_findings = current_case.get("KeyFindings", "N/A")
    knowledge_and_research = current_case.get("KnowledgeAndResearch", "N/A")

    # Format the differential diagnoses as a comma-separated string
    differential_diagnoses_str = ", ".join(differential_diagnoses) if differential_diagnoses else "N/A"
    # Construct the prompt
    prompt = f"""
            You are a frontline medical professional specializing in performing initial patient assessments based on dermatological images. Now you have known that the correct diagnosis of this image is {skin_disease}

            Your primary role is to organize preliminary medical observations from images and provide insights to support further diagnosis and agent collaboration.
            ---

            ### ⚠️ CRITICAL INSTRUCTIONS FOR KEY FINDINGS:

            When writing the "KeyFindings" section, you MUST:
            Start with anatomical location and context …
            Describe morphology in clinical terms …
            Highlight "red flags" or warning signs …
            Explicitly rate severity at the end …
            Avoid bullet points or lists …
            Additionally, immediately after the paragraph, append a separate sentence that begins exactly with:
            "Critical diagnostic differences:"
            and then list 1–2 ultra-short phrases that would help distinguish this condition from the most likely differential diagnoses (e.g., "silvery scale on extension, Auspitz-positive" or "spares nasolabial fold, no mucosal involvement").
            Do not use line breaks or bullets inside this sentence.
            ---

            ### ✅ FORMAT YOUR RESPONSE STRICTLY AS JSON:

            {{
                "KeyFindings": "[single cohesive paragraph + Critical diagnostic differences: ...]",
                "CriticalFeatures": ["phrase1", "phrase2"],
                "KnowledgeAndResearch": "..."
            }}

        
            ---

            Now analyze the provided image and generate your response in the exact JSON format above. You should control you output in 5000 words
            """
    
    return prompt


