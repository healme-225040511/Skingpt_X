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

from utils import safe_load_json


class ReasoningAgent:
    def __init__(self, model, api_key):
        """
        Initialize the ReasoningAgent with an API key and optional historical cases file path.
        
        Args:
            model (str): The name of the model to use.
            api_key (str): The API key for accessing OpenAI services.
        """
        self.model = model
        self.api_key = api_key

    def generate_report(self, current_case: Dict, prob_vec: list[float] = None) -> Dict:
        """
        Generate a comprehensive diagnostic report for the current case using OpenAI.
        
        Args:
            current_case (Dict): The current case data.
        
        Returns:
            Dict: The generated report.
        """
        # Prepare the prompt for OpenAI
        synthesizer, prompt = self._build_prompt(current_case, prob_vec)

        response = generate_response(
            engine=self.model, 
            temperature=0.5,
            max_tokens=2500,
            frequency_penalty=0,
            presence_penalty=0, 
            stop=None, 
            system_role=synthesizer, 
            user_input=prompt,
            api_key=self.api_key
        )

        return safe_load_json(response)

    def _build_prompt(self, analyses, prob_vec: list[float] = None):
        """
        Build a prompt for OpenAI based on the current case.
        
        Args:
            analyses (str): The total analyses from different agents.
        
        Returns:
            synthesizer (str): The synthesizer role in the conversation.
            prompt (str): The prompt for OpenAI.
        """
        synthesizer, prompt = get_synthesized_report_prompt(analyses, prob_vec)
        return synthesizer, prompt


if __name__ == "__main__":
    # Build the agent
    model = "gemini-2.5-pro"
    api_key = "AIzaSyC-9og_9OsxvKZ0rBXeMGboXBrMOpG5-do"
    reasoning_agent = ReasoningAgent(model=model, api_key=api_key)

    current_case = {
        "RAG": """
        ### 1. Image Region
        - **Anatomical Region:** The image displays the right cheek, temple, and periorbital (around the eye) region of the face.
        - **Contextual Details:** This is a seborrheic area with a high density of pilosebaceous units, making it a common site for inflammatory skin conditions like acne. The presence of fine vellus hair and early beard growth suggests a post-pubertal male.
        
        ### 2. Key Findings
        - **Lesion Description:**
            - **Location and Distribution:** Multiple, discrete, and some confluent erythematous papules and pustules are scattered across the cheek and temple.
            - **Morphology:** The primary lesions are inflammatory papules (small, red bumps) and pustules (papules with a visible white or yellow center of pus). Some lesions appear to be resolving, while others are in an active inflammatory state. Open or closed comedones (blackheads/whiteheads) are not clearly visible but are presumed to be present. There is no evidence of deep nodules or cysts.
            - **Color and Texture:** The lesions are predominantly erythematous (red) to pink. The surrounding skin shows mild inflammation.
            - **Borders:** Individual lesions are well-demarcated.
        - **Associated Symptoms:** Based on the inflammatory nature, associated symptoms would typically include mild tenderness or pruritus (itching).
        - **Severity:** **Moderate**. The severity is graded as moderate due to the presence of numerous inflammatory papules and pustules, without the presence of more severe nodulocystic lesions.
        
        ### 3. Diagnostic Assessment
        - **Primary Diagnosis:** **Acne Vulgaris** (Confidence: High)
            - **Supporting Evidence:** The diagnosis is strongly supported by the clinical presentation of inflammatory papules and pustules distributed across a seborrheic area of the face in a pattern characteristic of this common condition. The morphology of the lesions is pathognomonic for inflammatory acne.
        
        - **Differential Diagnoses:**
            1.  **Papulopustular Rosacea:** (Confidence: Low)
                - **Supporting Evidence:** Can also present with erythematous papules and pustules on the face.
                - **Distinguishing Features:** Rosacea is typically associated with a background of persistent erythema, flushing, and telangiectasias, and importantly, lacks comedones. The distribution in the image is more typical for acne than the central-facial pattern of rosacea.
            2.  **Staphylococcal Folliculitis:** (Confidence: Low)
                - **Supporting Evidence:** Presents as perifollicular pustules and papules.
                - **Distinguishing Features:** While morphologically similar, bacterial folliculitis can occur anywhere on hair-bearing skin and is not typically associated with the comedonal component of acne. The polymorphic nature of the lesions in the image (various stages of development) is more indicative of acne vulgaris.
            3.  **Drug-Induced Acne (Acneiform Eruption):** (Confidence: Low)
                - **Supporting Evidence:** Certain medications (e.g., steroids, anticonvulsants) can trigger an acne-like eruption.
                - **Distinguishing Features:** These eruptions are often monomorphic (lesions are all in the same stage of development) and can appear in atypical locations. A thorough patient history would be required to rule this out, but the presentation is classic for common acne vulgaris. The knowledge base suggests considering anabolic steroid use in refractory cases.
        
        - **Critical Findings:** There are no findings that suggest a medical emergency. However, moderate inflammatory acne can lead to scarring and significant psychosocial distress if not managed effectively.
        
        ### 4. Research Context
        The diagnosis of acne vulgaris is primarily clinical, based on the presence of characteristic lesions (comedones, papules, pustules, nodules) in typical locations (face, neck, chest, back, and shoulders).
        
        - **Diagnostic Criteria and Clinical Guidelines:**
            - The American Academy of Dermatology (AAD) guidelines emphasize diagnosis through a physical exam, identifying the type and severity of lesions to guide treatment.
            - For moderate acne, as seen here, guidelines recommend topical combination therapy (e.g., topical retinoid and benzoyl peroxide) with or without an oral antibiotic.
            - The knowledge base notes that in severe or refractory cases, clinicians should re-evaluate the patient history for exacerbating factors like medications or underlying endocrine conditions such as Polycystic Ovary Syndrome (PCOS) in females.
        
        - **Evidence-Based Insights:**
            - Acne is a chronic inflammatory disease of the pilosebaceous unit. Its pathogenesis is multifactorial, involving follicular hyperkeratinization, increased sebum production, proliferation of *Cutibacterium acnes* (*C. acnes*), and inflammation.
            - Treatment strategies are aimed at targeting these factors. Combination therapy is considered the standard of care as it addresses multiple pathogenic pathways and can reduce the risk of antibiotic resistance.
        
        - **Relevant Medical Resources:**
            - **Key Studies:** Clinical trials consistently demonstrate that combination therapy is more effective than monotherapy for moderate acne. For example, studies comparing a fixed-dose combination of adapalene-benzoyl peroxide to either agent alone show superior efficacy.
            - **Guidelines:**
                - Zaenglein, A. L., Pathy, A. L., Schlosser, B. J., Alikhan, A., Baldwin, H. E., Berson, D. S., ... & Keri, J. E. (2016). Guidelines of care for the management of acne vulgaris. *Journal of the American Academy of Dermatology*, 74(5), 945-973.
                - Thiboutot, D. M., Dréno, B., Abanmi, A., et al. (2018). Practical management of acne for clinicians: An international consensus from the Global Alliance to Improve Outcomes in Acne. *Journal of the American Academy of Dermatology*, 78(2S), S1-S23.
        
        - **Key ReferThe image displays a sun-exposed area of the face, specifically the right lateral aspect, which includes the cheek, temple, and extending towards the hairline. The skin appears to be relatively healthy with minimal signs of redness or inflammation. However, there are several areas of concern that require further examination and treatment. The acne lesions, particularly the papules and pustules, are located on the cheeks and templesences:**
            1.  James, W. D., Elston, D. M., Treat, J. R., Rosenbach, M. A., & Neuhaus, I. M. (2020). *Andrews' Diseases of the Skin: Clinical Dermatology*. (13th ed.). Elsevier.
            2.  Bolognia, J. L., Schaffer, J. V., & Cerroni, L. (2017). *Dermatology*. (4th ed.). Elsevier.
            3.  Zaenglein, A. L., et al. (2016). Guidelines of care for the management of acne vulgaris. *Journal of the American Academy of Dermatology*, 74(5), 945-973.        """,
        "WebSearch": """
           Based on a thorough analysis of the provided image, here is a detailed dermatological assessment.
            ### 1. Image Region
            - **Anatomical Region:** The image displays the right lateral aspect of the face, specifically the cheek, temple, and extending towards the hairline.
            - **Contextual Details:** This is a sun-exposed area with a high concentration of pilosebaceous units (hair follicles and sebaceous glands), making it a very common site for acne.
            
            ### 2. Key Findings
            - **Primary Observations:** The skin exhibits multiple scattered, discrete lesions against a background of normal skin tone.
            - **Lesion Description:**
                - **Location and Distribution:** Inflammatory lesions are spread across the cheek and temple.
                - **Type and Shape:** The predominant lesions are erythematous (red) papules (small, raised bumps) and some pustules (pus-filled bumps). They are generally round to oval in shape.
                - **Size and Color:** Lesions vary in size from approximately 2 to 5 mm in diameter. The color is primarily pink to red, indicative of active inflammation. Some lesions appear to be resolving, leaving faint red marks known as post-inflammatory erythema.
                - **Borders and Texture:** The borders of the lesions are fairly well-defined. The texture is characterized by raised, inflamed bumps. There are no visible open comedones (blackheads) or cysts.
            - **Associated Symptoms:** While not visible, this condition is often associated with mild soreness or tenderness upon palpation. It can also be asymptomatic.
            - **Severity Rating:** **Moderate**. This classification is based on the presence of numerous inflammatory papules and pustules, without evidence of more severe nodular or cystic lesions.
            
            ### 3. Diagnostic Assessment
            - **Primary Diagnosis:** **Moderate Inflammatory Acne Vulgaris** (Confidence: High)
                - **Supporting Evidence:** The clinical presentation of inflammatory papules and pustules on the face of what appears to be a younger individual is classic for acne vulgaris. The density and type of lesions are consistent with a moderate severity level, as defined by major dermatological guidelines.
            - **Differential Diagnoses:**
                1.  **Papulopustular Rosacea:** This condition also presents with red papules and pustules. However, it is typically accompanied by persistent background redness (erythema), flushing, and telangiectasias (visible small blood vessels), none of which are clearly evident here. Rosacea also lacks comedones, which are the primary lesion of acne (though not visible here, they are presumed to be part of the underlying process).
                2.  **Staphylococcal Folliculitis:** A bacterial infection of hair follicles can mimic inflammatory acne. However, folliculitis lesions are often more uniform in size, dome-shaped, and may have a central hair. The varied appearance of the lesions in the image makes acne more likely.
            - **Critical Findings:** There are no critical or urgent findings that suggest a medical emergency. However, moderate acne warrants medical attention to prevent potential long-term effects such as scarring and post-inflammatory hyperpigmentation, as well as to mitigate the significant psychosocial impact it can have.
            
            ### 4. Research Context
            Recent medical literature emphasizes the inflammatory nature of acne vulgaris, even at its earliest stages, and guidelines are continuously updated to reflect the best treatment practices.
            
            - **Diagnostic and Treatment Guidelines:**
                The American Academy of Dermatology (AAD) provides the most widely recognized guidelines for acne management. The most recent update in 2024 continues to classify acne based on the type and number of lesions (comedonal, papulopustular, nodular) and overall severity. For moderate inflammatory acne, as seen here, guidelines typically recommend a combination of topical therapies (e.g., retinoids, benzoyl peroxide, topical antibiotics) and potentially oral antibiotics or hormonal agents for females.
            
            - **Pathophysiology Insights:**
                Modern research highlights that acne is fundamentally an inflammatory disease. Inflammation is not just a consequence of bacterial growth (*Cutibacterium acnes*) but is involved from the very beginning of lesion formation. Factors such as the host's immune response to *C. acnes*, sebum composition, and genetic predispositions are key drivers of the inflammatory cascade that results in the visible papules and pustules.
            
            - **Relevant Medical Links and Key References:**
                Here are key resources and peer-reviewed articles that support this assessment and provide further information on diagnosis and management:
            
                1.  **American Academy of Dermatology (AAD) Acne Clinical Guideline (2024):** This guideline is the cornerstone for evidence-based management of acne vulgaris in the United States and is influential globally.
                    - **Reference:** Reynolds, R. V., et al. (2024). *Guidelines of care for the management of acne vulgaris*. Journal of the American Academy of Dermatology. (Note: The 2024 update provides the latest recommendations.)
            
                2.  **Review on Acne Pathophysiology and Treatment (2023):** This article provides a comprehensive overview of the inflammatory mechanisms underlying acne and discusses how current and future treatments target these pathways.
                    - **Reference:** Tang, Y., et al. (2023). *Acne vulgaris: A review of the pathophysiology, treatment, and recent nanotechnology based advances*. International Journal of Pharmaceutics. This article discusses the complex interplay of factors leading to acne lesions.
            
                3.  **Article on Targeting Inflammation in Acne (2023):** This recent review focuses specifically on the role of inflammation and the therapeutic agents designed to control it, which is directly relevant to the patient's presentation.
                    - **Reference:** Thiboutot, D. M., & Zaenglein, A. L. (2023). *Targeting Inflammation in Acne: Current Treatments and Future Prospects*. American Journal of Clinical Dermatology. This provides insight into the rationale for using anti-inflammatory treatments.
             - Running: search_knowledge_base(query=acne vulgaris diagnostic criteria and clinical guidelines)
           """,
        "SkinGPT":
        """  
        ### 1. Image Region
        - **Anatomical Location:** The images display the side of a person's face, including the cheek, temple, and jawline area.
        - **Contextual Details:** This is a seborrheic area (oil-producing) and a common site for follicular-based skin conditions. The presence of facial hair (stubble/sideburns) is noted.
        
        ### 2. Key Findings
        - **Lesion Description:** The skin exhibits multiple, scattered lesions of varying types.
            - **Distribution:** The lesions are widespread across the cheek and temple in a follicular pattern (centered around hair follicles).
            - **Primary Lesions:** The predominant lesions are inflammatory papules (small, red, raised bumps) and pustules (papules with a visible white/yellow pus-filled center). Some smaller, non-inflammatory bumps may represent closed comedones (whiteheads).
            - **Color:** Lesions are erythematous (pink to red), with some showing a central purulent collection.
            - **Texture:** The overall skin surface appears uneven and bumpy due to the numerous lesions.
            - **Associated Signs:** There is localized erythema surrounding the inflammatory lesions. No significant scaling, erosions, or honey-colored crusting is apparent.
        - **Severity:** Based on the number and type of inflammatory lesions, the condition is rated as **Moderate**. There are no visible nodules or cysts, which would indicate a more severe form.
        
        ### 3. Diagnostic Assessment
        - **Primary Diagnosis:** **Acne Vulgaris** (Confidence: High)
            - **Rationale:** The clinical presentation is classic for moderate inflammatory acne. The key supporting features are the patient's likely age demographic (adolescent or young adult), the location on the face, and the polymorphic nature of the lesions, including inflammatory papules and pustules, likely arising from comedones. This aligns strongly with the high prior probability (72.3%) provided by the AI.
        
        - **Differential Diagnoses:**
            1.  **Papulopustular Rosacea:** This is a reasonable differential but less likely. Rosacea typically presents in an older age group (30+) and is characterized by a background of persistent facial erythema and telangiectasias (visible small blood vessels), which are not apparent here. Crucially, rosacea lacks comedones, which are the primary lesion of acne.
            2.  **Bacterial Folliculitis:** This is an infection of the hair follicles, which can present with erythematous papules and pustules. However, folliculitis lesions are typically monomorphic (all appearing at the same stage of development), whereas the lesions in the image appear polymorphic (in various stages), which is more characteristic of acne. This diagnosis corresponds to the AI's second suggestion but is less likely than acne given the overall picture.
            3.  **Pityrosporum (Malassezia) Folliculitis:** This fungal condition presents as monomorphic, often itchy, papules and pustules, typically on the upper trunk, but can affect the face. The polymorphic appearance here makes it less probable.
        
        - **Critical Findings:** There are no critical or urgent findings requiring immediate emergency attention. However, moderate acne can lead to permanent scarring and significant psychosocial distress. Therefore, a consultation with a dermatologist or primary care provider for appropriate treatment is recommended.
        """
    }

    # generate report
    report = reasoning_agent.generate_report(current_case)
    print(json.dumps(report, indent=2))

