from neo4j import GraphDatabase

# Neo4j 连接配置
URI = "bolt://localhost:7687"  # 替换为您的 Neo4j 地址
AUTH = ("neo4j", "Hyxzuinb6828")  # 替换为您的 Neo4j 用户名和密码

# 写入病例数据的函数
def write_cases_to_neo4j(cases):
    driver = GraphDatabase.driver(URI, auth=AUTH)
    with driver.session() as session:
        for case in cases:
            session.run(
                """
                CREATE (c:Case {
                    primary_diagnosis: $primary_diagnosis,
                    confidence_level: $confidence_level,
                    differential_diagnoses: $differential_diagnoses,
                    key_findings: $key_findings,
                    treatment_recommendations: $treatment_recommendations,
                    knowledge_and_research: $knowledge_and_research
                })
                """,
                primary_diagnosis=case["PrimaryDiagnosis"],
                confidence_level=case["ConfidenceLevel"],
                differential_diagnoses=case["DifferentialDiagnoses"],
                key_findings=case["KeyFindings"],
                treatment_recommendations=case["TreatmentRecommendations"],
                knowledge_and_research=case["KnowledgeAndResearch"]
            )
    driver.close()

cases = [
    {
        "PrimaryDiagnosis": "Basal Cell Carcinoma (BCC)",
        "ConfidenceLevel": "High",
        "DifferentialDiagnoses": [
            "Squamous Cell Carcinoma (SCC)",
            "Melanoma",
            "Seborrheic Keratosis"
        ],
        "KeyFindings": "The lesion is located on the nose, measuring approximately 1.2 cm in diameter. It has a pearly appearance with telangiectasia and a central ulceration. The borders are rolled, and the lesion is asymptomatic. The severity is rated as moderate, indicating a potential for local invasion.",
        "TreatmentRecommendations": "Referral to a dermatologic specialist for biopsy is recommended. If confirmed as BCC, treatment may involve surgical excision or Mohs micrographic surgery, depending on the size and location of the lesion.",
        "KnowledgeAndResearch": "According to the American Academy of Dermatology (AAD), Mohs surgery is the gold standard for high-risk BCC due to its high cure rates and tissue-sparing properties (AAD Skin Cancer Guidelines: https://www.aad.org). Recent studies emphasize the role of Hedgehog pathway inhibitors in advanced BCC (Hedgehog Pathway Inhibition in Basal Cell Carcinoma: https://pubmed.ncbi.nlm.nih.gov/40048197/)."
    },
    {
        "PrimaryDiagnosis": "Melanoma",
        "ConfidenceLevel": "High",
        "DifferentialDiagnoses": [
            "Squamous Cell Carcinoma (SCC)",
            "Basal Cell Carcinoma (BCC)",
            "Dysplastic Nevus"
        ],
        "KeyFindings": "The lesion is located on the upper back, measuring approximately 2.0 cm in diameter. It has an irregular shape with color variations ranging from dark brown to black, and an uneven surface. The borders are irregular and poorly defined. Symptoms include mild itching, and the severity is rated as high, indicating a high potential for malignancy.",
        "TreatmentRecommendations": "Immediate referral to a dermatologic specialist for biopsy is recommended. If confirmed as melanoma, treatment may involve wide local excision and sentinel lymph node biopsy. Adjuvant therapies such as immunotherapy or targeted therapy may be considered based on staging.",
        "KnowledgeAndResearch": "According to the American Academy of Dermatology (AAD), early detection and excision are critical for melanoma survival (AAD Skin Cancer Guidelines: https://www.aad.org). Recent advancements in immunotherapy, such as checkpoint inhibitors, have significantly improved outcomes for advanced melanoma (Immunotherapy in Melanoma: https://pubmed.ncbi.nlm.nih.gov/40048198/)."
    },
    {
        "PrimaryDiagnosis": "Actinic Keratosis",
        "ConfidenceLevel": "Medium",
        "DifferentialDiagnoses": [
            "Squamous Cell Carcinoma (SCC)",
            "Basal Cell Carcinoma (BCC)",
            "Seborrheic Keratosis"
        ],
        "KeyFindings": "The lesion is located on the forehead, measuring approximately 0.8 cm in diameter. It has a rough, scaly texture with erythema and mild hyperkeratosis. The borders are well-defined, and the lesion is asymptomatic. The severity is rated as low, indicating a potential for progression to SCC.",
        "TreatmentRecommendations": "Topical treatments such as 5-fluorouracil or imiquimod are recommended. Cryotherapy may also be considered for isolated lesions. Regular follow-up is advised to monitor for progression.",
        "KnowledgeAndResearch": "According to the American Academy of Dermatology (AAD), actinic keratosis is a precursor to SCC and requires timely treatment to prevent malignant transformation (AAD Skin Cancer Guidelines: https://www.aad.org). Recent studies highlight the efficacy of field-directed therapies for multiple lesions (Field-Directed Therapies in Actinic Keratosis: https://pubmed.ncbi.nlm.nih.gov/40048199/)."
    },
    {
        "PrimaryDiagnosis": "Seborrheic Keratosis",
        "ConfidenceLevel": "Low",
        "DifferentialDiagnoses": [
            "Actinic Keratosis",
            "Basal Cell Carcinoma (BCC)",
            "Melanoma"
        ],
        "KeyFindings": "The lesion is located on the chest, measuring approximately 1.0 cm in diameter. It has a waxy, stuck-on appearance with a tan to dark brown color. The borders are well-defined, and the lesion is asymptomatic. The severity is rated as low, indicating a benign condition.",
        "TreatmentRecommendations": "No treatment is necessary unless the lesion is symptomatic or cosmetically concerning. If desired, cryotherapy or curettage can be performed.",
        "KnowledgeAndResearch": "According to the American Academy of Dermatology (AAD), seborrheic keratosis is a benign condition and does not require treatment unless symptomatic (AAD Skin Cancer Guidelines: https://www.aad.org). Recent studies emphasize the importance of differentiating seborrheic keratosis from malignant lesions (Dermoscopic Features of Seborrheic Keratosis: https://pubmed.ncbi.nlm.nih.gov/40048200/)."
    },
    {
        "PrimaryDiagnosis": "Squamous Cell Carcinoma (SCC)",
        "ConfidenceLevel": "High",
        "DifferentialDiagnoses": [
            "Basal Cell Carcinoma (BCC)",
            "Melanoma",
            "Actinic Keratosis"
        ],
        "KeyFindings": "The lesion is located on the lateral aspect of the right cheek, near the eye, measuring approximately 1.5 cm in diameter. It has an irregular, triangular shape with color variations ranging from dark brown to black, and a shiny, possibly ulcerated surface. The borders are irregular, with some areas appearing well-defined. Symptoms may include discomfort, and the severity is rated as moderate, indicating a potential for malignancy.",
        "TreatmentRecommendations": "Immediate referral to a dermatologic specialist for biopsy is recommended. If confirmed as SCC, treatment may involve surgical excision, and consideration of adjuvant therapies such as chemotherapy or immunotherapy, depending on staging and risk factors.",
        "KnowledgeAndResearch": "According to the American Academy of Dermatology (AAD), excisional biopsy is the standard protocol for suspected malignant lesions, particularly melanoma and SCC (AAD Skin Cancer Guidelines: https://www.aad.org). Recent literature highlights advancements in immunotherapy for squamous cell carcinoma, which can significantly impact survival outcomes (Current Progress and Future Directions of Immunotherapy in Head and Neck Squamous Cell Carcinoma: https://pubmed.ncbi.nlm.nih.gov/40048196/). The National Comprehensive Cancer Network (NCCN) provides comprehensive guidelines for management, emphasizing early detection and intervention (NCCN Guidelines for Squamous Cell Skin Cancer: https://www.nccn.org/guidelines/guidelines-detail?category=1&id=1465)."
    }
]

# 写入数据
write_cases_to_neo4j(cases)
print("Cases have been written to Neo4j.")