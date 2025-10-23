DISEASE_NAME = ['Acne and Rosacea Photos',
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
                'Warts Molluscum and other Viral Infections',
                'Other Diseases']
ISIC_DISEASE_NAME = [
    'Malignant',
    'Benign'
]
ISIC_PRECSV_PATH = '/Volumes/T7/SkinGPT-X-EvaluationResults/PanDerm_Base_LP_result/ISIC/PanDerm_Base_LP_predprob.csv'

MEDGAMMA_EVALUATION_PATH = '/Volumes/T7/SkinGPT-X-EvaluationResults/Experiments/Diagnosis/Medgamma/filename_to_medgamma_pred.csv'
REASONINGLAYER_EVALUATION_PATH = '/Volumes/T7/SkinGPT-X-EvaluationResults/Experiments/Diagnosis/SkinGPTX/Reasoning_output.csv'
CASEREVIEW_EVALUATION_PATH = '/Volumes/T7/SkinGPT-X-EvaluationResults/Experiments/Diagnosis/SkinGPTX/CaseReview_output.csv'
MEDGAMMA_LABELS_PATH = '/Volumes/T7/SkinGPT-X-EvaluationResults/Experiments/Diagnosis/filename_to_label.csv'
REASONING_LABELS_PATH = '/Volumes/T7/SkinGPT-X-EvaluationResults/Experiments/Diagnosis/SkinGPTX/filename_to_labels_Reasoning.csv'
CASEREVIEW_LABELS_PATH = '/Volumes/T7/SkinGPT-X-EvaluationResults/Experiments/Diagnosis/SkinGPTX/filename_to_labels_CaseReview.csv'
BASE_IMAGE_DIRECTORY = '/Volumes/T7/SkinGPT-X-Dataset/Dermnet/test'

MEDGAMMA_WORDHIT_OUTPUT = '/Volumes/T7/SkinGPT-X-EvaluationResults/Experiments/Diagnosis/SkinGPTX/MEDGAMMA_word_hit.csv'
CASEREVIEW_WORDHIT_OUTPUT = '/Volumes/T7/SkinGPT-X-EvaluationResults/Experiments/Diagnosis/SkinGPTX/CaseReview_word_hit.csv'
REASONING_WORDHIT_OUTPUT = '/Volumes/T7/SkinGPT-X-EvaluationResults/Experiments/Diagnosis/SkinGPTX/REASONING_word_hit.csv'

DERMNET_DATASET_ROOT = '/Volumes/T7/SkinGPT-X-Dataset/Dermnet/test'
ISIC_DATASET_ROOT = '/Volumes/T7/SkinGPT-X-Dataset/ISIC_2024_Resize224/test'