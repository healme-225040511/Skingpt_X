import pandas as pd
import json   # 仅需新增这一行

outputPath = '/Users/macbook/Desktop/SkinGPT-X-EvaluationResults/DDI/output/'
dataPath   = '/Users/macbook/Desktop/SkinGPT-X-Dataset/DDI/'

# ---------- 1. 读取4个诊断CSV ----------
file1 = outputPath + 'RAG_diagnostic_assessment.csv'
file2 = outputPath + 'SkinGPT_diagnostic_assessment.csv'
file3 = outputPath + 'WebSearch_diagnostic_assessment.csv'
file4 = outputPath + 'treatmentRecommend_diagnostic_assessment.csv'
groundtruth_file = dataPath + 'ddi_metadata.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)
df4 = pd.read_csv(file4)
groundtruth_df = pd.read_csv(groundtruth_file)

# 剔除失败行
for df in (df1, df2, df3):
    df.drop(df[df['DiagnosticAssessment'].str.contains('not success', na=False)].index, inplace=True)

# 重命名列
df1.rename(columns={'DiagnosticAssessment': 'RAG_diagnostic_assessment'}, inplace=True)
df2.rename(columns={'DiagnosticAssessment': 'SkinGPT_diagnostic_assessment'}, inplace=True)
df3.rename(columns={'DiagnosticAssessment': 'WebSearch_diagnostic_assessment'}, inplace=True)
df4.rename(columns={'DiagnosticAssessment': 'Treatment_diagnostic_assessment'}, inplace=True)

# ---------- 2. 合并 ----------
merged_df = df1.merge(df2, on='image_name', how='outer')\
               .merge(df3, on='image_name', how='outer')

# ---------- 3. 合并 groundtruth ----------
groundtruth_df = groundtruth_df[['DDI_file', 'disease']].rename(
    columns={'DDI_file': 'image_name', 'disease': 'groundtruth'})
merged_df = merged_df.merge(groundtruth_df, on='image_name', how='left')

# ---------- 4. 新增：合并 JSON PrimaryDiagnosis ----------
json_path = outputPath + 'TreatmentRecommend_output.json'
with open(json_path, 'r', encoding='utf-8') as f:
    tx_dict = json.load(f)

# 构造 DataFrame
tx_df = pd.DataFrame([
    {'image_name': k, 'PrimaryDiagnosis': v.get('PrimaryDiagnosis', '')}
    for k, v in tx_dict.items()
])

# 合并
merged_df = merged_df.merge(tx_df, on='image_name', how='left')

# ---------- 5. 保存 ----------
out_file = outputPath + 'merged_diagnostic_assessment_with_groundtruth.csv'
merged_df.to_csv(out_file, index=False)
print(f"合并完成（含 PrimaryDiagnosis），结果已保存到 {out_file}")