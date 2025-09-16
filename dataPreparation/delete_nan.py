# delete_empty_rows.py
import pandas as pd

IN_FILE = '/Users/macbook/Desktop/SkinGPT-X-EvaluationResults/DDI/output/' + 'merged_diagnostic_assessment_with_groundtruth.csv'
OUT_FILE = '/Users/macbook/Desktop/SkinGPT-X-EvaluationResults/DDI/output/'+ 'merged_diagnostic_assessment_with_groundtruth_clean.csv'

# 读取
df = pd.read_csv(IN_FILE)

# 把空字符串也当成 NaN，然后一次性 drop
df = df.replace('', pd.NA).dropna()

# 保存
df.to_csv(OUT_FILE, index=False)

print(f'已完成！共删除 {len(pd.read_csv(IN_FILE)) - len(df)} 行，结果写入 {OUT_FILE}')