import pandas as pd

# 读取CSV文件
file_path = "/Volumes/T7/SkinGPT-X-EvaluationResults/HAM10000/SkinGPT-X/Reasoning_output_softmax.csv"

df = pd.read_csv(file_path)

# 确保概率列是数值类型
# 假设第一列是文件名，最后一列是label，其余列是概率值
filename_column = df.columns[0]
label_column = df.columns[-1]
probability_columns = df.columns[1:-1]  # 排除文件名和label列

# 将概率列转换为数值类型，非数值的转换为NaN
for col in probability_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 检查是否有非数值数据（NaN）
if df[probability_columns].isnull().any().any():
    print("警告：数据中存在非数值数据，已将其转换为NaN。")
    # 可以选择填充NaN值，例如用0填充
    df[probability_columns] = df[probability_columns].fillna(0)

# 归一化每一行的概率
def normalize_row(row):
    probability_values = row[probability_columns]  # 获取概率列的值
    total_sum = probability_values.sum()
    if total_sum == 0:
        # 如果总和为0，将所有概率值设置为相等
        normalized_probabilities = [1.0 / len(probability_values)] * len(probability_values)
    else:
        normalized_probabilities = probability_values / total_sum
    return normalized_probabilities

# 应用归一化函数
normalized_data = df.apply(normalize_row, axis=1, result_type='expand')

# 将归一化后的数据与原始文件名列和label列合并
df_normalized = pd.concat([df[filename_column], normalized_data, df[label_column]], axis=1)

# 输出归一化后的结果
print(df_normalized)

# 保存归一化后的数据到新的CSV文件
output_file_path = "Reasoning_output_softmax_normalized.csv"
df_normalized.to_csv(output_file_path, index=False)
print(f"归一化后的数据已保存到 {output_file_path}")