import csv
from pathlib import Path

def read_misclassified_filenames(csv_path: str | Path) -> list[str]:
    """
    读取 misclassified_reasoning.csv 中的文件名（第一列），返回文件名列表。

    Parameters
    ----------
    csv_path : str | pathlib.Path
        CSV 文件路径。

    Returns
    -------
    list[str]
        所有出现在文件中的文件名（含子目录），例如：
        ['Benign/6342.jpg', 'Benign/6329.jpg', ...]
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"{csv_path} 不存在")

    filenames = []
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)          # 跳过表头
        for row in reader:
            if row:                 # 防止空行
                filenames.append(row[0])
    return filenames


# 使用示例
if __name__ == "__main__":
    files = read_misclassified_filenames("misclassified_reasoning.csv")
    print(f"共读取到 {len(files)} 个文件名")
    print(files[:10])  # 预览前 10 条