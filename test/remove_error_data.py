import json
from pathlib import Path

def drop_bad_entries(json_path: Path,
                     bad_value: str = "Unable to parse model output",
                     field: str = "PrimaryDiagnosis"):
    """
    删除 json 文件中指定字段值为 bad_value 的顶层对象
    :param json_path: 原始 json 文件
    :param bad_value: 要匹配的值
    :param field: 要检查的字段名
    :return: 已删除的 key 列表
    """
    data = json.loads(json_path.read_text(encoding='utf-8'))
    to_delete = [k for k, v in data.items()
                 if isinstance(v, dict) and v.get(field) == bad_value]

    for k in to_delete:
        del data[k]

    # 写回同名文件（可先备份）
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return to_delete


# ---------------- 调用示例 ----------------
if __name__ == "__main__":
    json_path = Path("/Volumes/T7/SkinGPT-X-EvaluationResults/HAM10000/SkinGPT-X/Reasoning_output.json")
    removed = drop_bad_entries(json_path)
    print(f"✅ 已删除 {len(removed)} 条无效数据：{removed}")