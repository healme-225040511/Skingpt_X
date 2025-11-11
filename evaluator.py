import os
import google.generativeai as genai
import pandas as pd
from typing import List, Dict

class EvaluatorAgent:
    """
    基于 Gemini 的零-shot 皮肤文本恶性肿瘤评估器
    支持两种调用方式：
      1. evaluate(list[str])          -> 单批次预测
      2. evaluate_csv(csv_path, ...)  -> 读取 CSV 批量预测
    """

    _PROMPT = """
You are a board-certified dermatologist.
Below is a list of predicted skin diseases (one per line).
For EACH line, answer exactly one word: **Malignant** or **Benign**.
Then give a final verdict for the whole list:  
**Final: Cancerous(Malignant)** or **Final: Non-Cancerous(Benign)**  
(= “只要出现任意一条 Malignant，Final 就是 Malignant”).
You should take account if prediction text contains **Malignant/Benign** to give your final answer.
Output format:
1. <Malignant|Benign>   
2. <Malignant|Benign>
...
Final: <Cancerous(Malignant)|Non-Cancerous(Benign)>

Predictions:
{predictions}
"""

    def __init__(self, api_key: str = None, model: str = "gemini-2.0-flash"):
        genai.configure(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(model)

    # ---------------- 核心评估逻辑 ----------------
    def evaluate(self, predictions: List[str]) -> Dict:
        text = "\n".join(predictions)
        response = self.model.generate_content(self._PROMPT.format(predictions=text))
        raw = response.text.strip()
        lines = [l for l in raw.splitlines() if l.startswith(
            ("Final:", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10."))]
        final_line = [l for l in lines if l.startswith("Final:")][-1]
        final = final_line.replace("Final:", "").strip()
        return {"raw": raw, "final": final, "per_line": lines[:-1]}

    # ---------------- 读取 CSV 批量评估 ----------------
    def evaluate_csv(self,
                     csv_path: str,
                     text_col: str = " medgamma_pred",
                     save_path: str = None) -> pd.DataFrame:
        """
        读取 CSV -> 评估 -> 追加两列 [per_line, final] -> 可选保存
        """
        df = pd.read_csv(csv_path)
        if text_col not in df.columns:
            raise KeyError(f"列 '{text_col}' 不存在于 {csv_path}")

        preds = df[text_col].astype(str).tolist()
        res = self.evaluate(preds)

        # 把逐条结果拆开
        per_line = [l.split(".", 1)[-1].strip() for l in res["per_line"]]
        df["gemini_per_line"] = per_line + [None] * (len(df) - len(per_line))
        df["gemini_final"] = res["final"]

        if save_path:
            df.to_csv(save_path, index=False)
            print(f"[+] 结果已保存至 {save_path}")
        return df


# ------------------- demo -------------------
if __name__ == "__main__":
    agent = EvaluatorAgent(api_key='AIzaSyDClRNJkcDgHv2wA90v6TODPvBlu8umIWU')          # 默认读环境变量 GEMINI_API_KEY

    # 1) 单批次快速测试
    preds = [
        "Plaque Psoriasis",
        "Basal Cell Carcinoma (nodular type)",
        "Atopic Dermatitis flare"
    ]
    print("=== 单批次测试 ===")
    print("Final verdict:", agent.evaluate(preds)["final"])

    # 2) 读取 CSV 批量评估
    csv_file = "/content/drive/MyDrive/Project/SkinGPT-X-EvaluationResults/Experiments/Diagnosis/ISIC/results.csv"        # 把你的文件放这儿
    out_file = "/content/drive/MyDrive/Project/SkinGPT-X-EvaluationResults/Experiments/Diagnosis/ISIC/isic2024_test_pred_labeled.csv"
    print("\n=== CSV 批量评估 ===")
    df_labeled = agent.evaluate_csv(csv_file, save_path=out_file)
    print(df_labeled[["image_path", "gemini_final"]].head())