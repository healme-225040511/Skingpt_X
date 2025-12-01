import json
import re
from pathlib import Path
from typing import Any, Dict, List

# ---------------- 1. 23 类固定顺序 ----------------
HAM10000_DISEASE_NAME = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
HAM10000_DISEASE_MAPPING_NAME = {"akiec": "Actinic keratoses and intraepithelial carcinoma / Bowen's disease",
                                 "bcc": "basal cell carcinoma",
                                 "bkl": "benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses)",
                                 "df": "dermatofibroma", "mel": "melanoma", "nv": "melanocytic nevi ",
                                 "vasc": "vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)"}
REVERSED_MAPPING = {
    "Actinic keratoses and intraepithelial carcinoma / Bowen's disease": "akiec",
    "basal cell carcinoma": "bcc",
    "benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses)": "bkl",
    "dermatofibroma": "df",
    "melanoma": "mel",
    "melanocytic nevi ": "nv",
    "vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)": "vasc"
}
MAPPING_NAME = [HAM10000_DISEASE_MAPPING_NAME[abbr] for abbr in HAM10000_DISEASE_NAME]
TARGET_SET = set(MAPPING_NAME)

# ---------------- 2. 万能概率解析 ----------------
def parse_prob(raw: Any) -> float:
    """遇 < 整个变 0；遇 % 再除以 100；其余转 float 并 4 位小数"""
    if isinstance(raw, dict):
        txt = str(raw.get("probability", "0"))
    else:
        txt = str(raw)

    if "<" in txt:
        return 0.0000

    txt = txt.strip().replace("%", "")
    try:
        val = float(txt)
    except ValueError:
        return 0.0000

    # 如果原始串带 %，则已经除了 100；否则不再除
    if isinstance(raw, str) and "%" in raw:
        val /= 100.0

    return round(max(0.0, min(1.0, val)), 4)

# ---------------- 3. 兜底解析：纯“疾病: 概率%”格式 ----------------
PAIR_RE = re.compile(r"^\s*([^:]+?)\s*:\s*([\d.]+)\s*%?\s*$")
def fallback_parse(raw_list: List[Any]) -> Dict[str, float]:
    """针对 ["疾病: 概率%", ...] 这种纯字符串做解析"""
    parsed: Dict[str, float] = {}
    for item in raw_list:
        if not isinstance(item, str):
            continue
        m = PAIR_RE.match(item)
        if not m:
            continue
        disease, prob_str = m.group(1), m.group(2)
        if disease not in TARGET_SET:
            continue
        # 手动构造一个带 % 的字符串，复用 parse_prob
        parsed[disease] = parse_prob(prob_str + "%")
    return parsed
# ---------- 稳健 key 匹配（大小写+空格容错） ---------- #
def _match_disease(name: str) -> str | None:
    """大小写不敏感 + 去首尾空格 + 去内部多余空格"""
    if not name:
        return None
    # 统一小写、去首尾空格、合并连续空格
    norm = " ".join(name.strip().lower().split())
    for std in TARGET_SET:
        if " ".join(std.lower().split()) == norm:
            return std
    return None
# ---------------- 4. 归一化函数（含兜底） ----------------
# ---------------- 4. 归一化函数（只改这里） ----------------
def normalize_distribution(raw_dist: List[Any]) -> List[Dict[str, float]]:
    parsed: Dict[str, float] = {}

    for item in raw_dist:
        # 4.1 已经是 dict
        if isinstance(item, dict) and "disease" in item:
            d_raw = item["disease"]
            # >>> 新增：大小写+空格容错
            d = _match_disease(d_raw)
            if d:
                parsed[d] = parse_prob(item)
            continue

        # 4.2 字符串：先尝试“疾病: 概率%”格式
        if isinstance(item, str):
            m = PAIR_RE.match(item)
            if m:
                d_raw, prob_str = m.group(1), m.group(2)
                d = _match_disease(d_raw)
                if d:
                    parsed[d] = parse_prob(prob_str + "%")
                continue

            # >>> 新增：单/双引号类 JSON 字符串，就地正则提取
            m2 = re.match(r"^\s*\{\s*['\"]disease['\"]\s*:\s*['\"](.*?)['\"]\s*,\s*['\"]probability['\"]\s*:\s*([\d.]+)\s*\}\s*$", item)
            if m2:
                d_raw, prob_str = m2.group(1), m2.group(2)
                d = _match_disease(d_raw)
                if d:
                    parsed[d] = parse_prob(prob_str)
                continue

            # 4.3 尝试真 JSON（保留原逻辑）
            try:
                s = re.sub(r"'", '"', str(item))
                obj = json.loads(s)
                d_raw, p = obj["disease"], parse_prob(obj)
                d = _match_disease(d_raw)
                if d:
                    parsed[d] = p
            except Exception:
                continue

    # 4.4 兜底解析（原逻辑）
    if not parsed:
        parsed = fallback_parse(raw_dist)

    # 4.5 排序输出（原逻辑）
    has_val = [(d, parsed[d]) for d in MAPPING_NAME if parsed.get(d, 0.0) != 0.0]
    zero_val = [(d, 0.0000) for d in MAPPING_NAME if parsed.get(d, 0.0) == 0.0]
    has_val.sort(key=lambda x: x[1], reverse=True)
    return [{"disease": REVERSED_MAPPING[d], "probability": p} for d, p in has_val + zero_val]
# ---------------- 5. 主流程（不变） ----------------
json_path = Path("/Volumes/T7/SkinGPT-X-EvaluationResults/HAM10000/SkinGPT-X/Reasoning_output.json")
out_path  = json_path.with_name("Reasoning_output_normalized.json")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

ok_cnt, empty_cnt = 0, 0
for fname, info in data.items():
    new_dist = normalize_distribution(info.get("ProbabilityDistribution", []))
    info["ProbabilityDistribution"] = new_dist
    if new_dist:
        ok_cnt += 1
    else:
        print(fname)
        empty_cnt += 1

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"✅ 归一化完成 → {out_path}")
print(f"   有效文件：{ok_cnt} 个 | 完全无效：{empty_cnt} 个")