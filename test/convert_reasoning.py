import json
import re
from pathlib import Path
from typing import Any, Dict, List

# ---------------- 1. 23 类固定顺序 ----------------
DERMNET_DISEASE_NAME = [
    'Acne and Rosacea Photos',
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
    'Warts Molluscum and other Viral Infections'
]
TARGET_SET = set(DERMNET_DISEASE_NAME)

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

# ---------------- 4. 归一化函数（含兜底） ----------------
def normalize_distribution(raw_dist: List[Any]) -> List[Dict[str, float]]:
    """23 项：非零值按降序，零值保持原顺序垫后"""
    parsed: Dict[str, float] = {}

    # 4.1 先按原逻辑解析 dict / 伪 JSON
    for item in raw_dist:
        if isinstance(item, dict) and "disease" in item:
            d, p = item["disease"], parse_prob(item)
        else:
            try:
                s = re.sub(r"'", '"', str(item))
                obj = json.loads(s)
                d, p = obj["disease"], parse_prob(obj)
            except Exception:
                continue
        if d in TARGET_SET:
            parsed[d] = p

    # 4.2 若一轮下来啥也没捞到，再走兜底解析
    if not parsed:
        parsed = fallback_parse(raw_dist)

    if not parsed:
        return []

    # 4.3 稳定排序：有值的按降序，其余保持原顺序
    has_val = [(d, parsed[d]) for d in DERMNET_DISEASE_NAME if parsed.get(d, 0.0) != 0.0]
    zero_val = [(d, 0.0000) for d in DERMNET_DISEASE_NAME if parsed.get(d, 0.0) == 0.0]
    has_val.sort(key=lambda x: x[1], reverse=True)

    return [{"disease": d, "probability": p} for d, p in has_val + zero_val]

# ---------------- 5. 主流程（不变） ----------------
json_path = Path("/Volumes/T7/SkinGPT-X-EvaluationResults/Dermnet/test/Reasoning_output.json")
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