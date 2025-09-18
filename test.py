import re

text = """
1.  **Syringomas:**
        - **Rationale:** These are benign tumors of the sweat ducts that also present as small, skin-colored or yellowish papules, commonly in the periorbital area. They can be difficult to distinguish from milia visually but are often flatter and may coalesce into plaques.
    2.  **Sebaceous Hyperplasia:**
        - **Rationale:** This condition involves enlarged sebaceous (oil) glands, presenting as yellowish papules. However, they typically have a central depression or umbilication, which is not clearly visible on the lesions in the image. They are also more common on the forehead and nose.
    3.  **Flat Warts (Verruca Plana):**
        - **Rationale:** These are small, flat-topped papules caused by the human papillomavirus (HPV). They can be numerous on the face but usually have a flatter, rougher surface compared to the smooth, dome-shaped appearance of the lesions shown.
"""

# 使用正则表达式匹配**和****之间的疾病名
pattern = re.compile(r'^\s*\d+\.\s*\*\*(.+?):\*\*', re.M)
diseases = [m.strip() for m in pattern.findall(text)]
print(diseases)