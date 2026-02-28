import json
import numpy as np
from neo4j import GraphDatabase

# --- 配置区 ---
NEO4J_URI = "bolt://100.88.26.154:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Czty100165188"

TRAIN_JSON_PATH = "/225040511/project/Evaluation_Results/HAM10000/SkinGPT-X/train_files.json"
TRAIN_FEATS_PATH = "/225040511/project/Evaluation_Results/HAM10000/SkinGPT-X/train_feats.npy"
BATCH_SIZE = 1000  # 每次提交1000条数据，防止内存或事务过大

def update_feature_vectors():
    # 1. 加载本地数据
    print("正在加载 train.json 和 train_feats.npy...")
    with open(TRAIN_JSON_PATH, 'r', encoding='utf-8') as f:
        image_paths = json.load(f)
    
    features = np.load(TRAIN_FEATS_PATH)
    print(f"特征矩阵加载完成，形状为: {features.shape}")

    # 2. 准备连接 Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    # 3. 构造待更新的数据列表
    # 结构: [{"path": "/train/...", "vec": [...]}, ...]
    update_data = []
    for idx, path in enumerate(image_paths):
        # 确保将 numpy 数组转为普通 list，并转为 float 类型
        vec = features[idx].astype(float).tolist()
        update_data.append({
            "path": path,
            "vec": vec
        })

    # 4. 执行批量更新
    total = len(update_data)
    print(f"开始更新 Neo4j，总计 {total} 条记录...")
    
    with driver.session() as session:
        for i in range(0, total, BATCH_SIZE):
            batch = update_data[i : i + BATCH_SIZE]
            
            # 使用 UNWIND 执行高效批量更新
            session.run("""
                UNWIND $batch AS item
                MATCH (c:Case {image_path: item.path})
                SET c.feature_vector = item.vec
            """, batch=batch)
            
            print(f"进度: {min(i + BATCH_SIZE, total)} / {total}")

    driver.close()
    print("所有 feature_vector 更新完毕！")

if __name__ == "__main__":
    update_feature_vectors()