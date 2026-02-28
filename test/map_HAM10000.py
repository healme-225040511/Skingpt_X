from neo4j import GraphDatabase

# 你的映射字典
HAM10000_DISEASE_MAPPING_NAME = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma / Bowen's disease",
    "bcc": "basal cell carcinoma",
    "bkl": "benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses)",
    "df": "dermatofibroma", 
    "mel": "melanoma", 
    "nv": "melanocytic nevi ",
    "vasc": "vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)"
}

class Neo4jUpdater:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def update_diagnoses(self):
        with self.driver.session() as session:
            for short_name, full_name in HAM10000_DISEASE_MAPPING_NAME.items():
                print(f"正在更新: {short_name} -> {full_name}")
                
                # 更新 Case 节点
                # 注意：这里同时更新了 primary_diagnosis 和 true_label
                session.run("""
                    MATCH (c:Case) 
                    WHERE c.primary_diagnosis = $short 
                    SET c.primary_diagnosis = $full, c.true_label = $full
                """, short=short_name, full=full_name)

                # 更新 Prototype 节点
                session.run("""
                    MATCH (p:Prototype) 
                    WHERE p.disease = $short 
                    SET p.disease = $full
                """, short=short_name, full=full_name)
        print("更新完成！")

# 使用示例
updater = Neo4jUpdater("bolt://100.88.26.154:7687", "neo4j", "Czty100165188")
updater.update_diagnoses()
# updater.close()