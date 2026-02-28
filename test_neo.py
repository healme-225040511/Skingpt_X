from neo4j import GraphDatabase
import json

# 配置信息（沿用你提供的）
uri = "bolt://100.91.219.47:7687"
user = "neo4j"
password = "Czty100165188"

def display_one_sample():
    driver = GraphDatabase.driver(uri, auth=(user, password), encrypted=False)
    
    # Cypher 查询：获取一个病例节点及其所属的原型节点
    query = """
    MATCH (c:Case)-[:BELONGS_TO]->(p:Prototype)
    RETURN c, p
    LIMIT 1
    """
    
    try:
        with driver.session() as session:
            result = session.run(query).single()
            
            if not result:
                print("抽样失败：数据库中可能还没有数据，或者节点之间没有建立 BELONGS_TO 关系。")
                return

            case_node = result['c']
            proto_node = result['p']

            print("="*60)
            print("🔍 数据库抽样展示")
            print("="*60)

            print(f"\n[病例节点 (Case)]")
            for key, value in case_node.items():
                # 对长向量进行截断处理，方便阅读
                if isinstance(value, list) and len(value) > 10:
                    print(f"  {key}: {value[:5]} ... (长度: {len(value)})")
                else:
                    print(f"  {key}: {value}")

            print(f"\n[关联原型 (Prototype)]")
            for key, value in proto_node.items():
                print(f"  {key}: {value}")

            print("\n" + "="*60)

    except Exception as e:
        print(f"查询出错: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    display_one_sample()