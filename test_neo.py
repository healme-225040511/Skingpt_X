from neo4j import GraphDatabase
import sys

uri = "bolt://100.91.178.230:7687"
user = "neo4j"
password = "Czty100165188"

print(f"尝试连接到 {uri} ...")

try:
    # 关键修改：显式禁用加密并增加连接超时
    driver = GraphDatabase.driver(
        uri, 
        auth=(user, password),
        encrypted=False,             # 很多本地安装版本默认不开启 SSL
        connection_timeout=10.0      # 设置 10 秒超时
    )
    
    # 验证连接
    driver.verify_connectivity()
    print("✅ [SUCCESS] Neo4j 连接验证成功！")
    
    # 执行一个简单查询
    with driver.session() as session:
        res = session.run("RETURN 'Hello Neo4j' AS message").single()
        print(f"📊 数据库响应: {res['message']}")
    
    driver.close()
    
except Exception as e:
    print(f"❌ [FAILED] 连接失败。")
    print(f"错误类型: {type(e).__name__}")
    print(f"详细错误: {e}")
    sys.exit(1)