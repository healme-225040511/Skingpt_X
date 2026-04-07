from neo4j import GraphDatabase
import json

uri = "bolt://100.88.61.60:7687"
user = "neo4j"
password = "Czty100165188"


def export_disease_prototype_versions(output_path="/225040511/project/Skingpt_X/disease_prototype_versions.json"):
    driver = GraphDatabase.driver(uri, auth=(user, password), encrypted=False)
    query = """
    MATCH (p:Prototype)
    OPTIONAL MATCH (p)-[:HAS_SNAPSHOT]->(pv:PrototypeVersion)
    WITH p, pv
    ORDER BY pv.version_idx
    WITH p, collect(
        CASE
            WHEN pv IS NULL THEN NULL
            ELSE pv{
                .version_idx,
                .summary,
                .used_case_count,
                .used_case_ids
            }
        END
    ) AS versions_raw
    RETURN
        p.disease AS disease,
        p{
            .summary,
            .version_count,
            .updated_at
        } AS prototype,
        [v IN versions_raw WHERE v IS NOT NULL] AS versions
    ORDER BY disease
    """

    try:
        with driver.session() as session:
            records = session.run(query)
            data = [
                {
                    "disease": record["disease"],
                    "prototype": record["prototype"],
                    "versions": record["versions"],
                }
                for record in records
            ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"已导出 {len(data)} 个 disease 到: {output_path}")
    except Exception as e:
        print(f"查询或导出失败: {e}")
    finally:
        driver.close()


if __name__ == "__main__":
    export_disease_prototype_versions()
