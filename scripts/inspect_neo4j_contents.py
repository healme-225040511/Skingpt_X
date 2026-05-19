#!/usr/bin/env python3
"""Inspect the current Neo4j database contents used by SkinGPT-X."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NEO4J_HOME = PROJECT_ROOT / "NEO4J_HOME"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neo4j-home", type=Path, default=DEFAULT_NEO4J_HOME)
    parser.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687"))
    parser.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", "Czty100165188"))
    parser.add_argument("--neo4j-database", default=os.environ.get("NEO4J_DATABASE", "neo4j"))
    parser.add_argument("--sample-limit", type=int, default=3, help="How many sample nodes to print per label.")
    parser.add_argument("--pretty-json", action="store_true", help="Print JSON with indentation.")
    return parser.parse_args()


def run_query(session: Any, query: str, **params: Any) -> list[dict[str, Any]]:
    return [record.data() for record in session.run(query, **params)]


def normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for record in records:
        item: dict[str, Any] = {}
        for key, value in record.items():
            if hasattr(value, "iso_format"):
                item[key] = value.iso_format()
            else:
                item[key] = value
        normalized.append(item)
    return normalized


def inspect_database(args: argparse.Namespace) -> dict[str, Any]:
    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    try:
        with driver.session(database=args.neo4j_database) as session:
            summary: dict[str, Any] = {
                "connection": {
                    "neo4j_home": str(args.neo4j_home),
                    "uri": args.neo4j_uri,
                    "database": args.neo4j_database,
                    "user": args.neo4j_user,
                },
                "databases": normalize_records(run_query(session, "SHOW DATABASES")),
                "labels": normalize_records(
                    run_query(
                        session,
                        """
                        MATCH (n)
                        UNWIND labels(n) AS label
                        RETURN label, count(*) AS count
                        ORDER BY count DESC, label ASC
                        """,
                    )
                ),
                "relationship_types": normalize_records(
                    run_query(
                        session,
                        """
                        MATCH ()-[r]->()
                        RETURN type(r) AS relationshipType, count(*) AS count
                        ORDER BY count DESC, relationshipType ASC
                        """,
                    )
                ),
                "indexes": normalize_records(run_query(session, "SHOW INDEXES")),
                "constraints": normalize_records(run_query(session, "SHOW CONSTRAINTS")),
            }

            label_samples: dict[str, list[dict[str, Any]]] = {}
            for entry in summary["labels"]:
                label = entry["label"]
                samples = run_query(
                    session,
                    """
                    MATCH (n)
                    WHERE $label IN labels(n)
                    RETURN properties(n) AS properties
                    LIMIT $sample_limit
                    """,
                    label=label,
                    sample_limit=args.sample_limit,
                )
                label_samples[label] = normalize_records(samples)
            summary["label_samples"] = label_samples
            return summary
    finally:
        driver.close()


def main() -> None:
    args = parse_args()
    summary = inspect_database(args)
    if args.pretty_json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
