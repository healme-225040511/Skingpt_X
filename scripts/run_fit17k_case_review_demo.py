#!/usr/bin/env python3
"""Run a 3-image Fitzpatrick17k demo through case_review_rag_agent only."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATASET_ROOT = WORKSPACE_ROOT / "Dataset" / "fitzpatrick17k" / "dataset"
EVAL_DIR = WORKSPACE_ROOT / "project" / "Evaluation_Results" / "fitzpatrick17k" / "SkinGPT-X"
NEO4J_HOME = PROJECT_ROOT / "NEO4J_HOME"
NEO4J_DATABASE = "neo4j"
DEMO_ROOT = EVAL_DIR / "case_review_rag_demo"
DEMO_ASSETS_ROOT = PROJECT_ROOT / "demo" / "fitzpatrick17k_case_review"
LANCEDB_CANDIDATES = [
    PROJECT_ROOT / "lancedb",
    PROJECT_ROOT.parent / "Skingpt-X" / "lancedb",
    PROJECT_ROOT.parent / "Skingpt_X_save" / "lancedb",
]

DEMO_CASES = [
    "/test/erythema annulare centrifigum/910b6e79d1e4cb38c6d86294fa0c9786.jpg",
    "/test/scabies/1ac0e3c9b31905e8460042d59cdeb234.jpg",
    "/test/pyogenic granuloma/82d51006e6d32b5d5471b13010d425aa.jpg",
]


class DatabaseSessionDriver:
    """Small wrapper that forces all CaseReviewAgent sessions onto one DB."""

    def __init__(self, driver: Any, database: str):
        self._driver = driver
        self._database = database

    def session(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("database", self._database)
        return self._driver.session(*args, **kwargs)

    def close(self) -> None:
        self._driver.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--demo-root", type=Path, default=DEMO_ROOT)
    parser.add_argument("--neo4j-home", type=Path, default=NEO4J_HOME)
    parser.add_argument("--neo4j-database", default=NEO4J_DATABASE)
    parser.add_argument("--neo4j-uri", default="bolt://127.0.0.1:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", "Czty100165188"))
    parser.add_argument("--lancedb-uri", type=Path, default=None)
    parser.add_argument("--no-start-neo4j", action="store_true", help="Use an already-running Neo4j instance.")
    parser.add_argument("--force-vision-cache", type=Path, default=None)
    return parser.parse_args()


def resolve_lancedb(user_value: Path | None) -> Path:
    if user_value:
        return user_value
    for candidate in LANCEDB_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No LanceDB directory found. Checked: {LANCEDB_CANDIDATES}")


def run_command(cmd: list[str], env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd))
    return subprocess.run(cmd, text=True, capture_output=True, check=check, env=env)


def prepare_demo_dataset(dataset_root: Path, demo_root: Path) -> Path:
    runtime_images_root = demo_root / "images"
    curated_images_root = DEMO_ASSETS_ROOT / "images"
    runtime_images_root.mkdir(parents=True, exist_ok=True)
    curated_images_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    for rel_path in DEMO_CASES:
        src = dataset_root / rel_path.lstrip("/")
        if not src.exists():
            raise FileNotFoundError(f"Demo image not found: {src}")

        runtime_dst = runtime_images_root / rel_path.lstrip("/")
        runtime_dst.parent.mkdir(parents=True, exist_ok=True)
        if runtime_dst.exists() or runtime_dst.is_symlink():
            runtime_dst.unlink()
        os.symlink(src, runtime_dst)

        curated_dst = curated_images_root / rel_path.lstrip("/")
        curated_dst.parent.mkdir(parents=True, exist_ok=True)
        if curated_dst.exists() or curated_dst.is_symlink():
            curated_dst.unlink()
        os.symlink(src, curated_dst)

        manifest.append(
            {
                "relative_path": rel_path,
                "source": str(src),
                "runtime_image": str(runtime_dst),
                "demo_image": str(curated_dst),
                "label": Path(rel_path).parent.name,
            }
        )

    task_text = "\n".join(DEMO_CASES) + "\n"
    task_file = demo_root / "task_paths.txt"
    task_file.write_text(task_text, encoding="utf-8")
    (demo_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (DEMO_ASSETS_ROOT / "task_paths.txt").write_text(task_text, encoding="utf-8")
    (DEMO_ASSETS_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return task_file


def neo4j_status(neo4j_home: Path) -> bool:
    proc = run_command([str(neo4j_home / "bin" / "neo4j"), "status"], check=False)
    return proc.returncode == 0 and "running" in (proc.stdout + proc.stderr).lower()


def make_demo_neo4j_conf(neo4j_home: Path, database: str, demo_root: Path) -> Path:
    conf_src = neo4j_home / "conf"
    conf_dst = demo_root / "neo4j_conf"
    if conf_dst.exists():
        shutil.rmtree(conf_dst)
    shutil.copytree(conf_src, conf_dst)

    conf_file = conf_dst / "neo4j.conf"
    with conf_file.open("a", encoding="utf-8") as f:
        f.write("\n# Added by SkinGPT-X Fitzpatrick17k case-review demo.\n")
        f.write(f"initial.dbms.default_database={database}\n")
    return conf_dst


def start_neo4j_if_needed(args: argparse.Namespace) -> None:
    if args.no_start_neo4j:
        print("Skipping Neo4j start because --no-start-neo4j was set.")
        return

    db_dir = args.neo4j_home / "data" / "databases" / args.neo4j_database
    if not db_dir.exists():
        raise FileNotFoundError(f"Neo4j database directory not found: {db_dir}")

    if neo4j_status(args.neo4j_home):
        print("Neo4j is already running; reusing the existing server.")
        return

    conf_dir = make_demo_neo4j_conf(args.neo4j_home, args.neo4j_database, args.demo_root)
    env = os.environ.copy()
    env["NEO4J_HOME"] = str(args.neo4j_home)
    env["NEO4J_CONF"] = str(conf_dir)

    proc = run_command([str(args.neo4j_home / "bin" / "neo4j"), "start"], env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to start Neo4j:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
            with driver.session(database=args.neo4j_database) as session:
                session.run("RETURN 1").single()
            driver.close()
            print(f"Neo4j is ready: {args.neo4j_uri} database={args.neo4j_database}")
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"Neo4j did not become ready within 60s: {args.neo4j_uri}")


def load_vision_findings(eval_dir: Path, forced_path: Path | None) -> dict[str, str]:
    candidates = [forced_path] if forced_path else [
        eval_dir / "test_vision_findings_gpu.json",
        eval_dir / "test_vision_findings_gpu1.json",
        eval_dir / "RAG_output.json",
    ]
    merged: dict[str, str] = {}
    for path in candidates:
        if not path or not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(data, dict):
            continue
        for key, value in data.items():
            if isinstance(value, dict):
                value = value.get("key_findings") or value.get("KeyFindings") or json.dumps(value, ensure_ascii=False)
            merged[key] = str(value)
    return merged


def lookup_vision_findings(cache: dict[str, str], rel_path: str) -> str:
    keys = [rel_path, rel_path.lstrip("/"), rel_path.removeprefix("/test/")]
    for key in keys:
        if key in cache and cache[key].strip():
            return cache[key]
    raise KeyError(f"No cached vision/RAG findings for {rel_path}. Use --force-vision-cache if needed.")


def load_top5(csv_path: Path) -> dict[str, list[dict[str, float | str]]]:
    from Constants import Fitzpatrick17k_DISEASE_NAME

    top5_by_file: dict[str, list[dict[str, float | str]]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        prob_cols = sorted(
            [col for col in reader.fieldnames or [] if col.startswith("probability_class_")],
            key=lambda col: int(col.rsplit("_", 1)[1]),
        )
        for row in reader:
            probs = np.array([float(row[col]) for col in prob_cols])
            indices = np.argsort(probs)[-5:][::-1]
            top5_by_file[row["filename"]] = [
                {"disease": Fitzpatrick17k_DISEASE_NAME[int(idx)], "probability": float(probs[int(idx)])}
                for idx in indices
            ]
    return top5_by_file


def run_demo(args: argparse.Namespace) -> None:
    args.demo_root.mkdir(parents=True, exist_ok=True)
    task_file = prepare_demo_dataset(args.dataset_root, args.demo_root)
    print(f"Demo dataset prepared at {args.demo_root}")
    print(f"Task file: {task_file}")

    start_neo4j_if_needed(args)

    lancedb_uri = resolve_lancedb(args.lancedb_uri)
    vision_cache = load_vision_findings(args.eval_dir, args.force_vision_cache)
    top5_by_file = load_top5(args.eval_dir / "panderm_test_predictions.csv")

    from case_review_rag_agent import CaseReviewAgent

    agent = CaseReviewAgent(
        model="Qwen2-VL-8B",
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        lancedb_uri=str(lancedb_uri),
        markdown_path=str(PROJECT_ROOT / "skin_handbook.md"),
        train_feat_path=str(args.eval_dir / "train_feats.npy"),
        train_json_path=str(args.eval_dir / "train_files.json"),
        test_feat_path=str(args.eval_dir / "test_feats.npy"),
        test_json_path=str(args.eval_dir / "test_files.json"),
    )
    agent.driver = DatabaseSessionDriver(agent.driver, args.neo4j_database)

    results: dict[str, Any] = {}
    prompts: dict[str, str] = {}
    for rel_path in DEMO_CASES:
        full_image_path = args.dataset_root / rel_path.lstrip("/")
        if rel_path not in top5_by_file:
            raise KeyError(f"No Panderm prediction row for {rel_path}")

        print(f"\n=== Running case-review RAG demo for {rel_path} ===")
        review, prompt = agent.review_case(
            vision_key_findings=lookup_vision_findings(vision_cache, rel_path),
            panderm_top5=top5_by_file[rel_path],
            image_path=rel_path,
            full_image_path=str(full_image_path),
        )
        results[rel_path] = {
            "image_path": str(full_image_path),
            "primary_decision": top5_by_file[rel_path][0],
            "case_review": review,
        }
        prompts[rel_path] = prompt

    agent.close()
    (args.demo_root / "case_review_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.demo_root / "case_review_prompts.json").write_text(
        json.dumps(prompts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved results to {args.demo_root / 'case_review_results.json'}")


if __name__ == "__main__":
    run_demo(parse_args())
