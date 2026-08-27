#!/usr/bin/env python3
"""Adaptive historical-case memory depth controller for SkinGPT-X v2.

The controller treats the already computed C0/C25/C50/C75/C100 snapshots as
counterfactual retrieval actions.  Policies are learned on rule_acceptance,
governed on gate_development, and evaluated once on heldout_evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .core import canonical_hash, label_key, write_json
from .pipeline import DATASETS, exact_mcnemar, read_jsonl


SCHEMA_VERSION = "self_evolving_memory_v2_case_depth_controller_v1"
CANDIDATE_DEPTHS = ("C0", "C25", "C50", "C75")


def _load_predictions(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    result: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            sid = str(row["sample_id"])
            condition = str(row["condition"])
            if condition in result[sid]:
                raise ValueError(f"duplicate prediction condition: {sid}::{condition}")
            result[sid][condition] = row
    return dict(result)


def _roles(root: Path) -> dict[str, str]:
    rows = read_jsonl([root / "base_q100_records.jsonl"])
    result = {str(row["sample_id"]): str(row["split_role"]) for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate base sample IDs")
    return result


def _group(c100: dict[str, str]) -> str:
    return f"{c100['dataset']}::{label_key(c100['prediction'])}"


def _one_sided_greater(gains: int, regressions: int) -> float:
    total = gains + regressions
    if total == 0:
        return 1.0
    return sum(math.comb(total, index) for index in range(gains, total + 1)) / (2**total)


def fit(args: argparse.Namespace) -> None:
    predictions = _load_predictions(args.predictions)
    roles = _roles(args.root)
    stats: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for sid, conditions in predictions.items():
        if roles.get(sid) != "rule_acceptance" or "C100" not in conditions:
            continue
        c100 = conditions["C100"]
        group = _group(c100)
        c_correct = int(c100["correct"])
        for depth in CANDIDATE_DEPTHS:
            if depth not in conditions:
                continue
            correct = int(conditions[depth]["correct"])
            values = stats[group][depth]
            values["samples"] += 1
            values["gains"] += int(c_correct == 0 and correct == 1)
            values["regressions"] += int(c_correct == 1 and correct == 0)
    policies: list[dict[str, Any]] = []
    for group, depth_stats in sorted(stats.items()):
        candidates = []
        for depth, values in depth_stats.items():
            gains = int(values["gains"])
            regressions = int(values["regressions"])
            discordances = gains + regressions
            one_sided = _one_sided_greater(gains, regressions)
            if (
                discordances >= args.minimum_discordances
                and gains > regressions
                and gains - regressions >= args.minimum_net_gain
                and one_sided <= args.maximum_one_sided_p
            ):
                candidates.append((gains - regressions, -regressions, gains, depth, values, one_sided))
        if not candidates:
            continue
        _, _, _, depth, values, one_sided = max(candidates)
        policies.append(
            {
                "policy_id": "cd_" + canonical_hash({"group": group, "depth": depth})[:16],
                "group": group,
                "selected_depth": depth,
                "status": "staging",
                "acceptance": {
                    "samples": int(values["samples"]),
                    "gains": int(values["gains"]),
                    "regressions": int(values["regressions"]),
                    "net_correct": int(values["gains"] - values["regressions"]),
                    "one_sided_exact_p": one_sided,
                },
            }
        )
    payload = {
        "release_id": "v2_case_depth_staging",
        "schema_version": SCHEMA_VERSION,
        "source_split": "rule_acceptance",
        "candidate_depths": list(CANDIDATE_DEPTHS),
        "policies": policies,
        "fit_policy": {
            "minimum_discordances": args.minimum_discordances,
            "minimum_net_gain": args.minimum_net_gain,
            "maximum_one_sided_p": args.maximum_one_sided_p,
        },
        "status": "staging" if policies else "rolled_back",
    }
    payload["content_sha256"] = canonical_hash(payload)
    write_json(args.output, payload)
    print(json.dumps({"policies": len(policies), "content_sha256": payload["content_sha256"]}, indent=2))


def govern(args: argparse.Namespace) -> None:
    staging = json.loads(args.staging_release.read_text(encoding="utf-8"))
    predictions = _load_predictions(args.predictions)
    roles = _roles(args.root)
    by_group = {str(row["group"]): dict(row) for row in staging["policies"]}
    stats: dict[str, Counter] = defaultdict(Counter)
    for sid, conditions in predictions.items():
        if roles.get(sid) != "gate_development" or "C100" not in conditions:
            continue
        c100 = conditions["C100"]
        group = _group(c100)
        policy = by_group.get(group)
        if policy is None or policy["selected_depth"] not in conditions:
            continue
        candidate = conditions[policy["selected_depth"]]
        c_correct = int(c100["correct"])
        correct = int(candidate["correct"])
        values = stats[group]
        values["samples"] += 1
        values["gains"] += int(c_correct == 0 and correct == 1)
        values["regressions"] += int(c_correct == 1 and correct == 0)
        values["changes"] += int(label_key(c100["prediction"]) != label_key(candidate["prediction"]))
    governed = []
    lifecycle: Counter[str] = Counter()
    for group, policy in sorted(by_group.items()):
        values = stats[group]
        gains = int(values["gains"])
        regressions = int(values["regressions"])
        if regressions > max(args.maximum_regressions, gains):
            status = "quarantine"
        elif gains >= regressions and regressions <= args.maximum_regressions:
            status = "active"
        else:
            status = "staging"
        lifecycle[status] += 1
        governed.append(
            {
                **policy,
                "status": status,
                "gate": {
                    "samples": int(values["samples"]),
                    "changes": int(values["changes"]),
                    "gains": gains,
                    "regressions": regressions,
                    "net_correct": gains - regressions,
                },
            }
        )
    payload = {
        "release_id": "v2_case_depth_governed",
        "parent_release_id": staging["release_id"],
        "schema_version": SCHEMA_VERSION,
        "source_splits": ["rule_acceptance", "gate_development"],
        "policies": governed,
        "lifecycle_counts": dict(lifecycle),
        "governance_policy": {"maximum_regressions": args.maximum_regressions, "require_nonnegative_gate_net": True},
        "status": "active" if lifecycle["active"] else "rolled_back",
    }
    payload["content_sha256"] = canonical_hash(payload)
    write_json(args.output, payload)
    print(json.dumps({"lifecycle_counts": dict(lifecycle), "content_sha256": payload["content_sha256"]}, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    release = json.loads(args.release.read_text(encoding="utf-8"))
    predictions = _load_predictions(args.predictions)
    roles = _roles(args.root)
    active = {str(row["group"]): row for row in release["policies"] if row["status"] == "active"}
    rows: list[dict[str, Any]] = []
    for sid, conditions in predictions.items():
        if roles.get(sid) != "heldout_evaluation" or "C100" not in conditions:
            continue
        c100 = conditions["C100"]
        policy = active.get(_group(c100))
        selected_depth = str(policy["selected_depth"]) if policy and policy["selected_depth"] in conditions else "C100"
        selected = conditions[selected_depth]
        rows.append(
            {
                "sample_id": sid,
                "dataset": c100["dataset"],
                "relative_path": c100["relative_path"],
                "ground_truth": c100["ground_truth"],
                "c100_prediction": c100["prediction"],
                "v2_prediction": selected["prediction"],
                "selected_depth": selected_depth,
                "policy_id": policy["policy_id"] if policy else "",
                "correct_C100": int(c100["correct"]),
                "correct_V2CaseDepth": int(selected["correct"]),
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = []
    for dataset in (*DATASETS, "Pooled"):
        subset = rows if dataset == "Pooled" else [row for row in rows if row["dataset"] == dataset]
        for condition in ("C100", "V2CaseDepth"):
            correct = sum(row[f"correct_{condition}"] for row in subset)
            metrics.append({"dataset": dataset, "condition": condition, "samples": len(subset), "correct": correct, "accuracy": correct / len(subset) if subset else 0.0})
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    c_only = sum(row["correct_C100"] == 1 and row["correct_V2CaseDepth"] == 0 for row in rows)
    v2_only = sum(row["correct_C100"] == 0 and row["correct_V2CaseDepth"] == 1 for row in rows)
    differences = np.asarray([row["correct_V2CaseDepth"] - row["correct_C100"] for row in rows], dtype=float)
    datasets = np.asarray([row["dataset"] for row in rows])
    rng = np.random.default_rng(args.seed)
    boot = []
    indices = {dataset: np.where(datasets == dataset)[0] for dataset in DATASETS}
    for _ in range(args.bootstraps):
        bootstrap_indices = np.concatenate([rng.choice(value, len(value), replace=True) for value in indices.values() if len(value)])
        boot.append(float(differences[bootstrap_indices].mean()))
    paired = {
        "comparison": "V2CaseDepth_vs_C100",
        "samples": len(rows),
        "difference": float(differences.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "c100_only_correct": c_only,
        "v2_only_correct": v2_only,
        "mcnemar_exact_p": exact_mcnemar(c_only, v2_only),
    }
    with (args.output / "paired_tests.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired))
        writer.writeheader()
        writer.writerow(paired)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_scope": "exploratory_reuse",
        "confirmatory": False,
        "samples": len(rows),
        "active_policies": len(active),
        "memory_depth_changed": sum(row["selected_depth"] != "C100" for row in rows),
        "metrics": metrics,
        "paired": paired,
        "status": "PASS",
    }
    write_json(args.output / "audit.json", audit)
    print(json.dumps(audit, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("--root", type=Path, required=True)
    fit_parser.add_argument("--predictions", type=Path, required=True)
    fit_parser.add_argument("--minimum-discordances", type=int, default=2)
    fit_parser.add_argument("--minimum-net-gain", type=int, default=1)
    fit_parser.add_argument("--maximum-one-sided-p", type=float, default=0.25)
    fit_parser.add_argument("--output", type=Path, required=True)
    fit_parser.set_defaults(func=fit)
    govern_parser = commands.add_parser("govern")
    govern_parser.add_argument("--root", type=Path, required=True)
    govern_parser.add_argument("--predictions", type=Path, required=True)
    govern_parser.add_argument("--staging-release", type=Path, required=True)
    govern_parser.add_argument("--maximum-regressions", type=int, default=1)
    govern_parser.add_argument("--output", type=Path, required=True)
    govern_parser.set_defaults(func=govern)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--root", type=Path, required=True)
    evaluate_parser.add_argument("--predictions", type=Path, required=True)
    evaluate_parser.add_argument("--release", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--bootstraps", type=int, default=10000)
    evaluate_parser.add_argument("--seed", type=int, default=20260827)
    evaluate_parser.set_defaults(func=evaluate)
    return result


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
