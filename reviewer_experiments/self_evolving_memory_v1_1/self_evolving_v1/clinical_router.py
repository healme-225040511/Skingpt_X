#!/usr/bin/env python3
"""SkinGPT-X v3 pilot: utility-reranked, contrastive episodic memory.

The pilot is deliberately additive.  It never mutates the v1/v2 snapshots and
fits only on ``rule_acceptance``.  ``gate_development`` selects an arm, and the
selected arm can then be evaluated once on ``heldout_evaluation``.

S0 is the frozen no-SkinHandbook C100 result.  New inference arms are:

* S1: the six most relevant precedents (similarity only);
* S2: a per-query memory router plus similarity/verified-utility reranking;
* S3: S2 plus label-balanced supportive and hard-negative precedents.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .core import canonical_hash, label_key, parse_primary_diagnosis, write_json
from .pipeline import evaluation_label, exact_mcnemar, read_jsonl


SCHEMA_VERSION = "skingpt_x_v3_clinical_router_pilot_v1"
PILOT_DATASETS = ("DermNet23", "Fit17k")
ARMS = ("S1", "S2", "S3")
CASE_HEADER_RE = re.compile(
    r"\[4\. Precedent Cases from the Same Frozen Training Checkpoint\]:.*?"
    r"(?=\n\n\[5\.|\n\n\[Instructions\]:)",
    flags=re.S,
)
HANDBOOK_RE = re.compile(
    r"\n\n\[5\. Frozen Static Handbook Knowledge\]:.*?(?=\n\n\[Instructions\]:)",
    flags=re.S,
)
PRECEDENT_RE = re.compile(
    r"^- Verified precedent \[(?P<label>[^\]]+)\] \| similarity=(?P<similarity>[0-9.]+)\s*\n"
    r"\s*Findings:\s*(?P<findings>.*?)"
    r"(?=\n- Verified precedent |\n\n\[5\.|\n\n\[Instructions\]:)",
    flags=re.M | re.S,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_predictions(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    output: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            sid, condition = str(row["sample_id"]), str(row["condition"])
            if condition in output[sid]:
                raise ValueError(f"duplicate prediction: {sid}::{condition}")
            output[sid][condition] = row
    return dict(output)


def parse_precedents(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse the frozen 12-case section and bind IDs by its audited order."""

    matches = list(PRECEDENT_RE.finditer(str(row["prompt"])))
    ids = [str(value) for value in row.get("retrieved_case_ids", [])]
    if len(matches) != len(ids):
        raise ValueError(
            f"precedent/ID mismatch for {row.get('sample_id')}: {len(matches)} != {len(ids)}"
        )
    return [
        {
            "case_id": case_id,
            "label": match.group("label").strip(),
            "similarity": float(match.group("similarity")),
            "findings": re.sub(r"\s+", " ", match.group("findings")).strip(),
        }
        for case_id, match in zip(ids, matches)
    ]


def _probabilities(row: dict[str, Any]) -> list[tuple[str, float]]:
    values = [(str(item[0]), float(item[1])) for item in row.get("candidate_probabilities", [])]
    if len(values) < 2:
        raise ValueError(f"missing candidate probabilities: {row.get('sample_id')}")
    return values


def _bin(value: float, cuts: tuple[float, ...]) -> str:
    return str(sum(value >= cut for cut in cuts))


def context_key(row: dict[str, Any], precedents: list[dict[str, Any]]) -> str:
    probs = _probabilities(row)
    confidence = probs[0][1]
    margin = probs[0][1] - probs[1][1]
    top = label_key(probs[0][0])
    agreement = sum(label_key(case["label"]) == top for case in precedents) / max(len(precedents), 1)
    return "::".join(
        (
            str(row["dataset"]),
            "c" + _bin(confidence, (0.20, 0.35)),
            "m" + _bin(margin, (0.05, 0.15)),
            "a" + _bin(agreement, (0.25, 0.50)),
        )
    )


def _utility(gains: int, regressions: int) -> tuple[float, float]:
    """Beta(2,2) posterior mean and a conservative one-SD lower score."""

    alpha, beta = gains + 2, regressions + 2
    mean = alpha / (alpha + beta)
    variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
    return mean, max(0.0, mean - math.sqrt(variance))


def fit_policy(args: argparse.Namespace) -> None:
    rows = [
        row
        for row in read_jsonl([args.root / "base_q100_records.jsonl"])
        if row["split_role"] == "rule_acceptance" and row["dataset"] in args.datasets
    ]
    predictions = _load_predictions(args.predictions)
    item_stats: dict[str, Counter[str]] = defaultdict(Counter)
    context_stats: dict[str, Counter[str]] = defaultdict(Counter)
    dataset_stats: dict[str, Counter[str]] = defaultdict(Counter)
    audited: Counter[str] = Counter()
    for row in rows:
        sid = str(row["sample_id"])
        pair = predictions.get(sid, {})
        if "C0" not in pair or "C100" not in pair:
            raise ValueError(f"missing frozen C0/C100 pair: {sid}")
        precedents = parse_precedents(row)
        delta = int(pair["C100"]["correct"]) - int(pair["C0"]["correct"])
        outcome = "gain" if delta > 0 else "regression" if delta < 0 else "neutral"
        context = context_key(row, precedents)
        context_stats[context]["samples"] += 1
        context_stats[context][outcome] += 1
        dataset_stats[str(row["dataset"])]["samples"] += 1
        dataset_stats[str(row["dataset"])][outcome] += 1
        for case in precedents:
            stats = item_stats[str(case["case_id"])]
            stats["exposures"] += 1
            stats[outcome] += 1
        audited[outcome] += 1

    items = []
    for case_id, stats in sorted(item_stats.items()):
        mean, lower = _utility(int(stats["gain"]), int(stats["regression"]))
        items.append(
            {
                "case_id": case_id,
                "exposures": int(stats["exposures"]),
                "gains": int(stats["gain"]),
                "regressions": int(stats["regression"]),
                "neutral": int(stats["neutral"]),
                "posterior_mean": mean,
                "posterior_lower_score": lower,
            }
        )

    contexts = []
    for context, stats in sorted(context_stats.items()):
        gains, regressions = int(stats["gain"]), int(stats["regression"])
        if int(stats["samples"]) < args.minimum_context_samples:
            status = "staging"
        elif gains >= regressions:
            status = "active"
        else:
            status = "quarantine"
        contexts.append(
            {
                "context": context,
                "status": status,
                "samples": int(stats["samples"]),
                "gains": gains,
                "regressions": regressions,
                "net_correct": gains - regressions,
            }
        )
    datasets = []
    for dataset, stats in sorted(dataset_stats.items()):
        datasets.append(
            {
                "dataset": dataset,
                "status": "active" if int(stats["gain"]) >= int(stats["regression"]) else "quarantine",
                "samples": int(stats["samples"]),
                "gains": int(stats["gain"]),
                "regressions": int(stats["regression"]),
                "net_correct": int(stats["gain"] - stats["regression"]),
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "release_id": "v3_pilot_staging",
        "fit_split": "rule_acceptance",
        "datasets": list(args.datasets),
        "source_predictions": str(args.predictions),
        "source_predictions_sha256": _sha256(args.predictions),
        "minimum_context_samples": args.minimum_context_samples,
        "outcomes": dict(audited),
        "items": items,
        "contexts": contexts,
        "dataset_fallbacks": datasets,
        "status": "staging",
    }
    payload["content_sha256"] = canonical_hash(payload)
    write_json(args.output, payload)
    print(json.dumps({"rows": len(rows), "items": len(items), "contexts": Counter(x["status"] for x in contexts), "outcomes": dict(audited)}, indent=2))


def _load_policy(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frozen = str(payload.get("content_sha256", ""))
    unhashed = dict(payload)
    unhashed.pop("content_sha256", None)
    if not frozen or canonical_hash(unhashed) != frozen:
        raise ValueError("policy content hash mismatch")
    return payload


def _memory_allowed(row: dict[str, Any], cases: list[dict[str, Any]], policy: dict[str, Any]) -> bool:
    contexts = {str(item["context"]): str(item["status"]) for item in policy["contexts"]}
    status = contexts.get(context_key(row, cases))
    if status is not None:
        return status == "active"
    datasets = {str(item["dataset"]): str(item["status"]) for item in policy["dataset_fallbacks"]}
    return datasets.get(str(row["dataset"])) == "active"


def _rank_cases(cases: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    utilities = {str(item["case_id"]): item for item in policy["items"]}
    output = []
    for case in cases:
        item = utilities.get(str(case["case_id"]), {})
        lower = float(item.get("posterior_lower_score", 0.25))
        score = 0.80 * float(case["similarity"]) + 0.20 * lower
        output.append({**case, "utility_score": lower, "retrieval_score": score})
    return sorted(output, key=lambda item: (-float(item["retrieval_score"]), str(item["case_id"])))


def select_cases(
    row: dict[str, Any], cases: list[dict[str, Any]], policy: dict[str, Any], arm: str, limit: int = 6
) -> tuple[list[dict[str, Any]], str]:
    if arm == "S1":
        ordered = sorted(cases, key=lambda item: (-float(item["similarity"]), str(item["case_id"])))
        return ordered[:limit], "relevance_only"
    if not _memory_allowed(row, cases, policy):
        return [], "router_no_memory"
    ordered = _rank_cases(cases, policy)
    if arm == "S2":
        return ordered[:limit], "utility_reranked"
    if arm != "S3":
        raise ValueError(f"unknown arm: {arm}")

    candidates = [label_key(label) for label, _ in _probabilities(row)[:3]]
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    # First guarantee contrast across the top three candidate labels when such
    # precedents exist, then fill by score with a per-label diversity cap.
    for candidate in candidates:
        match = next(
            (item for item in ordered if label_key(item["label"]) == candidate and item not in selected),
            None,
        )
        if match is not None:
            selected.append(match)
            counts[label_key(match["label"])] += 1
    for item in ordered:
        key = label_key(item["label"])
        if item in selected or counts[key] >= 3:
            continue
        selected.append(item)
        counts[key] += 1
        if len(selected) >= limit:
            break
    return selected[:limit], "contrastive_balanced"


def _render_cases(cases: list[dict[str, Any]], arm: str) -> str:
    if not cases:
        return "[4. Precedent Cases from the Same Frozen Training Checkpoint]:\nNo eligible precedent case was selected by the memory router."
    lines = ["[4. Precedent Cases from the Same Frozen Training Checkpoint]:"]
    for case in cases:
        suffix = ""
        if arm in {"S2", "S3"}:
            suffix = f" | verified-utility={float(case.get('utility_score', 0.0)):.3f}"
        lines.extend(
            (
                f"- Verified precedent [{case['label']}] | similarity={float(case['similarity']):.5f}{suffix}",
                f"  Findings: {case['findings']}",
            )
        )
    return "\n".join(lines)


def build_prompt(row: dict[str, Any], selected: list[dict[str, Any]], arm: str) -> str:
    prompt = HANDBOOK_RE.sub("", str(row["prompt"]), count=1)
    replacement = _render_cases(selected, arm)
    prompt, count = CASE_HEADER_RE.subn(replacement, prompt, count=1)
    if count != 1:
        raise ValueError(f"could not replace case section: {row.get('sample_id')}")
    if arm == "S3":
        instruction = (
            "\n- Compare supportive and competing precedents explicitly; do not diagnose by majority vote."
            "\n- Treat similarity and historical utility as retrieval evidence, not as diagnostic proof."
            "\n- If precedent evidence conflicts with the current lesion, prioritize current visual findings and frozen probabilities."
        )
        prompt = prompt.replace("\n[Instructions]:", "\n[Instructions]:" + instruction, 1)
    if "Frozen Static Handbook Knowledge" in prompt:
        raise ValueError("SkinHandbook section survived prompt construction")
    return prompt


def build_manifest(args: argparse.Namespace) -> None:
    policy = _load_policy(args.policy)
    base = [
        row
        for row in read_jsonl([args.root / "base_q100_records.jsonl"])
        if row["split_role"] == args.role and row["dataset"] in args.datasets
    ]
    if args.maximum_cases_per_dataset:
        limited = []
        for dataset in args.datasets:
            limited.extend([row for row in base if row["dataset"] == dataset][: args.maximum_cases_per_dataset])
        base = limited
    arms = tuple(args.arms)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / f"{args.role}_shard_{index}.jsonl" for index in range(args.shards)]
    handles = [path.open("w", encoding="utf-8") for path in paths]
    counts: Counter[str] = Counter()
    case_counts: Counter[str] = Counter()
    prompt_hashes = []
    total = 0
    try:
        for row in base:
            cases = parse_precedents(row)
            for arm in arms:
                selected, action = select_cases(row, cases, policy, arm, limit=args.case_limit)
                prompt = build_prompt(row, selected, arm)
                global_index = args.global_index_offset + total
                record = {
                    **{key: value for key, value in row.items() if key != "prompt"},
                    "global_index": global_index,
                    "source_control_global_index": row["global_index"],
                    "condition": arm,
                    "arm": "clinical_router_v3_pilot",
                    "memory_action": action,
                    "selected_case_ids": [item["case_id"] for item in selected],
                    "selected_case_labels": [item["label"] for item in selected],
                    "selected_case_count": len(selected),
                    "policy_sha256": policy["content_sha256"],
                    "handbook_mode": "removed",
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                }
                handles[total % args.shards].write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
                counts[f"{row['dataset']}::{arm}::{action}"] += 1
                case_counts[f"{arm}::{len(selected)}"] += 1
                prompt_hashes.append(record["prompt_sha256"])
    finally:
        for handle in handles:
            handle.close()
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "role": args.role,
        "datasets": list(args.datasets),
        "arms": list(arms),
        "base_cases": len(base),
        "records": total,
        "shards": {path.name: sum(1 for line in path.open() if line.strip()) for path in paths},
        "counts": dict(counts),
        "selected_case_counts": dict(case_counts),
        "skinhandbook_mentions": 0,
        "unique_prompt_hashes": len(set(prompt_hashes)),
        "policy_sha256": policy["content_sha256"],
    }
    if total != len(base) * len(arms) or len(set(prompt_hashes)) != total:
        audit["status"] = "FAIL"
        write_json(output / "manifest_audit.json", audit)
        raise ValueError("manifest completeness or uniqueness audit failed")
    write_json(output / "manifest_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def _result_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    seen: set[int] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") not in {"inference", "diagnosis"}:
                continue
            index = int(row["global_index"])
            if index in seen:
                raise ValueError(f"duplicate result global index: {index}")
            seen.add(index)
            rows.append(row)
    return rows


def _bootstrap(values: np.ndarray, datasets: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    groups = [np.where(datasets == dataset)[0] for dataset in PILOT_DATASETS if np.any(datasets == dataset)]
    estimates = []
    for _ in range(samples):
        selected = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        estimates.append(float(values[selected].mean()))
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def evaluate(args: argparse.Namespace) -> None:
    predictions = _load_predictions(args.predictions)
    results = _result_rows(args.results)
    rows = []
    for result in results:
        sid, arm = str(result["sample_id"]), str(result["condition"])
        if arm not in ARMS or str(result["dataset"]) not in args.datasets:
            continue
        allowed = [str(item[0]) for item in result["candidate_probabilities"]]
        prediction = parse_primary_diagnosis(str(result.get("response", "")), allowed)
        truth = str(result["ground_truth"])
        dataset = str(result["dataset"])
        baseline = predictions.get(sid, {}).get("C100")
        if baseline is None:
            raise ValueError(f"missing S0 baseline: {sid}")
        rows.append(
            {
                "sample_id": sid,
                "dataset": dataset,
                "split_role": result["split_role"],
                "arm": arm,
                "ground_truth": truth,
                "prediction": prediction,
                "correct": int(bool(prediction) and evaluation_label(dataset, prediction) == evaluation_label(dataset, truth)),
                "parse_valid": int(bool(prediction)),
                "s0_prediction": baseline["prediction"],
                "s0_correct": int(baseline["correct"]),
                "memory_action": result["memory_action"],
                "selected_case_count": int(result["selected_case_count"]),
            }
        )
    expected_arms = set(args.arms)
    by_sample: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_sample[row["sample_id"]].add(row["arm"])
    incomplete = {sid: sorted(expected_arms - arms) for sid, arms in by_sample.items() if arms != expected_arms}
    if incomplete:
        raise ValueError(f"incomplete arm pairing for {len(incomplete)} samples")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    metrics: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for arm_index, arm in enumerate(args.arms):
        arm_rows = [row for row in rows if row["arm"] == arm]
        for dataset in (*args.datasets, "Pooled"):
            subset = arm_rows if dataset == "Pooled" else [row for row in arm_rows if row["dataset"] == dataset]
            if not subset:
                continue
            metrics.extend(
                (
                    {"dataset": dataset, "arm": "S0", "samples": len(subset), "correct": sum(row["s0_correct"] for row in subset), "accuracy": np.mean([row["s0_correct"] for row in subset])},
                    {"dataset": dataset, "arm": arm, "samples": len(subset), "correct": sum(row["correct"] for row in subset), "accuracy": np.mean([row["correct"] for row in subset])},
                )
            )
        diffs = np.asarray([row["correct"] - row["s0_correct"] for row in arm_rows], dtype=float)
        datasets = np.asarray([row["dataset"] for row in arm_rows])
        low, high = _bootstrap(diffs, datasets, args.bootstraps, args.seed + arm_index)
        gains, regressions = int((diffs == 1).sum()), int((diffs == -1).sum())
        paired.append(
            {
                "comparison": f"{arm}_vs_S0",
                "samples": len(arm_rows),
                "difference": float(diffs.mean()),
                "ci_low": low,
                "ci_high": high,
                "gains": gains,
                "regressions": regressions,
                "net_correct": gains - regressions,
                "mcnemar_exact_p": exact_mcnemar(regressions, gains),
                "invalid_outputs": sum(1 - row["parse_valid"] for row in arm_rows),
                "memory_coverage": np.mean([row["selected_case_count"] > 0 for row in arm_rows]),
            }
        )
    for name, values in (("metrics.csv", metrics), ("paired_tests.csv", paired)):
        with (args.output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)

    eligible = []
    for arm in args.arms:
        arm_pair = next(row for row in paired if row["comparison"] == f"{arm}_vs_S0")
        dataset_nets = {}
        for dataset in args.datasets:
            subset = [row for row in rows if row["arm"] == arm and row["dataset"] == dataset]
            dataset_nets[dataset] = sum(row["correct"] - row["s0_correct"] for row in subset)
        if arm_pair["net_correct"] > 0 and all(value >= 0 for value in dataset_nets.values()):
            eligible.append((arm_pair["net_correct"], -arm_pair["regressions"], arm, dataset_nets))
    winner = max(eligible)[2] if eligible else "S0"
    governance = {
        "schema_version": SCHEMA_VERSION,
        "status": "active" if winner != "S0" else "rolled_back",
        "selected_arm": winner,
        "selection_split": args.role,
        "selection_rule": "positive pooled net correct and nonnegative net correct in each pilot dataset; maximize net correct then minimize regressions",
        "confirmatory": False,
        "scope": "exploratory_reuse_of_inspected_cohort",
        "paired": paired,
    }
    governance["content_sha256"] = canonical_hash(governance)
    write_json(args.output / "governance.json", governance)
    write_json(args.output / "audit.json", {"status": "PASS", "rows": len(rows), "samples": len(by_sample), "arms": list(args.arms), "invalid_outputs": sum(1 - row["parse_valid"] for row in rows)})
    print(json.dumps(governance, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit")
    fit.add_argument("--root", type=Path, required=True)
    fit.add_argument("--predictions", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--datasets", nargs="+", default=list(PILOT_DATASETS))
    fit.add_argument("--minimum-context-samples", type=int, default=12)
    fit.set_defaults(function=fit_policy)
    build = commands.add_parser("build-manifest")
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--policy", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--role", choices=("rule_acceptance", "gate_development", "heldout_evaluation"), required=True)
    build.add_argument("--datasets", nargs="+", default=list(PILOT_DATASETS))
    build.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    build.add_argument("--shards", type=int, default=2)
    build.add_argument("--case-limit", type=int, default=6)
    build.add_argument("--maximum-cases-per-dataset", type=int, default=0)
    build.add_argument("--global-index-offset", type=int, default=3_000_000)
    build.set_defaults(function=build_manifest)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--predictions", type=Path, required=True)
    evaluate_parser.add_argument("--results", nargs="+", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--role", required=True)
    evaluate_parser.add_argument("--datasets", nargs="+", default=list(PILOT_DATASETS))
    evaluate_parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    evaluate_parser.add_argument("--bootstraps", type=int, default=10000)
    evaluate_parser.add_argument("--seed", type=int, default=20260827)
    evaluate_parser.set_defaults(function=evaluate)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
