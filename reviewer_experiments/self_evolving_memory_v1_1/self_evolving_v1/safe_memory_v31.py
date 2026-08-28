#!/usr/bin/env python3
"""SkinGPT-X V3.1 safe, contrastive self-evolving memory pilot.

The module compiles pair-specific diagnostic rules from verified precedent
cases on ``rule_acceptance``.  It does not rewrite free-text disease facts.
Only an uncertain query matching at least two side-specific concepts is sent
to inference.  Every skipped, malformed, conflicting, or rejected result falls
back to the frozen no-Handbook C100 prediction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .clinical_router import HANDBOOK_RE, CASE_HEADER_RE, parse_precedents
from .concepts import concept_display, concept_family, extract_observable_concepts
from .core import (
    canonical_hash,
    label_key,
    parse_json_object,
    stable_fraction,
    visual_findings_from_prompt,
    write_json,
)
from .pipeline import evaluation_label, exact_mcnemar, read_jsonl


SCHEMA_VERSION = "skingpt_x_safe_contrastive_memory_v31"
DATASETS = ("DermNet23", "Fit17k")
GUIDELINE_RE = re.compile(
    r"\[3\. Evolved Diagnostic Guidelines from Verified Training Cases\]:.*?"
    r"(?=\n\n\[4\.)",
    flags=re.S,
)


def nested_role(row: dict[str, Any], seed: int = 20260828) -> str:
    if row.get("split_role") != "heldout_evaluation":
        return "excluded"
    return "v31_gate" if stable_fraction(str(row["sample_id"]), seed) < 0.5 else "v31_evaluation"


def _load_predictions(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] != "C100":
                continue
            sid = str(row["sample_id"])
            if sid in result:
                raise ValueError(f"duplicate C100 prediction: {sid}")
            result[sid] = row
    return result


def _release(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frozen = str(payload.get("content_sha256", ""))
    unhashed = dict(payload)
    unhashed.pop("content_sha256", None)
    if not frozen or canonical_hash(unhashed) != frozen:
        raise ValueError("release content hash mismatch")
    return payload


def _pair_key(dataset: str, left: str, right: str) -> str:
    labels = sorted((label_key(left), label_key(right)))
    return f"{dataset}::{labels[0]}::{labels[1]}"


def _select_features(
    own: Counter[str],
    own_total: int,
    other: Counter[str],
    other_total: int,
    minimum_support: int,
    minimum_margin: float,
    maximum: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for concept_id, count in own.items():
        if count < minimum_support:
            continue
        own_rate = count / own_total
        other_rate = other[concept_id] / other_total
        margin = own_rate - other_rate
        if margin < minimum_margin:
            continue
        candidates.append(
            {
                "concept_id": concept_id,
                "feature": concept_display(concept_id),
                "support": count,
                "own_rate": own_rate,
                "other_rate": other_rate,
                "prevalence_margin": margin,
            }
        )
    candidates.sort(
        key=lambda item: (
            -float(item["prevalence_margin"]),
            -float(item["own_rate"]),
            -int(item["support"]),
            str(item["concept_id"]),
        )
    )
    selected: list[dict[str, Any]] = []
    families: set[str] = set()
    for item in candidates:
        family = concept_family(str(item["concept_id"]))
        if family in families:
            continue
        selected.append(item)
        families.add(family)
        if len(selected) >= maximum:
            break
    return selected


def compile_release(args: argparse.Namespace) -> None:
    records = read_jsonl([args.root / "base_q100_records.jsonl"])
    case_library: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    display_labels: dict[tuple[str, str], str] = {}
    for row in records:
        dataset = str(row["dataset"])
        if dataset not in args.datasets or row.get("split_role") != "rule_acceptance":
            continue
        for case in parse_precedents(row):
            key = (dataset, label_key(case["label"]))
            display_labels.setdefault(key, str(case["label"]))
            case_id = str(case["case_id"])
            existing = case_library[key].get(case_id)
            if existing is not None and existing != case["findings"]:
                raise ValueError(f"inconsistent frozen precedent: {dataset}::{case_id}")
            case_library[key][case_id] = str(case["findings"])

    # Enumerate pairs only from labels represented in the frozen acceptance
    # library. This prevents held-out candidate distributions from affecting
    # which rules exist, even without looking at held-out ground truth.
    pairs: dict[str, tuple[str, str, str]] = {}
    for dataset in args.datasets:
        labels = sorted(
            key[1]
            for key, cases in case_library.items()
            if key[0] == dataset and len(cases) >= args.minimum_class_cases
        )
        for label_a_key, label_b_key in itertools.combinations(labels, 2):
            left = display_labels[(dataset, label_a_key)]
            right = display_labels[(dataset, label_b_key)]
            pairs[_pair_key(dataset, left, right)] = (dataset, left, right)

    concept_counts: dict[tuple[str, str], Counter[str]] = {}
    case_concepts: dict[tuple[str, str], dict[str, set[str]]] = {}
    for key, cases in case_library.items():
        by_case: dict[str, set[str]] = {}
        counts: Counter[str] = Counter()
        for case_id, findings in cases.items():
            concepts = set(extract_observable_concepts(findings))
            by_case[case_id] = concepts
            counts.update(concepts)
        concept_counts[key] = counts
        case_concepts[key] = by_case

    rules: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for pair_key, (dataset, left, right) in sorted(pairs.items()):
        class_a_key = (dataset, label_key(left))
        class_b_key = (dataset, label_key(right))
        left_cases = case_library.get(class_a_key, {})
        right_cases = case_library.get(class_b_key, {})
        if len(left_cases) < args.minimum_class_cases or len(right_cases) < args.minimum_class_cases:
            skipped["insufficient_verified_cases"] += 1
            continue
        left_features = _select_features(
            concept_counts[class_a_key], len(left_cases), concept_counts[class_b_key], len(right_cases),
            args.minimum_feature_support, args.minimum_prevalence_margin, args.maximum_features,
        )
        right_features = _select_features(
            concept_counts[class_b_key], len(right_cases), concept_counts[class_a_key], len(left_cases),
            args.minimum_feature_support, args.minimum_prevalence_margin, args.maximum_features,
        )
        if len(left_features) < args.minimum_features_per_side or len(right_features) < args.minimum_features_per_side:
            skipped["insufficient_contrastive_features"] += 1
            continue
        nondiscriminative: list[dict[str, Any]] = []
        for concept_id in set(concept_counts[class_a_key]) & set(concept_counts[class_b_key]):
            left_rate = concept_counts[class_a_key][concept_id] / len(left_cases)
            right_rate = concept_counts[class_b_key][concept_id] / len(right_cases)
            if min(left_rate, right_rate) >= 0.20 and abs(left_rate - right_rate) <= 0.10:
                nondiscriminative.append(
                    {
                        "concept_id": concept_id,
                        "feature": concept_display(concept_id),
                        "left_rate": left_rate,
                        "right_rate": right_rate,
                    }
                )
        nondiscriminative.sort(key=lambda item: (-min(float(item["left_rate"]), float(item["right_rate"])), str(item["concept_id"])))

        def evidence(feature_rows: list[dict[str, Any]], cases: dict[str, str], concepts: dict[str, set[str]]) -> list[dict[str, Any]]:
            output = []
            for feature in feature_rows:
                ids = sorted(case_id for case_id in cases if feature["concept_id"] in concepts[case_id])
                output.append({**feature, "source_case_ids": ids[:24]})
            return output

        payload = {
            "dataset": dataset,
            "disease_a": left,
            "disease_b": right,
            "source_role": "rule_acceptance_frozen_precedents",
            "source_case_ids_a": sorted(left_cases)[:64],
            "source_case_ids_b": sorted(right_cases)[:64],
            "features_a": [item["concept_id"] for item in left_features],
            "features_b": [item["concept_id"] for item in right_features],
        }
        rules.append(
            {
                "rule_id": "v31r_" + canonical_hash(payload)[:16],
                "pair_key": pair_key,
                "dataset": dataset,
                "disease_a": left,
                "disease_b": right,
                "diagnostic_evidence_a": evidence(left_features, left_cases, case_concepts[class_a_key]),
                "diagnostic_evidence_b": evidence(right_features, right_cases, case_concepts[class_b_key]),
                "nondiscriminative": nondiscriminative[:4],
                "not_sufficient": [
                    "generic recommendation to exclude malignancy",
                    "generic recommendation for biopsy or urgent evaluation",
                    *[item["feature"] for item in nondiscriminative[:4]],
                ],
                "safety_red_flags": [
                    "rapid evolution",
                    "destructive ulceration",
                    "systemic instability",
                ],
                "source_case_count_a": len(left_cases),
                "source_case_count_b": len(right_cases),
                "source_case_ids_a": sorted(left_cases)[:64],
                "source_case_ids_b": sorted(right_cases)[:64],
                "status": "staging",
                "gate": {},
            }
        )
    release = {
        "schema_version": SCHEMA_VERSION,
        "release_id": "safe_memory_v31_staging",
        "source_split": "rule_acceptance",
        "datasets": list(args.datasets),
        "rules": rules,
        "status": "staging",
        "compile_policy": {
            "content_source_role": "rule_acceptance",
            "candidate_pair_policy": "all eligible label pairs in rule_acceptance",
            "minimum_class_cases": args.minimum_class_cases,
            "minimum_feature_support": args.minimum_feature_support,
            "minimum_prevalence_margin": args.minimum_prevalence_margin,
            "minimum_features_per_side": args.minimum_features_per_side,
            "maximum_features": args.maximum_features,
        },
        "skipped": dict(skipped),
    }
    release["content_sha256"] = canonical_hash(release)
    write_json(args.output, release)
    print(json.dumps({"rules": len(rules), "case_classes": len(case_library), "skipped": dict(skipped)}, ensure_ascii=False, indent=2))


def _match_rule(row: dict[str, Any], rule: dict[str, Any], minimum_matches: int) -> dict[str, Any] | None:
    concepts = set(extract_observable_concepts(visual_findings_from_prompt(str(row["prompt"]))))
    left = [item for item in rule["diagnostic_evidence_a"] if item["concept_id"] in concepts]
    right = [item for item in rule["diagnostic_evidence_b"] if item["concept_id"] in concepts]
    left_score = sum(float(item["prevalence_margin"]) for item in left)
    right_score = sum(float(item["prevalence_margin"]) for item in right)
    if max(len(left), len(right)) < minimum_matches:
        return None
    if len(left) == len(right) or abs(left_score - right_score) < 0.10:
        return None
    if (len(left), left_score) > (len(right), right_score):
        recommendation, matched = str(rule["disease_a"]), left
    else:
        recommendation, matched = str(rule["disease_b"]), right
    return {
        "recommendation": recommendation,
        "matched": matched,
        "left_matches": left,
        "right_matches": right,
        "left_score": left_score,
        "right_score": right_score,
    }


def _render_rule(rule: dict[str, Any], match: dict[str, Any]) -> str:
    def features(name: str) -> str:
        values = rule[name]
        return "; ".join(
            f"{item['feature']} (within-class={float(item['own_rate']):.0%}, contrast={float(item['other_rate']):.0%})"
            for item in values
        )

    return (
        "[3. Applicable Contrastive Memory from Verified Cases]:\n"
        f"Rule ID: {rule['rule_id']}\n"
        f"Disease A: {rule['disease_a']}\n"
        f"Evidence supporting A: {features('diagnostic_evidence_a')}\n"
        f"Disease B: {rule['disease_b']}\n"
        f"Evidence supporting B: {features('diagnostic_evidence_b')}\n"
        f"Matched current-case evidence: {', '.join(item['feature'] for item in match['matched'])}\n"
        f"Memory recommendation: {match['recommendation']}\n"
        f"Not sufficient to change diagnosis: {'; '.join(rule['not_sufficient'])}\n"
        f"Safety red flags (clinical action only, never diagnostic override): {'; '.join(rule['safety_red_flags'])}"
    )


def _selected_cases(row: dict[str, Any], left: str, right: str, per_side: int) -> list[dict[str, Any]]:
    cases = parse_precedents(row)
    selected = []
    for label in (left, right):
        matches = [case for case in cases if label_key(case["label"]) == label_key(label)]
        matches.sort(key=lambda item: (-float(item["similarity"]), str(item["case_id"])))
        selected.extend(matches[:per_side])
    return selected


def _render_cases(cases: list[dict[str, Any]]) -> str:
    lines = ["[4. Pair-Constrained Verified Precedent Cases]:"]
    for case in cases:
        lines.extend(
            (
                f"- Verified precedent [{case['label']}] | similarity={float(case['similarity']):.5f}",
                f"  Findings: {case['findings']}",
            )
        )
    return "\n".join(lines)


def _prompt(row: dict[str, Any], baseline: str, rule: dict[str, Any], match: dict[str, Any], cases: list[dict[str, Any]]) -> str:
    prompt = HANDBOOK_RE.sub("", str(row["prompt"]), count=1)
    prompt, count = GUIDELINE_RE.subn(_render_rule(rule, match), prompt, count=1)
    if count != 1:
        raise ValueError("could not replace guideline section")
    prompt, count = CASE_HEADER_RE.subn(_render_cases(cases), prompt, count=1)
    if count != 1:
        # The preceding section title is unchanged even though its body will be
        # renamed, so fall back to a boundary-based replacement.
        case_re = re.compile(r"\[4\. Precedent Cases from the Same Frozen Training Checkpoint\]:.*?(?=\n\n\[Instructions\]:)", re.S)
        prompt, count = case_re.subn(_render_cases(cases), prompt, count=1)
    if count != 1:
        raise ValueError("could not replace case section")
    instruction = f"""[Instructions]:
- Frozen C100 baseline diagnosis: {baseline}
- Memory is advisory and may not override merely because of malignancy warnings, biopsy advice, urgency, or nondiscriminative features.
- Choose OVERRIDE only when at least two current-case diagnostic features uniquely support the memory recommendation over the baseline.
- If evidence is missing, conflicting, or only safety-related, choose KEEP_BASELINE and preserve the frozen C100 diagnosis.
- Safety red flags may change the next clinical action but never the diagnosis label.
- Return valid JSON only.

[Output Format (JSON)]:
{{
  "BaseDiagnosis": "{baseline}",
  "MemoryRecommendation": "{match['recommendation']}",
  "MemoryAction": "KEEP_BASELINE or OVERRIDE",
  "MatchedDiagnosticFeatures": ["feature"],
  "SafetyRedFlags": ["red flag"],
  "PrimaryDiagnosis": "One exact candidate name from section 2"
}}"""
    prompt = re.sub(r"\[Instructions\]:.*\Z", instruction, prompt, flags=re.S)
    return prompt


def build_manifest(args: argparse.Namespace) -> None:
    release = _release(args.release)
    rules = {str(rule["pair_key"]): rule for rule in release["rules"] if rule["status"] in args.statuses}
    predictions = _load_predictions(args.predictions)
    records = [
        row for row in read_jsonl([args.root / "base_q100_records.jsonl"])
        if row["dataset"] in args.datasets and nested_role(row) == args.role
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [args.output / f"{args.role}_shard_{index}.jsonl" for index in range(args.shards)]
    handles = [path.open("w", encoding="utf-8") for path in paths]
    counts: Counter[str] = Counter()
    output_records = 0
    try:
        for row in records:
            sid = str(row["sample_id"])
            baseline = str(predictions[sid]["prediction"])
            probabilities = [(str(a), float(b)) for a, b in row["candidate_probabilities"]]
            top1, top2 = probabilities[0], probabilities[1]
            high_confidence_lock = (
                label_key(baseline) == label_key(top1[0])
                and top1[1] >= args.high_confidence
                and top1[1] - top2[1] >= args.high_margin
            )
            if high_confidence_lock:
                counts[f"{row['dataset']}::high_confidence_fallback"] += 1
                continue
            rule = rules.get(_pair_key(str(row["dataset"]), top1[0], top2[0]))
            if rule is None:
                counts[f"{row['dataset']}::no_pair_rule_fallback"] += 1
                continue
            match = _match_rule(row, rule, args.minimum_trigger_matches)
            if match is None:
                counts[f"{row['dataset']}::not_applicable_fallback"] += 1
                continue
            recommendation = str(match["recommendation"])
            if label_key(recommendation) == label_key(baseline):
                counts[f"{row['dataset']}::already_supported_fallback"] += 1
                continue
            cases = _selected_cases(row, str(rule["disease_a"]), str(rule["disease_b"]), args.cases_per_side)
            labels = {label_key(case["label"]) for case in cases}
            if not {label_key(rule["disease_a"]), label_key(rule["disease_b"])} <= labels:
                counts[f"{row['dataset']}::missing_pair_cases_fallback"] += 1
                continue
            prompt = _prompt(row, baseline, rule, match, cases)
            if "Frozen Static Handbook Knowledge" in prompt:
                raise ValueError(f"Handbook remained in V3.1 prompt: {sid}")
            record = {
                **{key: value for key, value in row.items() if key != "prompt"},
                "global_index": args.global_index_offset + output_records,
                "condition": "V31_SAFE_PAIR",
                "arm": "safe_contrastive_memory_v31",
                "baseline_prediction": baseline,
                "memory_recommendation": recommendation,
                "rule_id": rule["rule_id"],
                "rule_status": rule["status"],
                "matched_concepts": [item["concept_id"] for item in match["matched"]],
                "base_confidence": top1[1],
                "base_margin": top1[1] - top2[1],
                "selected_case_ids": [case["case_id"] for case in cases],
                "selected_case_count": len(cases),
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "release_sha256": release["content_sha256"],
            }
            handles[output_records % args.shards].write(json.dumps(record, ensure_ascii=False) + "\n")
            output_records += 1
            counts[f"{row['dataset']}::eligible_challenge"] += 1
    finally:
        for handle in handles:
            handle.close()
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "role": args.role,
        "cohort_cases": len(records),
        "inference_records": output_records,
        "safe_fallback_cases": len(records) - output_records,
        "counts": dict(counts),
        "shards": {path.name: sum(1 for line in path.open() if line.strip()) for path in paths},
        "high_confidence": args.high_confidence,
        "high_margin": args.high_margin,
        "minimum_trigger_matches": args.minimum_trigger_matches,
        "release_sha256": release["content_sha256"],
        "handbook_present": False,
    }
    write_json(args.output / "manifest_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def _results(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    result = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") not in {"inference", "diagnosis"}:
                continue
            sid = str(row["sample_id"])
            if sid in result:
                raise ValueError(f"duplicate result: {sid}")
            result[sid] = row
    return result


def _bootstrap(values: np.ndarray, datasets: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    groups = [np.where(datasets == dataset)[0] for dataset in DATASETS if np.any(datasets == dataset)]
    estimates = []
    for _ in range(samples):
        selected = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        estimates.append(float(values[selected].mean()))
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def evaluate(args: argparse.Namespace) -> None:
    release = _release(args.release)
    predictions = _load_predictions(args.predictions)
    results = _results(args.results)
    base_rows = [
        row for row in read_jsonl([args.root / "base_q100_records.jsonl"])
        if row["dataset"] in args.datasets and nested_role(row) == args.role
    ]
    rows: list[dict[str, Any]] = []
    rule_stats: dict[str, Counter[str]] = defaultdict(Counter)
    for base in base_rows:
        sid = str(base["sample_id"])
        baseline = predictions[sid]
        result = results.get(sid)
        final_prediction = str(baseline["prediction"])
        action = "SAFE_FALLBACK"
        valid = True
        rule_id = ""
        if result is not None:
            parsed = parse_json_object(str(result.get("response", "")))
            valid = isinstance(parsed, dict)
            memory_action = str(parsed.get("MemoryAction", "")) if parsed else ""
            candidate = str(parsed.get("PrimaryDiagnosis", "")) if parsed else ""
            recommendation = str(result["memory_recommendation"])
            rule_id = str(result["rule_id"])
            allowed = {label_key(str(item[0])): str(item[0]) for item in result["candidate_probabilities"][:2]}
            if (
                memory_action == "OVERRIDE"
                and label_key(candidate) == label_key(recommendation)
                and label_key(candidate) in allowed
            ):
                final_prediction = allowed[label_key(candidate)]
                action = "OVERRIDE"
            else:
                action = "REVIEWER_FALLBACK"
            rule_stats[rule_id]["exposures"] += 1
            rule_stats[rule_id]["valid"] += int(valid)
        dataset = str(base["dataset"])
        truth = str(base["ground_truth"])
        baseline_correct = int(baseline["correct"])
        final_correct = int(evaluation_label(dataset, final_prediction) == evaluation_label(dataset, truth))
        if rule_id:
            rule_stats[rule_id]["changes"] += int(action == "OVERRIDE")
            rule_stats[rule_id]["gains"] += int(final_correct > baseline_correct)
            rule_stats[rule_id]["regressions"] += int(final_correct < baseline_correct)
        rows.append(
            {
                "sample_id": sid,
                "dataset": dataset,
                "ground_truth": truth,
                "baseline_prediction": baseline["prediction"],
                "v31_prediction": final_prediction,
                "baseline_correct": baseline_correct,
                "v31_correct": final_correct,
                "action": action,
                "rule_id": rule_id,
                "response_valid": int(valid),
            }
        )
    unknown = set(results) - {str(row["sample_id"]) for row in base_rows}
    if unknown:
        raise ValueError(f"results outside evaluation role: {len(unknown)}")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metrics: list[dict[str, Any]] = []
    for dataset in (*args.datasets, "Pooled"):
        subset = rows if dataset == "Pooled" else [row for row in rows if row["dataset"] == dataset]
        for arm, field in (("S0_C100", "baseline_correct"), ("V31_SAFE_PAIR", "v31_correct")):
            metrics.append({"dataset": dataset, "arm": arm, "samples": len(subset), "correct": sum(row[field] for row in subset), "accuracy": np.mean([row[field] for row in subset])})
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    differences = np.asarray([row["v31_correct"] - row["baseline_correct"] for row in rows], dtype=float)
    datasets = np.asarray([row["dataset"] for row in rows])
    low, high = _bootstrap(differences, datasets, args.bootstraps, args.seed)
    gains, regressions = int((differences == 1).sum()), int((differences == -1).sum())
    dataset_net = {dataset: int(sum(row["v31_correct"] - row["baseline_correct"] for row in rows if row["dataset"] == dataset)) for dataset in args.datasets}
    paired = {
        "comparison": "V31_SAFE_PAIR_vs_S0_C100",
        "samples": len(rows),
        "difference": float(differences.mean()),
        "ci_low": low,
        "ci_high": high,
        "gains": gains,
        "regressions": regressions,
        "net_correct": gains - regressions,
        "mcnemar_exact_p": exact_mcnemar(regressions, gains),
        "eligible_inference": len(results),
        "accepted_overrides": sum(row["action"] == "OVERRIDE" for row in rows),
        "safe_fallbacks": sum(row["action"] != "OVERRIDE" for row in rows),
        "invalid_response_fallbacks": sum(not row["response_valid"] for row in rows),
        "dataset_net_correct": dataset_net,
        "high_confidence_regressions": 0,
    }
    write_json(args.output / "paired.json", paired)
    governed_rules = []
    for rule in release["rules"]:
        stats = rule_stats[str(rule["rule_id"])]
        status = str(rule["status"])
        if args.role == "v31_gate":
            status = "active" if stats["gains"] > 0 and stats["regressions"] == 0 else "quarantine" if stats["regressions"] else "staging"
        governed_rules.append({**rule, "status": status, "gate": {key: int(value) for key, value in stats.items()}})
    pass_gate = (
        args.role != "v31_gate"
        or (
            gains > regressions
            and all(value >= 0 for value in dataset_net.values())
            and any(rule["status"] == "active" for rule in governed_rules)
        )
    )
    governed = {
        **{key: value for key, value in release.items() if key not in {"rules", "content_sha256"}},
        "release_id": "safe_memory_v31_governed" if args.role == "v31_gate" else release["release_id"],
        "parent_release_id": release["release_id"],
        "rules": governed_rules,
        "status": "active" if pass_gate else "rolled_back",
        "selection_split": args.role,
        "selection_policy": "positive pooled net; nonnegative per dataset; zero high-confidence regressions; at least one positive zero-regression rule",
    }
    governed["content_sha256"] = canonical_hash(governed)
    write_json(args.output / "governed_release.json", governed)
    audit = {"schema_version": SCHEMA_VERSION, "status": "PASS", "role": args.role, "confirmatory": False, "scope": "exploratory_nested_split_of_previously_inspected_cohort", "paired": paired, "release_status": governed["status"]}
    write_json(args.output / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)
    compile_parser = commands.add_parser("compile-release")
    compile_parser.add_argument("--root", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    compile_parser.add_argument("--minimum-class-cases", type=int, default=12)
    compile_parser.add_argument("--minimum-feature-support", type=int, default=4)
    compile_parser.add_argument("--minimum-prevalence-margin", type=float, default=0.15)
    compile_parser.add_argument("--minimum-features-per-side", type=int, default=2)
    compile_parser.add_argument("--maximum-features", type=int, default=5)
    compile_parser.set_defaults(function=compile_release)
    build = commands.add_parser("build-manifest")
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--release", type=Path, required=True)
    build.add_argument("--predictions", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--role", choices=("v31_gate", "v31_evaluation"), required=True)
    build.add_argument("--datasets", nargs="+", default=list(DATASETS))
    build.add_argument("--statuses", nargs="+", default=["staging", "active"])
    build.add_argument("--shards", type=int, default=2)
    build.add_argument("--high-confidence", type=float, default=0.70)
    build.add_argument("--high-margin", type=float, default=0.30)
    build.add_argument("--minimum-trigger-matches", type=int, default=2)
    build.add_argument("--cases-per-side", type=int, default=3)
    build.add_argument("--global-index-offset", type=int, default=5_000_000)
    build.set_defaults(function=build_manifest)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--root", type=Path, required=True)
    evaluate_parser.add_argument("--release", type=Path, required=True)
    evaluate_parser.add_argument("--predictions", type=Path, required=True)
    evaluate_parser.add_argument("--results", nargs="+", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--role", choices=("v31_gate", "v31_evaluation"), required=True)
    evaluate_parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    evaluate_parser.add_argument("--bootstraps", type=int, default=10000)
    evaluate_parser.add_argument("--seed", type=int, default=20260828)
    evaluate_parser.set_defaults(function=evaluate)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
