#!/usr/bin/env python3
"""Utility-aware memory governance and retrieval for SkinGPT-X v2.

V2 deliberately reuses the immutable V1.1 structured rules and frozen C100
predictions.  It changes the memory lifecycle and retrieval controller, not the
underlying diagnosis model or evaluation labels:

* every applicable memory item is evaluated on ``rule_acceptance``;
* item utility is represented by an auditable beta posterior;
* items move through staging, active, and quarantine states;
* retrieval considers rules involving the C100 diagnosis and the frozen top-k
  candidates, ranks them by utility/evidence/prior support, and abstains on
  near-tied conflicting recommendations;
* a small, pre-registered controller family is selected on
  ``gate_development`` and evaluated once on ``heldout_evaluation``.

No held-out label is read by governance or controller tuning.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .core import (
    RULE_SECTION_RE,
    canonical_hash,
    label_key,
    parse_primary_diagnosis,
    probabilities_from_prompt,
    rule_applicability,
    rule_key,
    strict_rule_from_response,
    visual_findings_from_prompt,
    write_json,
)
from .concepts import concept_display, concept_family, extract_observable_concepts
from .pipeline import DATASETS, evaluation_label, exact_mcnemar, load_c100, load_release, read_jsonl


SCHEMA_VERSION = "self_evolving_memory_v2_utility_controller_v1"
ARM = "utility_aware_memory_v2"
LIFECYCLE_STATES = ("staging", "active", "quarantine")


def fail_closed_response(proposal: dict[str, Any]) -> str:
    """Return a schema-valid abstention without inventing clinical content."""

    payload = {
        "schema_version": "v1.1",
        "matching_protocol": proposal["matching_protocol"],
        "concept_ledger_sha256": proposal["concept_ledger_sha256"],
        "publishable": False,
        "disease_a": proposal["disease_a"],
        "disease_b": proposal["disease_b"],
        "feature_ids_a": [],
        "feature_ids_b": [],
        "nondiscriminative_concept_ids": [],
        "required_context": [],
        "limitations": [],
        "abstention_codes": ["other_conservative_abstention"],
        "source_case_ids_a": proposal["source_case_ids_a"],
        "source_case_ids_b": proposal["source_case_ids_b"],
    }
    response = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    _, error = strict_rule_from_response(proposal, response)
    if error != "evidence_abstained":
        raise ValueError(f"fail-closed payload did not validate: {error}")
    return response


def close_invalid_results(args: argparse.Namespace) -> None:
    """Convert only residual malformed generations to audited abstentions.

    This is a safety repair, not a rule repair: no diagnostic feature or rule is
    added.  The output is intentionally a small override JSONL which is read
    after the original attempts by the V1.1 finalizer.
    """

    proposals = read_jsonl(sorted(args.manifest_dir.glob("rules_shard_*.jsonl")))
    by_index = {int(row["global_index"]): row for row in proposals}
    attempts = read_jsonl(_expand(args.results))
    latest = {
        int(row["global_index"]): row
        for row in attempts
        if row.get("record_type") == "inference" and "global_index" in row
    }
    repaired: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for index, proposal in sorted(by_index.items()):
        result = latest.get(index)
        error: str | None
        if result is None:
            error = "missing_result"
        else:
            _, error = strict_rule_from_response(proposal, str(result.get("response", "")))
        if error in (None, "evidence_abstained"):
            continue
        reasons[str(error)] += 1
        row = dict(result or {})
        row.update(
            {
                "record_type": "inference",
                "global_index": index,
                "sample_id": proposal.get("sample_id", proposal["rule_id"]),
                "rule_id": proposal["rule_id"],
                "response": fail_closed_response(proposal),
                "repair_action": "fail_closed_abstention",
                "repair_reason": str(error),
            }
        )
        repaired.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in repaired:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"repaired": len(repaired), "reasons": reasons, "output": str(args.output)}, ensure_ascii=False, indent=2))


def _expand(specs: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for spec in specs:
        matches = [Path(value) for value in glob.glob(spec)]
        if not matches and Path(spec).is_file():
            matches = [Path(spec)]
        paths.extend(matches)
    unique = sorted(set(paths))
    if not unique:
        raise ValueError("no result files matched")
    return unique


def _strict_results(specs: Sequence[str]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(_expand(specs)):
        if row.get("record_type") != "inference":
            continue
        key = (str(row.get("sample_id", "")), str(row.get("rule_id", "")))
        if not all(key):
            raise ValueError("inference result is missing sample_id or rule_id")
        if key in result:
            raise ValueError(f"duplicate inference result: {key}")
        result[key] = row
    return result


def _release_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("rules"), list):
        raise ValueError("release must contain a rules list")
    frozen = str(payload.get("content_sha256", ""))
    unhashed = dict(payload)
    unhashed.pop("content_sha256", None)
    if not frozen or canonical_hash(unhashed) != frozen:
        raise ValueError("release content hash mismatch")
    return payload


def _beta_probability_above_half(alpha: int, beta: int) -> float:
    """Exact P(Beta(alpha, beta) > 0.5) for positive integer shapes."""

    if alpha <= 0 or beta <= 0:
        raise ValueError("beta posterior shapes must be positive")
    n = alpha + beta - 1
    return sum(math.comb(n, k) for k in range(alpha)) / (2**n)


def _confidence_bin(probabilities: list[tuple[str, float]]) -> str:
    if not probabilities:
        return "missing"
    probability = float(probabilities[0][1])
    margin = probability - float(probabilities[1][1]) if len(probabilities) > 1 else probability
    if probability >= 0.75 or margin >= 0.35:
        return "high"
    if probability >= 0.50 or margin >= 0.15:
        return "medium"
    return "low"


def _v2_rule_block(rule: dict[str, Any]) -> str:
    def joined(field: str) -> str:
        values = [str(value).strip() for value in rule.get(field, []) if str(value).strip()]
        return "; ".join(values) if values else "None validated."

    utility = rule.get("utility", {})
    return (
        f"[Governed Memory Item {rule['rule_id']} | lifecycle={rule.get('status', 'staging')} | "
        f"utility_mean={float(utility.get('posterior_mean', 0.5)):.3f} | "
        f"P_positive={float(utility.get('probability_positive', 0.5)):.3f}]\n"
        f"- Distinguish: {rule['disease_a']} versus {rule['disease_b']}\n"
        f"- Supports {rule['disease_a']}: {joined('supports_a')}\n"
        f"- Supports {rule['disease_b']}: {joined('supports_b')}\n"
        f"- Non-discriminative: {joined('nondiscriminative')}\n"
        f"- Query-matched recommendation: {rule.get('_query_recommendation', '')}\n"
        f"- Query trigger support: {rule.get('_query_trigger_matches', {})}\n"
        f"- Limitations: {joined('limitations')}\n"
        "- This item may only distinguish its named disease pair. It must not support a third diagnosis."
    )


def _inject_v2(prompt: str, rule: dict[str, Any], baseline_prediction: str) -> str:
    recommendation = str(rule.get("_query_recommendation", ""))
    replacement = (
        "\\g<head>"
        + _v2_rule_block(rule)
        + "\n\n[Memory V2 Governance Constraints]:\n"
        + f"- Frozen C100 diagnosis: {baseline_prediction}.\n"
        + f"- The controller recommendation is {recommendation}.\n"
        + "- Change the C100 diagnosis only if the visible findings support the recommendation.\n"
        + "- Do not infer absent clinical history, pathology, or an unlisted diagnosis.\n"
        + "- When evidence is insufficient or conflicting, retain the frozen C100 diagnosis."
        + "\\g<tail>"
    )
    updated, count = RULE_SECTION_RE.subn(replacement, prompt, count=1)
    if count != 1:
        raise ValueError("unable to replace evolved-guideline section")
    return updated


def _candidate_rules(
    row: dict[str, Any],
    rules: dict[str, dict[str, Any]],
    c_prediction: str,
    *,
    top_k: int,
    minimum_trigger_matches: int,
) -> tuple[list[dict[str, Any]], Counter]:
    probabilities = [(str(item[0]), float(item[1])) for item in row["candidate_probabilities"]]
    labels = [label for label, _ in probabilities[:top_k]]
    reasons: Counter = Counter()
    if not c_prediction:
        reasons["missing_c100_anchor"] += 1
        return [], reasons
    if all(label_key(label) != label_key(c_prediction) for label in labels):
        labels.append(c_prediction)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for comparator in labels:
        if label_key(comparator) == label_key(c_prediction):
            continue
        key = rule_key(str(row["dataset"]), c_prediction, comparator)
        if key in seen:
            continue
        seen.add(key)
        rule = rules.get(key)
        if rule is None:
            reasons["no_rule_for_pair"] += 1
            continue
        applicable, direction, trigger_matches = rule_applicability(
            str(row["prompt"]), rule, minimum_trigger_matches=minimum_trigger_matches
        )
        if not applicable:
            reasons[direction] += 1
            continue
        side = direction.rsplit("_", 1)[-1]
        recommendation = str(rule["disease_a"] if side == "a" else rule["disease_b"])
        if label_key(recommendation) == label_key(c_prediction):
            reasons["trigger_supports_c100"] += 1
            continue
        probability_by_label = {label_key(label): probability for label, probability in probabilities}
        candidates.append(
            {
                "rule": rule,
                "recommendation": recommendation,
                "trigger_matches": trigger_matches,
                "candidate_probability": probability_by_label.get(label_key(recommendation), 0.0),
                "c100_probability": probability_by_label.get(label_key(c_prediction), 0.0),
                "applicability_reason": direction,
            }
        )
    return candidates, reasons


def _retrieval_score(candidate: dict[str, Any], top_k: int) -> float:
    rule = candidate["rule"]
    utility = rule.get("utility", {})
    posterior = float(utility.get("posterior_mean", 0.5))
    probability_positive = float(utility.get("probability_positive", 0.5))
    net = float(utility.get("net_correct", 0.0))
    exposures = float(utility.get("exposures", 0.0))
    trigger = candidate["trigger_matches"]
    strength = min(1.0, max(float(trigger.get("a", 0)), float(trigger.get("b", 0))) / 3.0)
    prior = float(candidate["candidate_probability"])
    regression_rate = float(utility.get("regressions", 0.0)) / max(1.0, exposures)
    evidence_term = math.tanh(max(0.0, net) / 3.0)
    source_quality = float(rule.get("source_quality", {}).get("mean_prevalence_margin", 0.0))
    return (
        0.30 * posterior
        + 0.15 * probability_positive
        + 0.15 * strength
        + 0.15 * prior
        + 0.15 * source_quality
        + 0.10 * evidence_term
        - 0.30 * regression_rate
    )


def _select_ledger_concepts(entries: list[dict[str, Any]], maximum: int = 4) -> list[dict[str, Any]]:
    """Choose high-lift concepts without selecting duplicate morphology families."""

    ordered = sorted(
        entries,
        key=lambda row: (
            -float(row.get("prevalence_margin", 0.0)),
            -float(row.get("own_rate", 0.0)),
            -int(row.get("own_count", 0)),
            str(row.get("concept_id", "")),
        ),
    )
    selected: list[dict[str, Any]] = []
    families: set[str] = set()
    for row in ordered:
        concept_id = str(row.get("concept_id", ""))
        if not concept_id:
            continue
        family = concept_family(concept_id)
        if family in families:
            continue
        selected.append(row)
        families.add(family)
        if len(selected) >= maximum:
            break
    return selected


def compile_ledger_release(args: argparse.Namespace) -> None:
    """Compile grounded fallback rules for eligible ledgers that the LLM abstained on.

    The operation introduces no new clinical text.  It selects canonical IDs
    from the already audited ledger by a fixed prevalence-lift ordering and
    copies the compiler-owned evidence spans verbatim.  The resulting items
    remain candidates until independently governed.
    """

    parent = _release_payload(args.parent_release)
    existing = {str(row["rule_key"]): dict(row) for row in parent["rules"]}
    proposals = read_jsonl(sorted(args.manifest_dir.glob("rules_shard_*.jsonl")))
    compiled: list[dict[str, Any]] = []
    skipped: Counter = Counter()
    for proposal in sorted(proposals, key=lambda row: int(row["global_index"])):
        key = str(proposal["rule_key"])
        if key in existing:
            ledger = proposal.get("concept_ledger", {})
            ledger_entries = {
                str(row["concept_id"]): row
                for side in ("candidate_a", "candidate_b")
                for row in ledger.get(side, [])
            }
            selected_ids = list(existing[key].get("trigger_concepts_a", [])) + list(existing[key].get("trigger_concepts_b", []))
            margins = [float(ledger_entries[value]["prevalence_margin"]) for value in selected_ids if value in ledger_entries]
            compiled.append({
                **existing[key],
                "write_operator": "llm_selected_canonical_ledger",
                "source_quality": {
                    "mean_prevalence_margin": sum(margins) / len(margins) if margins else 0.0,
                    "selected_concepts": len(margins),
                },
            })
            continue
        ledger = proposal.get("concept_ledger", {})
        if str(ledger.get("content_sha256", "")) != str(proposal.get("concept_ledger_sha256", "")):
            raise ValueError(f"concept ledger hash mismatch: {proposal.get('rule_id')}")
        selected_a = _select_ledger_concepts(list(ledger.get("candidate_a", [])))
        selected_b = _select_ledger_concepts(list(ledger.get("candidate_b", [])))
        if len(selected_a) < args.minimum_concepts_per_side or len(selected_b) < args.minimum_concepts_per_side:
            skipped["insufficient_distinct_concept_families"] += 1
            continue
        nondiscriminative = _select_ledger_concepts(list(ledger.get("nondiscriminative", [])), maximum=4)

        def evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [
                {
                    "concept_id": str(row["concept_id"]),
                    "feature": str(row["display"]),
                    "cases": list(row.get("evidence", [])),
                }
                for row in rows
            ]

        selection_payload = {
            "proposal_rule_id": proposal["rule_id"],
            "concept_ledger_sha256": proposal["concept_ledger_sha256"],
            "feature_ids_a": [row["concept_id"] for row in selected_a],
            "feature_ids_b": [row["concept_id"] for row in selected_b],
            "nondiscriminative": [row["concept_id"] for row in nondiscriminative],
            "operator": "deterministic_prevalence_lift_distinct_family_v1",
        }
        compiled.append(
            {
                "rule_id": "cr2d_" + canonical_hash(selection_payload)[:16],
                "dataset": proposal["dataset"],
                "disease_a": proposal["disease_a"],
                "disease_b": proposal["disease_b"],
                "supports_a": [str(row["display"]) for row in selected_a],
                "supports_b": [str(row["display"]) for row in selected_b],
                "nondiscriminative": [str(row["display"]) for row in nondiscriminative],
                "required_context": [],
                "limitations": [
                    "Deterministically compiled from prevalence-normalized observable concepts; pending independent outcome governance.",
                    "Applicable only when query-side canonical triggers support exactly one disease side.",
                ],
                "source_case_ids_a": list(proposal["source_case_ids_a"]),
                "source_case_ids_b": list(proposal["source_case_ids_b"]),
                "source_case_count": len(proposal["source_case_ids_a"]) + len(proposal["source_case_ids_b"]),
                "trigger_terms_a": [str(row["display"]) for row in selected_a],
                "trigger_terms_b": [str(row["display"]) for row in selected_b],
                "trigger_concepts_a": [str(row["concept_id"]) for row in selected_a],
                "trigger_concepts_b": [str(row["concept_id"]) for row in selected_b],
                "required_context_terms": [],
                "abstain_terms": [],
                "evidence_a": evidence(selected_a),
                "evidence_b": evidence(selected_b),
                "source_selection_a": list(proposal.get("source_selection_a", [])),
                "source_selection_b": list(proposal.get("source_selection_b", [])),
                "matching_protocol": proposal["matching_protocol"],
                "matcher_algorithm_revision": proposal["matcher_algorithm_revision"],
                "matcher_sha256": proposal["matcher_sha256"],
                "concept_ledger_sha256": proposal["concept_ledger_sha256"],
                "schema_version": "v1.1",
                "status": "candidate",
                "validation_exposures": 0,
                "validation_changes": 0,
                "validation_gains": 0,
                "validation_regressions": 0,
                "validation_effect": 0.0,
                "reliability": 0.5,
                "rule_key": key,
                "write_operator": "deterministic_prevalence_lift_distinct_family_v1",
                "selection_payload_sha256": canonical_hash(selection_payload),
                "minimum_compiled_concepts_per_side": args.minimum_concepts_per_side,
                "source_quality": {
                    "mean_prevalence_margin": sum(float(row["prevalence_margin"]) for row in selected_a + selected_b) / len(selected_a + selected_b),
                    "selected_concepts": len(selected_a) + len(selected_b),
                },
            }
        )
    if len({row["rule_key"] for row in compiled}) != len(compiled):
        raise ValueError("compiled release contains duplicate rule keys")
    payload = {
        "release_id": "v2_ledger_candidate",
        "parent_release_id": parent["release_id"],
        "source_stream_hashes": parent["source_stream_hashes"],
        "rules": compiled,
        "status": "candidate",
        "schema_version": SCHEMA_VERSION,
        "protocol": "LLM-selected audited rules plus deterministic canonical-ledger fallback; all items require independent utility governance",
        "minimum_compiled_concepts_per_side": args.minimum_concepts_per_side,
        "write_operator_counts": dict(Counter(row["write_operator"] for row in compiled)),
        "skipped": dict(skipped),
    }
    payload["content_sha256"] = canonical_hash(payload)
    write_json(args.output, payload)
    print(json.dumps({"rules": len(compiled), "write_operator_counts": payload["write_operator_counts"], "skipped": dict(skipped), "content_sha256": payload["content_sha256"]}, ensure_ascii=False, indent=2))


def compile_failure_memory(args: argparse.Namespace) -> None:
    """Distill directional correction skills from verified C100 failures.

    Only the declared development split is read.  For a directional confusion
    ``C100=A, truth=B``, observable concepts are contrasted against verified
    ``C100=A, truth=A`` controls.  The skill can recommend B only when an
    enriched failure concept is visible and B is in the frozen retrieval set.
    """

    base_rows = [
        row
        for row in read_jsonl([args.root / "base_q100_records.jsonl"])
        if row["split_role"] == args.source_split
    ]
    c100 = load_c100(args.predictions)
    records: list[dict[str, Any]] = []
    missing_c100 = 0
    for row in base_rows:
        sid = str(row["sample_id"])
        if sid not in c100:
            missing_c100 += 1
            continue
        prediction = str(c100[sid]["prediction"])
        truth = str(row["ground_truth"])
        probabilities = [(str(item[0]), float(item[1])) for item in row["candidate_probabilities"]]
        target_in_topk = any(
            evaluation_label(str(row["dataset"]), label) == evaluation_label(str(row["dataset"]), truth)
            for label, _ in probabilities[: args.top_k]
        )
        findings = visual_findings_from_prompt(str(row["prompt"]))
        records.append(
            {
                "row": row,
                "prediction": prediction,
                "truth": truth,
                "prediction_key": evaluation_label(str(row["dataset"]), prediction),
                "truth_key": evaluation_label(str(row["dataset"]), truth),
                "concepts": extract_observable_concepts(findings),
                "findings": findings,
                "target_in_topk": target_in_topk,
            }
        )
    by_prediction: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    errors: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        dataset = str(item["row"]["dataset"])
        by_prediction[(dataset, item["prediction_key"])].append(item)
        if item["prediction_key"] != item["truth_key"] and item["target_in_topk"]:
            errors[(dataset, item["prediction_key"], item["truth_key"])].append(item)

    directional: list[dict[str, Any]] = []
    skipped: Counter = Counter()
    for (dataset, prediction_key, truth_key), error_rows in sorted(errors.items()):
        if len(error_rows) < args.minimum_error_cases:
            skipped["insufficient_verified_errors"] += 1
            continue
        controls = [
            item for item in by_prediction[(dataset, prediction_key)]
            if item["truth_key"] == prediction_key
        ]
        if len(controls) < args.minimum_control_cases:
            skipped["insufficient_correct_controls"] += 1
            continue
        error_counts = Counter(concept_id for item in error_rows for concept_id in item["concepts"])
        control_counts = Counter(concept_id for item in controls for concept_id in item["concepts"])

        def enriched(
            own_counts: Counter,
            own_total: int,
            other_counts: Counter,
            other_total: int,
            minimum_support: int,
        ) -> list[dict[str, Any]]:
            values = []
            for concept_id, count in own_counts.items():
                if count < minimum_support:
                    continue
                own_rate = count / own_total
                other_rate = other_counts[concept_id] / other_total
                margin = own_rate - other_rate
                if margin < args.minimum_prevalence_margin:
                    continue
                values.append(
                    {
                        "concept_id": concept_id,
                        "display": concept_display(concept_id),
                        "own_count": count,
                        "own_rate": own_rate,
                        "opposite_count": other_counts[concept_id],
                        "opposite_rate": other_rate,
                        "prevalence_margin": margin,
                    }
                )
            return _select_ledger_concepts(values, maximum=args.maximum_concepts_per_side)

        corrector_features = enriched(
            error_counts, len(error_rows), control_counts, len(controls), args.minimum_feature_support
        )
        baseline_features = enriched(
            control_counts, len(controls), error_counts, len(error_rows), args.minimum_feature_support
        )
        if not corrector_features:
            skipped["no_enriched_failure_concept"] += 1
            continue
        prediction_label = str(error_rows[0]["prediction"])
        truth_label = str(error_rows[0]["truth"])
        key = rule_key(dataset, prediction_label, truth_label)

        def evidence(features: list[dict[str, Any]], sources: list[dict[str, Any]], side: str) -> list[dict[str, Any]]:
            output = []
            for feature in features:
                cases = []
                for item in sources:
                    match = item["concepts"].get(feature["concept_id"])
                    if not match:
                        continue
                    cases.append(
                        {
                            "case_id": item["row"]["sample_id"],
                            "side": side,
                            "role": args.source_split,
                            "quote": str(match["quote"]),
                            "quote_start": int(match["quote_start"]),
                            "quote_end": int(match["quote_end"]),
                            "quote_sha256": str(match["quote_sha256"]),
                        }
                    )
                output.append({"concept_id": feature["concept_id"], "feature": feature["display"], "cases": cases})
            return output

        quality = sum(float(row["prevalence_margin"]) for row in corrector_features) / len(corrector_features)
        rule_payload = {
            "dataset": dataset,
            "baseline_diagnosis": prediction_label,
            "corrected_diagnosis": truth_label,
            "source_split": args.source_split,
            "error_case_ids": sorted(item["row"]["sample_id"] for item in error_rows),
            "control_case_ids": sorted(item["row"]["sample_id"] for item in controls),
            "corrector_concepts": [row["concept_id"] for row in corrector_features],
            "baseline_concepts": [row["concept_id"] for row in baseline_features],
        }
        directional.append(
            {
                "rule_id": "cr2f_" + canonical_hash(rule_payload)[:16],
                "dataset": dataset,
                "disease_a": prediction_label,
                "disease_b": truth_label,
                "supports_a": [row["display"] for row in baseline_features],
                "supports_b": [row["display"] for row in corrector_features],
                "nondiscriminative": [],
                "required_context": [],
                "limitations": [
                    "Directional correction skill distilled from verified C100 failures on a development split.",
                    "Applicable only when the frozen C100 diagnosis is disease A and disease B is retrieved among frozen candidates.",
                ],
                "source_case_ids_a": sorted(item["row"]["sample_id"] for item in controls),
                "source_case_ids_b": sorted(item["row"]["sample_id"] for item in error_rows),
                "source_case_count": len(controls) + len(error_rows),
                "trigger_terms_a": [row["display"] for row in baseline_features],
                "trigger_terms_b": [row["display"] for row in corrector_features],
                "trigger_concepts_a": [row["concept_id"] for row in baseline_features],
                "trigger_concepts_b": [row["concept_id"] for row in corrector_features],
                "required_context_terms": [],
                "abstain_terms": [],
                "evidence_a": evidence(baseline_features, controls, "a"),
                "evidence_b": evidence(corrector_features, error_rows, "b"),
                "source_selection_a": [{"case_id": item["row"]["sample_id"], "selection_role": "verified_c100_correct_control"} for item in controls],
                "source_selection_b": [{"case_id": item["row"]["sample_id"], "selection_role": "verified_c100_failure"} for item in error_rows],
                "matching_protocol": "canonical_observable_concepts_v1",
                "matcher_algorithm_revision": str(args.matcher_algorithm_revision),
                "matcher_sha256": str(args.matcher_sha256),
                "concept_ledger_sha256": canonical_hash(rule_payload),
                "schema_version": "v1.1",
                "status": "candidate",
                "validation_exposures": 0,
                "validation_changes": 0,
                "validation_gains": 0,
                "validation_regressions": 0,
                "validation_effect": 0.0,
                "reliability": 0.5,
                "rule_key": key,
                "write_operator": "verified_failure_contrastive_skill_v1",
                "source_quality": {
                    "mean_prevalence_margin": quality,
                    "selected_concepts": len(corrector_features) + len(baseline_features),
                    "verified_errors": len(error_rows),
                    "verified_controls": len(controls),
                },
                "directional_source": {"from": prediction_label, "to": truth_label},
                "selection_payload_sha256": canonical_hash(rule_payload),
            }
        )

    # The release format has one item per unordered disease pair.  When both
    # directions are supported, retain the direction with more verified errors,
    # then the larger morphology margin, using rule_id only as a stable tie-break.
    best: dict[str, dict[str, Any]] = {}
    for rule in directional:
        key = str(rule["rule_key"])
        current = best.get(key)
        rank = (
            int(rule["source_quality"]["verified_errors"]),
            float(rule["source_quality"]["mean_prevalence_margin"]),
            str(rule["rule_id"]),
        )
        current_rank = (
            int(current["source_quality"]["verified_errors"]),
            float(current["source_quality"]["mean_prevalence_margin"]),
            str(current["rule_id"]),
        ) if current else (-1, -1.0, "")
        if rank > current_rank:
            best[key] = rule
    rules = sorted(best.values(), key=lambda row: (row["dataset"], row["rule_key"]))
    payload = {
        "release_id": "v2_failure_memory_candidate",
        "parent_release_id": "v0_c100",
        "source_stream_hashes": {},
        "rules": rules,
        "status": "candidate",
        "schema_version": SCHEMA_VERSION,
        "protocol": "directional verified-failure contrastive skills; canonical observable concepts; disjoint development source",
        "source_split": args.source_split,
        "source_records": len(base_rows),
        "missing_c100": missing_c100,
        "directional_candidates_before_pair_deduplication": len(directional),
        "skipped": dict(skipped),
    }
    payload["content_sha256"] = canonical_hash(payload)
    write_json(args.output, payload)
    print(json.dumps({"rules": len(rules), "directional_candidates": len(directional), "skipped": dict(skipped), "content_sha256": payload["content_sha256"]}, ensure_ascii=False, indent=2))


def build_cohort(args: argparse.Namespace) -> None:
    payload = _release_payload(args.release)
    statuses = set(args.statuses)
    rules = load_release(args.release, statuses)
    c100 = load_c100(args.predictions)
    base = [
        row
        for row in read_jsonl([args.root / "base_q100_records.jsonl"])
        if row["split_role"] == args.cohort
    ]
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    handles = [
        (output / f"{args.cohort}_shard_{index}.jsonl").open("w", encoding="utf-8")
        for index in range(args.shards)
    ]
    reason_counts: Counter = Counter()
    candidate_counts: Counter = Counter()
    records: list[dict[str, Any]] = []
    try:
        for row in base:
            sid = str(row["sample_id"])
            baseline = str(c100.get(sid, {}).get("prediction", ""))
            candidates, reasons = _candidate_rules(
                row,
                rules,
                baseline,
                top_k=args.top_k,
                minimum_trigger_matches=args.minimum_trigger_matches,
            )
            reason_counts.update(reasons)
            if not candidates:
                reason_counts["no_applicable_memory"] += 1
                continue
            if args.mode == "all":
                selected = candidates
            else:
                for candidate in candidates:
                    candidate["retrieval_score"] = _retrieval_score(candidate, args.top_k)
                ordered = sorted(
                    candidates,
                    key=lambda item: (-float(item["retrieval_score"]), str(item["rule"]["rule_id"])),
                )
                if len(ordered) > 1:
                    first, second = ordered[:2]
                    conflicting = label_key(first["recommendation"]) != label_key(second["recommendation"])
                    if conflicting and float(first["retrieval_score"]) - float(second["retrieval_score"]) < args.conflict_margin:
                        reason_counts["near_tied_conflict_abstention"] += 1
                        continue
                selected = ordered[:1]
            candidate_counts[len(candidates)] += 1
            for candidate in selected:
                rule = {
                    **candidate["rule"],
                    "_query_recommendation": candidate["recommendation"],
                    "_query_trigger_matches": candidate["trigger_matches"],
                }
                global_index = args.global_index_offset + len(records)
                prompt = _inject_v2(str(row["prompt"]), rule, baseline)
                record = {
                    **{key: value for key, value in row.items() if key != "prompt"},
                    "global_index": global_index,
                    "source_control_global_index": row["global_index"],
                    "condition": args.condition,
                    "arm": ARM,
                    "memory_schema_version": SCHEMA_VERSION,
                    "memory_release_id": payload["release_id"],
                    "memory_release_sha256": payload["content_sha256"],
                    "rule_id": rule["rule_id"],
                    "rule_key": rule["rule_key"],
                    "rule_disease_a": rule["disease_a"],
                    "rule_disease_b": rule["disease_b"],
                    "memory_lifecycle": rule.get("status", "staging"),
                    "memory_utility": rule.get("utility", {}),
                    "c100_anchor_prediction": baseline,
                    "controller_recommendation": candidate["recommendation"],
                    "candidate_probability": candidate["candidate_probability"],
                    "c100_probability": candidate["c100_probability"],
                    "retrieval_score": candidate.get("retrieval_score", 0.0),
                    "applicability_reason": candidate["applicability_reason"],
                    "trigger_matches": candidate["trigger_matches"],
                    "prompt_sha256": canonical_hash(prompt),
                    "prompt": prompt,
                }
                records.append(record)
                handles[(len(records) - 1) % args.shards].write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        for handle in handles:
            handle.close()
    audit = {
        "schema_version": SCHEMA_VERSION,
        "cohort": args.cohort,
        "mode": args.mode,
        "source_records": len(base),
        "inference_records": len(records),
        "unique_samples": len({row["sample_id"] for row in records}),
        "top_k": args.top_k,
        "minimum_trigger_matches": args.minimum_trigger_matches,
        "conflict_margin": args.conflict_margin,
        "candidate_count_distribution": {str(key): value for key, value in sorted(candidate_counts.items())},
        "reason_counts": dict(reason_counts),
        "release_id": payload["release_id"],
        "release_sha256": payload["content_sha256"],
        "status": "PASS" if records else "NO_APPLICABLE_MEMORY",
    }
    audit["content_sha256"] = canonical_hash(audit)
    write_json(output / "manifest_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def _outcome(row: dict[str, Any], c: dict[str, Any]) -> dict[str, Any]:
    labels = [str(item[0]) for item in row.get("candidate_probabilities", [])]
    llm = parse_primary_diagnosis(str(row.get("response", "")), labels)
    c_prediction = str(c["prediction"])
    recommendation = str(row["controller_recommendation"])
    pair = {
        evaluation_label(str(row["dataset"]), str(row["rule_disease_a"])),
        evaluation_label(str(row["dataset"]), str(row["rule_disease_b"])),
    }
    if not llm or evaluation_label(str(row["dataset"]), llm) not in pair:
        llm = c_prediction
        parse_mode = "invalid_or_third_class_fallback"
    elif evaluation_label(str(row["dataset"]), llm) != evaluation_label(str(row["dataset"]), recommendation):
        llm = c_prediction
        parse_mode = "direction_disagreement_fallback"
    else:
        parse_mode = "valid_directional_output"
    truth = evaluation_label(str(row["dataset"]), str(row["ground_truth"]))
    c_correct = int(evaluation_label(str(row["dataset"]), c_prediction) == truth)
    llm_correct = int(evaluation_label(str(row["dataset"]), llm) == truth)
    direct_correct = int(evaluation_label(str(row["dataset"]), recommendation) == truth)
    probabilities = [(str(item[0]), float(item[1])) for item in row["candidate_probabilities"]]
    return {
        "c_prediction": c_prediction,
        "llm_prediction": llm,
        "direct_prediction": recommendation,
        "c_correct": c_correct,
        "llm_correct": llm_correct,
        "direct_correct": direct_correct,
        "parse_mode": parse_mode,
        "confidence_bin": _confidence_bin(probabilities),
        "high_confidence": _confidence_bin(probabilities) == "high",
    }


def govern(args: argparse.Namespace) -> None:
    candidate = _release_payload(args.candidate_release)
    rules = {str(row["rule_id"]): dict(row) for row in candidate["rules"]}
    c100 = load_c100(args.predictions)
    if args.results:
        results = _strict_results(args.results)
        utility_source = "paired_model_outputs"
    else:
        manifest_rows = read_jsonl(_expand(args.manifests))
        results = {}
        for row in manifest_rows:
            key = (str(row.get("sample_id", "")), str(row.get("rule_id", "")))
            if not all(key) or key in results:
                raise ValueError(f"invalid or duplicate acceptance manifest key: {key}")
            results[key] = row
        utility_source = "deterministic_controller_recommendations"
    allowed_ids = {
        str(row["sample_id"])
        for row in read_jsonl([args.root / "base_q100_records.jsonl"])
        if row["split_role"] in set(args.source_splits)
    }
    stats: dict[str, Counter] = defaultdict(Counter)
    context: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for (sid, rule_id), row in results.items():
        if sid not in allowed_ids:
            raise ValueError(f"result outside declared governance splits: {sid}")
        if rule_id not in rules:
            raise ValueError(f"unknown rule in acceptance result: {rule_id}")
        if sid not in c100:
            raise ValueError(f"missing C100 result: {sid}")
        value = _outcome(row, c100[sid])
        bucket = stats[rule_id]
        bucket["exposures"] += 1
        bucket["llm_gains"] += int(value["c_correct"] == 0 and value["llm_correct"] == 1)
        bucket["llm_regressions"] += int(value["c_correct"] == 1 and value["llm_correct"] == 0)
        bucket["direct_gains"] += int(value["c_correct"] == 0 and value["direct_correct"] == 1)
        bucket["direct_regressions"] += int(value["c_correct"] == 1 and value["direct_correct"] == 0)
        bucket["invalid_fallbacks"] += int(value["parse_mode"] != "valid_directional_output")
        bucket["high_confidence_direct_regressions"] += int(
            value["high_confidence"] and value["c_correct"] == 1 and value["direct_correct"] == 0
        )
        ctx = context[rule_id][f"{row['dataset']}::{value['confidence_bin']}::{row['applicability_reason']}"]
        ctx["exposures"] += 1
        ctx["gains"] += int(value["c_correct"] == 0 and value["direct_correct"] == 1)
        ctx["regressions"] += int(value["c_correct"] == 1 and value["direct_correct"] == 0)

    governed: list[dict[str, Any]] = []
    lifecycle_counts: Counter = Counter()
    for rule_id, rule in sorted(rules.items()):
        values = stats[rule_id]
        gains = int(values["direct_gains"])
        regressions = int(values["direct_regressions"])
        alpha = gains + 1
        beta = regressions + 1
        probability_positive = _beta_probability_above_half(alpha, beta)
        posterior_mean = alpha / (alpha + beta)
        high_conf = int(values["high_confidence_direct_regressions"])
        if high_conf > 0 or regressions >= max(2, gains + 2):
            status = "quarantine"
        elif (
            gains >= args.minimum_gains
            and gains > regressions
            and probability_positive >= args.minimum_probability_positive
            and regressions <= args.maximum_regressions
        ):
            status = "active"
        else:
            status = "staging"
        lifecycle_counts[status] += 1
        utility = {
            "source_split": "+".join(args.source_splits),
            "exposures": int(values["exposures"]),
            "gains": gains,
            "regressions": regressions,
            "net_correct": gains - regressions,
            "posterior_alpha": alpha,
            "posterior_beta": beta,
            "posterior_mean": posterior_mean,
            "probability_positive": probability_positive,
            "high_confidence_regressions": high_conf,
            "llm_gains": int(values["llm_gains"]),
            "llm_regressions": int(values["llm_regressions"]),
            "invalid_or_direction_fallbacks": int(values["invalid_fallbacks"]),
            "context_ledger": {name: dict(counts) for name, counts in sorted(context[rule_id].items())},
        }
        governed.append({**rule, "status": status, "utility": utility, "reliability": posterior_mean})
    payload = {
        "release_id": "v2_utility_governed",
        "parent_release_id": candidate["release_id"],
        "source_stream_hashes": candidate["source_stream_hashes"],
        "rules": governed,
        "status": "active" if lifecycle_counts["active"] else "rolled_back",
        "schema_version": SCHEMA_VERSION,
        "protocol": "item-level beta utility; staged promotion; high-confidence-regression quarantine; heldout labels inaccessible",
        "utility_source": utility_source,
        "governance_policy": {
            "minimum_gains": args.minimum_gains,
            "minimum_probability_positive": args.minimum_probability_positive,
            "maximum_regressions": args.maximum_regressions,
            "high_confidence_regressions_allowed": 0,
        },
        "lifecycle_counts": dict(lifecycle_counts),
        "acceptance_result_records": len(results),
    }
    payload["content_sha256"] = canonical_hash(payload)
    write_json(args.output, payload)
    write_json(
        args.output.with_name("active_release.json"),
        {
            "release_id": payload["release_id"] if lifecycle_counts["active"] else "v0_c100",
            "content_sha256": payload["content_sha256"] if lifecycle_counts["active"] else "",
            "status": payload["status"],
            "path": str(args.output),
        },
    )
    print(json.dumps({"release_id": payload["release_id"], "lifecycle_counts": dict(lifecycle_counts), "acceptance_results": len(results)}, ensure_ascii=False, indent=2))


CONTROLLER_POLICIES = (
    {"policy_id": "v2_conservative_consensus", "minimum_score": 0.46, "probability_cap": 0.65, "margin_cap": 0.15, "execution_mode": "consensus"},
    {"policy_id": "v2_balanced_consensus", "minimum_score": 0.40, "probability_cap": 0.75, "margin_cap": 0.25, "execution_mode": "consensus"},
    {"policy_id": "v2_balanced_direct", "minimum_score": 0.46, "probability_cap": 0.65, "margin_cap": 0.18, "execution_mode": "direct"},
)


def _policy_prediction(row: dict[str, Any], outcome: dict[str, Any], policy: dict[str, Any]) -> tuple[str, str]:
    probabilities = [(str(item[0]), float(item[1])) for item in row["candidate_probabilities"]]
    probability = probabilities[0][1] if probabilities else 1.0
    margin = probability - probabilities[1][1] if len(probabilities) > 1 else probability
    score = float(row.get("retrieval_score", 0.0))
    if score < float(policy["minimum_score"]):
        return outcome["c_prediction"], "score_below_threshold"
    if probability > float(policy["probability_cap"]) or margin > float(policy["margin_cap"]):
        return outcome["c_prediction"], "protected_strong_prior"
    if policy["execution_mode"] == "consensus":
        if evaluation_label(str(row["dataset"]), outcome["llm_prediction"]) != evaluation_label(str(row["dataset"]), outcome["direct_prediction"]):
            return outcome["c_prediction"], "reviewer_disagreement"
        return outcome["llm_prediction"], "consensus_override"
    return outcome["direct_prediction"], "direct_override"


def tune_controller(args: argparse.Namespace) -> None:
    c100 = load_c100(args.predictions)
    results = _strict_results(args.results)
    gate_ids = {
        str(row["sample_id"])
        for row in read_jsonl([args.root / "base_q100_records.jsonl"])
        if row["split_role"] == "gate_development"
    }
    if any(sid not in gate_ids for sid, _ in results):
        raise ValueError("controller results include non-gate samples")
    audits: list[dict[str, Any]] = []
    for policy in CONTROLLER_POLICIES:
        gains = regressions = overrides = high_conf_regressions = 0
        by_dataset: dict[str, Counter] = defaultdict(Counter)
        for (sid, _), row in results.items():
            outcome = _outcome(row, c100[sid])
            prediction, reason = _policy_prediction(row, outcome, policy)
            correct = int(evaluation_label(str(row["dataset"]), prediction) == evaluation_label(str(row["dataset"]), str(row["ground_truth"])))
            changed = evaluation_label(str(row["dataset"]), prediction) != evaluation_label(str(row["dataset"]), outcome["c_prediction"])
            gain = int(outcome["c_correct"] == 0 and correct == 1)
            regression = int(outcome["c_correct"] == 1 and correct == 0)
            gains += gain
            regressions += regression
            overrides += int(changed)
            high_conf_regressions += int(regression and outcome["high_confidence"])
            by_dataset[str(row["dataset"])]["gains"] += gain
            by_dataset[str(row["dataset"])]["regressions"] += regression
            by_dataset[str(row["dataset"])]["overrides"] += int(changed)
        discordances = gains + regressions
        one_sided = sum(math.comb(discordances, k) for k in range(gains, discordances + 1)) / (2**discordances) if discordances else 1.0
        represented_negative = [
            dataset for dataset, values in by_dataset.items()
            if values["overrides"] >= args.minimum_dataset_overrides and values["gains"] < values["regressions"]
        ]
        passed = (
            overrides >= args.minimum_overrides
            and gains > regressions
            and gains - regressions >= args.minimum_net_gain
            and high_conf_regressions == 0
            and not represented_negative
            and one_sided <= args.maximum_one_sided_p
        )
        audits.append({
            "policy": policy,
            "overrides": overrides,
            "gains": gains,
            "regressions": regressions,
            "net_correct": gains - regressions,
            "high_confidence_regressions": high_conf_regressions,
            "one_sided_exact_p": one_sided,
            "represented_negative_datasets": represented_negative,
            "by_dataset": {dataset: dict(values) for dataset, values in sorted(by_dataset.items())},
            "passed": passed,
        })
    passing = [row for row in audits if row["passed"]]
    if passing:
        selected = max(passing, key=lambda row: (row["net_correct"], -row["regressions"], row["gains"], -row["overrides"]))
        policy = dict(selected["policy"])
        fallback = False
    else:
        policy = {"policy_id": "v2_c100_fallback", "minimum_score": 2.0, "probability_cap": 0.0, "margin_cap": 0.0, "execution_mode": "consensus"}
        selected = {"overrides": 0, "gains": 0, "regressions": 0, "net_correct": 0}
        fallback = True
    audit = {
        "schema_version": SCHEMA_VERSION,
        "source_split": "gate_development",
        "applicable_results": len(results),
        "selected": policy,
        "selected_outcomes": {key: selected[key] for key in ("overrides", "gains", "regressions", "net_correct")},
        "fallback": fallback,
        "pre_registered_policy_audits": audits,
    }
    payload = {**policy, "development_audit": audit}
    payload["content_sha256"] = canonical_hash(payload)
    write_json(args.output, payload)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def _bootstrap(rows: list[dict[str, Any]], field: str, samples: int, seed: int) -> tuple[float, float]:
    differences = np.asarray([row[field] - row["correct_C100"] for row in rows], dtype=float)
    datasets = np.asarray([row["dataset"] for row in rows])
    indices = {dataset: np.where(datasets == dataset)[0] for dataset in DATASETS}
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = np.concatenate([rng.choice(value, len(value), replace=True) for value in indices.values() if len(value)])
        values[index] = differences[selected].mean()
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def finalize_evaluation(args: argparse.Namespace) -> None:
    controller = json.loads(args.controller.read_text(encoding="utf-8"))
    c100 = load_c100(args.predictions)
    if args.results:
        results = _strict_results(args.results)
    elif controller.get("policy_id") == "v2_c100_fallback":
        # A failed development gate deterministically selects C100 for every
        # held-out case.  Running the VLM would add cost but cannot change an
        # output, so emit the complete frozen evaluation without inference.
        results = {}
    else:
        raise ValueError("heldout inference results are required for a non-fallback controller")
    base = [
        row for row in read_jsonl([args.root / "base_q100_records.jsonl"])
        if row["split_role"] == "heldout_evaluation"
    ]
    by_sample: dict[str, dict[str, Any]] = {}
    for (sid, _), row in results.items():
        if sid in by_sample:
            raise ValueError(f"multiple selected v2 rules for heldout sample: {sid}")
        by_sample[sid] = row
    base_ids = {str(row["sample_id"]) for row in base}
    if set(by_sample) - base_ids:
        raise ValueError("heldout results include unexpected samples")
    rows: list[dict[str, Any]] = []
    for item in base:
        sid = str(item["sample_id"])
        c = c100[sid]
        prediction = str(c["prediction"])
        direct_prediction = prediction
        llm_prediction = prediction
        reason = "no_selected_memory"
        rule_id = ""
        score = 0.0
        if sid in by_sample:
            result = by_sample[sid]
            outcome = _outcome(result, c)
            prediction, reason = _policy_prediction(result, outcome, controller)
            direct_prediction = outcome["direct_prediction"]
            llm_prediction = outcome["llm_prediction"]
            rule_id = str(result["rule_id"])
            score = float(result.get("retrieval_score", 0.0))
        truth = evaluation_label(str(item["dataset"]), str(item["ground_truth"]))
        c_correct = int(str(c["correct"]) in {"1", "True", "true"})
        direct_correct = int(evaluation_label(str(item["dataset"]), direct_prediction) == truth)
        llm_correct = int(evaluation_label(str(item["dataset"]), llm_prediction) == truth)
        final_correct = int(evaluation_label(str(item["dataset"]), prediction) == truth)
        rows.append({
            "sample_id": sid,
            "dataset": item["dataset"],
            "relative_path": item["relative_path"],
            "ground_truth": item["ground_truth"],
            "prediction_C100": c["prediction"],
            "prediction_V2Direct": direct_prediction,
            "prediction_V2Reviewer": llm_prediction,
            "prediction_V2Final": prediction,
            "correct_C100": c_correct,
            "correct_V2Direct": direct_correct,
            "correct_V2Reviewer": llm_correct,
            "correct_V2Final": final_correct,
            "memory_selected": int(sid in by_sample),
            "override_allowed": int(evaluation_label(str(item["dataset"]), prediction) != evaluation_label(str(item["dataset"]), str(c["prediction"]))),
            "controller_reason": reason,
            "rule_id": rule_id,
            "retrieval_score": score,
        })
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics: list[dict[str, Any]] = []
    for dataset in (*DATASETS, "Pooled"):
        subset = rows if dataset == "Pooled" else [row for row in rows if row["dataset"] == dataset]
        for condition in ("C100", "V2Direct", "V2Reviewer", "V2Final"):
            correct = sum(int(row[f"correct_{condition}"]) for row in subset)
            metrics.append({"dataset": dataset, "condition": condition, "samples": len(subset), "correct": correct, "accuracy": correct / len(subset) if subset else 0.0})
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    paired: list[dict[str, Any]] = []
    for condition in ("V2Direct", "V2Reviewer", "V2Final"):
        a_only = sum(row["correct_C100"] == 1 and row[f"correct_{condition}"] == 0 for row in rows)
        b_only = sum(row["correct_C100"] == 0 and row[f"correct_{condition}"] == 1 for row in rows)
        low, high = _bootstrap(rows, f"correct_{condition}", args.bootstraps, args.seed)
        paired.append({
            "comparison": f"{condition}_vs_C100",
            "samples": len(rows),
            "difference": (b_only - a_only) / len(rows),
            "ci_low": low,
            "ci_high": high,
            "c100_only_correct": a_only,
            "comparison_only_correct": b_only,
            "mcnemar_exact_p": exact_mcnemar(a_only, b_only),
        })
    with (args.output / "paired_tests.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_scope": "exploratory_reuse",
        "confirmatory": False,
        "samples": len(rows),
        "memory_selected": sum(row["memory_selected"] for row in rows),
        "overrides": sum(row["override_allowed"] for row in rows),
        "controller_id": controller["policy_id"],
        "controller_sha256": controller["content_sha256"],
        "metrics": metrics,
        "paired": paired,
        "status": "PASS",
        "claim_boundary": "This inspected cohort is exploratory; a fresh temporal/external stream is required for confirmatory recognition claims.",
    }
    write_json(args.output / "audit.json", audit)
    print(json.dumps({"samples": len(rows), "selected": audit["memory_selected"], "overrides": audit["overrides"], "paired": paired}, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)

    close_invalid = commands.add_parser("close-invalid-results")
    close_invalid.add_argument("--manifest-dir", type=Path, required=True)
    close_invalid.add_argument("--results", nargs="+", required=True)
    close_invalid.add_argument("--output", type=Path, required=True)
    close_invalid.set_defaults(func=close_invalid_results)

    compile_release = commands.add_parser("compile-ledger-release")
    compile_release.add_argument("--parent-release", type=Path, required=True)
    compile_release.add_argument("--manifest-dir", type=Path, required=True)
    compile_release.add_argument("--minimum-concepts-per-side", type=int, choices=(1, 2), default=1)
    compile_release.add_argument("--output", type=Path, required=True)
    compile_release.set_defaults(func=compile_ledger_release)

    failure_release = commands.add_parser("compile-failure-memory")
    failure_release.add_argument("--root", type=Path, required=True)
    failure_release.add_argument("--predictions", type=Path, required=True)
    failure_release.add_argument("--source-split", choices=("rule_acceptance", "gate_development"), default="rule_acceptance")
    failure_release.add_argument("--top-k", type=int, default=5)
    failure_release.add_argument("--minimum-error-cases", type=int, default=3)
    failure_release.add_argument("--minimum-control-cases", type=int, default=5)
    failure_release.add_argument("--minimum-feature-support", type=int, default=2)
    failure_release.add_argument("--minimum-prevalence-margin", type=float, default=0.20)
    failure_release.add_argument("--maximum-concepts-per-side", type=int, default=4)
    failure_release.add_argument("--matcher-algorithm-revision", required=True)
    failure_release.add_argument("--matcher-sha256", required=True)
    failure_release.add_argument("--output", type=Path, required=True)
    failure_release.set_defaults(func=compile_failure_memory)

    build = commands.add_parser("build-cohort")
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--release", type=Path, required=True)
    build.add_argument("--predictions", type=Path, required=True)
    build.add_argument("--cohort", choices=("rule_acceptance", "gate_development", "heldout_evaluation"), required=True)
    build.add_argument("--condition", required=True)
    build.add_argument("--mode", choices=("all", "select"), required=True)
    build.add_argument("--statuses", nargs="+", default=["candidate"])
    build.add_argument("--top-k", type=int, default=3)
    build.add_argument("--minimum-trigger-matches", type=int, default=2)
    build.add_argument("--conflict-margin", type=float, default=0.05)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--global-index-offset", type=int, default=700000)
    build.add_argument("--shards", type=int, default=2)
    build.set_defaults(func=build_cohort)

    governance = commands.add_parser("govern")
    governance.add_argument("--root", type=Path, required=True)
    governance.add_argument("--candidate-release", type=Path, required=True)
    governance.add_argument("--predictions", type=Path, required=True)
    governance_inputs = governance.add_mutually_exclusive_group(required=True)
    governance_inputs.add_argument("--results", nargs="+")
    governance_inputs.add_argument("--manifests", nargs="+")
    governance.add_argument("--minimum-gains", type=int, default=1)
    governance.add_argument("--minimum-probability-positive", type=float, default=0.75)
    governance.add_argument("--maximum-regressions", type=int, default=1)
    governance.add_argument("--source-splits", nargs="+", choices=("rule_acceptance", "gate_development"), default=["rule_acceptance"])
    governance.add_argument("--output", type=Path, required=True)
    governance.set_defaults(func=govern)

    tune = commands.add_parser("tune-controller")
    tune.add_argument("--root", type=Path, required=True)
    tune.add_argument("--predictions", type=Path, required=True)
    tune.add_argument("--results", nargs="+", required=True)
    tune.add_argument("--minimum-overrides", type=int, default=10)
    tune.add_argument("--minimum-net-gain", type=int, default=2)
    tune.add_argument("--minimum-dataset-overrides", type=int, default=10)
    tune.add_argument("--maximum-one-sided-p", type=float, default=0.20)
    tune.add_argument("--output", type=Path, required=True)
    tune.set_defaults(func=tune_controller)

    final = commands.add_parser("finalize-evaluation")
    final.add_argument("--root", type=Path, required=True)
    final.add_argument("--predictions", type=Path, required=True)
    final.add_argument("--results", nargs="*", default=[])
    final.add_argument("--controller", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    final.add_argument("--bootstraps", type=int, default=10000)
    final.add_argument("--seed", type=int, default=20260827)
    final.set_defaults(func=finalize_evaluation)
    return result


def main() -> None:
    args = parser().parse_args()
    if getattr(args, "top_k", 3) < 2:
        raise ValueError("top-k must be at least two")
    args.func(args)


if __name__ == "__main__":
    main()
