#!/usr/bin/env python3
"""Strict, reproducible audit for a frozen self-evolving-memory initialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .concepts import (
    CONCEPT_LEXICON_SHA256,
    CONCEPT_MATCHER_ALGORITHM_REVISION,
    CONCEPT_MATCHER_SHA256,
    CONCEPT_PROTOCOL,
    build_concept_ledger,
)
from .core import canonical_hash, probabilities_from_prompt, rule_key, split_role, write_json
from .pipeline import (
    DATASETS,
    prompt_source_cases,
    q100_controls,
    rule_prompt,
    sentence_aware_excerpt,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_jsonl(path: Path, issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                issues.append({"file": str(path), "line": line_number, "reason": "blank_line"})
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                issues.append({
                    "file": str(path),
                    "line": line_number,
                    "reason": "invalid_json",
                    "detail": str(error),
                })
                continue
            if not isinstance(value, dict):
                issues.append({"file": str(path), "line": line_number, "reason": "non_object"})
                continue
            rows.append(value)
    return rows


def normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def hash_without_content_sha256(payload: dict[str, Any]) -> str:
    value = dict(payload)
    value.pop("content_sha256", None)
    return canonical_hash(value)


def add_issue(issues: list[dict[str, Any]], reason: str, **details: Any) -> None:
    issues.append({"reason": reason, **details})


def audit(args: argparse.Namespace) -> dict[str, Any]:
    experiment_root: Path = args.experiment_root.resolve()
    root: Path = args.root.resolve()
    issues: list[dict[str, Any]] = []

    audit_path = root / "initialization_audit.json"
    initialization = json.loads(audit_path.read_text(encoding="utf-8"))
    if initialization.get("content_sha256") != hash_without_content_sha256(initialization):
        add_issue(issues, "initialization_audit_hash_mismatch")
    if initialization.get("matching_protocol") != CONCEPT_PROTOCOL:
        add_issue(issues, "initialization_matching_protocol_mismatch")
    if initialization.get("concept_lexicon_sha256") != CONCEPT_LEXICON_SHA256:
        add_issue(issues, "initialization_concept_lexicon_hash_mismatch")
    if initialization.get("matcher_algorithm_revision") != CONCEPT_MATCHER_ALGORITHM_REVISION:
        add_issue(issues, "initialization_matcher_algorithm_revision_mismatch")
    if initialization.get("matcher_sha256") != CONCEPT_MATCHER_SHA256:
        add_issue(issues, "initialization_matcher_hash_mismatch")

    streams: dict[str, dict[str, Any]] = {}
    case_lookups: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    class_finding_lookups: dict[str, dict[tuple[str, str], str]] = {}
    normalized_train: dict[str, np.ndarray] = {}
    for dataset in DATASETS:
        stream_path = experiment_root / "streams" / f"{dataset}_stream.json"
        stream = json.loads(stream_path.read_text(encoding="utf-8"))
        streams[dataset] = stream
        case_lookups[dataset] = {
            (str(row.get("disease", "")), str(row.get("case_id", ""))): row
            for row in stream.get("cases", [])
        }
        class_finding_lookups[dataset] = {
            (str(disease), str(row.get("case_id", ""))): str(row.get("key_findings") or "").strip()
            for disease, rows in stream.get("classes", {}).items()
            for row in rows
            if row.get("case_id")
        }
        expected_stream_hash = initialization.get("source_stream_hashes", {}).get(dataset)
        if expected_stream_hash != stream.get("content_sha256"):
            add_issue(issues, "source_stream_hash_mismatch", dataset=dataset)
        matrix = np.asarray(np.load(stream["train_feats"], mmap_mode="r"), dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized_train[dataset] = matrix / np.maximum(norms, 1e-12)

    base_path = root / "base_q100_records.jsonl"
    base_rows = strict_jsonl(base_path, issues)
    frozen_rows = q100_controls(experiment_root)
    frozen_by_sid = {f"{row['dataset']}::{row['relative_path']}": row for row in frozen_rows}
    seen_base_indices: set[int] = set()
    seen_base_ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    for row in base_rows:
        index = int(row.get("global_index", -1))
        sid = str(row.get("sample_id", ""))
        if index in seen_base_indices:
            add_issue(issues, "duplicate_base_global_index", global_index=index)
        if sid in seen_base_ids:
            add_issue(issues, "duplicate_base_sample_id", sample_id=sid)
        seen_base_indices.add(index)
        seen_base_ids.add(sid)
        expected = frozen_by_sid.get(sid)
        if expected is None:
            add_issue(issues, "base_missing_from_frozen_c100", sample_id=sid)
            continue
        stripped = {key: value for key, value in row.items() if key not in {"sample_id", "split_role", "candidate_probabilities"}}
        if stripped != expected:
            add_issue(issues, "base_record_drift", sample_id=sid)
        expected_role = split_role(sid, int(initialization["seed"]))
        if row.get("split_role") != expected_role:
            add_issue(issues, "split_recompute_mismatch", sample_id=sid)
        split_counts[f"{row.get('dataset')}::{row.get('split_role')}"] += 1
        recomputed_probabilities = [list(value) for value in probabilities_from_prompt(str(row.get("prompt", "")))]
        if row.get("candidate_probabilities") != recomputed_probabilities:
            add_issue(issues, "candidate_probability_recompute_mismatch", sample_id=sid)
    if set(frozen_by_sid) != seen_base_ids:
        add_issue(
            issues,
            "frozen_c100_coverage_mismatch",
            frozen=len(frozen_by_sid),
            audited=len(seen_base_ids),
        )

    manifest_paths = sorted((root / "rule_manifests").glob("rules_shard_*.jsonl"))
    proposals: list[dict[str, Any]] = []
    manifest_hashes: dict[str, str] = {}
    for path in manifest_paths:
        proposals.extend(strict_jsonl(path, issues))
        manifest_hashes[str(path.relative_to(root))] = file_sha256(path)

    seen_indices: set[int] = set()
    seen_rule_ids: set[str] = set()
    seen_rule_keys: set[str] = set()
    source_roles: Counter[str] = Counter()
    exact_prompt_recomputations = 0
    exact_concept_ledger_recomputations = 0
    embedding_recomputations = 0
    fully_preserved_source_findings = 0
    source_findings = 0
    max_source_finding_characters = 0
    for row in proposals:
        index = int(row.get("global_index", -1))
        rule_id = str(row.get("rule_id", ""))
        key = str(row.get("rule_key", ""))
        dataset = str(row.get("dataset", ""))
        disease_a = str(row.get("disease_a", ""))
        disease_b = str(row.get("disease_b", ""))
        if index in seen_indices:
            add_issue(issues, "duplicate_rule_global_index", global_index=index)
        if rule_id in seen_rule_ids:
            add_issue(issues, "duplicate_rule_id", rule_id=rule_id)
        if key in seen_rule_keys:
            add_issue(issues, "duplicate_rule_key", rule_key=key)
        seen_indices.add(index)
        seen_rule_ids.add(rule_id)
        seen_rule_keys.add(key)
        expected_key = rule_key(dataset, disease_a, disease_b)
        if key != expected_key:
            add_issue(issues, "rule_key_recompute_mismatch", global_index=index)

        source_cases_a = row.get("source_cases_a", [])
        source_cases_b = row.get("source_cases_b", [])
        selection_a = row.get("source_selection_a", [])
        selection_b = row.get("source_selection_b", [])
        ids_a = [str(item.get("case_id", "")) for item in source_cases_a]
        ids_b = [str(item.get("case_id", "")) for item in source_cases_b]
        if ids_a != [str(value) for value in row.get("source_case_ids_a", [])]:
            add_issue(issues, "source_ids_a_mismatch", global_index=index)
        if ids_b != [str(value) for value in row.get("source_case_ids_b", [])]:
            add_issue(issues, "source_ids_b_mismatch", global_index=index)
        if set(ids_a) & set(ids_b):
            add_issue(issues, "cross_label_source_id_overlap", global_index=index)
        if len(ids_a) != len(set(ids_a)) or len(ids_b) != len(set(ids_b)):
            add_issue(issues, "within_label_source_id_duplicate", global_index=index)

        selection_by_id_a = {str(item.get("case_id", "")): item for item in selection_a}
        selection_by_id_b = {str(item.get("case_id", "")): item for item in selection_b}
        if set(selection_by_id_a) != set(ids_a) or set(selection_by_id_b) != set(ids_b):
            add_issue(issues, "source_selection_coverage_mismatch", global_index=index)
            continue
        cases_a = []
        cases_b = []
        vectors_a = []
        vectors_b = []
        for side, disease, cases, selections, rebuilt, vectors in (
            ("A", disease_a, source_cases_a, selection_by_id_a, cases_a, vectors_a),
            ("B", disease_b, source_cases_b, selection_by_id_b, cases_b, vectors_b),
        ):
            for case in cases:
                case_id = str(case.get("case_id", ""))
                selection = selections[case_id]
                finding = str(case.get("findings", ""))
                finding_excerpt = sentence_aware_excerpt(finding)
                original = case_lookups.get(dataset, {}).get((disease, case_id))
                if original is None:
                    add_issue(issues, "source_case_not_in_frozen_stream", global_index=index, side=side, case_id=case_id)
                    continue
                expected_finding = (
                    class_finding_lookups.get(dataset, {}).get((disease, case_id))
                    or str(original.get("key_findings") or "").strip()
                )
                if expected_finding != finding:
                    add_issue(issues, "source_finding_drift", global_index=index, side=side, case_id=case_id)
                if canonical_hash(finding) != selection.get("findings_sha256"):
                    add_issue(issues, "source_finding_hash_mismatch", global_index=index, side=side, case_id=case_id)
                if canonical_hash(original.get("image_path", "")) != selection.get("image_path_sha256"):
                    add_issue(issues, "source_path_hash_mismatch", global_index=index, side=side, case_id=case_id)
                if case.get("findings_excerpt") != finding_excerpt:
                    add_issue(issues, "source_excerpt_recompute_mismatch", global_index=index, side=side, case_id=case_id)
                if case.get("full_findings_sha256") != canonical_hash(finding):
                    add_issue(issues, "source_full_finding_hash_mismatch", global_index=index, side=side, case_id=case_id)
                if case.get("excerpt_sha256") != canonical_hash(finding_excerpt):
                    add_issue(issues, "source_excerpt_hash_mismatch", global_index=index, side=side, case_id=case_id)
                role = str(selection.get("selection_role", ""))
                source_roles[f"{dataset}::{side}::{role}"] += 1
                rebuilt.append({"case_id": case_id, "findings": finding, "selection_role": role})
                source_findings += 1
                normalized = normalized_text(finding)
                max_source_finding_characters = max(max_source_finding_characters, len(normalized))
                if sentence_aware_excerpt(finding) == normalized:
                    fully_preserved_source_findings += 1
                else:
                    add_issue(issues, "source_finding_not_fully_preserved", global_index=index, side=side, case_id=case_id)
                feature_index = int(selection.get("feature_index", -1))
                matrix = normalized_train.get(dataset)
                if matrix is None or feature_index < 0 or feature_index >= matrix.shape[0]:
                    add_issue(issues, "invalid_source_feature_index", global_index=index, side=side, case_id=case_id)
                else:
                    vectors.append(matrix[feature_index])
        if len(cases_a) == len(source_cases_a) and len(cases_b) == len(source_cases_b):
            expected_ledger = build_concept_ledger(
                prompt_source_cases(cases_a),
                prompt_source_cases(cases_b),
                minimum_support=int(row.get("minimum_feature_support", 3)),
            )
            exact_concept_ledger_recomputations += 1
            if row.get("concept_ledger") != expected_ledger:
                add_issue(issues, "concept_ledger_recompute_mismatch", global_index=index)
            if row.get("concept_ledger_sha256") != expected_ledger.get("content_sha256"):
                add_issue(issues, "concept_ledger_hash_mismatch", global_index=index)
            expected_prompt = rule_prompt(
                dataset,
                disease_a,
                disease_b,
                cases_a,
                cases_b,
                concept_ledger=expected_ledger,
                minimum_feature_support=int(row.get("minimum_feature_support", 3)),
            )
            exact_prompt_recomputations += 1
            if row.get("prompt") != expected_prompt:
                add_issue(issues, "rule_prompt_recompute_mismatch", global_index=index)

        selection_payload = {
            "schema_version": "v1.1",
            "key": key,
            "selection_protocol": "representative_cluster_medoids+diverse_mmr+quality_screened_positive_margin_embedding_hard",
            "matching_protocol": CONCEPT_PROTOCOL,
            "concept_lexicon_sha256": CONCEPT_LEXICON_SHA256,
            "matcher_algorithm_revision": CONCEPT_MATCHER_ALGORITHM_REVISION,
            "matcher_sha256": CONCEPT_MATCHER_SHA256,
            "a": [{"case_id": case_id, "role": selection_by_id_a[case_id].get("selection_role")} for case_id in ids_a],
            "b": [{"case_id": case_id, "role": selection_by_id_b[case_id].get("selection_role")} for case_id in ids_b],
        }
        expected_pool_hash = canonical_hash(selection_payload)
        if row.get("source_pool_sha256") != expected_pool_hash:
            add_issue(issues, "source_pool_hash_mismatch", global_index=index)
        if rule_id != "cr11c_" + expected_pool_hash[:16]:
            add_issue(issues, "rule_id_recompute_mismatch", global_index=index)
        if row.get("matching_protocol") != CONCEPT_PROTOCOL:
            add_issue(issues, "matching_protocol_mismatch", global_index=index)
        if row.get("concept_lexicon_sha256") != CONCEPT_LEXICON_SHA256:
            add_issue(issues, "concept_lexicon_hash_mismatch", global_index=index)
        if row.get("matcher_algorithm_revision") != CONCEPT_MATCHER_ALGORITHM_REVISION:
            add_issue(issues, "matcher_algorithm_revision_mismatch", global_index=index)
        if row.get("matcher_sha256") != CONCEPT_MATCHER_SHA256:
            add_issue(issues, "matcher_hash_mismatch", global_index=index)
        if vectors_a and vectors_b:
            # Take the maximum over the complete A-by-B matrix.
            recomputed_similarity = float((np.asarray(vectors_a) @ np.asarray(vectors_b).T).max())
            embedding_recomputations += 1
            if not math.isclose(recomputed_similarity, float(row.get("maximum_cross_label_source_similarity", -2.0)), abs_tol=5e-7):
                add_issue(
                    issues,
                    "cross_label_similarity_recompute_mismatch",
                    global_index=index,
                    stored=row.get("maximum_cross_label_source_similarity"),
                    recomputed=round(recomputed_similarity, 9),
                )
            if recomputed_similarity >= 0.9995:
                add_issue(issues, "cross_label_near_duplicate_embedding", global_index=index)

    if initialization.get("rule_proposals") != len(proposals):
        add_issue(issues, "proposal_count_mismatch", stored=initialization.get("rule_proposals"), recomputed=len(proposals))
    if initialization.get("base_records") != len(base_rows):
        add_issue(issues, "base_count_mismatch", stored=initialization.get("base_records"), recomputed=len(base_rows))
    if dict(sorted(split_counts.items())) != dict(sorted(initialization.get("role_counts", {}).items())):
        add_issue(issues, "split_role_count_mismatch")
    if dict(sorted(source_roles.items())) != dict(sorted(initialization.get("selection_role_counts", {}).items())):
        add_issue(issues, "source_selection_role_count_mismatch")

    if args.pair_release:
        pair_release = json.loads(args.pair_release.read_text(encoding="utf-8"))
        expected_pair_hash = pair_release.get("content_sha256") or canonical_hash(pair_release)
        if initialization.get("pair_release_sha256") != expected_pair_hash:
            add_issue(issues, "pair_release_hash_mismatch")

    release_path = root / "releases" / "v0_c100.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("content_sha256") != hash_without_content_sha256(release):
        add_issue(issues, "baseline_release_hash_mismatch")
    active = json.loads((root / "releases" / "active_release.json").read_text(encoding="utf-8"))
    if active.get("release_id") != release.get("release_id") or active.get("content_sha256") != release.get("content_sha256"):
        add_issue(issues, "active_release_pointer_mismatch")

    output = {
        "schema_version": "v1.1-initialization-full-audit-v2",
        "status": "PASS" if not issues else "FAIL",
        "root": str(root),
        "base_records": len(base_rows),
        "rule_proposals": len(proposals),
        "manifest_files": len(manifest_paths),
        "exact_prompt_recomputations": exact_prompt_recomputations,
        "exact_concept_ledger_recomputations": exact_concept_ledger_recomputations,
        "embedding_similarity_recomputations": embedding_recomputations,
        "source_findings": source_findings,
        "fully_preserved_source_findings": fully_preserved_source_findings,
        "max_source_finding_characters": max_source_finding_characters,
        "split_role_counts": dict(sorted(split_counts.items())),
        "source_selection_role_counts": dict(sorted(source_roles.items())),
        "file_sha256": {
            "base_q100_records.jsonl": file_sha256(base_path),
            "initialization_audit.json": file_sha256(audit_path),
            "releases/v0_c100.json": file_sha256(release_path),
            **manifest_hashes,
        },
        "issue_count": len(issues),
        "issues": issues,
    }
    output["content_sha256"] = canonical_hash(output)
    return output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--experiment-root", type=Path, required=True)
    result.add_argument("--root", type=Path, required=True)
    result.add_argument("--pair-release", type=Path)
    result.add_argument("--output", type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    payload = audit(args)
    output = args.output or (args.root / "initialization_full_audit.json")
    write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
