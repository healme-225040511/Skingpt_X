#!/usr/bin/env python3
"""Build, validate, publish, gate and evaluate SkinGPT-X self-evolving memory v1."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .core import (
    GatePolicy,
    MemoryRelease,
    StructuredRule,
    canonical_hash,
    inject_rule,
    label_key,
    normalize_overlapping_features,
    normalized_phrase,
    parse_primary_diagnosis,
    probabilities_from_prompt,
    read_jsonl,
    rule_applicability,
    rule_key,
    split_role,
    strict_rule_from_response,
    write_json,
)


DATASETS = ("DDI", "DermNet23", "Fit17k", "HAM10000")
HAM_SYNONYMS = {
    "akiec": ("actinic keratos", "bowen", "intraepithelial carcinoma"),
    "bcc": ("basal cell carcinoma",),
    "bkl": ("benign keratos", "solar lentig", "seborrheic keratos"),
    "df": ("dermatofibroma",),
    "mel": ("melanoma",),
    "nv": ("melanocytic nev", "melanocytic nae", "melanocytic nevi"),
    "vasc": ("vascular lesion", "angioma", "pyogenic granuloma"),
}


def evaluation_label(dataset: str, value: str) -> str:
    normalized = label_key(value)
    if dataset != "HAM10000":
        return normalized
    if normalized in HAM_SYNONYMS:
        return normalized
    for code, needles in HAM_SYNONYMS.items():
        if any(label_key(needle) in normalized for needle in needles):
            return code
    return normalized


def sample_id(row: dict[str, Any]) -> str:
    return f"{row['dataset']}::{row['relative_path']}"


def q100_controls(experiment_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((experiment_root / "diagnosis_manifests").glob("controls_shard_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("condition") == "C100":
                rows.append(row)
    rows.sort(key=lambda row: (row["dataset"], row["relative_path"]))
    return rows


def choose_sources(cases: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(cases) <= limit:
        return cases
    positions = np.linspace(0, len(cases) - 1, num=limit, dtype=int)
    return [cases[int(index)] for index in positions]


def ontology_conflict(disease_a: str, disease_b: str) -> bool:
    """Reject obvious synonym or parent/child label pairs before rule generation."""
    left = label_key(disease_a)
    right = label_key(disease_b)
    if not left or not right or left == right:
        return True
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if left_tokens <= right_tokens or right_tokens <= left_tokens:
        return min(len(left_tokens), len(right_tokens)) <= 2
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens) >= 0.80


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def sentence_aware_excerpt(text: str, maximum_characters: int = 4000) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= maximum_characters:
        return value
    sentences = re.split(r"(?<=[.!?])\s+", value)
    selected: list[str] = []
    length = 0
    for sentence in sentences:
        addition = len(sentence) + (1 if selected else 0)
        if selected and length + addition > maximum_characters:
            break
        selected.append(sentence)
        length += addition
    return " ".join(selected).strip() if selected else value[:maximum_characters].rsplit(" ", 1)[0].strip()


def density_proxy(matrix: np.ndarray, maximum_landmarks: int = 256, neighbors: int = 10) -> np.ndarray:
    """Scalable deterministic approximation to same-class local density."""
    count = len(matrix)
    if count <= 1:
        return np.ones(count, dtype=np.float32)
    landmark_indices = np.linspace(0, count - 1, num=min(count, maximum_landmarks), dtype=int)
    similarities = matrix @ matrix[landmark_indices].T
    top_k = min(neighbors, similarities.shape[1])
    partitioned = np.partition(similarities, similarities.shape[1] - top_k, axis=1)[:, -top_k:]
    return partitioned.mean(axis=1).astype(np.float32)


def spherical_cluster_medoids(matrix: np.ndarray, indices: list[int], clusters: int) -> list[int]:
    if not indices or clusters <= 0:
        return []
    values = matrix[indices]
    clusters = min(clusters, len(indices))
    seeds = [int(np.argmax(values.mean(axis=0) @ values.T))]
    while len(seeds) < clusters:
        similarity = values @ values[seeds].T
        next_seed = int(np.argmin(similarity.max(axis=1)))
        if next_seed in seeds:
            break
        seeds.append(next_seed)
    centers = values[seeds]
    assignments = np.zeros(len(values), dtype=int)
    for _ in range(12):
        assignments = np.argmax(values @ centers.T, axis=1)
        updated = []
        for cluster in range(len(centers)):
            members = values[assignments == cluster]
            if not len(members):
                updated.append(centers[cluster])
                continue
            center = members.mean(axis=0)
            center /= max(float(np.linalg.norm(center)), 1e-12)
            updated.append(center)
        new_centers = np.asarray(updated, dtype=np.float32)
        if np.allclose(new_centers, centers, atol=1e-5):
            centers = new_centers
            break
        centers = new_centers
    medoids = []
    for cluster in range(len(centers)):
        member_positions = np.where(assignments == cluster)[0]
        if not len(member_positions):
            continue
        local = int(member_positions[np.argmax(values[member_positions] @ centers[cluster])])
        medoids.append(indices[local])
    return medoids


def prepare_generator_class(
    disease: str,
    class_rows: list[dict[str, Any]],
    case_lookup: dict[tuple[str, str], dict[str, Any]],
    train_features: np.ndarray,
    generator_fraction: float,
) -> dict[str, Any] | None:
    generator_count = max(1, math.floor(len(class_rows) * generator_fraction))
    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    excluded = Counter()
    seen_cases: set[str] = set()
    seen_feature_indices: set[int] = set()
    seen_image_paths: set[str] = set()
    seen_findings: set[str] = set()
    for class_row in class_rows[:generator_count]:
        case_id = str(class_row.get("case_id", ""))
        metadata = case_lookup.get((disease, case_id))
        if not case_id or metadata is None or metadata.get("feature_index") is None:
            excluded["missing_feature"] += 1
            continue
        feature_index = int(metadata["feature_index"])
        findings = str(class_row.get("key_findings") or metadata.get("key_findings") or "").strip()
        image_path = str(metadata.get("image_path", "")).strip()
        if len(findings) < 120:
            excluded["short_findings"] += 1
            continue
        vector = np.asarray(train_features[feature_index], dtype=np.float32)
        if not np.isfinite(vector).all() or float(np.linalg.norm(vector)) <= 1e-12:
            excluded["invalid_feature"] += 1
            continue
        if case_id in seen_cases or feature_index in seen_feature_indices:
            excluded["duplicate_id_or_feature"] += 1
            continue
        normalized_findings = normalized_phrase(findings)
        if (image_path and image_path in seen_image_paths) or normalized_findings in seen_findings:
            excluded["duplicate_path_or_findings"] += 1
            continue
        seen_cases.add(case_id)
        seen_feature_indices.add(feature_index)
        if image_path:
            seen_image_paths.add(image_path)
        seen_findings.add(normalized_findings)
        rows.append({
            "case_id": case_id,
            "findings": findings,
            "image_path": image_path,
            "feature_index": feature_index,
        })
        vectors.append(vector)
    if len(rows) < 2:
        return None
    # Make every downstream tie-break independent of source-file row order.
    paired = sorted(zip(rows, vectors), key=lambda item: str(item[0]["case_id"]))
    rows = [item[0] for item in paired]
    vectors = [item[1] for item in paired]
    matrix = normalize_rows(np.asarray(vectors, dtype=np.float32))
    centroid = matrix.mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
    density = density_proxy(matrix)
    own_similarity = matrix @ centroid
    density_cutoff = float(np.quantile(density, 0.25))
    similarity_cutoff = float(np.quantile(own_similarity, 0.10))
    clean_indices = [
        index
        for index in range(len(rows))
        if float(density[index]) >= density_cutoff and float(own_similarity[index]) >= similarity_cutoff
    ]
    if len(clean_indices) < min(6, len(rows)):
        clean_indices = list(range(len(rows)))
    return {
        "rows": rows,
        "matrix": matrix,
        "centroid": centroid,
        "density": density,
        "own_similarity": own_similarity,
        "clean_indices": clean_indices,
        "excluded": dict(excluded),
        "generator_requested": generator_count,
    }


def select_pair_sources(
    prepared: dict[str, Any],
    comparator_centroid: np.ndarray,
    limit: int,
) -> list[dict[str, Any]]:
    rows = prepared["rows"]
    matrix = prepared["matrix"]
    density = prepared["density"]
    own_similarity = prepared["own_similarity"]
    clean = list(prepared["clean_indices"])
    effective_limit = min(limit, len(clean))
    representative_quota = max(1, math.ceil(effective_limit / 3))
    diverse_quota = max(1, math.ceil((effective_limit - representative_quota) / 2))
    hard_quota = max(0, effective_limit - representative_quota - diverse_quota)
    quotas = (representative_quota, diverse_quota, hard_quota)

    selected: list[int] = []
    roles: dict[int, str] = {}
    near_duplicate_similarity = 0.9995

    def distinct_from_selected(index: int) -> bool:
        return not selected or float((matrix[index] @ matrix[selected].T).max()) < near_duplicate_similarity

    representative = spherical_cluster_medoids(matrix, clean, quotas[0])
    for index in representative:
        if not distinct_from_selected(index):
            continue
        selected.append(index)
        roles[index] = "representative"

    density_values = density[clean]
    low, high = float(density_values.min()), float(density_values.max())
    density_percentile = (density - low) / max(high - low, 1e-12)
    for _ in range(quotas[1]):
        candidates = [index for index in clean if index not in selected and distinct_from_selected(index)]
        if not candidates:
            break
        if selected:
            minimum_distances = 1.0 - (matrix[candidates] @ matrix[selected].T).max(axis=1)
        else:
            minimum_distances = np.ones(len(candidates), dtype=np.float32)
        scores = 0.75 * minimum_distances + 0.25 * density_percentile[candidates]
        chosen = candidates[int(np.argmax(scores))]
        selected.append(chosen)
        roles[chosen] = "diverse"

    comparator_similarity = matrix @ comparator_centroid
    margins = own_similarity - comparator_similarity
    clean_margins = margins[clean]
    lower_margin = 0.0
    hard_candidates = [
        index
        for index in clean
        if index not in selected
        and distinct_from_selected(index)
        and float(margins[index]) > lower_margin
    ]
    hard_candidates.sort(key=lambda index: (abs(float(margins[index])), -float(density[index]), rows[index]["case_id"]))
    for index in hard_candidates[:quotas[2]]:
        selected.append(index)
        roles[index] = "quality_screened_embedding_hard"

    if len(selected) < effective_limit:
        fillers = [index for index in clean if index not in selected and distinct_from_selected(index)]
        fillers.sort(key=lambda index: (-float(density[index]), rows[index]["case_id"]))
        for index in fillers[: effective_limit - len(selected)]:
            selected.append(index)
            roles[index] = "representative_fill"

    result = []
    for index in selected[:limit]:
        if len(selected) > 1:
            other_indices = [value for value in selected if value != index]
            minimum_distance = float(1.0 - (matrix[index] @ matrix[other_indices].T).max()) if other_indices else 1.0
        else:
            minimum_distance = 1.0
        result.append({
            **rows[index],
            "selection_role": roles[index],
            "density_proxy": round(float(density[index]), 6),
            "own_similarity": round(float(own_similarity[index]), 6),
            "comparator_similarity": round(float(comparator_similarity[index]), 6),
            "pair_margin": round(float(margins[index]), 6),
            "minimum_distance_to_selected": round(minimum_distance, 6),
        })
    return result


def rule_prompt(dataset: str, disease_a: str, disease_b: str, cases_a: list[dict[str, Any]], cases_b: list[dict[str, Any]]) -> str:
    def compact(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
        return [
            {
                "case_id": str(row["case_id"]),
                "selection_role": str(row["selection_role"]),
                "findings": sentence_aware_excerpt(str(row["findings"])),
            }
            for row in rows
        ]

    return f"""You are proposing an evidence-traceable contrastive dermatology memory rule.

Dataset-internal closed-set labels:
- Disease A: {disease_a}
- Disease B: {disease_b}

Immutable training cases for Disease A:
{json.dumps(compact(cases_a), ensure_ascii=False, indent=2)}

Immutable training cases for Disease B:
{json.dumps(compact(cases_b), ensure_ascii=False, indent=2)}

The cases were selected with balanced representative, diverse, and quality-screened embedding-hard proxy roles. Create a conservative A-versus-B rule only when the supplied evidence supports a stable contrast across these roles. The embedding-hard cases are not doctor-verified clean labels and cannot support a feature on their own. Do not force a rule for synonymous labels, broad heterogeneous categories, or overlapping appearances.

For every discriminative feature, cite at least three different same-class cases and copy one exact contiguous 2-24 word quote from each cited finding. Use all supplied source IDs in source_case_ids_a/source_case_ids_b; feature-level citations may use subsets. A feature seen on both sides must be nondiscriminative. Keep each feature under 24 words.

Executable trigger terms must be exact, observable phrases copied from the findings, occur in at least three cases on that side, and occur in at least two fewer cases on the opposite side. Required-context and abstention terms must also be literal phrases that can be matched in a current frozen finding description. If these requirements cannot be met, return publishable=false rather than inventing a contrast.

Return valid JSON only:
{{
  "schema_version": "v1.1",
  "publishable": true,
  "disease_a": {json.dumps(disease_a, ensure_ascii=False)},
  "disease_b": {json.dumps(disease_b, ensure_ascii=False)},
  "supports_a": ["2-4 pair-specific features"],
  "supports_b": ["2-4 pair-specific features"],
  "nondiscriminative": ["2-6 shared/generic features"],
  "trigger_terms_a": ["2-6 exact observable phrases"],
  "trigger_terms_b": ["2-6 exact observable phrases"],
  "required_context_terms": ["0-4 literal context phrases"],
  "abstain_terms": ["0-4 literal abstention phrases"],
  "required_context": ["0-4 required image or clinical contexts"],
  "limitations": ["1-4 limitations or abstention conditions"],
  "evidence_a": [{{"feature": "exact supports_a feature", "cases": [{{"case_id": "A ID", "quote": "exact quote"}}]}}],
  "evidence_b": [{{"feature": "exact supports_b feature", "cases": [{{"case_id": "B ID", "quote": "exact quote"}}]}}],
  "source_case_ids_a": {json.dumps([str(row['case_id']) for row in cases_a], ensure_ascii=False)},
  "source_case_ids_b": {json.dumps([str(row['case_id']) for row in cases_b], ensure_ascii=False)}
}}"""


def initialize(args: argparse.Namespace) -> None:
    root: Path = args.experiment_root
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    streams = {dataset: json.loads((root / "streams" / f"{dataset}_stream.json").read_text(encoding="utf-8")) for dataset in DATASETS}
    source_hashes = {dataset: stream["content_sha256"] for dataset, stream in streams.items()}

    generator: dict[str, dict[str, dict[str, Any]]] = {}
    pair_set: set[tuple[str, str, str]] = set()
    source_audit: dict[str, Any] = {}
    skipped_pair_reasons = Counter()
    for dataset, stream in streams.items():
        generator[dataset] = {}
        labels = sorted(stream["classes"], key=label_key)
        centroids: dict[str, np.ndarray] = {}
        train = np.load(stream["train_feats"], mmap_mode="r")
        case_lookup = {
            (str(row["disease"]), str(row["case_id"])): row
            for row in stream["cases"]
            if row.get("case_id")
        }
        excluded = Counter()
        for disease in labels:
            prepared = prepare_generator_class(
                disease,
                [row for row in stream["classes"][disease] if row.get("case_id")],
                case_lookup,
                train,
                args.generator_fraction,
            )
            if prepared is None:
                excluded["class_without_two_usable_cases"] += 1
                continue
            generator[dataset][disease] = prepared
            centroids[disease] = prepared["centroid"]
            excluded.update(prepared["excluded"])
        if args.pair_release is None:
            for disease, centroid in centroids.items():
                others = []
                for comparator, other_centroid in centroids.items():
                    if comparator == disease:
                        continue
                    others.append((float(centroid @ other_centroid), comparator))
                for _, comparator in sorted(others, key=lambda item: (-item[0], label_key(item[1])))[: args.neighbor_pairs]:
                    if ontology_conflict(disease, comparator):
                        skipped_pair_reasons["ontology_conflict"] += 1
                        continue
                    ordered = sorted((disease, comparator), key=label_key)
                    pair_set.add((dataset, ordered[0], ordered[1]))
        source_audit[dataset] = {
            "classes": len(labels),
            "usable_classes": len(generator[dataset]),
            "generator_cases": sum(len(value["rows"]) for value in generator[dataset].values()),
            "centroid_classes": len(centroids),
            "excluded": dict(excluded),
        }

    pair_source = "row_normalized_nearest_centroids"
    pair_release_sha256 = None
    frozen_pair_audit = {"raw_rules": 0, "unique_eligible_pairs": 0}
    if args.pair_release is not None:
        pair_source = "frozen_release"
        frozen = json.loads(args.pair_release.read_text(encoding="utf-8"))
        pair_release_sha256 = str(frozen.get("content_sha256") or canonical_hash(frozen))
        frozen_rules = frozen.get("rules")
        if not isinstance(frozen_rules, list):
            raise ValueError("pair release must contain a rules list")
        frozen_pair_audit["raw_rules"] = len(frozen_rules)
        for rule in frozen_rules:
            dataset = str(rule.get("dataset", ""))
            disease_a = str(rule.get("disease_a", ""))
            disease_b = str(rule.get("disease_b", ""))
            if dataset not in DATASETS or not disease_a or not disease_b:
                skipped_pair_reasons["invalid_frozen_pair"] += 1
                continue
            if ontology_conflict(disease_a, disease_b):
                skipped_pair_reasons["ontology_conflict"] += 1
                continue
            ordered = sorted((disease_a, disease_b), key=label_key)
            pair_set.add((dataset, ordered[0], ordered[1]))
        frozen_pair_audit["unique_eligible_pairs"] = len(pair_set)

    base_rows = q100_controls(root)
    role_counts = Counter()
    with (output / "base_q100_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in base_rows:
            sid = sample_id(row)
            role = split_role(sid, args.seed)
            role_counts[(row["dataset"], role)] += 1
            value = {**row, "sample_id": sid, "split_role": role, "candidate_probabilities": probabilities_from_prompt(row["prompt"])}
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")

    proposal_records = []
    selection_role_counts = Counter()
    for global_index, (dataset, disease_a, disease_b) in enumerate(sorted(pair_set, key=lambda row: (row[0], label_key(row[1]), label_key(row[2]))), 300000):
        prepared_a = generator[dataset].get(disease_a)
        prepared_b = generator[dataset].get(disease_b)
        if prepared_a is None or prepared_b is None:
            skipped_pair_reasons["missing_prepared_class"] += 1
            continue
        cases_a = select_pair_sources(prepared_a, prepared_b["centroid"], args.source_cases_per_class)
        cases_b = select_pair_sources(prepared_b, prepared_a["centroid"], args.source_cases_per_class)
        if len(cases_a) < args.minimum_source_cases or len(cases_b) < args.minimum_source_cases:
            skipped_pair_reasons["insufficient_balanced_sources"] += 1
            continue
        paths_a = {row["image_path"] for row in cases_a if row.get("image_path")}
        paths_b = {row["image_path"] for row in cases_b if row.get("image_path")}
        if paths_a & paths_b:
            skipped_pair_reasons["cross_label_duplicate_path"] += 1
            continue
        vectors_a = {row["case_id"]: prepared_a["matrix"][index] for index, row in enumerate(prepared_a["rows"])}
        vectors_b = {row["case_id"]: prepared_b["matrix"][index] for index, row in enumerate(prepared_b["rows"])}
        selected_a = np.asarray([vectors_a[row["case_id"]] for row in cases_a], dtype=np.float32)
        selected_b = np.asarray([vectors_b[row["case_id"]] for row in cases_b], dtype=np.float32)
        maximum_cross_similarity = float((selected_a @ selected_b.T).max())
        if maximum_cross_similarity >= 0.9995:
            skipped_pair_reasons["cross_label_near_duplicate_embedding"] += 1
            continue
        for row in cases_a:
            selection_role_counts[f"{dataset}::A::{row['selection_role']}"] += 1
        for row in cases_b:
            selection_role_counts[f"{dataset}::B::{row['selection_role']}"] += 1
        key = rule_key(dataset, disease_a, disease_b)
        selection_payload = {
            "schema_version": "v1.1",
            "key": key,
            "selection_protocol": "representative_cluster_medoids+diverse_mmr+quality_screened_positive_margin_embedding_hard",
            "a": [{"case_id": row["case_id"], "role": row["selection_role"]} for row in cases_a],
            "b": [{"case_id": row["case_id"], "role": row["selection_role"]} for row in cases_b],
        }
        rule_id = "cr11_" + canonical_hash(selection_payload)[:16]
        source_cases_a = [{"case_id": row["case_id"], "findings": row["findings"]} for row in cases_a]
        source_cases_b = [{"case_id": row["case_id"], "findings": row["findings"]} for row in cases_b]
        source_selection_a = [
            {
                **{key2: value for key2, value in row.items() if key2 not in {"findings", "image_path"}},
                "findings_sha256": canonical_hash(row["findings"]),
                "image_path_sha256": canonical_hash(row["image_path"]),
            }
            for row in cases_a
        ]
        source_selection_b = [
            {
                **{key2: value for key2, value in row.items() if key2 not in {"findings", "image_path"}},
                "findings_sha256": canonical_hash(row["findings"]),
                "image_path_sha256": canonical_hash(row["image_path"]),
            }
            for row in cases_b
        ]
        proposal_records.append({
            "global_index": global_index,
            "record_type": "rule_proposal",
            "schema_version": "v1.1",
            "rule_id": rule_id,
            "rule_key": key,
            "dataset": dataset,
            "disease_a": disease_a,
            "disease_b": disease_b,
            "source_case_ids_a": [row["case_id"] for row in cases_a],
            "source_case_ids_b": [row["case_id"] for row in cases_b],
            "source_cases_a": source_cases_a,
            "source_cases_b": source_cases_b,
            "source_selection_a": source_selection_a,
            "source_selection_b": source_selection_b,
            "minimum_feature_support": args.minimum_feature_support,
            "selection_policy": "representative_cluster_medoids+diverse_mmr+quality_screened_positive_margin_embedding_hard",
            "maximum_cross_label_source_similarity": round(maximum_cross_similarity, 6),
            "source_pool_sha256": canonical_hash(selection_payload),
            "source_stream_sha256": source_hashes[dataset],
            "prompt": rule_prompt(dataset, disease_a, disease_b, cases_a, cases_b),
        })
    manifests = output / "rule_manifests"
    manifests.mkdir(exist_ok=True)
    handles = [(manifests / f"rules_shard_{index}.jsonl").open("w", encoding="utf-8") for index in range(args.shards)]
    try:
        for index, row in enumerate(proposal_records):
            handles[index % args.shards].write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        for handle in handles:
            handle.close()
    audit = {
        "schema_version": "v1.1",
        "protocol": "generator-only immutable cases; row-normalized embeddings; representative/diverse/pair-embedding-hard sources; label-blind deterministic evaluation split",
        "seed": args.seed,
        "generator_fraction": args.generator_fraction,
        "neighbor_pairs": args.neighbor_pairs,
        "pair_source": pair_source,
        "pair_release_sha256": pair_release_sha256,
        "frozen_pair_audit": frozen_pair_audit,
        "frozen_pair_count": len(pair_set) if args.pair_release is not None else None,
        "source_cases_per_class": args.source_cases_per_class,
        "minimum_source_cases": args.minimum_source_cases,
        "minimum_feature_support": args.minimum_feature_support,
        "source_stream_hashes": source_hashes,
        "source_audit": source_audit,
        "selection_role_counts": dict(selection_role_counts),
        "skipped_pair_reasons": dict(skipped_pair_reasons),
        "rule_proposals": len(proposal_records),
        "role_counts": {f"{dataset}::{role}": count for (dataset, role), count in sorted(role_counts.items())},
        "base_records": len(base_rows),
    }
    audit["content_sha256"] = canonical_hash(audit)
    write_json(output / "initialization_audit.json", audit)
    write_baseline_release(output, source_hashes)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def write_baseline_release(root: Path, source_hashes: dict[str, str]) -> dict[str, Any]:
    release = MemoryRelease(
        release_id="v0_c100",
        parent_release_id=None,
        source_stream_hashes=source_hashes,
        rules=(),
        status="active",
        protocol="frozen C100 case-memory baseline with no evolved contrastive rules",
    ).payload()
    path = root / "releases" / "v0_c100.json"
    write_json(path, release)
    write_json(
        root / "releases" / "active_release.json",
        {
            "release_id": release["release_id"],
            "content_sha256": release["content_sha256"],
            "status": "active",
            "path": str(path),
        },
    )
    return release


def create_baseline(args: argparse.Namespace) -> None:
    audit = json.loads((args.root / "initialization_audit.json").read_text(encoding="utf-8"))
    release = write_baseline_release(args.root, audit["source_stream_hashes"])
    print(json.dumps({"release_id": release["release_id"], "content_sha256": release["content_sha256"]}, ensure_ascii=False, indent=2))


def finalize_rules(args: argparse.Namespace) -> None:
    proposal_rows = read_jsonl(sorted(args.manifest_dir.glob("rules_shard_*.jsonl")))
    by_index = {int(row["global_index"]): row for row in proposal_rows}
    outputs = read_jsonl([Path(path) for pattern in args.results for path in glob.glob(pattern)])
    latest = {int(row["global_index"]): row for row in outputs if row.get("record_type") == "inference"}
    rules = []
    issues = []
    abstentions = []
    for index, proposal in sorted(by_index.items()):
        result = latest.get(index)
        if result is None:
            issues.append({"global_index": index, "rule_id": proposal["rule_id"], "reason": "missing_result"})
            continue
        rule, error = strict_rule_from_response(proposal, result.get("response", ""))
        if error == "evidence_abstained":
            abstentions.append({"global_index": index, "rule_id": proposal["rule_id"], "reason": error})
            continue
        if error:
            issues.append({"global_index": index, "rule_id": proposal["rule_id"], "reason": error})
            continue
        rules.append(rule)
    initialization = json.loads((args.root / "initialization_audit.json").read_text(encoding="utf-8"))
    source_hashes = initialization["source_stream_hashes"]
    is_v11 = initialization.get("schema_version") == "v1.1"
    release_id = "v1_1_candidate" if is_v11 else "v1_candidate"
    release = MemoryRelease(
        release_id=release_id,
        parent_release_id=None,
        source_stream_hashes=source_hashes,
        rules=tuple(rules),
        status="candidate",
        protocol=(
            "v1.1 evidence-quoted contrastive rules from representative/diverse/pair-embedding-hard immutable training cases"
            if is_v11
            else "strict structured contrastive rules grounded in immutable generator-only training cases"
        ),
    ).payload()
    write_json(args.root / "releases" / f"{release_id}.json", release)
    audit = {
        "proposals": len(proposal_rows),
        "result_attempts": len([row for row in outputs if row.get("record_type") == "inference"]),
        "unique_results": len(latest),
        "retry_attempts": max(0, len([row for row in outputs if row.get("record_type") == "inference"]) - len(latest)),
        "valid_rules": len(rules),
        "evidence_abstentions": len(abstentions),
        "abstentions": abstentions,
        "invalid_or_missing": len(issues),
        "issues": issues,
        "release_sha256": release["content_sha256"],
        "status": "PASS" if rules and not issues else "FAIL",
    }
    write_json(args.root / "rule_parse_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def build_rule_retry(args: argparse.Namespace) -> None:
    proposal_rows = read_jsonl(sorted(args.manifest_dir.glob("rules_shard_*.jsonl")))
    by_index = {int(row["global_index"]): row for row in proposal_rows}
    outputs = read_jsonl([Path(path) for pattern in args.results for path in glob.glob(pattern)])
    latest = {int(row["global_index"]): row for row in outputs if row.get("record_type") == "inference"}
    retry_rows = []
    reasons = Counter()
    for index, proposal in sorted(by_index.items()):
        result = latest.get(index)
        if result is None:
            reason = "missing_result"
        else:
            _, reason = strict_rule_from_response(proposal, result.get("response", ""))
        if reason in (None, "evidence_abstained"):
            continue
        reasons[reason] += 1
        previous_response = "" if result is None else str(result.get("response", ""))
        correction = (
            f"\n\nTHE PREVIOUS RESPONSE WAS REJECTED FOR: {reason}. "
            "Return one corrected, complete JSON object and no other text. "
            "Each finding must appear in exactly one of supports_a, supports_b, or nondiscriminative; "
            "shared findings belong only to nondiscriminative. "
            "For v1.1, every supports feature needs at least three valid exact-quote citations, and executable trigger terms must satisfy the source-frequency constraints. "
            f"source_case_ids_a must contain only these IDs: {json.dumps(proposal['source_case_ids_a'], ensure_ascii=False)}. "
            f"source_case_ids_b must contain only these IDs: {json.dumps(proposal['source_case_ids_b'], ensure_ascii=False)}. "
            f"Correct this rejected response without repeating its defect:\n{previous_response}"
        )
        retry_rows.append({**proposal, "prompt": proposal["prompt"] + correction, "retry_reason": reason})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in retry_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = {
        "proposals": len(proposal_rows),
        "latest_results": len(latest),
        "retry_records": len(retry_rows),
        "reasons": dict(reasons),
        "source_results": args.results,
        "status": "PASS" if retry_rows else "NO_RETRY_NEEDED",
    }
    write_json(args.output.with_suffix(args.output.suffix + ".audit.json"), audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def repair_rule_overlaps(args: argparse.Namespace) -> None:
    proposals = {
        int(row["global_index"]): row
        for row in read_jsonl([args.manifest])
    }
    outputs = read_jsonl([Path(path) for pattern in args.results for path in glob.glob(pattern)])
    latest = {int(row["global_index"]): row for row in outputs if row.get("record_type") == "inference"}
    repaired_rows = []
    issues = []
    overlap_counts = Counter()
    for index, proposal in sorted(proposals.items()):
        result = latest.get(index)
        if result is None:
            issues.append({"global_index": index, "reason": "missing_result"})
            continue
        try:
            parsed = json.loads(str(result.get("response", "")).strip())
        except json.JSONDecodeError:
            issues.append({"global_index": index, "reason": "invalid_json"})
            continue
        normalized, overlaps = normalize_overlapping_features(parsed)
        if not overlaps:
            issues.append({"global_index": index, "reason": "no_overlap_to_repair"})
            continue
        for value in overlaps:
            overlap_counts[value] += 1
        response = json.dumps(normalized, ensure_ascii=False, indent=2)
        _, error = strict_rule_from_response(proposal, response)
        if error:
            issues.append({"global_index": index, "reason": error, "overlaps": overlaps})
            continue
        repaired_rows.append({
            **result,
            "response": response,
            "repair_type": "deterministic_overlap_to_nondiscriminative",
            "repair_overlaps": overlaps,
            "original_response_sha256": canonical_hash(str(result.get("response", ""))),
            "repaired_response_sha256": canonical_hash(response),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"record_type": "deterministic_repair_run", "records": len(proposals)}, ensure_ascii=False) + "\n")
        for row in repaired_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = {
        "requested": len(proposals),
        "repaired": len(repaired_rows),
        "issues": issues,
        "normalized_overlap_counts": dict(overlap_counts),
        "status": "PASS" if len(repaired_rows) == len(proposals) and not issues else "FAIL",
    }
    write_json(args.output.with_suffix(args.output.suffix + ".audit.json"), audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def build_diagnosis_retry(args: argparse.Namespace) -> None:
    manifests = read_jsonl([Path(path) for pattern in args.manifests for path in glob.glob(pattern)])
    by_index = {int(row["global_index"]): row for row in manifests}
    outputs = read_jsonl([Path(path) for pattern in args.results for path in glob.glob(pattern)])
    latest = {int(row["global_index"]): row for row in outputs if row.get("record_type") == "inference"}
    retry_rows = []
    reasons = Counter()
    for index, manifest in sorted(by_index.items()):
        result = latest.get(index)
        labels = [str(item[0]) for item in manifest.get("candidate_probabilities", [])]
        if result is None:
            reason = "missing_result"
            previous_response = ""
        else:
            previous_response = str(result.get("response", ""))
            reason = None if parse_primary_diagnosis(previous_response, labels) else "invalid_primary_diagnosis"
        if reason is None:
            continue
        reasons[reason] += 1
        correction = (
            f"\n\nFORMAT-ONLY CORRECTION: The previous response was rejected for {reason}. "
            "Return one complete JSON object and no other text. "
            f"PrimaryDiagnosis must exactly equal one of these closed-set labels: {json.dumps(labels, ensure_ascii=False)}. "
            f"Do not repeat this rejected response:\n{previous_response}"
        )
        retry_rows.append({**manifest, "prompt": manifest["prompt"] + correction, "retry_reason": reason})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in retry_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = {
        "manifest_records": len(manifests),
        "latest_results": len(latest),
        "retry_records": len(retry_rows),
        "reasons": dict(reasons),
        "status": "RETRY_REQUIRED" if retry_rows else "PASS",
    }
    write_json(args.output.with_suffix(args.output.suffix + ".audit.json"), audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def load_release(path: Path, statuses: set[str]) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    release_id = payload["release_id"]
    rules = {}
    for row in payload["rules"]:
        if row.get("status") in statuses:
            value = {**row, "release_id": release_id}
            rules[row["rule_key"]] = value
    return rules


def build_cohort(args: argparse.Namespace) -> None:
    release_payload = json.loads(args.release.read_text(encoding="utf-8"))
    rules = load_release(args.release, set(args.statuses))
    c100 = load_c100(args.predictions) if args.predictions else {}
    rows = read_jsonl([args.root / "base_q100_records.jsonl"])
    selected = [row for row in rows if row["split_role"] == args.cohort]
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = [(output_dir / f"{args.cohort}_shard_{index}.jsonl").open("w", encoding="utf-8") for index in range(args.shards)]
    included = 0
    missing = 0
    pair_matched = 0
    reason_counts = Counter()
    try:
        for row in selected:
            if args.maximum_records and included >= args.maximum_records:
                reason_counts["maximum_records_reached"] += 1
                continue
            probabilities = [(str(item[0]), float(item[1])) for item in row["candidate_probabilities"]]
            if len(probabilities) < 2:
                missing += 1
                reason_counts["missing_probabilities"] += 1
                continue
            key = rule_key(row["dataset"], probabilities[0][0], probabilities[1][0])
            rule = rules.get(key)
            if rule is None:
                missing += 1
                reason_counts["no_rule_for_pair"] += 1
                continue
            pair_matched += 1
            sid = row["sample_id"]
            baseline_prediction = c100.get(sid, {}).get("prediction", "")
            if rule.get("schema_version") == "v1.1":
                pair_labels = {
                    evaluation_label(row["dataset"], rule["disease_a"]),
                    evaluation_label(row["dataset"], rule["disease_b"]),
                }
                if not baseline_prediction:
                    missing += 1
                    reason_counts["missing_c100_anchor"] += 1
                    continue
                if evaluation_label(row["dataset"], baseline_prediction) not in pair_labels:
                    missing += 1
                    reason_counts["c100_outside_rule_pair"] += 1
                    continue
                applicable, applicability_reason, trigger_matches = rule_applicability(
                    row["prompt"],
                    rule,
                    minimum_trigger_matches=args.minimum_trigger_matches,
                )
                if not applicable:
                    missing += 1
                    reason_counts[applicability_reason] += 1
                    continue
            else:
                applicability_reason = "legacy_pair_match"
                trigger_matches = {"a": 0, "b": 0}
            global_index = args.global_index_offset + included
            prompt = inject_rule(row["prompt"], rule, baseline_prediction=baseline_prediction)
            record = {
                **{key2: value for key2, value in row.items() if key2 != "prompt"},
                "global_index": global_index,
                "source_control_global_index": row["global_index"],
                "condition": args.condition,
                "arm": "structured_evolving_memory_v1_1" if rule.get("schema_version") == "v1.1" else "structured_evolving_memory_v1",
                "guideline_diseases": [probabilities[0][0], probabilities[1][0]],
                "memory_release_id": rule["release_id"],
                "memory_release_sha256": release_payload["content_sha256"],
                "rule_id": rule["rule_id"],
                "rule_key": key,
                "rule_disease_a": rule["disease_a"],
                "rule_disease_b": rule["disease_b"],
                "rule_reliability": float(rule.get("reliability", 0.5)),
                "c100_anchor_prediction": baseline_prediction,
                "applicability_reason": applicability_reason,
                "trigger_matches": trigger_matches,
                "prompt_sha256": canonical_hash(prompt),
                "prompt": prompt,
            }
            handles[included % args.shards].write(json.dumps(record, ensure_ascii=False) + "\n")
            included += 1
    finally:
        for handle in handles:
            handle.close()
    audit = {
        "cohort": args.cohort,
        "cohort_records": len(selected),
        "pair_matched": pair_matched,
        "included_with_rule": included,
        "fallback_to_c100": missing,
        "coverage": included / len(selected) if selected else 0.0,
        "reason_counts": dict(reason_counts),
        "release": str(args.release),
        "release_sha256": release_payload["content_sha256"],
        "status": "PASS" if included else "FAIL",
    }
    write_json(output_dir / "manifest_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def load_c100(predictions: Path) -> dict[str, dict[str, Any]]:
    result = {}
    with predictions.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] == "C100":
                result[row["sample_id"]] = row
    return result


def load_evolved_results(patterns: list[str]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in read_jsonl([Path(path) for pattern in patterns for path in glob.glob(pattern)]):
        if row.get("record_type") == "inference":
            sid = row["sample_id"]
            if sid in result:
                raise ValueError(f"duplicate evolved result for sample_id {sid}")
            result[sid] = row
    return result


def parsed_e_prediction(row: dict[str, Any]) -> str:
    labels = [str(item[0]) for item in row.get("candidate_probabilities", [])]
    return parse_primary_diagnosis(row.get("response", ""), labels)


def pair_constrained_prediction(row: dict[str, Any], c_prediction: str) -> tuple[str, str]:
    proposal = parsed_e_prediction(row)
    if not proposal:
        return c_prediction, "invalid_output_fallback"
    if row.get("arm") != "structured_evolving_memory_v1_1":
        return proposal, "legacy_output"
    pair = {
        evaluation_label(row["dataset"], row.get("rule_disease_a", "")),
        evaluation_label(row["dataset"], row.get("rule_disease_b", "")),
    }
    if evaluation_label(row["dataset"], proposal) not in pair:
        return c_prediction, "third_class_fallback"
    return proposal, "pair_constrained_output"


def validate_rules(args: argparse.Namespace) -> None:
    candidate = json.loads(args.candidate_release.read_text(encoding="utf-8"))
    rules_by_id = {row["rule_id"]: row for row in candidate["rules"]}
    c100 = load_c100(args.predictions)
    evolved = load_evolved_results(args.results)
    acceptance_ids = {
        row["sample_id"]
        for row in read_jsonl([args.root / "base_q100_records.jsonl"])
        if row["split_role"] == "rule_acceptance"
    }
    unexpected_ids = sorted(set(evolved) - acceptance_ids)
    if unexpected_ids:
        raise ValueError(f"validation results contain {len(unexpected_ids)} non-acceptance sample IDs")
    stats: dict[str, Counter] = defaultdict(Counter)
    for sid, row in evolved.items():
        if sid not in c100:
            raise ValueError(f"missing C100 baseline for validation sample {sid}")
        c = c100[sid]
        rule_id = row["rule_id"]
        if rule_id not in rules_by_id:
            raise ValueError(f"unknown rule_id in validation output: {rule_id}")
        stats[rule_id]["exposures"] += 1
        raw_prediction = parsed_e_prediction(row)
        if not raw_prediction:
            stats[rule_id]["invalid"] += 1
            continue
        c_prediction = c["prediction"]
        e_prediction, constraint_reason = pair_constrained_prediction(row, c_prediction)
        if constraint_reason == "third_class_fallback":
            stats[rule_id]["third_class_fallbacks"] += 1
        c_correct = int(c["correct"])
        e_correct = int(evaluation_label(row["dataset"], e_prediction) == evaluation_label(row["dataset"], row["ground_truth"]))
        changed = evaluation_label(row["dataset"], c_prediction) != evaluation_label(row["dataset"], e_prediction)
        if changed:
            stats[rule_id]["changes"] += 1
        if c_correct == 0 and e_correct == 1:
            stats[rule_id]["gains"] += 1
            probabilities = [(str(item[0]), float(item[1])) for item in row["candidate_probabilities"]]
            if probabilities and evaluation_label(row["dataset"], e_prediction) == evaluation_label(row["dataset"], probabilities[0][0]):
                stats[rule_id]["restore_top1_gains"] += 1
            else:
                stats[rule_id]["beyond_top1_gains"] += 1
        if c_correct == 1 and e_correct == 0:
            stats[rule_id]["regressions"] += 1
            probabilities = [(str(item[0]), float(item[1])) for item in row["candidate_probabilities"]]
            if probabilities[0][1] >= 0.70 and probabilities[0][1] - probabilities[1][1] >= 0.20:
                stats[rule_id]["high_confidence_regressions"] += 1
        if changed and c_correct == 0 and e_correct == 0:
            stats[rule_id]["changed_wrong"] += 1

    updated = []
    status_counts = Counter()
    is_v11 = candidate.get("release_id") == "v1_1_candidate"
    for rule_id, row in rules_by_id.items():
        item = stats.get(rule_id, Counter())
        exposures = int(item["exposures"])
        changes = int(item["changes"])
        gains = int(item["gains"])
        regressions = int(item["regressions"])
        discordances = gains + regressions
        one_sided_p = (
            sum(math.comb(discordances, index) for index in range(gains, discordances + 1)) / 2**discordances
            if discordances else 1.0
        )
        if is_v11:
            enough_information = exposures >= args.minimum_exposures and changes >= args.minimum_changes
            safe = (
                regressions <= args.maximum_regressions
                and item["high_confidence_regressions"] == 0
                and item["third_class_fallbacks"] == 0
                and item["invalid"] == 0
            )
            beneficial = (
                gains >= args.minimum_gains
                and gains - regressions >= args.minimum_net_gain
                and one_sided_p < args.maximum_one_sided_p
            )
            if not safe:
                status = "rejected"
            elif enough_information and beneficial:
                status = "active"
            elif not enough_information:
                status = "provisional"
            else:
                status = "rejected"
        else:
            if changes >= args.minimum_changes and gains >= regressions + args.minimum_net_gain and item["high_confidence_regressions"] == 0:
                status = "active"
            elif changes < args.minimum_changes:
                status = "provisional"
            else:
                status = "rejected"
        reliability = (gains + 1.0) / (exposures + 2.0) if is_v11 else (gains + 1.0) / (gains + regressions + 2.0)
        change_precision = (gains + 1.0) / (gains + regressions + int(item["changed_wrong"]) + 2.0)
        updated_row = {
            **row,
            "status": status,
            "validation_exposures": exposures,
            "validation_changes": changes,
            "validation_gains": gains,
            "validation_regressions": regressions,
            "validation_changed_wrong": int(item["changed_wrong"]),
            "validation_invalid": int(item["invalid"]),
            "validation_third_class_fallbacks": int(item["third_class_fallbacks"]),
            "validation_restore_top1_gains": int(item["restore_top1_gains"]),
            "validation_beyond_top1_gains": int(item["beyond_top1_gains"]),
            "validation_high_confidence_regressions": int(item["high_confidence_regressions"]),
            "validation_one_sided_exact_p": one_sided_p,
            "validation_effect": (gains - regressions) / exposures if exposures else 0.0,
            "validation_change_precision": change_precision,
            "reliability": reliability,
        }
        updated.append(updated_row)
        status_counts[status] += 1
    payload = {
        "release_id": "v1_1_active" if is_v11 else "v1_active",
        "parent_release_id": candidate["release_id"],
        "source_stream_hashes": candidate["source_stream_hashes"],
        "rules": updated,
        "status": "active" if status_counts["active"] else "rolled_back",
        "protocol": "candidate rules accepted only on deterministic rule_acceptance split; no heldout labels used",
        "validation_policy": {
            "minimum_exposures": args.minimum_exposures,
            "minimum_changes": args.minimum_changes,
            "minimum_gains": args.minimum_gains,
            "minimum_net_gain": args.minimum_net_gain,
            "maximum_regressions": args.maximum_regressions,
            "maximum_one_sided_p": args.maximum_one_sided_p,
            "forbid_high_confidence_regression": True,
            "forbid_third_class_output": is_v11,
            "reliability_denominator": "all_applicable_exposures" if is_v11 else "correctness_flips",
        },
    }
    payload["content_sha256"] = canonical_hash(payload)
    write_json(args.output, payload)
    if status_counts["active"]:
        pointer = {"release_id": payload["release_id"], "content_sha256": payload["content_sha256"], "status": "active", "path": str(args.output)}
    else:
        baseline_path = args.output.parent / "v0_c100.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        pointer = {"release_id": baseline["release_id"], "content_sha256": baseline["content_sha256"], "status": "rolled_back", "path": str(baseline_path)}
    write_json(args.output.parent / "active_release.json", pointer)
    audit = {
        "acceptance_ids": len(acceptance_ids),
        "evaluated_ids": len(evolved),
        "unexpected_ids": unexpected_ids,
        "status_counts": dict(status_counts),
        "rule_stats": {key: dict(value) for key, value in stats.items()},
        "release_status": payload["status"],
        "release_sha256": payload["content_sha256"],
    }
    write_json(args.output.parent / "validation_audit.json", audit)
    print(json.dumps({"status_counts": dict(status_counts), "release_status": payload["status"], "release_sha256": payload["content_sha256"]}, ensure_ascii=False, indent=2))


def policy_rows(c100: dict[str, dict[str, Any]], evolved: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for sid, e in evolved.items():
        c = c100[sid]
        e_prediction, constraint_reason = pair_constrained_prediction(e, c["prediction"])
        probabilities = [(str(item[0]), float(item[1])) for item in e["candidate_probabilities"]]
        c_correct = int(c["correct"])
        e_correct = int(evaluation_label(e["dataset"], e_prediction) == evaluation_label(e["dataset"], e["ground_truth"]))
        top1_prediction = probabilities[0][0] if probabilities else c["prediction"]
        top1_correct = int(evaluation_label(e["dataset"], top1_prediction) == evaluation_label(e["dataset"], e["ground_truth"]))
        rows.append({
            "sample_id": sid,
            "dataset": e["dataset"],
            "c_prediction": c["prediction"],
            "e_prediction": e_prediction,
            "c_correct": c_correct,
            "e_correct": e_correct,
            "top1_prediction": top1_prediction,
            "top1_correct": top1_correct,
            "constraint_reason": constraint_reason,
            "probabilities": probabilities,
            "reliability": float(e.get("rule_reliability", 0.5)),
        })
    return rows


def tune_gate(args: argparse.Namespace) -> None:
    c100 = load_c100(args.predictions)
    evolved = load_evolved_results(args.results)
    cohort_sids = {row["sample_id"] for row in read_jsonl([args.root / "base_q100_records.jsonl"]) if row["split_role"] == "gate_development"}
    unexpected_ids = sorted(set(evolved) - cohort_sids)
    if unexpected_ids:
        raise ValueError(f"gate results contain {len(unexpected_ids)} non-gate sample IDs")
    missing_c100 = sorted(set(evolved) - set(c100))
    if missing_c100:
        raise ValueError(f"gate results have {len(missing_c100)} samples without C100 baselines")
    rows = policy_rows(c100, evolved)
    dataset_applicable = Counter(row["dataset"] for row in rows)
    baseline_correct = sum(int(c100[sid]["correct"]) for sid in cohort_sids)
    release_payload = json.loads(args.release.read_text(encoding="utf-8"))
    is_v11 = str(release_payload.get("release_id", "")).startswith("v1_1")
    if is_v11:
        # Three pre-registered policies only. This avoids selecting from a broad
        # threshold grid after observing gate outcomes.
        policy_specs = (
            (False, 0.40, 0.15, 0.10),
            (False, 0.50, 0.25, 0.10),
            (True, 0.50, 0.25, 0.10),
        )
    else:
        policy_specs = tuple(
            (allow_non_top, probability_cap, margin_cap, reliability)
            for allow_non_top in (False, True)
            for probability_cap in (0.40, 0.50, 0.60, 0.70, 0.80)
            for margin_cap in (0.15, 0.25, 0.40, 0.60)
            for reliability in (0.50, 0.60, 0.67)
        )
    raw_gains = sum(row["c_correct"] == 0 and row["e_correct"] == 1 for row in rows)
    raw_regressions = sum(row["c_correct"] == 1 and row["e_correct"] == 0 for row in rows)
    candidates = []
    policy_audits = []
    for allow_non_top, probability_cap, margin_cap, reliability in policy_specs:
        policy = GatePolicy(
            policy_id="candidate",
            allow_when_c_not_top1=allow_non_top,
            top1_probability_cap=probability_cap,
            top1_margin_cap=margin_cap,
            minimum_rule_reliability=reliability,
        )
        correct = baseline_correct
        overrides = gains = regressions = high_conf_regressions = beyond_top1_gains = 0
        dataset_delta = Counter()
        for row in rows:
            allow, _ = policy.allow(c_prediction=row["c_prediction"], evolved_prediction=row["e_prediction"], candidate_probabilities=row["probabilities"], rule_reliability=row["reliability"])
            if not allow:
                continue
            overrides += 1
            correct += row["e_correct"] - row["c_correct"]
            dataset_delta[row["dataset"]] += row["e_correct"] - row["c_correct"]
            if row["c_correct"] == 0 and row["e_correct"] == 1:
                gains += 1
                if not row["top1_correct"]:
                    beyond_top1_gains += 1
            if row["c_correct"] == 1 and row["e_correct"] == 0:
                regressions += 1
                p = row["probabilities"]
                if p[0][1] >= 0.70 and p[0][1] - p[1][1] >= 0.20:
                    high_conf_regressions += 1
        delta = correct - baseline_correct
        discordances = gains + regressions
        one_sided_p = (
            sum(math.comb(discordances, index) for index in range(gains, discordances + 1)) / 2**discordances
            if discordances else 1.0
        )
        unsafe_dataset_drop = any(
            dataset_applicable[dataset] >= 20 and dataset_delta[dataset] < 0
            for dataset in dataset_applicable
        )
        retains_raw_gains = gains >= math.ceil(0.80 * raw_gains) if raw_gains else True
        removes_raw_regressions = regressions <= math.floor(0.50 * raw_regressions) if raw_regressions else True
        passes = (
            len(rows) >= args.minimum_applicable
            and overrides >= args.minimum_overrides
            and discordances >= args.minimum_discordances
            and delta >= args.minimum_net_gain
            and gains > regressions
            and one_sided_p < 0.10
            and high_conf_regressions <= args.maximum_high_confidence_regressions
            and not unsafe_dataset_drop
            and beyond_top1_gains >= args.minimum_beyond_top1_gains
            and retains_raw_gains
            and removes_raw_regressions
        )
        item = {
            "policy": policy.to_dict(),
            "net_correct": delta,
            "gains": gains,
            "regressions": regressions,
            "beyond_top1_gains": beyond_top1_gains,
            "overrides": overrides,
            "discordances": discordances,
            "one_sided_exact_p": one_sided_p,
            "high_confidence_regressions": high_conf_regressions,
            "dataset_net": dict(dataset_delta),
            "retains_80_percent_raw_gains": retains_raw_gains,
            "removes_50_percent_raw_regressions": removes_raw_regressions,
            "passes": passes,
        }
        policy_audits.append(item)
        if passes:
            candidates.append(item)
    if candidates:
        best = max(candidates, key=lambda item: (item["net_correct"], -item["regressions"], -item["overrides"]))
        policy = GatePolicy(
            policy_id="gate_v1_1" if is_v11 else "gate_v1",
            allow_when_c_not_top1=bool(best["policy"]["allow_when_c_not_top1"]),
            top1_probability_cap=float(best["policy"]["top1_probability_cap"]),
            top1_margin_cap=float(best["policy"]["top1_margin_cap"]),
            minimum_rule_reliability=float(best["policy"]["minimum_rule_reliability"]),
        )
        audit = {"selected": policy.to_dict(), **{key: value for key, value in best.items() if key != "policy"}, "candidate_policies_passing": len(candidates), "fallback": False}
    else:
        policy = GatePolicy("gate_v1_1_fallback" if is_v11 else "gate_v1_fallback", False, 0.0, 0.0, 1.01)
        audit = {"selected": policy.to_dict(), "net_correct": 0, "gains": 0, "regressions": 0, "overrides": 0, "candidate_policies_passing": 0, "fallback": True, "status": "NO_APPLICABLE_RESULTS" if not rows else "NO_SAFE_POSITIVE_POLICY"}
    audit["unexpected_ids"] = unexpected_ids
    audit["applicable_cases"] = len(rows)
    audit["release_id"] = release_payload.get("release_id")
    audit["pre_registered_policy_audits"] = policy_audits
    audit["raw_pair_constrained"] = {"gains": raw_gains, "regressions": raw_regressions, "net_correct": raw_gains - raw_regressions}
    top1_gains = sum(row["c_correct"] == 0 and row["top1_correct"] == 1 for row in rows)
    top1_regressions = sum(row["c_correct"] == 1 and row["top1_correct"] == 0 for row in rows)
    audit["top1_only_same_applicable_cases"] = {
        "cases": len(rows),
        "gains": top1_gains,
        "regressions": top1_regressions,
        "net_correct": top1_gains - top1_regressions,
    }
    payload = {**policy.to_dict(), "development_audit": audit, "content_sha256": canonical_hash(audit)}
    write_json(args.output, payload)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def exact_mcnemar(a_only: int, b_only: int) -> float:
    total = a_only + b_only
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, index) for index in range(min(a_only, b_only) + 1)) / 2**total
    return min(1.0, 2.0 * tail)


def summarize(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    values = [int(row[f"correct_{condition}"]) for row in rows]
    return {"condition": condition, "samples": len(values), "correct": sum(values), "accuracy": sum(values) / len(values) if values else 0.0}


def finalize_evaluation(args: argparse.Namespace) -> None:
    if args.bootstraps <= 0:
        raise ValueError("bootstraps must be positive")
    c100 = load_c100(args.predictions)
    evolved = load_evolved_results(args.results)
    raw_e = {}
    with args.predictions.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] == "E100":
                raw_e[row["sample_id"]] = row
    gate_payload = json.loads(args.gate.read_text(encoding="utf-8"))
    policy = GatePolicy(
        policy_id=gate_payload["policy_id"],
        allow_when_c_not_top1=bool(gate_payload["allow_when_c_not_top1"]),
        top1_probability_cap=float(gate_payload["top1_probability_cap"]),
        top1_margin_cap=float(gate_payload["top1_margin_cap"]),
        minimum_rule_reliability=float(gate_payload["minimum_rule_reliability"]),
    )
    base = [row for row in read_jsonl([args.root / "base_q100_records.jsonl"]) if row["split_role"] == "heldout_evaluation"]
    base_ids = [row["sample_id"] for row in base]
    if len(base_ids) != len(set(base_ids)):
        raise ValueError("duplicate heldout sample IDs")
    missing_c100 = sorted(set(base_ids) - set(c100))
    missing_raw = sorted(set(base_ids) - set(raw_e))
    if missing_c100 or missing_raw:
        raise ValueError(f"incomplete frozen baselines: missing C100={len(missing_c100)}, RawE100={len(missing_raw)}")
    extra_result_ids = sorted(set(evolved) - set(base_ids))
    if extra_result_ids:
        raise ValueError(f"final results contain {len(extra_result_ids)} non-heldout sample IDs")
    mismatched_release = [
        sid for sid, row in evolved.items()
        if str(row.get("memory_release_sha256", "")) != str(args.release_sha256)
    ]
    if mismatched_release:
        raise ValueError(f"final results contain {len(mismatched_release)} release hash mismatches")
    rows = []
    for item in base:
        sid = item["sample_id"]
        c = c100[sid]
        raw = raw_e[sid]
        e = evolved.get(sid)
        c_correct = int(c["correct"])
        raw_correct = int(raw["correct"])
        structured_prediction = c["prediction"]
        constraint_reason = "no_active_rule"
        allow = False
        reason = "no_active_rule"
        rule_id = ""
        raw_proposal = ""
        probabilities = [(str(value[0]), float(value[1])) for value in item.get("candidate_probabilities", [])]
        top1_prediction = probabilities[0][0] if e is not None and probabilities else c["prediction"]
        if e is not None:
            raw_proposal = parsed_e_prediction(e)
            structured_prediction, constraint_reason = pair_constrained_prediction(e, c["prediction"])
            probabilities = [(str(value[0]), float(value[1])) for value in e["candidate_probabilities"]]
            top1_prediction = probabilities[0][0] if probabilities else c["prediction"]
            allow, reason = policy.allow(
                c_prediction=c["prediction"],
                evolved_prediction=structured_prediction,
                candidate_probabilities=probabilities,
                rule_reliability=float(e.get("rule_reliability", 0.5)),
            )
            rule_id = e["rule_id"]
        structured_correct = int(evaluation_label(item["dataset"], structured_prediction) == evaluation_label(item["dataset"], item["ground_truth"]))
        top1_correct = int(evaluation_label(item["dataset"], top1_prediction) == evaluation_label(item["dataset"], item["ground_truth"]))
        gated_prediction = structured_prediction if allow else c["prediction"]
        gated_correct = int(evaluation_label(item["dataset"], gated_prediction) == evaluation_label(item["dataset"], item["ground_truth"]))
        top1_gate_prediction = top1_prediction if allow else c["prediction"]
        top1_gate_correct = int(evaluation_label(item["dataset"], top1_gate_prediction) == evaluation_label(item["dataset"], item["ground_truth"]))
        top1_probability = probabilities[0][1] if probabilities else 0.0
        top1_margin = probabilities[0][1] - probabilities[1][1] if len(probabilities) >= 2 else 0.0
        final_gain_type = "none"
        if c_correct == 0 and gated_correct == 1:
            final_gain_type = "restore_top1" if top1_correct else "beyond_top1"
        rows.append({
            "sample_id": sid,
            "dataset": item["dataset"],
            "relative_path": item["relative_path"],
            "ground_truth": item["ground_truth"],
            "prediction_C100": c["prediction"],
            "prediction_RawE100": raw["prediction"],
            "prediction_Top1Applicable": top1_prediction,
            "prediction_V1_1RawPairConstrained": structured_prediction,
            "prediction_Top1GateMatched": top1_gate_prediction,
            "prediction_V1_1FinalGated": gated_prediction,
            "correct_C100": c_correct,
            "correct_RawE100": raw_correct,
            "correct_Top1Applicable": top1_correct,
            "correct_V1_1RawPairConstrained": structured_correct,
            "correct_Top1GateMatched": top1_gate_correct,
            "correct_V1_1FinalGated": gated_correct,
            "memory_applicable": int(e is not None),
            "override_allowed": int(allow),
            "gate_reason": reason,
            "constraint_reason": constraint_reason,
            "raw_proposal": raw_proposal,
            "candidate_top1": probabilities[0][0] if probabilities else "",
            "top1_probability": top1_probability,
            "top1_margin": top1_margin,
            "final_gain_type": final_gain_type,
            "rule_id": rule_id,
            "rule_key": e.get("rule_key", "") if e else "",
            "rule_disease_a": e.get("rule_disease_a", "") if e else "",
            "rule_disease_b": e.get("rule_disease_b", "") if e else "",
            "rule_reliability": float(e.get("rule_reliability", 0.0)) if e else 0.0,
            "applicability_reason": e.get("applicability_reason", "") if e else "",
            "trigger_matches": json.dumps(e.get("trigger_matches", {}), ensure_ascii=False, sort_keys=True) if e else "{}",
            "memory_release_id": e.get("memory_release_id", "") if e else "",
            "memory_release_sha256": e.get("memory_release_sha256", "") if e else "",
            "gate_policy_id": gate_payload["policy_id"],
            "gate_sha256": gate_payload["content_sha256"],
        })
    args.output.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("heldout evaluation contains no records")
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = []
    for dataset in (*DATASETS, "Pooled"):
        subset = rows if dataset == "Pooled" else [row for row in rows if row["dataset"] == dataset]
        for condition in ("C100", "RawE100", "Top1Applicable", "V1_1RawPairConstrained", "Top1GateMatched", "V1_1FinalGated"):
            metrics.append({"dataset": dataset, **summarize(subset, condition)})
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    paired = []
    for comparison in ("RawE100", "Top1Applicable", "V1_1RawPairConstrained", "Top1GateMatched", "V1_1FinalGated"):
        a_only = sum(row["correct_C100"] == 1 and row[f"correct_{comparison}"] == 0 for row in rows)
        b_only = sum(row["correct_C100"] == 0 and row[f"correct_{comparison}"] == 1 for row in rows)
        differences = np.asarray([row[f"correct_{comparison}"] - row["correct_C100"] for row in rows], dtype=float)
        rng = np.random.default_rng(args.seed)
        boot = np.empty(args.bootstraps, dtype=float)
        dataset_rows = {dataset: np.where(np.asarray([row["dataset"] for row in rows]) == dataset)[0] for dataset in DATASETS}
        for index in range(args.bootstraps):
            sampled = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in dataset_rows.values() if len(indices)])
            boot[index] = differences[sampled].mean()
        paired.append({
            "comparison": f"{comparison}_vs_C100",
            "samples": len(rows),
            "difference": differences.mean(),
            "ci_low": np.percentile(boot, 2.5),
            "ci_high": np.percentile(boot, 97.5),
            "c_only_correct": a_only,
            "comparison_only_correct": b_only,
            "mcnemar_exact_p": exact_mcnemar(a_only, b_only),
        })
    with (args.output / "paired_tests.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    def outcome_decomposition(condition: str) -> dict[str, int]:
        gains = [row for row in rows if row["correct_C100"] == 0 and row[f"correct_{condition}"] == 1]
        regressions = [row for row in rows if row["correct_C100"] == 1 and row[f"correct_{condition}"] == 0]
        return {
            "gains": len(gains),
            "regressions": len(regressions),
            "net_correct": len(gains) - len(regressions),
            "restore_top1_gains": sum(row["correct_Top1Applicable"] == 1 for row in gains),
            "beyond_top1_gains": sum(row["correct_Top1Applicable"] == 0 for row in gains),
            "changed_wrong": sum(
                row["correct_C100"] == 0
                and row[f"correct_{condition}"] == 0
                and evaluation_label(row["dataset"], row[f"prediction_{condition}"])
                != evaluation_label(row["dataset"], row["prediction_C100"])
                for row in rows
            ),
        }

    audit = {
        "protocol": "exploratory heldout_evaluation split fixed before v1 parsing; this split was inspected during V1 development and is not confirmatory for V1.1",
        "samples": len(rows),
        "evaluation_scope": "exploratory_reuse",
        "confirmatory": False,
        "reason": "V1.1 design was informed by prior inspection of this cohort",
        "active_rule_outputs": len(set(evolved) & set(base_ids)),
        "extra_result_ids": extra_result_ids,
        "semantic_applicable": sum(row["memory_applicable"] for row in rows),
        "overrides": sum(row["override_allowed"] for row in rows),
        "invalid_output_fallbacks": sum(row["constraint_reason"] == "invalid_output_fallback" for row in rows),
        "third_class_fallbacks": sum(row["constraint_reason"] == "third_class_fallback" for row in rows),
        "gate_sha256": gate_payload["content_sha256"],
        "release_sha256": args.release_sha256,
        "metrics": metrics,
        "paired": paired,
        "outcomes": {
            condition: outcome_decomposition(condition)
            for condition in ("Top1Applicable", "V1_1RawPairConstrained", "Top1GateMatched", "V1_1FinalGated")
        },
        "claim_boundary": "Only beyond-top1 gains can support improved recognition; restore-top1 gains support reduced retrieval-induced error only.",
        "status": "PASS",
    }
    write_json(args.output / "audit.json", audit)
    print(json.dumps({"samples": len(rows), "overrides": audit["overrides"], "pooled": [row for row in metrics if row["dataset"] == "Pooled"], "paired": paired}, ensure_ascii=False, indent=2))


def rollback(args: argparse.Namespace) -> None:
    target = json.loads(args.target_release.read_text(encoding="utf-8"))
    pointer = {"release_id": target["release_id"], "content_sha256": target["content_sha256"], "status": "rolled_back", "path": str(args.target_release), "reason": args.reason}
    write_json(args.active_pointer, pointer)
    print(json.dumps(pointer, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)
    init = commands.add_parser("initialize")
    init.add_argument("--experiment-root", type=Path, required=True)
    init.add_argument("--output", type=Path, required=True)
    init.add_argument("--generator-fraction", type=float, default=0.80)
    init.add_argument("--neighbor-pairs", type=int, default=4)
    init.add_argument("--source-cases-per-class", type=int, default=12)
    init.add_argument("--minimum-source-cases", type=int, default=6)
    init.add_argument("--minimum-feature-support", type=int, default=3)
    init.add_argument("--pair-release", type=Path)
    init.add_argument("--shards", type=int, default=2)
    init.add_argument("--seed", type=int, default=20260825)
    init.set_defaults(func=initialize)

    baseline = commands.add_parser("create-baseline")
    baseline.add_argument("--root", type=Path, required=True)
    baseline.set_defaults(func=create_baseline)

    finish = commands.add_parser("finalize-rules")
    finish.add_argument("--root", type=Path, required=True)
    finish.add_argument("--manifest-dir", type=Path, required=True)
    finish.add_argument("--results", nargs="+", required=True)
    finish.set_defaults(func=finalize_rules)

    retry = commands.add_parser("build-rule-retry")
    retry.add_argument("--manifest-dir", type=Path, required=True)
    retry.add_argument("--results", nargs="+", required=True)
    retry.add_argument("--output", type=Path, required=True)
    retry.set_defaults(func=build_rule_retry)

    repair = commands.add_parser("repair-rule-overlaps")
    repair.add_argument("--manifest", type=Path, required=True)
    repair.add_argument("--results", nargs="+", required=True)
    repair.add_argument("--output", type=Path, required=True)
    repair.set_defaults(func=repair_rule_overlaps)

    diagnosis_retry = commands.add_parser("build-diagnosis-retry")
    diagnosis_retry.add_argument("--manifests", nargs="+", required=True)
    diagnosis_retry.add_argument("--results", nargs="+", required=True)
    diagnosis_retry.add_argument("--output", type=Path, required=True)
    diagnosis_retry.set_defaults(func=build_diagnosis_retry)

    build = commands.add_parser("build-cohort")
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--release", type=Path, required=True)
    build.add_argument("--statuses", nargs="+", default=["candidate"])
    build.add_argument("--cohort", choices=("rule_acceptance", "gate_development", "heldout_evaluation"), required=True)
    build.add_argument("--condition", required=True)
    build.add_argument("--predictions", type=Path)
    build.add_argument("--minimum-trigger-matches", type=int, default=2)
    build.add_argument("--maximum-records", type=int, default=0)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--global-index-offset", type=int, default=400000)
    build.add_argument("--shards", type=int, default=2)
    build.set_defaults(func=build_cohort)

    validate = commands.add_parser("validate-rules")
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument("--candidate-release", type=Path, required=True)
    validate.add_argument("--predictions", type=Path, required=True)
    validate.add_argument("--results", nargs="+", required=True)
    validate.add_argument("--minimum-changes", type=int, default=2)
    validate.add_argument("--minimum-exposures", type=int, default=20)
    validate.add_argument("--minimum-gains", type=int, default=5)
    validate.add_argument("--minimum-net-gain", type=int, default=4)
    validate.add_argument("--maximum-regressions", type=int, default=1)
    validate.add_argument("--maximum-one-sided-p", type=float, default=0.10)
    validate.add_argument("--output", type=Path, required=True)
    validate.set_defaults(func=validate_rules)

    gate = commands.add_parser("tune-gate")
    gate.add_argument("--root", type=Path, required=True)
    gate.add_argument("--release", type=Path, required=True)
    gate.add_argument("--predictions", type=Path, required=True)
    gate.add_argument("--results", nargs="+", required=True)
    gate.add_argument("--minimum-applicable", type=int, default=100)
    gate.add_argument("--minimum-overrides", type=int, default=30)
    gate.add_argument("--minimum-discordances", type=int, default=10)
    gate.add_argument("--minimum-net-gain", type=int, default=3)
    gate.add_argument("--minimum-beyond-top1-gains", type=int, default=3)
    gate.add_argument("--maximum-high-confidence-regressions", type=int, default=0)
    gate.add_argument("--output", type=Path, required=True)
    gate.set_defaults(func=tune_gate)

    final = commands.add_parser("finalize-evaluation")
    final.add_argument("--root", type=Path, required=True)
    final.add_argument("--predictions", type=Path, required=True)
    final.add_argument("--results", nargs="+", required=True)
    final.add_argument("--gate", type=Path, required=True)
    final.add_argument("--release-sha256", required=True)
    final.add_argument("--output", type=Path, required=True)
    final.add_argument("--bootstraps", type=int, default=10000)
    final.add_argument("--seed", type=int, default=20260825)
    final.set_defaults(func=finalize_evaluation)

    back = commands.add_parser("rollback")
    back.add_argument("--target-release", type=Path, required=True)
    back.add_argument("--active-pointer", type=Path, required=True)
    back.add_argument("--reason", required=True)
    back.set_defaults(func=rollback)
    return result


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
