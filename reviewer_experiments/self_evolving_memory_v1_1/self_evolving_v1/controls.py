#!/usr/bin/env python3
"""Build paired neutral and within-pair shuffled controls for V1.1 manifests.

The builder is deliberately independent from the release/validation pipeline.  It
accepts only frozen artifacts, validates their one-to-one provenance, and writes
new inference manifests without changing the source cohort or release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

if __package__:
    from .core import (
        RULE_SECTION_RE,
        canonical_hash,
        inject_rule,
        label_key,
        probabilities_from_prompt,
        render_rule,
        rule_key,
    )
else:  # Support the shebang/direct-script CLI in addition to ``python -m``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from self_evolving_v1.core import (
        RULE_SECTION_RE,
        canonical_hash,
        inject_rule,
        label_key,
        probabilities_from_prompt,
        render_rule,
        rule_key,
    )


CONTROL_SCHEMA_VERSION = "v1.1-paired-controls-v1"
NEUTRAL_CONDITION = "V1_1NeutralControl"
SHUFFLED_CONDITION = "V1_1ShuffledControl"
V11_ARM = "structured_evolving_memory_v1_1"
SAFETY_MARKER = "\n\n[Memory Safety Constraints]:"
SHARD_RE = re.compile(r"^(?P<cohort>.+)_shard_(?P<index>\d+)\.jsonl$")


class ControlManifestError(ValueError):
    """Raised before output is written when frozen inputs fail validation."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ControlManifestError(f"missing JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControlManifestError(f"invalid JSON file: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ControlManifestError(f"JSON root must be an object: {path}")
    return value


def _strict_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ControlManifestError(f"missing JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ControlManifestError(f"invalid UTF-8 JSONL file: {path}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ControlManifestError(
                f"invalid JSONL record: {path}:{line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise ControlManifestError(
                f"JSONL record must be an object: {path}:{line_number}"
            )
        row = dict(row)
        row["_source_manifest_file"] = path.name
        row["_source_manifest_line"] = line_number
        rows.append(row)
    if not rows and not allow_empty:
        raise ControlManifestError(f"empty JSONL input: {path}")
    return rows


def _require_fields(row: dict[str, Any], fields: Iterable[str], context: str) -> None:
    missing = [field for field in fields if field not in row or row[field] is None]
    if missing:
        raise ControlManifestError(f"{context} missing fields: {', '.join(missing)}")


def _require_integer_field(row: dict[str, Any], field: str, context: str) -> None:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ControlManifestError(f"{context} field {field} must be a JSON integer")


def _unique_index(
    rows: list[dict[str, Any]], field: str, context: str
) -> dict[Any, dict[str, Any]]:
    result: dict[Any, dict[str, Any]] = {}
    duplicates: list[Any] = []
    for row in rows:
        if field not in row:
            raise ControlManifestError(f"{context} row missing {field}")
        value = row[field]
        if value in result:
            duplicates.append(value)
        else:
            result[value] = row
    if duplicates:
        rendered = ", ".join(str(value) for value in sorted(set(duplicates), key=str))
        raise ControlManifestError(f"duplicate {field} in {context}: {rendered}")
    return result


def _validated_prompt_probabilities(
    prompt: str, metadata: Any, context: str
) -> list[tuple[str, float]]:
    if not isinstance(metadata, list) or len(metadata) < 2:
        raise ControlManifestError(f"{context} candidate_probabilities must contain at least two rows")
    normalized: list[tuple[str, float]] = []
    for index, item in enumerate(metadata):
        if not isinstance(item, list) or len(item) != 2:
            raise ControlManifestError(f"{context} candidate probability row {index} is malformed")
        label, probability = item
        if not isinstance(label, str) or not label.strip():
            raise ControlManifestError(f"{context} candidate label {index} is invalid")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(float(probability))
            or not 0.0 <= float(probability) <= 1.0
        ):
            raise ControlManifestError(f"{context} candidate probability {index} is invalid")
        normalized.append((label, float(probability)))
    parsed = probabilities_from_prompt(prompt)
    if len(parsed) != len(normalized):
        raise ControlManifestError(
            f"{context} prompt/metadata candidate count mismatch: prompt={len(parsed)} metadata={len(normalized)}"
        )
    for index, ((metadata_label, metadata_probability), (prompt_label, prompt_probability)) in enumerate(
        zip(normalized, parsed)
    ):
        if label_key(metadata_label) != label_key(prompt_label):
            raise ControlManifestError(f"{context} prompt/metadata candidate label mismatch at {index}")
        if not math.isclose(metadata_probability, prompt_probability, rel_tol=0.0, abs_tol=1e-9):
            raise ControlManifestError(
                f"{context} prompt/metadata candidate probability mismatch at {index}"
            )
    return normalized


def _section_parts(prompt: str) -> tuple[re.Match[str], str, str]:
    matches = list(RULE_SECTION_RE.finditer(str(prompt)))
    if len(matches) != 1:
        raise ControlManifestError(
            f"expected exactly one evolved-rule section, found {len(matches)}"
        )
    match = matches[0]
    section = str(prompt)[match.end("head") : match.start("tail")]
    if section.count(SAFETY_MARKER) != 1:
        raise ControlManifestError(
            "true-rule prompt must contain exactly one Memory Safety Constraints block"
        )
    rule_block, safety_suffix = section.split(SAFETY_MARKER, 1)
    return match, rule_block.strip(), safety_suffix


def _prompt_without_rule_section(prompt: str) -> str:
    matches = list(RULE_SECTION_RE.finditer(str(prompt)))
    if len(matches) != 1:
        raise ControlManifestError(
            f"expected exactly one evolved-rule section, found {len(matches)}"
        )
    match = matches[0]
    return (
        str(prompt)[: match.end("head")]
        + "<FROZEN_RULE_SECTION>"
        + str(prompt)[match.start("tail") :]
    )


def _replace_rule_block(prompt: str, replacement: str) -> str:
    match, _, safety_suffix = _section_parts(prompt)
    return (
        str(prompt)[: match.end("head")]
        + replacement.strip()
        + SAFETY_MARKER
        + safety_suffix
        + str(prompt)[match.start("tail") :]
    )


def _neutralize_text(value: str) -> str:
    """Preserve character and whitespace shape while removing all semantics."""

    return "".join(character if character.isspace() else "x" for character in value)


def _placeholder_series(values: Any) -> str:
    if not isinstance(values, list) or not values:
        return "None validated."
    return "; ".join(_neutralize_text(str(value).strip()) for value in values)


def render_neutral_rule(rule: dict[str, Any]) -> str:
    """Mirror the true rule's visible structure without diagnostic content."""

    return (
        f"[Validated Contrastive Rule {rule['rule_id']} | release={rule.get('release_id', 'candidate')} | "
        f"reliability={float(rule.get('reliability', 0.5)):.2f}]\n"
        f"- Distinguish: {rule['disease_a']} versus {rule['disease_b']}\n"
        f"- Supports {rule['disease_a']}: {_placeholder_series(rule.get('supports_a'))}\n"
        f"- Supports {rule['disease_b']}: {_placeholder_series(rule.get('supports_b'))}\n"
        "- Non-discriminative (must not drive an override alone): "
        f"{_placeholder_series(rule.get('nondiscriminative'))}\n"
        f"- Executable triggers for {rule['disease_a']}: "
        f"{_placeholder_series(rule.get('trigger_terms_a'))}\n"
        f"- Executable triggers for {rule['disease_b']}: "
        f"{_placeholder_series(rule.get('trigger_terms_b'))}\n"
        f"- Required context terms: {_placeholder_series(rule.get('required_context_terms'))}\n"
        f"- Executable abstention terms: {_placeholder_series(rule.get('abstain_terms'))}\n"
        f"- Required context: {_placeholder_series(rule.get('required_context'))}\n"
        f"- Limitations: {_placeholder_series(rule.get('limitations'))}\n"
        f"- Evidence: {int(rule.get('source_case_count', 0))} immutable training cases; "
        f"validation gains={int(rule.get('validation_gains', 0))}, "
        f"regressions={int(rule.get('validation_regressions', 0))}."
    )


def _seeded_order(values: Any, *, seed: int, namespace: str) -> list[Any]:
    if not isinstance(values, list):
        raise ControlManifestError(f"shuffle source must be a list: {namespace}")
    items = list(values)
    return [
        item
        for _, item in sorted(
            enumerate(items),
            key=lambda indexed: hashlib.sha256(
                f"{seed}::{namespace}::{indexed[0]}::{indexed[1]}".encode("utf-8")
            ).digest(),
        )
    ]


def shuffled_rule(rule: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """Seeded A/B side swap that keeps the disease pair itself fixed."""

    value = dict(rule)
    rule_id = str(rule.get("rule_id", ""))
    value["supports_a"] = _seeded_order(
        rule.get("supports_b", []), seed=seed, namespace=f"{rule_id}:supports_b_to_a"
    )
    value["supports_b"] = _seeded_order(
        rule.get("supports_a", []), seed=seed, namespace=f"{rule_id}:supports_a_to_b"
    )
    value["trigger_terms_a"] = _seeded_order(
        rule.get("trigger_terms_b", []), seed=seed, namespace=f"{rule_id}:triggers_b_to_a"
    )
    value["trigger_terms_b"] = _seeded_order(
        rule.get("trigger_terms_a", []), seed=seed, namespace=f"{rule_id}:triggers_a_to_b"
    )
    return value


def _release_rules(release: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    stored_hash = str(release.get("content_sha256", ""))
    calculated_hash = canonical_hash(
        {key: value for key, value in release.items() if key != "content_sha256"}
    )
    if not stored_hash or stored_hash != calculated_hash:
        raise ControlManifestError(
            f"release content hash mismatch: stored={stored_hash or '<missing>'} calculated={calculated_hash}"
        )
    release_id = str(release.get("release_id", ""))
    raw_rules = release.get("rules")
    if not release_id.startswith("v1_1") or not isinstance(raw_rules, list):
        raise ControlManifestError("release must be a V1.1 release with a rules list")
    by_id: dict[str, dict[str, Any]] = {}
    by_key: dict[str, dict[str, Any]] = {}
    for index, raw_rule in enumerate(raw_rules):
        if not isinstance(raw_rule, dict):
            raise ControlManifestError(f"release rule {index} is not an object")
        _require_fields(
            raw_rule,
            (
                "rule_id",
                "dataset",
                "disease_a",
                "disease_b",
                "supports_a",
                "supports_b",
                "trigger_terms_a",
                "trigger_terms_b",
                "schema_version",
                "reliability",
            ),
            f"release rule {index}",
        )
        if raw_rule["schema_version"] != "v1.1":
            raise ControlManifestError(f"non-V1.1 rule in release: {raw_rule['rule_id']}")
        reliability = raw_rule["reliability"]
        if (
            isinstance(reliability, bool)
            or not isinstance(reliability, (int, float))
            or not math.isfinite(float(reliability))
            or not 0.0 <= float(reliability) <= 1.0
        ):
            raise ControlManifestError(
                f"release rule {raw_rule['rule_id']} reliability must be a finite number in [0, 1]"
            )
        rendered_list_fields = (
            "supports_a",
            "supports_b",
            "nondiscriminative",
            "trigger_terms_a",
            "trigger_terms_b",
            "required_context_terms",
            "abstain_terms",
            "required_context",
            "limitations",
        )
        for field in rendered_list_fields:
            values = raw_rule.get(field)
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ControlManifestError(
                    f"release rule {raw_rule['rule_id']} field {field} must be a list of non-empty strings"
                )
        expected_key = rule_key(
            str(raw_rule["dataset"]),
            str(raw_rule["disease_a"]),
            str(raw_rule["disease_b"]),
        )
        if raw_rule.get("rule_key") != expected_key:
            raise ControlManifestError(f"invalid rule_key for rule {raw_rule['rule_id']}")
        rule = {**raw_rule, "release_id": release_id}
        rule_id = str(rule["rule_id"])
        if rule_id in by_id:
            raise ControlManifestError(f"duplicate rule_id in release: {rule_id}")
        if expected_key in by_key:
            raise ControlManifestError(f"duplicate rule_key in release: {expected_key}")
        by_id[rule_id] = rule
        by_key[expected_key] = rule
    return by_id, by_key


def _discover_true_manifests(directory: Path) -> tuple[dict[str, Any], list[Path]]:
    if not directory.is_dir():
        raise ControlManifestError(f"missing true-rule manifest directory: {directory}")
    audit = _strict_json(directory / "manifest_audit.json")
    cohort = str(audit.get("cohort", "")).strip()
    if not cohort:
        raise ControlManifestError("manifest audit missing cohort")
    all_shards = sorted(directory.glob("*_shard_*.jsonl"))
    files = sorted(directory.glob(f"{cohort}_shard_*.jsonl"))
    if not files:
        raise ControlManifestError(f"no true-rule shards found for cohort {cohort}")
    if set(all_shards) != set(files):
        extras = sorted(path.name for path in set(all_shards) - set(files))
        raise ControlManifestError(f"ambiguous extra manifest shards: {extras}")
    indices: list[int] = []
    for path in files:
        match = SHARD_RE.match(path.name)
        if match is None or match.group("cohort") != cohort:
            raise ControlManifestError(f"invalid shard name: {path.name}")
        indices.append(int(match.group("index")))
    if sorted(indices) != list(range(len(indices))):
        raise ControlManifestError(f"missing or duplicate shard index: {indices}")
    return audit, files


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _condition_row(
    source: dict[str, Any],
    *,
    control_arm: str,
    condition: str,
    prompt: str,
    global_index: int,
    seed: int,
) -> dict[str, Any]:
    value = {
        key: item
        for key, item in source.items()
        if not key.startswith("_source_manifest_") and key != "prompt"
    }
    original_trigger_matches = source.get("trigger_matches", {"a": 0, "b": 0})
    source_supported_side = str(source.get("trigger_supported_side", ""))
    if control_arm == "neutral":
        control_trigger_matches = {"a": 0, "b": 0}
        control_supported_side = ""
        control_supported_disease = ""
    else:
        control_trigger_matches = {
            "a": int(original_trigger_matches.get("b", 0)),
            "b": int(original_trigger_matches.get("a", 0)),
        }
        control_supported_side = (
            "b" if source_supported_side == "a" else "a" if source_supported_side == "b" else ""
        )
        control_supported_disease = (
            str(source.get("rule_disease_a", ""))
            if control_supported_side == "a"
            else str(source.get("rule_disease_b", ""))
            if control_supported_side == "b"
            else ""
        )
    transform = {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "control_arm": control_arm,
        "seed": seed,
        "sample_id": source["sample_id"],
        "source_true_global_index": int(source["global_index"]),
        "source_rule_id": source["rule_id"],
        "source_prompt_sha256": canonical_hash(source["prompt"]),
        "prompt_sha256": canonical_hash(prompt),
    }
    _, source_rule_block, _ = _section_parts(source["prompt"])
    _, control_rule_block, _ = _section_parts(prompt)
    source_characters = len(source_rule_block)
    control_characters = len(control_rule_block)
    if not source_characters:
        raise ControlManifestError("source true-rule block is empty")
    length_ratio = control_characters / source_characters
    if not 0.90 <= length_ratio <= 1.10:
        raise ControlManifestError(
            f"control rule-block length ratio outside [0.90, 1.10]: {length_ratio:.4f}"
        )
    value.update(
        {
            "global_index": global_index,
            "source_true_global_index": int(source["global_index"]),
            "source_true_condition": source.get("condition", ""),
            "source_true_prompt_sha256": canonical_hash(source["prompt"]),
            "source_manifest_file": source["_source_manifest_file"],
            "source_manifest_line": int(source["_source_manifest_line"]),
            "condition": condition,
            "arm": V11_ARM,
            "control_arm": control_arm,
            "control_schema_version": CONTROL_SCHEMA_VERSION,
            "control_seed": seed,
            "control_pair_labels": [
                source["rule_disease_a"],
                source["rule_disease_b"],
            ],
            "source_applicability_reason": source.get("applicability_reason", ""),
            "source_trigger_matches": original_trigger_matches,
            "source_trigger_supported_side": source_supported_side,
            "source_trigger_supported_disease": source.get("trigger_supported_disease", ""),
            "applicability_reason": "paired_control_from_true_rule_applicable",
            "trigger_matches": control_trigger_matches,
            "trigger_supported_side": control_supported_side,
            "trigger_supported_disease": control_supported_disease,
            "control_transform_sha256": canonical_hash(transform),
            "source_rule_block_characters": source_characters,
            "control_rule_block_characters": control_characters,
            "control_rule_block_length_ratio": length_ratio,
            "source_rule_block_words": len(source_rule_block.split()),
            "control_rule_block_words": len(control_rule_block.split()),
            "prompt_sha256": canonical_hash(prompt),
            "prompt": prompt,
        }
    )
    return value


def build_control_manifests(
    *,
    base_q100: Path,
    release_path: Path,
    true_manifest_dir: Path,
    output_dir: Path,
    seed: int = 20260826,
    shards: int | None = None,
    global_index_offset: int = 800000,
) -> dict[str, Any]:
    """Validate frozen inputs and write paired neutral/shuffled manifests."""

    base_rows = _strict_jsonl(base_q100)
    for row in base_rows:
        _require_integer_field(row, "global_index", "base_q100 row")
    base_by_sid = _unique_index(base_rows, "sample_id", "base_q100")
    _unique_index(base_rows, "global_index", "base_q100")

    release = _strict_json(release_path)
    rules_by_id, rules_by_key = _release_rules(release)
    release_id = str(release["release_id"])
    release_sha256 = str(release["content_sha256"])

    manifest_audit, manifest_files = _discover_true_manifests(true_manifest_dir)
    true_rows: list[dict[str, Any]] = []
    for path in manifest_files:
        true_rows.extend(_strict_jsonl(path, allow_empty=True))
    if not true_rows:
        raise ControlManifestError("true-rule cohort contains no records")
    for row in true_rows:
        _require_integer_field(row, "global_index", "true-rule row")
        _require_integer_field(row, "source_control_global_index", "true-rule row")
    true_by_sid = _unique_index(true_rows, "sample_id", "true-rule cohort")
    _unique_index(true_rows, "global_index", "true-rule cohort")
    expected_records = manifest_audit.get("included_with_rule")
    if not isinstance(expected_records, int) or expected_records != len(true_rows):
        raise ControlManifestError(
            f"manifest audit count mismatch: audit={expected_records} actual={len(true_rows)}"
        )
    if manifest_audit.get("status") != "PASS":
        raise ControlManifestError("true-rule manifest audit is not PASS")
    if manifest_audit.get("release_sha256") != release_sha256:
        raise ControlManifestError("manifest audit release hash does not match frozen release")

    required_true_fields = (
        "sample_id",
        "global_index",
        "source_control_global_index",
        "dataset",
        "relative_path",
        "ground_truth",
        "split_role",
        "candidate_probabilities",
        "c100_anchor_prediction",
        "memory_release_id",
        "memory_release_sha256",
        "arm",
        "rule_id",
        "rule_key",
        "rule_disease_a",
        "rule_disease_b",
        "rule_reliability",
        "prompt",
        "prompt_sha256",
    )
    validated: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for sid, source in sorted(
        true_by_sid.items(), key=lambda item: int(item[1]["global_index"])
    ):
        _require_fields(source, required_true_fields, f"true-rule record {sid}")
        if not str(source["c100_anchor_prediction"]).strip():
            raise ControlManifestError(f"missing C100 anchor for {sid}")
        base = base_by_sid.get(sid)
        if base is None:
            raise ControlManifestError(f"true-rule record missing from base_q100: {sid}")
        _require_fields(
            base,
            (
                "sample_id",
                "global_index",
                "dataset",
                "relative_path",
                "ground_truth",
                "split_role",
                "candidate_probabilities",
                "prompt",
            ),
            f"base_q100 record {sid}",
        )
        if int(source["source_control_global_index"]) != int(base["global_index"]):
            raise ControlManifestError(f"source index mismatch for {sid}")
        for field in ("dataset", "relative_path", "ground_truth"):
            if source[field] != base[field]:
                raise ControlManifestError(f"{field} mismatch against base_q100 for {sid}")
        if source["split_role"] != manifest_audit["cohort"]:
            raise ControlManifestError(f"true-rule split/cohort mismatch for {sid}")
        if base["split_role"] != manifest_audit["cohort"]:
            raise ControlManifestError(f"base_q100 split/cohort mismatch for {sid}")
        _validated_prompt_probabilities(
            str(base["prompt"]), base["candidate_probabilities"], f"base_q100 record {sid}"
        )
        if canonical_hash(source["candidate_probabilities"]) != canonical_hash(
            base["candidate_probabilities"]
        ):
            raise ControlManifestError(f"candidate probabilities mismatch for {sid}")
        if canonical_hash(source["prompt"]) != source["prompt_sha256"]:
            raise ControlManifestError(f"true prompt hash mismatch for {sid}")
        if _prompt_without_rule_section(source["prompt"]) != _prompt_without_rule_section(
            base["prompt"]
        ):
            raise ControlManifestError(f"non-rule prompt content changed for {sid}")
        if source["memory_release_id"] != release_id:
            raise ControlManifestError(f"release ID mismatch for {sid}")
        if source["memory_release_sha256"] != release_sha256:
            raise ControlManifestError(f"release hash mismatch for {sid}")
        if source["arm"] != V11_ARM:
            raise ControlManifestError(f"source arm is not V1.1 for {sid}")
        rule = rules_by_id.get(str(source["rule_id"]))
        if rule is None:
            raise ControlManifestError(f"missing release rule for {sid}: {source['rule_id']}")
        if rules_by_key.get(str(source["rule_key"])) is not rule:
            raise ControlManifestError(f"rule ID/key mismatch for {sid}")
        expected_key = rule_key(
            str(source["dataset"]),
            str(source["rule_disease_a"]),
            str(source["rule_disease_b"]),
        )
        if expected_key != source["rule_key"]:
            raise ControlManifestError(f"record disease pair mismatch for {sid}")
        if label_key(source["rule_disease_a"]) != label_key(rule["disease_a"]):
            raise ControlManifestError(f"disease_a mismatch for {sid}")
        if label_key(source["rule_disease_b"]) != label_key(rule["disease_b"]):
            raise ControlManifestError(f"disease_b mismatch for {sid}")
        source_reliability = source["rule_reliability"]
        if (
            isinstance(source_reliability, bool)
            or not isinstance(source_reliability, (int, float))
            or not math.isfinite(float(source_reliability))
        ):
            raise ControlManifestError(f"rule_reliability must be a finite number for {sid}")
        if float(source_reliability) != float(rule["reliability"]):
            raise ControlManifestError(f"rule_reliability mismatch against release for {sid}")
        guideline = source.get("guideline_diseases")
        if not isinstance(guideline, list) or len(guideline) != 2:
            raise ControlManifestError(f"missing two-label guideline pair for {sid}")
        if rule_key(str(source["dataset"]), str(guideline[0]), str(guideline[1])) != expected_key:
            raise ControlManifestError(f"guideline pair is not the frozen rule pair for {sid}")
        probabilities = source["candidate_probabilities"]
        if (
            not isinstance(probabilities, list)
            or len(probabilities) < 2
            or not all(isinstance(item, list) and len(item) >= 2 for item in probabilities[:2])
        ):
            raise ControlManifestError(f"missing frozen top-2 probabilities for {sid}")
        candidate_key = rule_key(
            str(source["dataset"]), str(probabilities[0][0]), str(probabilities[1][0])
        )
        if candidate_key != expected_key:
            raise ControlManifestError(f"rule pair is not the frozen candidate top-2 for {sid}")
        if any(
            label_key(str(guideline[index])) != label_key(str(probabilities[index][0]))
            for index in range(2)
        ):
            raise ControlManifestError(f"guideline pair order differs from frozen top-2 for {sid}")
        anchor_key = label_key(str(source["c100_anchor_prediction"]))
        candidate_pair_keys = {
            label_key(str(probabilities[0][0])),
            label_key(str(probabilities[1][0])),
        }
        if anchor_key not in candidate_pair_keys:
            raise ControlManifestError(f"C100 anchor is outside the frozen candidate pair for {sid}")
        _, true_rule_block, _ = _section_parts(source["prompt"])
        if true_rule_block != render_rule(rule).strip():
            raise ControlManifestError(f"true prompt rule block does not match release for {sid}")
        expected_true_prompt = inject_rule(
            base["prompt"], rule, baseline_prediction=source["c100_anchor_prediction"]
        )
        if source["prompt"] != expected_true_prompt:
            raise ControlManifestError(
                f"true prompt safety/rule injection does not match frozen inputs for {sid}"
            )
        validated.append((source, rule))

    shard_count = len(manifest_files) if shards is None else shards
    if shard_count <= 0:
        raise ControlManifestError("shards must be positive")
    if isinstance(global_index_offset, bool) or not isinstance(global_index_offset, int):
        raise ControlManifestError("global_index_offset must be an integer")
    if global_index_offset < 0:
        raise ControlManifestError("global_index_offset must be non-negative")
    generated_indices = set(range(global_index_offset, global_index_offset + 2 * len(validated)))
    source_indices = {int(row["global_index"]) for row, _ in validated}
    base_indices = {int(row["global_index"]) for row in base_rows}
    if generated_indices & (source_indices | base_indices):
        raise ControlManifestError("generated global indices collide with frozen source indices")

    neutral_rows: list[dict[str, Any]] = []
    shuffled_rows: list[dict[str, Any]] = []
    for position, (source, rule) in enumerate(validated):
        neutral_prompt = _replace_rule_block(source["prompt"], render_neutral_rule(rule))
        swapped = shuffled_rule(rule, seed=seed)
        shuffled_prompt = _replace_rule_block(source["prompt"], render_rule(swapped))
        neutral_rows.append(
            _condition_row(
                source,
                control_arm="neutral",
                condition=NEUTRAL_CONDITION,
                prompt=neutral_prompt,
                global_index=global_index_offset + position,
                seed=seed,
            )
        )
        shuffled_rows.append(
            _condition_row(
                source,
                control_arm="shuffled_within_pair",
                condition=SHUFFLED_CONDITION,
                prompt=shuffled_prompt,
                global_index=global_index_offset + len(validated) + position,
                seed=seed,
            )
        )

    audit_path = output_dir / "control_manifest_audit.json"
    existing = (
        sorted(path for path in output_dir.rglob("*") if path.is_file())
        if output_dir.exists()
        else []
    )
    if existing:
        raise ControlManifestError(
            "refusing to write into a non-empty control output directory: "
            + ", ".join(str(path) for path in existing)
        )

    rows_by_arm = {"neutral": neutral_rows, "shuffled": shuffled_rows}
    output_files: dict[str, dict[str, str]] = {"neutral": {}, "shuffled": {}}
    for arm, arm_rows in rows_by_arm.items():
        for shard_index in range(shard_count):
            path = output_dir / arm / f"{manifest_audit['cohort']}_{arm}_shard_{shard_index}.jsonl"
            shard_rows = [
                row for index, row in enumerate(arm_rows) if index % shard_count == shard_index
            ]
            _write_jsonl(path, shard_rows)
            output_files[arm][str(path.relative_to(output_dir))] = file_sha256(path)

    dataset_counts = Counter(str(row["dataset"]) for row, _ in validated)
    rule_counts = Counter(str(row["rule_id"]) for row, _ in validated)
    pairing = [
        {
            "sample_id": source["sample_id"],
            "base_global_index": int(source["source_control_global_index"]),
            "true_global_index": int(source["global_index"]),
            "neutral_global_index": neutral_rows[index]["global_index"],
            "shuffled_global_index": shuffled_rows[index]["global_index"],
            "rule_id": source["rule_id"],
            "rule_key": source["rule_key"],
        }
        for index, (source, _) in enumerate(validated)
    ]
    audit: dict[str, Any] = {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "status": "PASS",
        "protocol": (
            "paired controls derived only from a frozen true-rule V1.1 cohort; neutral replaces only "
            "the visible rule block with non-clinical placeholders; shuffled swaps supports/triggers "
            "within the same fixed A/B pair"
        ),
        "seed": seed,
        "cohort": manifest_audit["cohort"],
        "records_per_arm": len(validated),
        "shards_per_arm": shard_count,
        "global_index_offset": global_index_offset,
        "conditions": {
            "neutral": NEUTRAL_CONDITION,
            "shuffled": SHUFFLED_CONDITION,
        },
        "invariants": {
            "sample_pairing_complete": True,
            "candidate_probabilities_preserved": True,
            "c100_anchor_preserved": True,
            "c100_anchor_pair_membership_verified": True,
            "rule_reliability_matches_release": True,
            "source_indices_preserved": True,
            "prompt_outside_rule_section_preserved": True,
            "shuffled_pair_labels_preserved": True,
            "duplicate_sample_ids": 0,
            "duplicate_global_indices": 0,
            "missing_base_records": 0,
            "missing_release_rules": 0,
        },
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "rule_counts": dict(sorted(rule_counts.items())),
        "rule_block_length": {
            "allowed_ratio": [0.90, 1.10],
            "neutral_min_ratio": min(
                row["control_rule_block_length_ratio"] for row in neutral_rows
            ),
            "neutral_max_ratio": max(
                row["control_rule_block_length_ratio"] for row in neutral_rows
            ),
            "shuffled_min_ratio": min(
                row["control_rule_block_length_ratio"] for row in shuffled_rows
            ),
            "shuffled_max_ratio": max(
                row["control_rule_block_length_ratio"] for row in shuffled_rows
            ),
            "neutral_max_absolute_character_delta": max(
                abs(
                    row["control_rule_block_characters"]
                    - row["source_rule_block_characters"]
                )
                for row in neutral_rows
            ),
            "shuffled_max_absolute_character_delta": max(
                abs(
                    row["control_rule_block_characters"]
                    - row["source_rule_block_characters"]
                )
                for row in shuffled_rows
            ),
        },
        "base_q100": {
            "path": str(base_q100),
            "records": len(base_rows),
            "file_sha256": file_sha256(base_q100),
            "content_sha256": canonical_hash(
                [
                    {key: value for key, value in row.items() if not key.startswith("_source_manifest_")}
                    for row in base_rows
                ]
            ),
        },
        "release": {
            "path": str(release_path),
            "release_id": release_id,
            "file_sha256": file_sha256(release_path),
            "content_sha256": release_sha256,
        },
        "true_manifest": {
            "directory": str(true_manifest_dir),
            "records": len(true_rows),
            "audit_file_sha256": file_sha256(true_manifest_dir / "manifest_audit.json"),
            "files": {path.name: file_sha256(path) for path in manifest_files},
            "content_sha256": canonical_hash(
                [
                    {key: value for key, value in row.items() if not key.startswith("_source_manifest_")}
                    for row, _ in validated
                ]
            ),
        },
        "controls": {
            "neutral": {
                "records": len(neutral_rows),
                "content_sha256": canonical_hash(neutral_rows),
                "files": output_files["neutral"],
            },
            "shuffled": {
                "records": len(shuffled_rows),
                "content_sha256": canonical_hash(shuffled_rows),
                "files": output_files["shuffled"],
            },
        },
        "pairing_sha256": canonical_hash(pairing),
        "pairing": pairing,
    }
    audit["audit_sha256"] = canonical_hash(audit)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Build paired neutral and within-pair shuffled V1.1 control manifests."
    )
    result.add_argument("--base-q100", type=Path, required=True)
    result.add_argument("--release", type=Path, required=True)
    result.add_argument("--true-manifest-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--seed", type=int, default=20260826)
    result.add_argument("--shards", type=int)
    result.add_argument("--global-index-offset", type=int, default=800000)
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        audit = build_control_manifests(
            base_q100=args.base_q100,
            release_path=args.release,
            true_manifest_dir=args.true_manifest_dir,
            output_dir=args.output,
            seed=args.seed,
            shards=args.shards,
            global_index_offset=args.global_index_offset,
        )
    except ControlManifestError as error:
        raise SystemExit(f"control manifest validation failed: {error}") from error
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
