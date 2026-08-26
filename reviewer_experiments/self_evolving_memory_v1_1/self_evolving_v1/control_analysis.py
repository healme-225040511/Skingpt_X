#!/usr/bin/env python3
"""Strict paired analysis for true, neutral, and shuffled V1.1 arms.

This module intentionally revalidates the frozen inputs instead of treating the
control-manifest audit as a certificate.  In particular, it reconstructs both
control manifests from the true arm and frozen release, binds every inference
file to one manifest shard, and applies the same closed-set parsing and
pair-constrained fallback used by the V1.1 evaluation pipeline.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

if __package__:
    from .controls import (
        CONTROL_SCHEMA_VERSION,
        NEUTRAL_CONDITION,
        SHUFFLED_CONDITION,
        V11_ARM,
        _condition_row,
        _release_rules,
        _replace_rule_block,
        file_sha256,
        render_neutral_rule,
        shuffled_rule,
    )
    from .core import (
        canonical_hash,
        inject_rule,
        label_key,
        parse_primary_diagnosis,
        probabilities_from_prompt,
        render_rule,
        rule_key,
    )
    from .pipeline import evaluation_label, pair_constrained_prediction
else:  # Allow both ``python -m`` and direct-script execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from self_evolving_v1.controls import (
        CONTROL_SCHEMA_VERSION,
        NEUTRAL_CONDITION,
        SHUFFLED_CONDITION,
        V11_ARM,
        _condition_row,
        _release_rules,
        _replace_rule_block,
        file_sha256,
        render_neutral_rule,
        shuffled_rule,
    )
    from self_evolving_v1.core import (
        canonical_hash,
        inject_rule,
        label_key,
        parse_primary_diagnosis,
        probabilities_from_prompt,
        render_rule,
        rule_key,
    )
    from self_evolving_v1.pipeline import evaluation_label, pair_constrained_prediction


ANALYSIS_SCHEMA_VERSION = "v1.1-paired-control-analysis-v1"
ARM_ORDER = ("true", "neutral", "shuffled")
CONTROL_ARMS = {
    "neutral": "neutral",
    "shuffled": "shuffled_within_pair",
}
RESULT_EXTRA_FIELDS = {
    "record_type",
    "response",
    "input_tokens",
    "output_tokens",
    "batch_seconds",
    "peak_memory_gib",
}
RUN_COMPARABILITY_FIELDS = (
    "model",
    "model_revision",
    "model_config_sha256",
    "model_index_sha256",
    "code_revision",
    "batch_size",
    "max_new_tokens",
    "do_sample",
    "torch_version",
    "transformers_version",
    "cuda_version",
    "python_version",
)


class ControlAnalysisError(ValueError):
    """Raised when a frozen artifact or a paired result fails validation."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ControlAnalysisError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ControlAnalysisError(f"non-finite JSON number is forbidden: {value}")


def _parse_json(text: str, context: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ControlAnalysisError(f"invalid JSON in {context}: {error}") from error


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ControlAnalysisError(f"missing JSON file: {path}")
    value = _parse_json(path.read_text(encoding="utf-8"), str(path))
    if not isinstance(value, dict):
        raise ControlAnalysisError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(
    path: Path,
    *,
    allow_empty: bool = False,
) -> list[tuple[dict[str, Any], int]]:
    if not path.is_file():
        raise ControlAnalysisError(f"missing JSONL file: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ControlAnalysisError(f"invalid UTF-8 JSONL file: {path}") from error
    rows: list[tuple[dict[str, Any], int]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise ControlAnalysisError(f"blank JSONL line: {path}:{line_number}")
        row = _parse_json(line, f"{path}:{line_number}")
        if not isinstance(row, dict):
            raise ControlAnalysisError(
                f"JSONL record must be an object: {path}:{line_number}"
            )
        rows.append((row, line_number))
    if not rows and not allow_empty:
        raise ControlAnalysisError(f"empty JSONL file: {path}")
    return rows


def _require_fields(row: dict[str, Any], fields: Iterable[str], context: str) -> None:
    missing = [field for field in fields if field not in row or row[field] is None]
    if missing:
        raise ControlAnalysisError(f"{context} missing fields: {', '.join(missing)}")


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ControlAnalysisError(f"{context} must be a JSON integer")
    return value


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ControlAnalysisError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ControlAnalysisError(f"{context} must be finite")
    return result


def _unique_index(
    rows: Iterable[dict[str, Any]], field: str, context: str
) -> dict[Any, dict[str, Any]]:
    result: dict[Any, dict[str, Any]] = {}
    for row in rows:
        if field not in row:
            raise ControlAnalysisError(f"{context} row missing {field}")
        key = row[field]
        if key in result:
            raise ControlAnalysisError(f"duplicate {field} in {context}: {key}")
        result[key] = row
    return result


def _expand_paths(specs: Sequence[str | Path], context: str) -> list[Path]:
    paths: list[Path] = []
    for raw_spec in specs:
        spec = os.fspath(raw_spec)
        matches = [Path(value) for value in glob.glob(spec)]
        if not matches:
            candidate = Path(spec)
            if candidate.is_dir():
                matches = sorted(candidate.glob("*.jsonl"))
            elif candidate.exists():
                matches = [candidate]
        for match in matches:
            if match.is_dir():
                paths.extend(sorted(match.glob("*.jsonl")))
            elif match.is_file():
                paths.append(match)
    if not paths:
        raise ControlAnalysisError(f"no files matched {context}")
    resolved: dict[Path, Path] = {}
    for path in paths:
        identity = path.resolve()
        if identity in resolved:
            raise ControlAnalysisError(
                f"duplicate file in {context}: {path} and {resolved[identity]}"
            )
        resolved[identity] = path
    return sorted(paths, key=lambda path: (path.name, str(path.resolve())))


def _validate_candidates(row: dict[str, Any], context: str) -> list[tuple[str, float]]:
    values = row.get("candidate_probabilities")
    if not isinstance(values, list) or len(values) < 2:
        raise ControlAnalysisError(
            f"{context} candidate_probabilities must contain at least two rows"
        )
    result: list[tuple[str, float]] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, list) or len(value) != 2:
            raise ControlAnalysisError(f"{context} candidate row {index} is malformed")
        label = str(value[0]).strip() if isinstance(value[0], str) else ""
        if not label:
            raise ControlAnalysisError(f"{context} candidate label {index} is invalid")
        key = label_key(label)
        if key in seen:
            raise ControlAnalysisError(f"{context} duplicate candidate label: {label}")
        seen.add(key)
        probability = _finite_number(value[1], f"{context} candidate probability {index}")
        if not 0.0 <= probability <= 1.0:
            raise ControlAnalysisError(
                f"{context} candidate probability {index} is outside [0, 1]"
            )
        result.append((label, probability))
    return result


def _validate_prompt_candidates(row: dict[str, Any], context: str) -> None:
    metadata = _validate_candidates(row, context)
    parsed = probabilities_from_prompt(str(row.get("prompt", "")))
    if len(parsed) != len(metadata):
        raise ControlAnalysisError(f"{context} prompt/metadata candidate count mismatch")
    for index, ((left_label, left_probability), (right_label, right_probability)) in enumerate(
        zip(metadata, parsed)
    ):
        if label_key(left_label) != label_key(right_label) or not math.isclose(
            left_probability, right_probability, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ControlAnalysisError(
                f"{context} prompt/metadata candidate mismatch at position {index}"
            )


def _safe_relative_file(root: Path, relative: str, context: str) -> Path:
    raw = Path(str(relative))
    if raw.is_absolute() or ".." in raw.parts:
        raise ControlAnalysisError(f"unsafe relative path in {context}: {relative}")
    root_resolved = root.resolve()
    path = (root / raw).resolve()
    try:
        path.relative_to(root_resolved)
    except ValueError as error:
        raise ControlAnalysisError(f"path escapes control root in {context}: {relative}") from error
    if not path.is_file():
        raise ControlAnalysisError(f"missing audited file in {context}: {relative}")
    return path


def _validate_self_hash(payload: dict[str, Any], field: str, context: str) -> str:
    stored = str(payload.get(field, ""))
    calculated = canonical_hash({key: value for key, value in payload.items() if key != field})
    if not stored or stored != calculated:
        raise ControlAnalysisError(
            f"{context} {field} mismatch: stored={stored or '<missing>'} calculated={calculated}"
        )
    return stored


def _load_base(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = [row for row, _ in _read_jsonl(path)]
    required = (
        "sample_id",
        "global_index",
        "dataset",
        "relative_path",
        "ground_truth",
        "split_role",
        "candidate_probabilities",
        "prompt",
    )
    for row in rows:
        sid = str(row.get("sample_id", ""))
        _require_fields(row, required, f"base_q100 record {sid or '<unknown>'}")
        _integer(row["global_index"], f"base_q100 global_index for {sid}")
        _validate_prompt_candidates(row, f"base_q100 record {sid}")
    by_sid = _unique_index(rows, "sample_id", "base_q100")
    _unique_index(rows, "global_index", "base_q100")
    return rows, by_sid


def _load_true_manifests(
    paths: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, int]]]:
    rows: list[dict[str, Any]] = []
    locations: dict[str, tuple[str, int]] = {}
    for path in paths:
        for row, line_number in _read_jsonl(path, allow_empty=True):
            sid = str(row.get("sample_id", ""))
            if not sid:
                raise ControlAnalysisError(f"true manifest row missing sample_id: {path}:{line_number}")
            if sid in locations:
                raise ControlAnalysisError(f"duplicate sample_id in true manifests: {sid}")
            rows.append(row)
            locations[sid] = (path.name, line_number)
    if not rows:
        raise ControlAnalysisError("true manifests contain no records")
    _unique_index(rows, "global_index", "true manifests")
    rows.sort(key=lambda row: _integer(row["global_index"], "true global_index"))
    return rows, locations


def _manifest_files_from_audit(
    control_root: Path, audit: dict[str, Any], arm: str
) -> list[Path]:
    control = audit.get("controls", {}).get(arm)
    if not isinstance(control, dict) or not isinstance(control.get("files"), dict):
        raise ControlAnalysisError(f"control audit missing {arm} manifest files")
    paths: list[Path] = []
    for relative, expected_hash in sorted(control["files"].items()):
        path = _safe_relative_file(control_root, str(relative), f"{arm} files")
        if path.parts[-2] != arm:
            raise ControlAnalysisError(f"{arm} manifest is outside its arm directory: {relative}")
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            raise ControlAnalysisError(
                f"{arm} manifest file SHA mismatch for {relative}: {actual_hash}"
            )
        paths.append(path)
    if not paths:
        raise ControlAnalysisError(f"control audit contains no {arm} manifest files")
    return paths


def _load_control_rows(paths: list[Path], arm: str) -> list[dict[str, Any]]:
    rows = [row for path in paths for row, _ in _read_jsonl(path, allow_empty=True)]
    if not rows:
        raise ControlAnalysisError(f"{arm} manifests contain no records")
    _unique_index(rows, "sample_id", f"{arm} manifests")
    _unique_index(rows, "global_index", f"{arm} manifests")
    rows.sort(key=lambda row: _integer(row["global_index"], f"{arm} global_index"))
    return rows


def _validate_true_row(
    row: dict[str, Any],
    *,
    base: dict[str, Any],
    pairing: dict[str, Any],
    release: dict[str, Any],
    rule: dict[str, Any],
    cohort: str,
) -> None:
    sid = str(row.get("sample_id", ""))
    required = (
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
    _require_fields(row, required, f"true manifest record {sid}")
    true_index = _integer(row["global_index"], f"true global_index for {sid}")
    base_index = _integer(
        row["source_control_global_index"], f"source_control_global_index for {sid}"
    )
    if true_index != _integer(pairing.get("true_global_index"), f"pairing true index for {sid}"):
        raise ControlAnalysisError(f"true global_index disagrees with pairing for {sid}")
    if base_index != _integer(pairing.get("base_global_index"), f"pairing base index for {sid}"):
        raise ControlAnalysisError(f"base global_index disagrees with pairing for {sid}")
    if base_index != base["global_index"]:
        raise ControlAnalysisError(f"true source index disagrees with base_q100 for {sid}")
    for field in ("dataset", "relative_path", "ground_truth", "split_role"):
        if row[field] != base[field]:
            raise ControlAnalysisError(f"true {field} disagrees with base_q100 for {sid}")
    if row["split_role"] != cohort:
        raise ControlAnalysisError(f"true split_role disagrees with control cohort for {sid}")
    if canonical_hash(row["candidate_probabilities"]) != canonical_hash(
        base["candidate_probabilities"]
    ):
        raise ControlAnalysisError(f"true candidate probabilities disagree with base_q100 for {sid}")
    _validate_candidates(row, f"true manifest record {sid}")
    if canonical_hash(row["prompt"]) != row["prompt_sha256"]:
        raise ControlAnalysisError(f"true prompt SHA mismatch for {sid}")
    if row["memory_release_id"] != release["release_id"]:
        raise ControlAnalysisError(f"true release ID mismatch for {sid}")
    if row["memory_release_sha256"] != release["content_sha256"]:
        raise ControlAnalysisError(f"true release SHA mismatch for {sid}")
    if row["arm"] != V11_ARM:
        raise ControlAnalysisError(f"true arm is not V1.1 for {sid}")
    if str(row["rule_id"]) != str(rule["rule_id"]):
        raise ControlAnalysisError(f"true rule ID mismatch for {sid}")
    expected_key = rule_key(str(row["dataset"]), str(row["rule_disease_a"]), str(row["rule_disease_b"]))
    if row["rule_key"] != expected_key or row["rule_key"] != rule["rule_key"]:
        raise ControlAnalysisError(f"true rule key mismatch for {sid}")
    if pairing.get("rule_id") != row["rule_id"] or pairing.get("rule_key") != row["rule_key"]:
        raise ControlAnalysisError(f"true rule disagrees with pairing for {sid}")
    if label_key(row["rule_disease_a"]) != label_key(rule["disease_a"]) or label_key(
        row["rule_disease_b"]
    ) != label_key(rule["disease_b"]):
        raise ControlAnalysisError(f"true disease pair disagrees with release for {sid}")
    reliability = _finite_number(row["rule_reliability"], f"rule reliability for {sid}")
    if reliability != _finite_number(rule["reliability"], f"release reliability for {sid}"):
        raise ControlAnalysisError(f"true rule reliability mismatch for {sid}")
    probabilities = _validate_candidates(row, f"true manifest record {sid}")
    guideline = row.get("guideline_diseases")
    if not isinstance(guideline, list) or len(guideline) != 2 or any(
        label_key(str(guideline[index])) != label_key(probabilities[index][0])
        for index in range(2)
    ):
        raise ControlAnalysisError(f"true guideline order disagrees with frozen top-2 for {sid}")
    expected_pair = rule_key(str(row["dataset"]), probabilities[0][0], probabilities[1][0])
    if expected_pair != row["rule_key"]:
        raise ControlAnalysisError(f"true rule is not the frozen candidate top-2 for {sid}")
    pair_labels = {
        evaluation_label(str(row["dataset"]), str(row["rule_disease_a"])),
        evaluation_label(str(row["dataset"]), str(row["rule_disease_b"])),
    }
    if evaluation_label(str(row["dataset"]), str(row["c100_anchor_prediction"])) not in pair_labels:
        raise ControlAnalysisError(f"true C100 anchor is outside the frozen pair for {sid}")
    side = str(row.get("trigger_supported_side", ""))
    disease = str(row.get("trigger_supported_disease", ""))
    if side not in {"", "a", "b"}:
        raise ControlAnalysisError(f"invalid true trigger direction for {sid}: {side}")
    expected_disease = (
        str(row["rule_disease_a"])
        if side == "a"
        else str(row["rule_disease_b"])
        if side == "b"
        else ""
    )
    if label_key(disease) != label_key(expected_disease):
        raise ControlAnalysisError(f"true trigger direction/disease mismatch for {sid}")
    expected_prompt = inject_rule(
        str(base["prompt"]), rule, baseline_prediction=str(row["c100_anchor_prediction"])
    )
    if row["prompt"] != expected_prompt:
        raise ControlAnalysisError(f"true prompt does not replay from base/release for {sid}")


def _validate_control_reconstruction(
    *,
    source: dict[str, Any],
    source_location: tuple[str, int],
    rule: dict[str, Any],
    actual: dict[str, Any],
    arm: str,
    pairing: dict[str, Any],
    seed: int,
) -> None:
    sid = str(source["sample_id"])
    source_with_location = dict(source)
    source_with_location["_source_manifest_file"] = source_location[0]
    source_with_location["_source_manifest_line"] = source_location[1]
    if arm == "neutral":
        prompt = _replace_rule_block(source["prompt"], render_neutral_rule(rule))
        condition = NEUTRAL_CONDITION
        global_index = _integer(
            pairing.get("neutral_global_index"), f"pairing neutral index for {sid}"
        )
    else:
        swapped = shuffled_rule(rule, seed=seed)
        prompt = _replace_rule_block(source["prompt"], render_rule(swapped))
        condition = SHUFFLED_CONDITION
        global_index = _integer(
            pairing.get("shuffled_global_index"), f"pairing shuffled index for {sid}"
        )
    expected = _condition_row(
        source_with_location,
        control_arm=CONTROL_ARMS[arm],
        condition=condition,
        prompt=prompt,
        global_index=global_index,
        seed=seed,
    )
    if canonical_hash(actual) != canonical_hash(expected) or actual != expected:
        raise ControlAnalysisError(f"{arm} manifest reconstruction mismatch for {sid}")


def _load_results(
    *,
    specs: Sequence[str | Path],
    manifests: list[Path],
    manifest_rows: list[dict[str, Any]],
    arm: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    result_paths = _expand_paths(specs, f"{arm} results")
    manifest_keys: dict[tuple[str, str], Path] = {}
    rows_by_manifest: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    manifest_by_index: dict[int, dict[str, Any]] = {}
    for path in manifests:
        key = (path.name, file_sha256(path))
        if key in manifest_keys:
            raise ControlAnalysisError(f"duplicate manifest identity for {arm}: {key}")
        manifest_keys[key] = path
        for row, _ in _read_jsonl(path, allow_empty=True):
            index = _integer(row.get("global_index"), f"{arm} manifest global_index")
            if index in manifest_by_index:
                raise ControlAnalysisError(f"duplicate global_index in {arm} manifests: {index}")
            manifest_by_index[index] = row
            rows_by_manifest[key].append(row)
    if len(manifest_by_index) != len(manifest_rows):
        raise ControlAnalysisError(f"internal {arm} manifest record count mismatch")

    seen_manifest_keys: set[tuple[str, str]] = set()
    results_by_sid: dict[str, dict[str, Any]] = {}
    headers: list[dict[str, Any]] = []
    result_hashes: dict[str, str] = {}
    for result_path in result_paths:
        parsed = [row for row, _ in _read_jsonl(result_path)]
        run_rows = [row for row in parsed if row.get("record_type") == "run"]
        inference_rows = [row for row in parsed if row.get("record_type") == "inference"]
        unknown = sorted(
            {str(row.get("record_type")) for row in parsed}
            - {"run", "inference"}
        )
        if unknown:
            raise ControlAnalysisError(f"unknown record types in {result_path}: {unknown}")
        if len(run_rows) != 1:
            raise ControlAnalysisError(
                f"{result_path} must contain exactly one run header, found {len(run_rows)}"
            )
        header = run_rows[0]
        _require_fields(
            header,
            (
                "manifest",
                "manifest_sha256",
                "records",
                "do_sample",
                "model_revision",
                "code_revision",
            ),
            f"run header {result_path}",
        )
        key = (Path(str(header["manifest"])).name, str(header["manifest_sha256"]))
        if key not in manifest_keys:
            raise ControlAnalysisError(
                f"{arm} result header is not bound to an audited manifest: {result_path}"
            )
        if key in seen_manifest_keys:
            raise ControlAnalysisError(f"multiple {arm} result files for manifest {key[0]}")
        seen_manifest_keys.add(key)
        expected_rows = rows_by_manifest[key]
        if _integer(header["records"], f"run records in {result_path}") != len(expected_rows):
            raise ControlAnalysisError(f"run header record count mismatch in {result_path}")
        if header["do_sample"] is not False:
            raise ControlAnalysisError(f"non-deterministic inference run in {result_path}")
        if len(inference_rows) != len(expected_rows):
            raise ControlAnalysisError(
                f"incomplete {arm} results in {result_path}: expected={len(expected_rows)} actual={len(inference_rows)}"
            )
        expected_by_index = {
            _integer(row["global_index"], f"{arm} manifest global_index"): row
            for row in expected_rows
        }
        seen_indices: set[int] = set()
        for result in inference_rows:
            _require_fields(
                result,
                ("global_index", "sample_id", "response", "candidate_probabilities"),
                f"{arm} inference row in {result_path}",
            )
            index = _integer(result["global_index"], f"{arm} result global_index")
            if index in seen_indices:
                raise ControlAnalysisError(
                    f"duplicate {arm} result global_index in {result_path}: {index}"
                )
            seen_indices.add(index)
            manifest = expected_by_index.get(index)
            if manifest is None:
                raise ControlAnalysisError(f"unexpected {arm} result global_index: {index}")
            unknown_fields = set(result) - set(manifest) - RESULT_EXTRA_FIELDS
            if unknown_fields:
                raise ControlAnalysisError(
                    f"unexpected fields in {arm} result {index}: {sorted(unknown_fields)}"
                )
            if "prompt" in result:
                raise ControlAnalysisError(f"prompt must not be copied into {arm} result {index}")
            for field, expected_value in manifest.items():
                if field == "prompt":
                    continue
                if field not in result or canonical_hash(result[field]) != canonical_hash(expected_value):
                    raise ControlAnalysisError(
                        f"{arm} result/manifest mismatch for {result.get('sample_id')} field {field}"
                    )
            if not isinstance(result["response"], str):
                raise ControlAnalysisError(f"{arm} response must be a string for index {index}")
            for token_field in ("input_tokens", "output_tokens"):
                _integer(result.get(token_field), f"{arm} {token_field} for index {index}")
            for timing_field in ("batch_seconds", "peak_memory_gib"):
                if _finite_number(
                    result.get(timing_field), f"{arm} {timing_field} for index {index}"
                ) < 0.0:
                    raise ControlAnalysisError(
                        f"{arm} {timing_field} must be non-negative for index {index}"
                    )
            sid = str(result["sample_id"])
            if sid in results_by_sid:
                raise ControlAnalysisError(f"duplicate sample_id in {arm} results: {sid}")
            results_by_sid[sid] = result
        if seen_indices != set(expected_by_index):
            raise ControlAnalysisError(f"{arm} results do not exactly cover {key[0]}")
        headers.append(header)
        result_hashes[str(result_path)] = file_sha256(result_path)
    if seen_manifest_keys != set(manifest_keys):
        missing = sorted(key[0] for key in set(manifest_keys) - seen_manifest_keys)
        raise ControlAnalysisError(f"missing {arm} result files for manifests: {missing}")
    expected_sids = {str(row["sample_id"]) for row in manifest_rows}
    if set(results_by_sid) != expected_sids:
        raise ControlAnalysisError(f"{arm} results do not exactly cover audited sample IDs")
    return results_by_sid, headers, result_hashes


def _outcome(
    manifest: dict[str, Any], result: dict[str, Any], arm: str
) -> dict[str, Any]:
    sid = str(manifest["sample_id"])
    candidates = _validate_candidates(manifest, f"{arm} manifest record {sid}")
    raw_prediction = parse_primary_diagnosis(
        result.get("response", ""), [label for label, _ in candidates]
    )
    c100_prediction = str(manifest["c100_anchor_prediction"])
    authoritative = {**manifest, "response": result.get("response", "")}
    final_prediction, constraint_reason = pair_constrained_prediction(
        authoritative, c100_prediction
    )
    dataset = str(manifest["dataset"])
    ground_truth = str(manifest["ground_truth"])
    c100_correct = int(
        evaluation_label(dataset, c100_prediction)
        == evaluation_label(dataset, ground_truth)
    )
    final_correct = int(
        evaluation_label(dataset, final_prediction)
        == evaluation_label(dataset, ground_truth)
    )
    changed = evaluation_label(dataset, final_prediction) != evaluation_label(
        dataset, c100_prediction
    )
    if not changed:
        category = "unchanged"
    elif c100_correct == 0 and final_correct == 1:
        category = "gain"
    elif c100_correct == 1 and final_correct == 0:
        category = "regression"
    elif c100_correct == 0 and final_correct == 0:
        category = "changed_wrong"
    else:  # Defensive: evaluation-label equivalence should make this impossible.
        raise ControlAnalysisError(f"unclassifiable paired outcome for {arm} sample {sid}")
    pair = {
        evaluation_label(dataset, str(manifest["rule_disease_a"])),
        evaluation_label(dataset, str(manifest["rule_disease_b"])),
    }
    if evaluation_label(dataset, final_prediction) not in pair:
        raise ControlAnalysisError(f"pair safety failed for {arm} sample {sid}")
    supported = str(manifest.get("trigger_supported_disease", "")).strip()
    if arm != "neutral" and changed and supported and evaluation_label(
        dataset, final_prediction
    ) != evaluation_label(dataset, supported):
        raise ControlAnalysisError(f"trigger-direction safety failed for {arm} sample {sid}")
    return {
        "sample_id": sid,
        "dataset": dataset,
        "rule_id": str(manifest["rule_id"]),
        "rule_key": str(manifest["rule_key"]),
        "ground_truth": ground_truth,
        "c100_prediction": c100_prediction,
        "raw_prediction": raw_prediction,
        "final_prediction": final_prediction,
        "valid": int(bool(raw_prediction)),
        "c100_correct": c100_correct,
        "correct": final_correct,
        "changed": int(changed),
        "outcome": category,
        "constraint_reason": constraint_reason,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = len(rows)
    reasons = Counter(str(row["constraint_reason"]) for row in rows)
    value: dict[str, Any] = {
        "samples": samples,
        "valid": sum(int(row["valid"]) for row in rows),
        "invalid": sum(not int(row["valid"]) for row in rows),
        "correct": sum(int(row["correct"]) for row in rows),
        "accuracy": sum(int(row["correct"]) for row in rows) / samples if samples else 0.0,
        "unchanged": sum(row["outcome"] == "unchanged" for row in rows),
        "changed": sum(int(row["changed"]) for row in rows),
        "changed_wrong": sum(row["outcome"] == "changed_wrong" for row in rows),
        "gain": sum(row["outcome"] == "gain" for row in rows),
        "regression": sum(row["outcome"] == "regression" for row in rows),
    }
    value["net"] = value["gain"] - value["regression"]
    value["constraint_reasons"] = dict(sorted(reasons.items()))
    if value["valid"] + value["invalid"] != samples:
        raise ControlAnalysisError("validity decomposition is incomplete")
    if (
        value["unchanged"]
        + value["changed_wrong"]
        + value["gain"]
        + value["regression"]
        != samples
    ):
        raise ControlAnalysisError("outcome decomposition is incomplete")
    return value


def _stratified_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_dataset_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)
        by_rule[str(row["rule_id"])].append(row)
        by_dataset_rule[f"{row['dataset']}::{row['rule_id']}"].append(row)
    return {
        "overall": _summary(rows),
        "by_dataset": {key: _summary(by_dataset[key]) for key in sorted(by_dataset)},
        "by_rule": {key: _summary(by_rule[key]) for key in sorted(by_rule)},
        "by_dataset_rule": {
            key: _summary(by_dataset_rule[key]) for key in sorted(by_dataset_rule)
        },
    }


def _binomial_lower_tail(total: int, maximum: int) -> float:
    if maximum < 0:
        return 0.0
    if maximum >= total:
        return 1.0
    logs = [
        math.lgamma(total + 1)
        - math.lgamma(index + 1)
        - math.lgamma(total - index + 1)
        - total * math.log(2.0)
        for index in range(maximum + 1)
    ]
    anchor = max(logs)
    return min(1.0, math.exp(anchor) * math.fsum(math.exp(value - anchor) for value in logs))


def exact_mcnemar_two_sided(true_only: int, control_only: int) -> float:
    total = true_only + control_only
    if total == 0:
        return 1.0
    return min(1.0, 2.0 * _binomial_lower_tail(total, min(true_only, control_only)))


def exact_binomial_true_superior(true_only: int, control_only: int) -> float:
    total = true_only + control_only
    if total == 0:
        return 1.0
    threshold = true_only
    if threshold <= total // 2:
        return max(0.0, 1.0 - _binomial_lower_tail(total, threshold - 1))
    return _binomial_lower_tail(total, total - threshold)


def _bootstrap(
    differences: np.ndarray,
    datasets: list[str],
    *,
    bootstraps: int,
    seed: int,
) -> dict[str, Any]:
    dataset_array = np.asarray(datasets, dtype=object)
    strata = {
        dataset: np.where(dataset_array == dataset)[0]
        for dataset in sorted(set(datasets))
    }
    rng = np.random.default_rng(seed)
    values = np.empty(bootstraps, dtype=float)
    for index in range(bootstraps):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in strata.values()]
        )
        values[index] = float(differences[sampled].mean())
    return {
        "method": "dataset-stratified paired nonparametric bootstrap",
        "bootstraps": bootstraps,
        "seed": seed,
        "point_difference": float(differences.mean()),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
    }


def _comparison(
    true_rows: dict[str, dict[str, Any]],
    control_rows: dict[str, dict[str, Any]],
    *,
    control_arm: str,
    bootstraps: int,
    seed: int,
) -> dict[str, Any]:
    if set(true_rows) != set(control_rows):
        raise ControlAnalysisError(f"true/{control_arm} analyzed sample sets differ")
    samples: list[dict[str, Any]] = []
    for sid in sorted(true_rows):
        true = true_rows[sid]
        control = control_rows[sid]
        if true["dataset"] != control["dataset"] or true["rule_id"] != control["rule_id"]:
            raise ControlAnalysisError(f"true/{control_arm} stratum mismatch for {sid}")
        samples.append(
            {
                "sample_id": sid,
                "dataset": true["dataset"],
                "rule_id": true["rule_id"],
                "c100_correct": true["c100_correct"],
                "true_valid": true["valid"],
                f"{control_arm}_valid": control["valid"],
                "true_prediction": true["final_prediction"],
                f"{control_arm}_prediction": control["final_prediction"],
                "true_correct": true["correct"],
                f"{control_arm}_correct": control["correct"],
                "correctness_difference": int(true["correct"] - control["correct"]),
                "true_outcome": true["outcome"],
                f"{control_arm}_outcome": control["outcome"],
                "true_constraint_reason": true["constraint_reason"],
                f"{control_arm}_constraint_reason": control["constraint_reason"],
            }
        )
    true_only = sum(
        row["true_correct"] == 1 and row[f"{control_arm}_correct"] == 0
        for row in samples
    )
    control_only = sum(
        row["true_correct"] == 0 and row[f"{control_arm}_correct"] == 1
        for row in samples
    )
    differences = np.asarray(
        [row["correctness_difference"] for row in samples], dtype=float
    )
    return {
        "comparison": f"true_vs_{control_arm}",
        "samples": len(samples),
        "mean_correctness_difference": float(differences.mean()),
        "true_only_correct": true_only,
        "control_only_correct": control_only,
        "both_correct": sum(
            row["true_correct"] == 1 and row[f"{control_arm}_correct"] == 1
            for row in samples
        ),
        "neither_correct": sum(
            row["true_correct"] == 0 and row[f"{control_arm}_correct"] == 0
            for row in samples
        ),
        "exact_mcnemar_two_sided_p": exact_mcnemar_two_sided(true_only, control_only),
        "exact_binomial_one_sided_true_superior_p": exact_binomial_true_superior(
            true_only, control_only
        ),
        "bootstrap_95_ci": _bootstrap(
            differences,
            [str(row["dataset"]) for row in samples],
            bootstraps=bootstraps,
            seed=seed,
        ),
        "sample_level_differences": samples,
    }


def analyze_paired_controls(
    *,
    base_q100: Path,
    release_path: Path,
    true_manifests: Sequence[str | Path],
    true_results: Sequence[str | Path],
    control_root: Path,
    neutral_results: Sequence[str | Path],
    shuffled_results: Sequence[str | Path],
    output: Path,
    bootstraps: int = 10000,
    seed: int = 20260826,
) -> dict[str, Any]:
    """Validate all frozen artifacts and write one self-hashed analysis JSON."""

    if isinstance(bootstraps, bool) or not isinstance(bootstraps, int) or bootstraps <= 0:
        raise ControlAnalysisError("bootstraps must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ControlAnalysisError("seed must be an integer")
    base_q100 = Path(base_q100)
    release_path = Path(release_path)
    control_root = Path(control_root)
    output = Path(output)

    base_rows, base_by_sid = _load_base(base_q100)
    release = _read_json(release_path)
    try:
        rules_by_id, _ = _release_rules(release)
    except ValueError as error:
        raise ControlAnalysisError(f"invalid frozen release: {error}") from error

    audit_path = control_root / "control_manifest_audit.json"
    audit = _read_json(audit_path)
    if audit.get("schema_version") != CONTROL_SCHEMA_VERSION:
        raise ControlAnalysisError("control audit schema version mismatch")
    if audit.get("status") != "PASS":
        raise ControlAnalysisError("control audit is not PASS")
    _validate_self_hash(audit, "audit_sha256", "control audit")
    pairing = audit.get("pairing")
    if not isinstance(pairing, list) or not pairing:
        raise ControlAnalysisError("control audit pairing must be a non-empty list")
    if audit.get("pairing_sha256") != canonical_hash(pairing):
        raise ControlAnalysisError("control audit pairing SHA mismatch")
    if _integer(audit.get("records_per_arm"), "control records_per_arm") != len(pairing):
        raise ControlAnalysisError("control audit pairing count mismatch")
    required_invariants = (
        "sample_pairing_complete",
        "candidate_probabilities_preserved",
        "c100_anchor_preserved",
        "c100_anchor_pair_membership_verified",
        "rule_reliability_matches_release",
        "source_indices_preserved",
        "prompt_outside_rule_section_preserved",
        "shuffled_pair_labels_preserved",
    )
    invariants = audit.get("invariants")
    if not isinstance(invariants, dict) or any(invariants.get(key) is not True for key in required_invariants):
        raise ControlAnalysisError("control audit required invariants are not all true")

    base_audit = audit.get("base_q100")
    if not isinstance(base_audit, dict):
        raise ControlAnalysisError("control audit missing base_q100 provenance")
    if base_audit.get("file_sha256") != file_sha256(base_q100):
        raise ControlAnalysisError("base_q100 file SHA disagrees with control audit")
    if _integer(base_audit.get("records"), "audited base records") != len(base_rows):
        raise ControlAnalysisError("base_q100 record count disagrees with control audit")
    if base_audit.get("content_sha256") != canonical_hash(base_rows):
        raise ControlAnalysisError("base_q100 content SHA disagrees with control audit")

    release_audit = audit.get("release")
    if not isinstance(release_audit, dict):
        raise ControlAnalysisError("control audit missing release provenance")
    if release_audit.get("file_sha256") != file_sha256(release_path):
        raise ControlAnalysisError("release file SHA disagrees with control audit")
    if release_audit.get("content_sha256") != release.get("content_sha256"):
        raise ControlAnalysisError("release content SHA disagrees with control audit")
    if release_audit.get("release_id") != release.get("release_id"):
        raise ControlAnalysisError("release ID disagrees with control audit")

    true_paths = _expand_paths(true_manifests, "true manifests")
    true_audit = audit.get("true_manifest")
    if not isinstance(true_audit, dict) or not isinstance(true_audit.get("files"), dict):
        raise ControlAnalysisError("control audit missing true-manifest provenance")
    expected_true_files = {str(name): str(value) for name, value in true_audit["files"].items()}
    if {path.name for path in true_paths} != set(expected_true_files):
        raise ControlAnalysisError("provided true manifests do not exactly match the control audit")
    for path in true_paths:
        if file_sha256(path) != expected_true_files[path.name]:
            raise ControlAnalysisError(f"true manifest file SHA mismatch: {path.name}")
    parent_dirs = {path.parent.resolve() for path in true_paths}
    if len(parent_dirs) != 1:
        raise ControlAnalysisError("true manifests must share one manifest-audit directory")
    true_manifest_audit_path = next(iter(parent_dirs)) / "manifest_audit.json"
    if file_sha256(true_manifest_audit_path) != true_audit.get("audit_file_sha256"):
        raise ControlAnalysisError("true manifest audit file SHA mismatch")
    true_manifest_audit = _read_json(true_manifest_audit_path)
    if true_manifest_audit.get("status") != "PASS":
        raise ControlAnalysisError("true manifest audit is not PASS")
    if true_manifest_audit.get("release_sha256") != release["content_sha256"]:
        raise ControlAnalysisError("true manifest audit release SHA mismatch")
    cohort = str(audit.get("cohort", ""))
    if not cohort or true_manifest_audit.get("cohort") != cohort:
        raise ControlAnalysisError("true manifest cohort mismatch")

    true_rows, true_locations = _load_true_manifests(true_paths)
    if _integer(true_audit.get("records"), "audited true records") != len(true_rows):
        raise ControlAnalysisError("true manifest record count disagrees with control audit")
    if true_audit.get("content_sha256") != canonical_hash(true_rows):
        raise ControlAnalysisError("true manifest content SHA disagrees with control audit")
    if _integer(true_manifest_audit.get("included_with_rule"), "true included_with_rule") != len(true_rows):
        raise ControlAnalysisError("true manifest audit included count mismatch")

    pairing_by_sid = _unique_index(pairing, "sample_id", "control pairing")
    true_by_sid = _unique_index(true_rows, "sample_id", "true manifests")
    if set(pairing_by_sid) != set(true_by_sid):
        raise ControlAnalysisError("pairing does not exactly cover true-manifest samples")
    for sid, row in true_by_sid.items():
        base = base_by_sid.get(sid)
        if base is None:
            raise ControlAnalysisError(f"true sample is missing from base_q100: {sid}")
        rule = rules_by_id.get(str(row.get("rule_id", "")))
        if rule is None:
            raise ControlAnalysisError(f"true sample references unknown release rule: {sid}")
        _validate_true_row(
            row,
            base=base,
            pairing=pairing_by_sid[sid],
            release=release,
            rule=rule,
            cohort=cohort,
        )
    expected_dataset_counts = dict(
        sorted(Counter(str(row["dataset"]) for row in true_rows).items())
    )
    expected_rule_counts = dict(
        sorted(Counter(str(row["rule_id"]) for row in true_rows).items())
    )
    if audit.get("dataset_counts") != expected_dataset_counts:
        raise ControlAnalysisError("control audit dataset counts do not replay")
    if audit.get("rule_counts") != expected_rule_counts:
        raise ControlAnalysisError("control audit rule counts do not replay")
    if audit.get("conditions") != {
        "neutral": NEUTRAL_CONDITION,
        "shuffled": SHUFFLED_CONDITION,
    }:
        raise ControlAnalysisError("control audit condition labels mismatch")

    arm_manifest_paths: dict[str, list[Path]] = {"true": true_paths}
    arm_manifest_rows: dict[str, list[dict[str, Any]]] = {"true": true_rows}
    for arm in ("neutral", "shuffled"):
        paths = _manifest_files_from_audit(control_root, audit, arm)
        rows = _load_control_rows(paths, arm)
        arm_audit = audit["controls"][arm]
        if _integer(arm_audit.get("records"), f"audited {arm} records") != len(rows):
            raise ControlAnalysisError(f"{arm} manifest record count disagrees with control audit")
        if arm_audit.get("content_sha256") != canonical_hash(rows):
            raise ControlAnalysisError(f"{arm} manifest content SHA disagrees with control audit")
        by_sid = _unique_index(rows, "sample_id", f"{arm} manifests")
        if set(by_sid) != set(true_by_sid):
            raise ControlAnalysisError(f"{arm} manifests do not exactly cover paired samples")
        for sid, source in true_by_sid.items():
            rule = rules_by_id[str(source["rule_id"])]
            _validate_control_reconstruction(
                source=source,
                source_location=true_locations[sid],
                rule=rule,
                actual=by_sid[sid],
                arm=arm,
                pairing=pairing_by_sid[sid],
                seed=_integer(audit.get("seed"), "control seed"),
            )
        arm_manifest_paths[arm] = paths
        arm_manifest_rows[arm] = rows

    all_indices: list[int] = []
    for rows in arm_manifest_rows.values():
        all_indices.extend(_integer(row["global_index"], "manifest global_index") for row in rows)
    if len(all_indices) != len(set(all_indices)):
        raise ControlAnalysisError("global_index collision across the three paired arms")
    base_indices = {
        _integer(row["global_index"], "base_q100 global_index") for row in base_rows
    }
    if base_indices.intersection(all_indices):
        raise ControlAnalysisError("paired-arm global_index collides with base_q100")

    result_specs = {
        "true": true_results,
        "neutral": neutral_results,
        "shuffled": shuffled_results,
    }
    result_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    headers_by_arm: dict[str, list[dict[str, Any]]] = {}
    result_hashes_by_arm: dict[str, dict[str, str]] = {}
    for arm in ARM_ORDER:
        results, headers, hashes = _load_results(
            specs=result_specs[arm],
            manifests=arm_manifest_paths[arm],
            manifest_rows=arm_manifest_rows[arm],
            arm=arm,
        )
        result_by_arm[arm] = results
        headers_by_arm[arm] = headers
        result_hashes_by_arm[arm] = hashes
    sample_sets = {arm: set(result_by_arm[arm]) for arm in ARM_ORDER}
    if len({frozenset(values) for values in sample_sets.values()}) != 1:
        raise ControlAnalysisError("three result arms do not contain exactly one shared sample set")

    reference_provenance: dict[str, Any] | None = None
    for arm in ARM_ORDER:
        for header in headers_by_arm[arm]:
            provenance = {field: header.get(field) for field in RUN_COMPARABILITY_FIELDS}
            missing = [field for field, value in provenance.items() if value is None]
            if missing:
                raise ControlAnalysisError(f"{arm} run header missing comparability fields: {missing}")
            if reference_provenance is None:
                reference_provenance = provenance
            elif provenance != reference_provenance:
                raise ControlAnalysisError(f"inference provenance differs across arms: {arm}")

    analyzed_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    for arm in ARM_ORDER:
        manifests_by_sid = _unique_index(
            arm_manifest_rows[arm], "sample_id", f"{arm} manifests"
        )
        analyzed = {
            sid: _outcome(manifests_by_sid[sid], result_by_arm[arm][sid], arm)
            for sid in sorted(manifests_by_sid)
        }
        analyzed_by_arm[arm] = analyzed
        summaries[arm] = _stratified_summary(list(analyzed.values()))

    comparisons = {
        "true_vs_neutral": _comparison(
            analyzed_by_arm["true"],
            analyzed_by_arm["neutral"],
            control_arm="neutral",
            bootstraps=bootstraps,
            seed=seed,
        ),
        "true_vs_shuffled": _comparison(
            analyzed_by_arm["true"],
            analyzed_by_arm["shuffled"],
            control_arm="shuffled",
            bootstraps=bootstraps,
            seed=seed + 1,
        ),
    }

    payload: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "PASS",
        "protocol": (
            "strict three-arm sample pairing; exact frozen-artifact SHA replay; existing "
            "closed-set diagnosis parser; V1.1 pair-constrained fallback; neutral has no "
            "trigger direction and shuffled uses the direction already swapped in its manifest"
        ),
        "cohort": cohort,
        "samples_per_arm": len(true_rows),
        "bootstraps": bootstraps,
        "seed": seed,
        "input_provenance": {
            "base_q100": {
                "path": str(base_q100),
                "file_sha256": file_sha256(base_q100),
                "content_sha256": canonical_hash(base_rows),
            },
            "release": {
                "path": str(release_path),
                "release_id": release["release_id"],
                "file_sha256": file_sha256(release_path),
                "content_sha256": release["content_sha256"],
            },
            "true_manifests": {
                str(path): file_sha256(path) for path in true_paths
            },
            "control_manifest_audit": {
                "path": str(audit_path),
                "file_sha256": file_sha256(audit_path),
                "audit_sha256": audit["audit_sha256"],
                "pairing_sha256": audit["pairing_sha256"],
            },
            "results": result_hashes_by_arm,
            "inference_comparability": reference_provenance,
        },
        "arms": summaries,
        "comparisons": comparisons,
    }
    payload["content_sha256"] = canonical_hash(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Strict paired analysis of V1.1 true, neutral, and shuffled results."
    )
    result.add_argument("--base-q100", type=Path, required=True)
    result.add_argument("--release", dest="release_path", type=Path, required=True)
    result.add_argument("--true-manifests", nargs="+", required=True)
    result.add_argument("--true-results", nargs="+", required=True)
    result.add_argument("--control-root", type=Path, required=True)
    result.add_argument("--neutral-results", nargs="+", required=True)
    result.add_argument("--shuffled-results", nargs="+", required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--bootstraps", type=int, default=10000)
    result.add_argument("--seed", type=int, default=20260826)
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        payload = analyze_paired_controls(
            base_q100=args.base_q100,
            release_path=args.release_path,
            true_manifests=args.true_manifests,
            true_results=args.true_results,
            control_root=args.control_root,
            neutral_results=args.neutral_results,
            shuffled_results=args.shuffled_results,
            output=args.output,
            bootstraps=args.bootstraps,
            seed=args.seed,
        )
    except (ControlAnalysisError, OSError) as error:
        raise SystemExit(f"paired control analysis failed: {error}") from error
    print(
        json.dumps(
            {
                "status": payload["status"],
                "samples_per_arm": payload["samples_per_arm"],
                "arms": {arm: payload["arms"][arm]["overall"] for arm in ARM_ORDER},
                "comparisons": {
                    key: {
                        field: value
                        for field, value in comparison.items()
                        if field != "sample_level_differences"
                    }
                    for key, comparison in payload["comparisons"].items()
                },
                "content_sha256": payload["content_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
