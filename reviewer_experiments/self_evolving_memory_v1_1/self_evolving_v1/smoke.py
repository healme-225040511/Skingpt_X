#!/usr/bin/env python3
"""Build small, deterministic, label-blind inference smoke manifests.

This module only selects frozen records.  It never edits a selected record or
renumbers its identifiers.  Both builders validate all source shards before
creating the output directory and emit byte-, content-, and per-record hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Sequence


DATASETS = ("DDI", "DermNet23", "Fit17k", "HAM10000")
RULE_SCHEMA_VERSION = "v1.1-rule-smoke-v1"
DIAGNOSIS_SCHEMA_VERSION = "v1.1-diagnosis-smoke-v1"
MAX_DIAGNOSIS_SMOKE_RECORDS = 96
RULE_SHARD_RE = re.compile(r"^rules_shard_(?P<index>\d+)\.jsonl$")
DIAGNOSIS_SHARD_RE = re.compile(r"^(?P<cohort>.+)_shard_(?P<index>\d+)\.jsonl$")


class SmokeManifestError(ValueError):
    """Raised when a frozen input is incomplete, duplicated, or corrupted."""


class _DuplicateJSONKey(ValueError):
    pass


def canonical_hash(value: Any) -> str:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise SmokeManifestError(f"value is not canonical JSON: {error}") from error
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _loads_strict(raw: str, context: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, _DuplicateJSONKey, ValueError) as error:
        raise SmokeManifestError(f"invalid JSON at {context}: {error}") from error


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SmokeManifestError(f"missing JSON file: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise SmokeManifestError(f"invalid UTF-8 JSON file: {path}") from error
    value = _loads_strict(raw, str(path))
    if not isinstance(value, dict):
        raise SmokeManifestError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SmokeManifestError(f"missing JSONL file: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise SmokeManifestError(f"invalid UTF-8 JSONL file: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise SmokeManifestError(f"blank JSONL record at {path}:{line_number}")
        value = _loads_strict(line, f"{path}:{line_number}")
        if not isinstance(value, dict):
            raise SmokeManifestError(
                f"JSONL record must be an object at {path}:{line_number}"
            )
        rows.append(value)
    return rows


def _discover_rule_shards(manifest_dir: Path) -> list[Path]:
    if not manifest_dir.is_dir():
        raise SmokeManifestError(f"missing rule manifest directory: {manifest_dir}")
    indexed: list[tuple[int, Path]] = []
    for path in manifest_dir.iterdir():
        if not path.is_file() or not path.name.startswith("rules_shard_"):
            continue
        match = RULE_SHARD_RE.fullmatch(path.name)
        if match is None:
            raise SmokeManifestError(f"malformed rule shard filename: {path.name}")
        indexed.append((int(match.group("index")), path))
    return _contiguous_shards(indexed, "rule")


def _discover_diagnosis_shards(manifest_dir: Path) -> tuple[str, list[Path]]:
    if not manifest_dir.is_dir():
        raise SmokeManifestError(
            f"missing diagnosis manifest directory: {manifest_dir}"
        )
    indexed: list[tuple[int, Path]] = []
    cohorts: set[str] = set()
    for path in manifest_dir.iterdir():
        if not path.is_file() or path.suffix != ".jsonl":
            continue
        match = DIAGNOSIS_SHARD_RE.fullmatch(path.name)
        if match is None:
            continue
        cohort = match.group("cohort")
        cohorts.add(cohort)
        indexed.append((int(match.group("index")), path))
    if len(cohorts) != 1:
        rendered = ", ".join(sorted(cohorts)) or "none"
        raise SmokeManifestError(
            f"expected exactly one build-cohort shard prefix, found: {rendered}"
        )
    return next(iter(cohorts)), _contiguous_shards(indexed, "diagnosis")


def _contiguous_shards(indexed: list[tuple[int, Path]], kind: str) -> list[Path]:
    if not indexed:
        raise SmokeManifestError(f"no {kind} manifest shards found")
    indices = [index for index, _ in indexed]
    if len(indices) != len(set(indices)):
        raise SmokeManifestError(f"duplicate {kind} shard index")
    expected = list(range(len(indices)))
    if sorted(indices) != expected:
        raise SmokeManifestError(
            f"non-contiguous {kind} shard indices: found={sorted(indices)} expected={expected}"
        )
    return [path for _, path in sorted(indexed)]


def _require_fields(
    row: dict[str, Any], fields: Iterable[str], context: str
) -> None:
    missing = [field for field in fields if field not in row or row[field] is None]
    if missing:
        raise SmokeManifestError(
            f"{context} missing fields: {', '.join(sorted(missing))}"
        )


def _require_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SmokeManifestError(f"{context} must be a JSON integer")
    return value


def _require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SmokeManifestError(f"{context} must be a non-empty string")
    return value


def _load_shards(
    paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    file_audit: list[dict[str, Any]] = []
    record_audit: list[dict[str, Any]] = []
    for path in paths:
        file_rows = _read_jsonl(path)
        rows.extend(file_rows)
        file_audit.append(
            {
                "path": path.name,
                "records": len(file_rows),
                "file_sha256": file_sha256(path),
                "content_sha256": canonical_hash(file_rows),
            }
        )
        for line_number, row in enumerate(file_rows, 1):
            record_audit.append(
                {
                    "source_file": path.name,
                    "source_line": line_number,
                    "global_index": row.get("global_index"),
                    "sample_id": row.get("sample_id"),
                    "rule_id": row.get("rule_id"),
                    "content_sha256": canonical_hash(row),
                }
            )
    if not rows:
        raise SmokeManifestError("source manifest contains no records")
    return rows, file_audit, record_audit


def _unique_field(rows: Sequence[dict[str, Any]], field: str, context: str) -> None:
    seen: set[Any] = set()
    duplicates: set[Any] = set()
    for row in rows:
        value = row[field]
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        rendered = ", ".join(str(value) for value in sorted(duplicates, key=str))
        raise SmokeManifestError(f"duplicate {field} in {context}: {rendered}")


def _validate_prompt_hash(row: dict[str, Any], context: str) -> None:
    if "prompt_sha256" not in row:
        return
    stored = _require_nonempty_string(row["prompt_sha256"], f"{context} prompt_sha256")
    expected = canonical_hash(row["prompt"])
    if stored != expected:
        raise SmokeManifestError(
            f"prompt hash mismatch in {context}: stored={stored} expected={expected}"
        )


def _stable_rank(seed: int, namespace: str, *parts: Any) -> str:
    payload = "::".join([str(seed), namespace, *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_output_target(output_dir: Path, shards: int) -> None:
    if isinstance(shards, bool) or not isinstance(shards, int) or shards <= 0:
        raise SmokeManifestError("shards must be a positive integer")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SmokeManifestError(
            f"refusing to write into non-empty output directory: {output_dir}"
        )


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _write_outputs(
    output_dir: Path,
    selected: Sequence[dict[str, Any]],
    *,
    shards: int,
    filename_prefix: str,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_audit: list[dict[str, Any]] = []
    written_rows: list[dict[str, Any]] = []
    for shard_index in range(shards):
        shard_rows = list(selected[shard_index::shards])
        path = output_dir / f"{filename_prefix}_shard_{shard_index}.jsonl"
        _write_jsonl(path, shard_rows)
        reread = _read_jsonl(path)
        if reread != shard_rows:
            raise SmokeManifestError(f"written output verification failed: {path}")
        written_rows.extend(reread)
        output_audit.append(
            {
                "path": path.name,
                "records": len(shard_rows),
                "file_sha256": file_sha256(path),
                "content_sha256": canonical_hash(shard_rows),
            }
        )
    expected_hashes = Counter(canonical_hash(row) for row in selected)
    written_hashes = Counter(canonical_hash(row) for row in written_rows)
    if expected_hashes != written_hashes:
        raise SmokeManifestError("output shards do not exactly cover selected records")
    return output_audit


def _selected_record_audit(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "global_index": row.get("global_index"),
            "sample_id": row.get("sample_id"),
            "rule_id": row.get("rule_id"),
            "dataset": row.get("dataset"),
            "content_sha256": canonical_hash(row),
        }
        for row in rows
    ]


def _finish_audit(output_dir: Path, audit: dict[str, Any]) -> dict[str, Any]:
    audit["audit_sha256"] = canonical_hash(audit)
    path = output_dir / "smoke_manifest_audit.json"
    path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    reread = _read_json_object(path)
    stored = reread.pop("audit_sha256", None)
    if stored != canonical_hash(reread):
        raise SmokeManifestError("written smoke audit hash verification failed")
    return audit


def build_rule_smoke_manifests(
    *,
    manifest_dir: Path,
    output_dir: Path,
    per_dataset: int = 2,
    shards: int = 2,
    seed: int = 20260826,
    datasets: Sequence[str] = DATASETS,
) -> dict[str, Any]:
    """Select an equal stable-hash sample of frozen rule proposals."""
    if (
        isinstance(per_dataset, bool)
        or not isinstance(per_dataset, int)
        or per_dataset <= 0
    ):
        raise SmokeManifestError("per_dataset must be a positive integer")
    if len(datasets) != len(set(datasets)) or not datasets:
        raise SmokeManifestError("datasets must be a non-empty unique sequence")
    paths = _discover_rule_shards(manifest_dir)
    rows, input_files, input_records = _load_shards(paths)
    allowed = set(datasets)
    for position, row in enumerate(rows):
        context = f"rule proposal record {position}"
        _require_fields(
            row, ("global_index", "dataset", "rule_id", "rule_key", "prompt"), context
        )
        _require_integer(row["global_index"], f"{context} global_index")
        dataset = _require_nonempty_string(row["dataset"], f"{context} dataset")
        if dataset not in allowed:
            raise SmokeManifestError(f"unexpected dataset in {context}: {dataset}")
        _require_nonempty_string(row["rule_id"], f"{context} rule_id")
        _require_nonempty_string(row["rule_key"], f"{context} rule_key")
        _require_nonempty_string(row["prompt"], f"{context} prompt")
        _validate_prompt_hash(row, context)
    _unique_field(rows, "global_index", "rule proposal manifests")
    _unique_field(rows, "rule_id", "rule proposal manifests")
    _unique_field(rows, "rule_key", "rule proposal manifests")
    content_hashes = [canonical_hash(row) for row in rows]
    if len(content_hashes) != len(set(content_hashes)):
        raise SmokeManifestError("duplicate complete record in rule proposal manifests")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    insufficient = {
        dataset: len(grouped.get(dataset, []))
        for dataset in datasets
        if len(grouped.get(dataset, [])) < per_dataset
    }
    if insufficient:
        raise SmokeManifestError(
            f"insufficient records for equal dataset smoke selection: {insufficient}"
        )
    selected: list[dict[str, Any]] = []
    for dataset in datasets:
        ordered = sorted(
            grouped[dataset],
            key=lambda row: (
                _stable_rank(
                    seed,
                    "rule",
                    dataset,
                    row["rule_id"],
                    row["global_index"],
                ),
                str(row["rule_id"]),
                int(row["global_index"]),
            ),
        )
        selected.extend(ordered[:per_dataset])

    _validate_output_target(output_dir, shards)
    output_files = _write_outputs(
        output_dir,
        selected,
        shards=shards,
        filename_prefix="rules_smoke",
    )
    input_dataset_counts = Counter(str(row["dataset"]) for row in rows)
    selected_dataset_counts = Counter(str(row["dataset"]) for row in selected)
    audit: dict[str, Any] = {
        "schema_version": RULE_SCHEMA_VERSION,
        "status": "PASS",
        "selection_protocol": (
            "equal per-dataset deterministic SHA-256 ranking; source records are not mutated"
        ),
        "seed": seed,
        "selection_fields": ["dataset", "rule_id", "global_index"],
        "label_fields_used_for_selection": [],
        "parameters": {
            "datasets": list(datasets),
            "per_dataset": per_dataset,
            "shards": shards,
        },
        "input": {
            "records": len(rows),
            "dataset_counts": dict(sorted(input_dataset_counts.items())),
            "files": input_files,
            "content_sha256": canonical_hash(rows),
            "record_hashes": input_records,
        },
        "selection": {
            "records": len(selected),
            "dataset_counts": dict(sorted(selected_dataset_counts.items())),
            "equal_dataset_coverage": len(set(selected_dataset_counts.values())) == 1,
            "content_sha256": canonical_hash(selected),
            "identity_sha256": canonical_hash(
                [
                    [row["dataset"], row["rule_id"], row["global_index"]]
                    for row in selected
                ]
            ),
            "record_hashes": _selected_record_audit(selected),
        },
        "output": {
            "records": len(selected),
            "files": output_files,
            "content_sha256": canonical_hash(selected),
            "exact_selected_record_coverage": True,
        },
    }
    return _finish_audit(output_dir, audit)


def _round_robin_rules(
    dataset: str, rows: Sequence[dict[str, Any]], seed: int
) -> deque[dict[str, Any]]:
    by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_rule[str(row["rule_id"])].append(row)
    ordered_rule_ids = sorted(
        by_rule,
        key=lambda rule_id: (_stable_rank(seed, "diagnosis-rule", dataset, rule_id), rule_id),
    )
    queues: dict[str, deque[dict[str, Any]]] = {}
    for rule_id in ordered_rule_ids:
        ordered = sorted(
            by_rule[rule_id],
            key=lambda row: (
                _stable_rank(
                    seed,
                    "diagnosis-case",
                    dataset,
                    rule_id,
                    row["sample_id"],
                    row["global_index"],
                ),
                str(row["sample_id"]),
                int(row["global_index"]),
            ),
        )
        queues[rule_id] = deque(ordered)
    result: deque[dict[str, Any]] = deque()
    active = list(ordered_rule_ids)
    while active:
        next_active: list[str] = []
        for rule_id in active:
            queue = queues[rule_id]
            result.append(queue.popleft())
            if queue:
                next_active.append(rule_id)
        active = next_active
    return result


def _balanced_diagnosis_selection(
    rows: Sequence[dict[str, Any]], maximum_records: int, seed: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    dataset_order = sorted(
        grouped,
        key=lambda dataset: (_stable_rank(seed, "diagnosis-dataset", dataset), dataset),
    )
    queues = {
        dataset: _round_robin_rules(dataset, grouped[dataset], seed)
        for dataset in dataset_order
    }
    selected: list[dict[str, Any]] = []
    active = list(dataset_order)
    while active and len(selected) < maximum_records:
        next_active: list[str] = []
        for dataset in active:
            if len(selected) >= maximum_records:
                break
            queue = queues[dataset]
            selected.append(queue.popleft())
            if queue:
                next_active.append(dataset)
        active = next_active
    return selected


def build_diagnosis_smoke_manifests(
    *,
    manifest_dir: Path,
    output_dir: Path,
    maximum_records: int = MAX_DIAGNOSIS_SMOKE_RECORDS,
    shards: int = 2,
    seed: int = 20260826,
    datasets: Sequence[str] = DATASETS,
) -> dict[str, Any]:
    """Select a dataset/rule-balanced, label-blind diagnosis smoke cohort."""
    if (
        isinstance(maximum_records, bool)
        or not isinstance(maximum_records, int)
        or not 1 <= maximum_records <= MAX_DIAGNOSIS_SMOKE_RECORDS
    ):
        raise SmokeManifestError(
            f"maximum_records must be an integer in [1, {MAX_DIAGNOSIS_SMOKE_RECORDS}]"
        )
    if len(datasets) != len(set(datasets)) or not datasets:
        raise SmokeManifestError("datasets must be a non-empty unique sequence")
    cohort, paths = _discover_diagnosis_shards(manifest_dir)
    rows, input_files, input_records = _load_shards(paths)
    manifest_audit_path = manifest_dir / "manifest_audit.json"
    manifest_audit = _read_json_object(manifest_audit_path)
    if manifest_audit.get("status") != "PASS":
        raise SmokeManifestError("source build-cohort manifest audit is not PASS")
    if manifest_audit.get("cohort") != cohort:
        raise SmokeManifestError(
            "source build-cohort manifest audit cohort does not match shard names"
        )
    included = manifest_audit.get("included_with_rule")
    if isinstance(included, bool) or not isinstance(included, int) or included != len(rows):
        raise SmokeManifestError(
            "source build-cohort manifest audit record count does not match shards"
        )

    allowed = set(datasets)
    for position, row in enumerate(rows):
        context = f"diagnosis record {position}"
        _require_fields(
            row,
            ("global_index", "sample_id", "dataset", "rule_id", "prompt", "split_role"),
            context,
        )
        _require_integer(row["global_index"], f"{context} global_index")
        sample_id = _require_nonempty_string(row["sample_id"], f"{context} sample_id")
        dataset = _require_nonempty_string(row["dataset"], f"{context} dataset")
        if dataset not in allowed:
            raise SmokeManifestError(f"unexpected dataset in {context}: {dataset}")
        if not sample_id.startswith(dataset + "::"):
            raise SmokeManifestError(f"sample_id/dataset mismatch in {context}")
        _require_nonempty_string(row["rule_id"], f"{context} rule_id")
        _require_nonempty_string(row["prompt"], f"{context} prompt")
        if row["split_role"] != cohort:
            raise SmokeManifestError(f"split/cohort mismatch in {context}")
        _validate_prompt_hash(row, context)
    _unique_field(rows, "global_index", "diagnosis manifests")
    _unique_field(rows, "sample_id", "diagnosis manifests")
    content_hashes = [canonical_hash(row) for row in rows]
    if len(content_hashes) != len(set(content_hashes)):
        raise SmokeManifestError("duplicate complete record in diagnosis manifests")

    selected = _balanced_diagnosis_selection(rows, maximum_records, seed)
    _validate_output_target(output_dir, shards)
    output_files = _write_outputs(
        output_dir,
        selected,
        shards=shards,
        filename_prefix=f"{cohort}_smoke",
    )
    input_dataset_counts = Counter(str(row["dataset"]) for row in rows)
    selected_dataset_counts = Counter(str(row["dataset"]) for row in selected)
    input_rule_counts = Counter(str(row["rule_id"]) for row in rows)
    selected_rule_counts = Counter(str(row["rule_id"]) for row in selected)
    dataset_coverage = {
        dataset: {
            "input_records": input_dataset_counts.get(dataset, 0),
            "selected_records": selected_dataset_counts.get(dataset, 0),
            "record_fraction": (
                selected_dataset_counts.get(dataset, 0) / input_dataset_counts[dataset]
                if input_dataset_counts.get(dataset, 0)
                else 0.0
            ),
            "input_rules": len(
                {row["rule_id"] for row in rows if row["dataset"] == dataset}
            ),
            "selected_rules": len(
                {row["rule_id"] for row in selected if row["dataset"] == dataset}
            ),
        }
        for dataset in datasets
    }
    audit: dict[str, Any] = {
        "schema_version": DIAGNOSIS_SCHEMA_VERSION,
        "status": "PASS",
        "selection_protocol": (
            "deterministic dataset round-robin containing deterministic within-dataset "
            "rule round-robin; case ranks use only dataset, rule_id, sample_id, and global_index"
        ),
        "seed": seed,
        "cohort": cohort,
        "selection_fields": ["dataset", "rule_id", "sample_id", "global_index"],
        "label_fields_used_for_selection": [],
        "parameters": {
            "datasets": list(datasets),
            "maximum_records": maximum_records,
            "shards": shards,
        },
        "input": {
            "records": len(rows),
            "dataset_counts": dict(sorted(input_dataset_counts.items())),
            "rule_counts": dict(sorted(input_rule_counts.items())),
            "files": input_files,
            "content_sha256": canonical_hash(rows),
            "record_hashes": input_records,
            "build_cohort_audit": {
                "path": manifest_audit_path.name,
                "file_sha256": file_sha256(manifest_audit_path),
                "content_sha256": canonical_hash(manifest_audit),
            },
        },
        "selection": {
            "records": len(selected),
            "record_fraction": len(selected) / len(rows),
            "dataset_counts": dict(sorted(selected_dataset_counts.items())),
            "rule_counts": dict(sorted(selected_rule_counts.items())),
            "unique_rule_ids": len(selected_rule_counts),
            "dataset_coverage": dataset_coverage,
            "content_sha256": canonical_hash(selected),
            "identity_sha256": canonical_hash(
                [
                    [
                        row["dataset"],
                        row["rule_id"],
                        row["sample_id"],
                        row["global_index"],
                    ]
                    for row in selected
                ]
            ),
            "record_hashes": _selected_record_audit(selected),
        },
        "output": {
            "records": len(selected),
            "files": output_files,
            "content_sha256": canonical_hash(selected),
            "exact_selected_record_coverage": True,
            "global_indices_preserved": True,
            "sample_ids_preserved": True,
        },
    }
    return _finish_audit(output_dir, audit)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Build deterministic auditable rule/diagnosis smoke manifests."
    )
    commands = result.add_subparsers(dest="command", required=True)

    rules = commands.add_parser("rules", help="select rule-generation proposals")
    rules.add_argument("--manifest-dir", type=Path, required=True)
    rules.add_argument("--output", type=Path, required=True)
    rules.add_argument("--per-dataset", type=int, default=2)
    rules.add_argument("--shards", type=int, default=2)
    rules.add_argument("--seed", type=int, default=20260826)

    diagnosis = commands.add_parser(
        "diagnosis", help="select build-cohort diagnosis records"
    )
    diagnosis.add_argument("--manifest-dir", type=Path, required=True)
    diagnosis.add_argument("--output", type=Path, required=True)
    diagnosis.add_argument(
        "--maximum-records", type=int, default=MAX_DIAGNOSIS_SMOKE_RECORDS
    )
    diagnosis.add_argument("--shards", type=int, default=2)
    diagnosis.add_argument("--seed", type=int, default=20260826)
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        if args.command == "rules":
            audit = build_rule_smoke_manifests(
                manifest_dir=args.manifest_dir,
                output_dir=args.output,
                per_dataset=args.per_dataset,
                shards=args.shards,
                seed=args.seed,
            )
        else:
            audit = build_diagnosis_smoke_manifests(
                manifest_dir=args.manifest_dir,
                output_dir=args.output,
                maximum_records=args.maximum_records,
                shards=args.shards,
                seed=args.seed,
            )
    except SmokeManifestError as error:
        raise SystemExit(f"smoke manifest validation failed: {error}") from error
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
