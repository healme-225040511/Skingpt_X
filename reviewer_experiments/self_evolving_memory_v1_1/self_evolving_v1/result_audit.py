#!/usr/bin/env python3
"""Audit frozen-manifest coverage and strict JSON responses from formal inference.

The audit deliberately does not repair or normalize model output.  A result is
either byte-traceable to one frozen manifest record and strictly parseable, or
the audit fails and the affected record must be retried by the inference
pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "v1.1-formal-result-audit-v1"
RUN_FIELDS = ("model_revision", "code_revision", "batch_size", "max_new_tokens")
TRANSPORT_FIELDS = {
    "record_type",
    "response",
    "input_tokens",
    "output_tokens",
    "batch_seconds",
    "peak_memory_gib",
}


class _DuplicateJSONKey(ValueError):
    """Raised when a JSON object contains an ambiguous duplicate key."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _loads_strict(raw: str) -> Any:
    return json.loads(
        raw,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_json(left: Any, right: Any) -> bool:
    """Compare canonical JSON, preserving distinctions such as true versus 1."""
    return _canonical_bytes(left) == _canonical_bytes(right)


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing JSONL file: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"invalid UTF-8 JSONL file: {path}") from error
    if not lines:
        raise ValueError(f"empty JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(lines, 1):
        if not raw.strip():
            raise ValueError(f"blank line in {path} at line {line_number}")
        try:
            value = _loads_strict(raw)
        except (json.JSONDecodeError, _DuplicateJSONKey, ValueError) as error:
            raise ValueError(
                f"invalid JSON in {path} at line {line_number}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(f"non-object JSON in {path} at line {line_number}")
        rows.append(value)
    return rows


def _issue(issues: list[dict[str, Any]], code: str, **details: Any) -> None:
    issues.append({"code": code, **details})


def _candidate_labels(row: dict[str, Any]) -> list[str]:
    """Return the exact closed-set labels frozen into a diagnosis manifest."""
    if "candidate_labels" in row:
        raw = row["candidate_labels"]
        if not isinstance(raw, list) or not raw:
            raise ValueError("candidate_labels must be a non-empty list")
        labels = raw
    elif "candidate_probabilities" in row:
        probabilities = row["candidate_probabilities"]
        if not isinstance(probabilities, list) or not probabilities:
            raise ValueError("candidate_probabilities must be a non-empty list")
        labels = []
        for position, item in enumerate(probabilities):
            if not isinstance(item, list) or not item:
                raise ValueError(
                    f"candidate_probabilities[{position}] must be a non-empty list"
                )
            labels.append(item[0])
    elif "guideline_diseases" in row:
        raw = row["guideline_diseases"]
        if not isinstance(raw, list) or not raw:
            raise ValueError("guideline_diseases must be a non-empty list")
        labels = raw
    else:
        raise ValueError("manifest has no candidate label field")
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("candidate labels must be non-empty strings")
    if len(labels) != len(set(labels)):
        raise ValueError("candidate labels must be exactly unique")
    return labels


def _strict_response(response: Any, mode: str, manifest: dict[str, Any]) -> str | None:
    if not isinstance(response, str):
        return "response_not_string"
    try:
        parsed = _loads_strict(response)
    except (json.JSONDecodeError, _DuplicateJSONKey, ValueError):
        return "response_not_strict_json"
    if not isinstance(parsed, dict):
        return "response_not_json_object"
    if mode == "rule":
        return None
    try:
        labels = _candidate_labels(manifest)
    except ValueError:
        return "invalid_manifest_candidate_labels"
    diagnosis = parsed.get("PrimaryDiagnosis")
    if not isinstance(diagnosis, str):
        return "missing_or_nonstring_primary_diagnosis"
    if diagnosis not in labels:
        return "primary_diagnosis_outside_candidate_labels"
    return None


def _safe_file_metadata(path: Path, issues: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    value: dict[str, Any] = {"path": str(path.resolve())}
    try:
        value["byte_size"] = path.stat().st_size
        value["sha256"] = file_sha256(path)
    except OSError as error:
        _issue(issues, f"{kind}_file_unreadable", path=str(path), error=str(error))
    return value


def audit_results(
    *,
    manifests: Sequence[Path],
    results: Sequence[Path],
    mode: str,
) -> dict[str, Any]:
    """Return a self-contained audit payload; validation problems become issues."""
    if mode not in {"rule", "diagnosis"}:
        raise ValueError("mode must be 'rule' or 'diagnosis'")
    manifest_paths = [Path(path).resolve() for path in manifests]
    result_paths = [Path(path).resolve() for path in results]
    issues: list[dict[str, Any]] = []
    if not manifest_paths:
        _issue(issues, "no_manifests")
    if not result_paths:
        _issue(issues, "no_results")
    if len(manifest_paths) != len(set(manifest_paths)):
        _issue(issues, "duplicate_manifest_path_argument")
    if len(result_paths) != len(set(result_paths)):
        _issue(issues, "duplicate_result_path_argument")

    manifest_files: list[dict[str, Any]] = []
    result_files: list[dict[str, Any]] = []
    manifest_rows_by_path: dict[str, list[dict[str, Any]]] = {}
    manifest_by_index: dict[int, dict[str, Any]] = {}
    manifest_index_locations: dict[int, list[str]] = {}

    for path in manifest_paths:
        metadata = _safe_file_metadata(path, issues, "manifest")
        try:
            rows = _read_jsonl(path)
        except ValueError as error:
            _issue(issues, "invalid_manifest_jsonl", path=str(path), error=str(error))
            metadata.update({"records": None, "global_index_sha256": None})
            manifest_files.append(metadata)
            continue
        indices: list[int] = []
        valid_rows = True
        for line_number, row in enumerate(rows, 1):
            index = row.get("global_index")
            if not _is_integer(index):
                _issue(
                    issues,
                    "invalid_manifest_global_index",
                    path=str(path),
                    line=line_number,
                )
                valid_rows = False
                continue
            indices.append(index)
            manifest_index_locations.setdefault(index, []).append(f"{path}:{line_number}")
            manifest_by_index.setdefault(index, row)
        duplicates = sorted(index for index, count in Counter(indices).items() if count > 1)
        if duplicates:
            _issue(
                issues,
                "duplicate_manifest_global_index_within_file",
                path=str(path),
                global_indices=duplicates,
            )
        resolved = str(path)
        if resolved in manifest_rows_by_path:
            _issue(issues, "duplicate_resolved_manifest", path=resolved)
        elif valid_rows:
            manifest_rows_by_path[resolved] = rows
        metadata.update(
            {
                "records": len(rows),
                "global_index_sha256": canonical_hash(indices),
            }
        )
        manifest_files.append(metadata)

    cross_manifest_duplicates = sorted(
        index for index, locations in manifest_index_locations.items() if len(locations) > 1
    )
    if cross_manifest_duplicates:
        _issue(
            issues,
            "duplicate_manifest_global_index_global",
            global_indices=cross_manifest_duplicates,
            locations={
                str(index): manifest_index_locations[index]
                for index in cross_manifest_duplicates
            },
        )

    all_inference_rows: list[tuple[Path, int, dict[str, Any]]] = []
    result_for_manifest: dict[str, str] = {}
    run_headers: list[tuple[Path, dict[str, Any]]] = []

    for path in result_paths:
        metadata = _safe_file_metadata(path, issues, "result")
        try:
            rows = _read_jsonl(path)
        except ValueError as error:
            _issue(issues, "invalid_result_jsonl", path=str(path), error=str(error))
            metadata.update({"records": None, "run_headers": None, "inference_records": None})
            result_files.append(metadata)
            continue
        headers = [(line, row) for line, row in enumerate(rows, 1) if row.get("record_type") == "run"]
        inference = [
            (line, row)
            for line, row in enumerate(rows, 1)
            if row.get("record_type") == "inference"
        ]
        unknown = [
            line
            for line, row in enumerate(rows, 1)
            if row.get("record_type") not in {"run", "inference"}
        ]
        if unknown:
            _issue(issues, "unknown_result_record_type", path=str(path), lines=unknown)
        if len(headers) != 1:
            _issue(
                issues,
                "result_run_header_count",
                path=str(path),
                expected=1,
                actual=len(headers),
            )
        if headers and headers[0][0] != 1:
            _issue(issues, "run_header_not_first_record", path=str(path), line=headers[0][0])
        if len(headers) == 1:
            header = headers[0][1]
            run_headers.append((path, header))
            header_manifest = header.get("manifest")
            if not isinstance(header_manifest, str) or not Path(header_manifest).is_absolute():
                _issue(issues, "header_manifest_not_absolute", path=str(path))
            elif header_manifest not in manifest_rows_by_path:
                _issue(
                    issues,
                    "header_manifest_not_supplied",
                    path=str(path),
                    manifest=header_manifest,
                )
            else:
                if header_manifest in result_for_manifest:
                    _issue(
                        issues,
                        "multiple_results_for_manifest",
                        manifest=header_manifest,
                        result_paths=[result_for_manifest[header_manifest], str(path)],
                    )
                else:
                    result_for_manifest[header_manifest] = str(path)
                expected_rows = manifest_rows_by_path[header_manifest]
                expected_path = Path(header_manifest)
                if header_manifest != str(expected_path.resolve()):
                    _issue(issues, "header_manifest_path_not_canonical", path=str(path))
                if header.get("manifest_sha256") != file_sha256(expected_path):
                    _issue(issues, "header_manifest_sha256_mismatch", path=str(path))
                if not _is_integer(header.get("records")) or header.get("records") != len(expected_rows):
                    _issue(
                        issues,
                        "header_manifest_record_count_mismatch",
                        path=str(path),
                        expected=len(expected_rows),
                        actual=header.get("records"),
                    )
                expected_indices = {int(row["global_index"]) for row in expected_rows}
                shard_indices = [row.get("global_index") for _, row in inference]
                valid_shard_indices = [index for index in shard_indices if _is_integer(index)]
                duplicates = sorted(
                    index for index, count in Counter(valid_shard_indices).items() if count > 1
                )
                if duplicates:
                    _issue(
                        issues,
                        "duplicate_result_global_index_within_file",
                        path=str(path),
                        global_indices=duplicates,
                    )
                missing = sorted(expected_indices - set(valid_shard_indices))
                extra = sorted(set(valid_shard_indices) - expected_indices)
                if missing:
                    _issue(
                        issues,
                        "missing_result_indices_for_manifest",
                        path=str(path),
                        global_indices=missing,
                    )
                if extra:
                    _issue(
                        issues,
                        "extra_result_indices_for_manifest",
                        path=str(path),
                        global_indices=extra,
                    )
        for line_number, row in inference:
            index = row.get("global_index")
            if not _is_integer(index):
                _issue(
                    issues,
                    "invalid_result_global_index",
                    path=str(path),
                    line=line_number,
                )
            all_inference_rows.append((path, line_number, row))
        metadata.update(
            {
                "records": len(rows),
                "run_headers": len(headers),
                "inference_records": len(inference),
            }
        )
        result_files.append(metadata)

    for manifest_path in sorted(manifest_rows_by_path):
        if manifest_path not in result_for_manifest:
            _issue(issues, "manifest_has_no_result_file", manifest=manifest_path)

    common_run_configuration: dict[str, Any] = {}
    for field in RUN_FIELDS:
        values: list[Any] = []
        for path, header in run_headers:
            if field not in header:
                _issue(issues, "missing_run_header_field", path=str(path), field=field)
                continue
            value = header[field]
            if field in {"model_revision", "code_revision"}:
                valid = isinstance(value, str) and bool(value)
            else:
                valid = _is_integer(value) and value > 0
            if not valid:
                _issue(
                    issues,
                    "invalid_run_header_field",
                    path=str(path),
                    field=field,
                )
            values.append(value)
        unique = {_canonical_bytes(value) for value in values}
        if len(values) != len(run_headers) or len(unique) != 1:
            _issue(issues, "inconsistent_run_header_field", field=field)
        elif values:
            common_run_configuration[field] = values[0]
    for path, header in run_headers:
        if header.get("do_sample") is not False:
            _issue(issues, "run_is_not_deterministic", path=str(path))

    valid_result_indices = [
        row["global_index"]
        for _, _, row in all_inference_rows
        if _is_integer(row.get("global_index"))
    ]
    duplicate_result_indices = sorted(
        index for index, count in Counter(valid_result_indices).items() if count > 1
    )
    if duplicate_result_indices:
        _issue(
            issues,
            "duplicate_result_global_index_global",
            global_indices=duplicate_result_indices,
        )
    manifest_indices = set(manifest_by_index) if not cross_manifest_duplicates else set(manifest_by_index)
    result_indices = set(valid_result_indices)
    missing_global = sorted(manifest_indices - result_indices)
    extra_global = sorted(result_indices - manifest_indices)
    if missing_global:
        _issue(issues, "missing_result_global_indices", global_indices=missing_global)
    if extra_global:
        _issue(issues, "extra_result_global_indices", global_indices=extra_global)

    frozen_records_checked = 0
    response_objects_checked = 0
    for path, line_number, row in all_inference_rows:
        index = row.get("global_index")
        if not _is_integer(index) or index not in manifest_by_index:
            continue
        manifest = manifest_by_index[index]
        if "prompt" in row:
            _issue(
                issues,
                "prompt_leaked_into_result",
                path=str(path),
                line=line_number,
                global_index=index,
            )
        for key, expected in manifest.items():
            if key in {"prompt", "record_type"}:
                continue
            if key not in row:
                _issue(
                    issues,
                    "missing_frozen_result_field",
                    path=str(path),
                    line=line_number,
                    global_index=index,
                    field=key,
                )
            elif not _same_json(row[key], expected):
                _issue(
                    issues,
                    "frozen_result_field_mismatch",
                    path=str(path),
                    line=line_number,
                    global_index=index,
                    field=key,
                )
        unexpected = sorted(
            set(row) - (set(manifest) - {"prompt", "record_type"}) - TRANSPORT_FIELDS
        )
        if unexpected:
            _issue(
                issues,
                "unexpected_result_fields",
                path=str(path),
                line=line_number,
                global_index=index,
                fields=unexpected,
            )
        frozen_records_checked += 1
        response_error = _strict_response(row.get("response"), mode, manifest)
        if response_error:
            _issue(
                issues,
                response_error,
                path=str(path),
                line=line_number,
                global_index=index,
            )
        else:
            response_objects_checked += 1
        output_tokens = row.get("output_tokens")
        token_budget = common_run_configuration.get("max_new_tokens")
        if output_tokens is not None:
            if not _is_integer(output_tokens) or output_tokens < 0:
                _issue(
                    issues,
                    "invalid_output_tokens",
                    path=str(path),
                    line=line_number,
                    global_index=index,
                )
            elif _is_integer(token_budget) and output_tokens > token_budget:
                _issue(
                    issues,
                    "output_tokens_exceed_budget",
                    path=str(path),
                    line=line_number,
                    global_index=index,
                )
        for field in ("batch_seconds", "peak_memory_gib"):
            value = row.get(field)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                _issue(
                    issues,
                    "invalid_runtime_metric",
                    path=str(path),
                    line=line_number,
                    global_index=index,
                    field=field,
                )

    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not issues else "FAIL",
        "mode": mode,
        "manifests": manifest_files,
        "results": result_files,
        "common_run_configuration": common_run_configuration,
        "coverage": {
            "manifest_files": len(manifest_paths),
            "result_files": len(result_paths),
            "manifest_records": sum(
                value.get("records") or 0 for value in manifest_files
            ),
            "inference_records": len(all_inference_rows),
            "unique_manifest_global_indices": len(manifest_indices),
            "unique_result_global_indices": len(result_indices),
            "missing_global_indices": missing_global,
            "extra_global_indices": extra_global,
            "duplicate_manifest_global_indices": cross_manifest_duplicates,
            "duplicate_result_global_indices": duplicate_result_indices,
        },
        "checks": {
            "frozen_records_checked": frozen_records_checked,
            "strict_response_objects_checked": response_objects_checked,
        },
        "issue_count": len(issues),
        "issue_code_counts": dict(sorted(Counter(issue["code"] for issue in issues).items())),
        "issues": issues,
    }
    audit["audit_sha256"] = canonical_hash(audit)
    return audit


def write_audit(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifests", type=Path, nargs="+", required=True)
    result.add_argument("--results", type=Path, nargs="+", required=True)
    result.add_argument("--mode", choices=("rule", "diagnosis"), required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    payload = audit_results(
        manifests=args.manifests,
        results=args.results,
        mode=args.mode,
    )
    write_audit(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
