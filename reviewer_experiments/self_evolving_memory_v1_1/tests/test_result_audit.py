from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from self_evolving_v1.result_audit import audit_results, canonical_hash, file_sha256


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def manifest_row(index: int, *, diagnosis: bool = False) -> dict:
    row = {
        "global_index": index,
        "record_type": "diagnosis_proposal" if diagnosis else "rule_proposal",
        "dataset": "DDI",
        "sample_id": f"DDI::{index}",
        "frozen_nested": {"value": [index, True]},
        "prompt": f"private prompt {index}",
    }
    if diagnosis:
        row["candidate_probabilities"] = [["Disease A", 0.6], ["Disease B", 0.4]]
    return row


def result_row(manifest: dict, response: str) -> dict:
    frozen = {key: value for key, value in manifest.items() if key != "prompt"}
    return {
        **frozen,
        "record_type": "inference",
        "response": response,
        "input_tokens": 10,
        "output_tokens": 8,
        "batch_seconds": 0.5,
        "peak_memory_gib": 1.25,
    }


def write_result(
    result: Path,
    manifest: Path,
    rows: list[dict],
    *,
    model_revision: str = "model-rev",
    code_revision: str = "code-rev",
    batch_size: int = 2,
    max_new_tokens: int = 64,
) -> None:
    header = {
        "record_type": "run",
        "manifest": str(manifest.resolve()),
        "manifest_sha256": file_sha256(manifest),
        "records": len(rows),
        "model_revision": model_revision,
        "code_revision": code_revision,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    write_jsonl(result, [header, *rows])


def make_two_shards(tmp_path: Path, *, diagnosis: bool = False) -> tuple[list[Path], list[Path], list[dict]]:
    manifests: list[Path] = []
    results: list[Path] = []
    rows: list[dict] = []
    for shard, index in enumerate((101, 102)):
        manifest = tmp_path / f"manifest_{shard}.jsonl"
        result = tmp_path / f"result_{shard}.jsonl"
        row = manifest_row(index, diagnosis=diagnosis)
        response = (
            json.dumps({"PrimaryDiagnosis": "Disease A"})
            if diagnosis
            else json.dumps({"status": "candidate", "rule": index})
        )
        write_jsonl(manifest, [row])
        write_result(result, manifest, [result_row(row, response)])
        manifests.append(manifest)
        results.append(result)
        rows.append(row)
    return manifests, results, rows


def test_rule_audit_passes_with_exact_coverage_and_self_hash(tmp_path: Path) -> None:
    manifests, results, _ = make_two_shards(tmp_path)
    audit = audit_results(manifests=manifests, results=reversed(results), mode="rule")
    assert audit["status"] == "PASS"
    assert audit["issue_count"] == 0
    assert audit["coverage"]["manifest_records"] == 2
    stored = audit["audit_sha256"]
    assert canonical_hash({key: value for key, value in audit.items() if key != "audit_sha256"}) == stored


def test_diagnosis_requires_exact_closed_set_label(tmp_path: Path) -> None:
    manifests, results, rows = make_two_shards(tmp_path, diagnosis=True)
    write_result(
        results[1],
        manifests[1],
        [result_row(rows[1], json.dumps({"PrimaryDiagnosis": "Disease C"}))],
    )
    audit = audit_results(manifests=manifests, results=results, mode="diagnosis")
    assert audit["status"] == "FAIL"
    assert audit["issue_code_counts"]["primary_diagnosis_outside_candidate_labels"] == 1


@pytest.mark.parametrize(
    "response,code",
    [
        ('```json\n{"status":"candidate"}\n```', "response_not_strict_json"),
        ('{"status":"candidate"} trailing', "response_not_strict_json"),
        ('["not", "an", "object"]', "response_not_json_object"),
        ('{"status":1,"status":2}', "response_not_strict_json"),
    ],
)
def test_response_must_be_one_strict_json_object(
    tmp_path: Path, response: str, code: str
) -> None:
    manifests, results, rows = make_two_shards(tmp_path)
    write_result(results[0], manifests[0], [result_row(rows[0], response)])
    audit = audit_results(manifests=manifests, results=results, mode="rule")
    assert audit["status"] == "FAIL"
    assert audit["issue_code_counts"][code] == 1


def test_rejects_frozen_field_drift_and_prompt_leak(tmp_path: Path) -> None:
    manifests, results, rows = make_two_shards(tmp_path)
    changed = result_row(rows[0], json.dumps({"status": "candidate"}))
    changed["frozen_nested"] = {"value": [101, False]}
    changed["prompt"] = rows[0]["prompt"]
    write_result(results[0], manifests[0], [changed])
    audit = audit_results(manifests=manifests, results=results, mode="rule")
    assert audit["status"] == "FAIL"
    assert audit["issue_code_counts"]["frozen_result_field_mismatch"] == 1
    assert audit["issue_code_counts"]["prompt_leaked_into_result"] == 1


def test_rejects_missing_extra_duplicate_indices_and_inconsistent_run(tmp_path: Path) -> None:
    manifests, results, rows = make_two_shards(tmp_path)
    extra = {**rows[1], "global_index": 999}
    write_result(
        results[0],
        manifests[0],
        [
            result_row(rows[0], json.dumps({"ok": True})),
            result_row(rows[0], json.dumps({"ok": True})),
        ],
    )
    write_result(
        results[1],
        manifests[1],
        [result_row(extra, json.dumps({"ok": True}))],
        batch_size=3,
    )
    audit = audit_results(manifests=manifests, results=results, mode="rule")
    assert audit["status"] == "FAIL"
    assert "duplicate_result_global_index_global" in audit["issue_code_counts"]
    assert "missing_result_global_indices" in audit["issue_code_counts"]
    assert "extra_result_global_indices" in audit["issue_code_counts"]
    assert "inconsistent_run_header_field" in audit["issue_code_counts"]


@pytest.mark.parametrize("bad_line", ["", "{bad", '{"a":1,"a":2}'])
def test_strict_jsonl_failure_is_recorded(tmp_path: Path, bad_line: str) -> None:
    manifests, results, _ = make_two_shards(tmp_path)
    results[0].write_text(bad_line + "\n", encoding="utf-8")
    audit = audit_results(manifests=manifests, results=results, mode="rule")
    assert audit["status"] == "FAIL"
    assert audit["issue_code_counts"]["invalid_result_jsonl"] == 1


def test_cli_writes_failed_audit_and_exits_nonzero(tmp_path: Path) -> None:
    manifests, results, _ = make_two_shards(tmp_path)
    output = tmp_path / "audit.json"
    results[0].write_text("{truncated\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "self_evolving_v1.result_audit",
            "--manifests",
            *(str(path) for path in manifests),
            "--results",
            *(str(path) for path in results),
            "--mode",
            "rule",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    stored = payload.pop("audit_sha256")
    assert canonical_hash(payload) == stored
