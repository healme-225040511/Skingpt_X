from __future__ import annotations

import json
from pathlib import Path

import pytest

from self_evolving_v1.run_inference import load_completed


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_valid_resume_returns_completed_indices(tmp_path: Path) -> None:
    output = tmp_path / "results.jsonl"
    header = {"record_type": "run", "manifest_sha256": "abc", "records": 2}
    write_rows(output, [header, {"record_type": "inference", "global_index": 7}])
    assert load_completed(output, header) == {7}


def test_resume_rejects_changed_frozen_configuration(tmp_path: Path) -> None:
    output = tmp_path / "results.jsonl"
    write_rows(output, [{"record_type": "run", "manifest_sha256": "old", "records": 2}])
    with pytest.raises(ValueError, match="incompatible resume"):
        load_completed(output, {"record_type": "run", "manifest_sha256": "new", "records": 2})


def test_resume_rejects_duplicate_or_malformed_output(tmp_path: Path) -> None:
    output = tmp_path / "results.jsonl"
    header = {"record_type": "run", "manifest_sha256": "abc"}
    write_rows(
        output,
        [
            header,
            {"record_type": "inference", "global_index": 7},
            {"record_type": "inference", "global_index": 7},
        ],
    )
    with pytest.raises(ValueError, match="duplicate inference"):
        load_completed(output, header)
    output.write_text(json.dumps(header) + "\n{truncated", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_completed(output, header)
