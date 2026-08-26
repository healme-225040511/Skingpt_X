from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from self_evolving_v1.smoke import (
    DATASETS,
    SmokeManifestError,
    build_diagnosis_smoke_manifests,
    build_rule_smoke_manifests,
    canonical_hash,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_smoke_rows(output: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(output.glob("*_smoke_shard_*.jsonl")):
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def _rule_fixture(root: Path, records_per_dataset: int = 6) -> list[dict]:
    rows: list[dict] = []
    index = 300000
    for dataset in DATASETS:
        for offset in range(records_per_dataset):
            rows.append(
                {
                    "global_index": index,
                    "dataset": dataset,
                    "rule_id": f"{dataset}-rule-{offset}",
                    "rule_key": f"{dataset}::disease-{offset}::other-{offset}",
                    "prompt": f"Frozen proposal {dataset} {offset}",
                    "opaque_payload": {"order": offset, "preserve": True},
                }
            )
            index += 1
    # Deliberately separate adjacent records so the selector cannot depend on
    # source-shard ordering for its scientific balancing.
    _write_jsonl(root / "rules_shard_0.jsonl", rows[::2])
    _write_jsonl(root / "rules_shard_1.jsonl", rows[1::2])
    return rows


def _diagnosis_fixture(
    root: Path, *, rules_per_dataset: int = 8, cases_per_rule: int = 4
) -> list[dict]:
    rows: list[dict] = []
    index = 400000
    for dataset in DATASETS:
        for rule_offset in range(rules_per_dataset):
            for case_offset in range(cases_per_rule):
                prompt = f"Frozen diagnosis {dataset} rule={rule_offset} case={case_offset}"
                rows.append(
                    {
                        "global_index": index,
                        "sample_id": f"{dataset}::case-{rule_offset}-{case_offset}.jpg",
                        "dataset": dataset,
                        "rule_id": f"{dataset}-rule-{rule_offset}",
                        "split_role": "gate_development",
                        "ground_truth": f"secret-label-{case_offset % 2}",
                        "prompt": prompt,
                        "prompt_sha256": canonical_hash(prompt),
                        "opaque_payload": [rule_offset, case_offset],
                    }
                )
                index += 1
    _write_jsonl(root / "gate_development_shard_0.jsonl", rows[::2])
    _write_jsonl(root / "gate_development_shard_1.jsonl", rows[1::2])
    _write_json(
        root / "manifest_audit.json",
        {
            "cohort": "gate_development",
            "included_with_rule": len(rows),
            "status": "PASS",
        },
    )
    return rows


def test_rule_smoke_is_deterministic_equal_and_record_preserving(tmp_path: Path) -> None:
    source = tmp_path / "rules"
    source_rows = _rule_fixture(source)

    first = build_rule_smoke_manifests(
        manifest_dir=source, output_dir=tmp_path / "first", seed=91
    )
    second = build_rule_smoke_manifests(
        manifest_dir=source, output_dir=tmp_path / "second", seed=91
    )
    first_rows = _read_smoke_rows(tmp_path / "first")
    second_rows = _read_smoke_rows(tmp_path / "second")

    assert len(first_rows) == 8
    assert Counter(row["dataset"] for row in first_rows) == Counter(
        {dataset: 2 for dataset in DATASETS}
    )
    assert {canonical_hash(row) for row in first_rows} <= {
        canonical_hash(row) for row in source_rows
    }
    assert first_rows == second_rows
    assert first["selection"]["identity_sha256"] == second["selection"]["identity_sha256"]
    assert first["selection"]["content_sha256"] == first["output"]["content_sha256"]
    assert len(first["input"]["record_hashes"]) == len(source_rows)
    assert len(first["output"]["files"]) == 2
    stored = first["audit_sha256"]
    assert canonical_hash({key: value for key, value in first.items() if key != "audit_sha256"}) == stored


def test_diagnosis_smoke_caps_at_96_balances_datasets_and_diversifies_rules(
    tmp_path: Path,
) -> None:
    source = tmp_path / "diagnosis"
    source_rows = _diagnosis_fixture(source)

    audit = build_diagnosis_smoke_manifests(
        manifest_dir=source, output_dir=tmp_path / "smoke", seed=17
    )
    rows = _read_smoke_rows(tmp_path / "smoke")
    source_by_id = {row["sample_id"]: row for row in source_rows}

    assert len(rows) == 96
    assert Counter(row["dataset"] for row in rows) == Counter(
        {dataset: 24 for dataset in DATASETS}
    )
    assert len({row["rule_id"] for row in rows}) == 32
    assert all(row == source_by_id[row["sample_id"]] for row in rows)
    assert all(row["global_index"] == source_by_id[row["sample_id"]]["global_index"] for row in rows)
    assert audit["selection"]["unique_rule_ids"] == 32
    assert audit["output"]["global_indices_preserved"] is True
    assert audit["output"]["sample_ids_preserved"] is True
    assert all(
        audit["selection"]["dataset_coverage"][dataset]["selected_rules"] == 8
        for dataset in DATASETS
    )


def test_diagnosis_selection_is_deterministic_and_label_blind(tmp_path: Path) -> None:
    original = tmp_path / "original"
    changed = tmp_path / "changed"
    rows = _diagnosis_fixture(original, rules_per_dataset=4, cases_per_rule=3)
    changed_rows = [
        {**row, "ground_truth": f"different-{position}"}
        for position, row in enumerate(rows)
    ]
    _write_jsonl(changed / "gate_development_shard_0.jsonl", changed_rows[::2])
    _write_jsonl(changed / "gate_development_shard_1.jsonl", changed_rows[1::2])
    _write_json(
        changed / "manifest_audit.json",
        {
            "cohort": "gate_development",
            "included_with_rule": len(changed_rows),
            "status": "PASS",
        },
    )

    first = build_diagnosis_smoke_manifests(
        manifest_dir=original,
        output_dir=tmp_path / "first",
        maximum_records=16,
        seed=51,
    )
    repeated = build_diagnosis_smoke_manifests(
        manifest_dir=original,
        output_dir=tmp_path / "repeat",
        maximum_records=16,
        seed=51,
    )
    relabeled = build_diagnosis_smoke_manifests(
        manifest_dir=changed,
        output_dir=tmp_path / "relabeled",
        maximum_records=16,
        seed=51,
    )

    assert first["selection"]["identity_sha256"] == repeated["selection"]["identity_sha256"]
    assert first["selection"]["identity_sha256"] == relabeled["selection"]["identity_sha256"]
    assert first["selection"]["content_sha256"] != relabeled["selection"]["content_sha256"]
    assert first["selection_fields"] == ["dataset", "rule_id", "sample_id", "global_index"]
    assert first["label_fields_used_for_selection"] == []
    selected = _read_smoke_rows(tmp_path / "first")
    assert Counter(row["dataset"] for row in selected) == Counter(
        {dataset: 4 for dataset in DATASETS}
    )
    assert len({row["rule_id"] for row in selected}) == 16


@pytest.mark.parametrize("duplicate_field", ["global_index", "rule_id", "rule_key"])
def test_rule_smoke_rejects_duplicate_identities_before_writing(
    tmp_path: Path, duplicate_field: str
) -> None:
    source = tmp_path / "rules"
    rows = _rule_fixture(source)
    rows[1] = {**rows[1], duplicate_field: rows[0][duplicate_field]}
    _write_jsonl(source / "rules_shard_0.jsonl", rows[::2])
    _write_jsonl(source / "rules_shard_1.jsonl", rows[1::2])

    output = tmp_path / "output"
    with pytest.raises(SmokeManifestError, match=f"duplicate {duplicate_field}"):
        build_rule_smoke_manifests(manifest_dir=source, output_dir=output)
    assert not output.exists()


def test_rule_smoke_rejects_corrupt_json_before_writing(tmp_path: Path) -> None:
    source = tmp_path / "rules"
    _rule_fixture(source)
    with (source / "rules_shard_1.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"broken":\n')

    output = tmp_path / "output"
    with pytest.raises(SmokeManifestError, match="invalid JSON"):
        build_rule_smoke_manifests(manifest_dir=source, output_dir=output)
    assert not output.exists()


def test_diagnosis_smoke_rejects_duplicates_and_prompt_corruption(
    tmp_path: Path,
) -> None:
    duplicate_source = tmp_path / "duplicate"
    rows = _diagnosis_fixture(duplicate_source, rules_per_dataset=2, cases_per_rule=2)
    rows[1] = {**rows[1], "sample_id": rows[0]["sample_id"]}
    _write_jsonl(duplicate_source / "gate_development_shard_0.jsonl", rows[::2])
    _write_jsonl(duplicate_source / "gate_development_shard_1.jsonl", rows[1::2])
    with pytest.raises(SmokeManifestError, match="duplicate sample_id"):
        build_diagnosis_smoke_manifests(
            manifest_dir=duplicate_source,
            output_dir=tmp_path / "duplicate-output",
            maximum_records=8,
        )

    corrupt_source = tmp_path / "corrupt"
    corrupt_rows = _diagnosis_fixture(
        corrupt_source, rules_per_dataset=2, cases_per_rule=2
    )
    corrupt_rows[0] = {**corrupt_rows[0], "prompt": "tampered prompt"}
    _write_jsonl(
        corrupt_source / "gate_development_shard_0.jsonl", corrupt_rows[::2]
    )
    with pytest.raises(SmokeManifestError, match="prompt hash mismatch"):
        build_diagnosis_smoke_manifests(
            manifest_dir=corrupt_source,
            output_dir=tmp_path / "corrupt-output",
            maximum_records=8,
        )


def test_diagnosis_smoke_rejects_source_audit_count_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "diagnosis"
    rows = _diagnosis_fixture(source, rules_per_dataset=2, cases_per_rule=2)
    _write_json(
        source / "manifest_audit.json",
        {
            "cohort": "gate_development",
            "included_with_rule": len(rows) + 1,
            "status": "PASS",
        },
    )
    with pytest.raises(SmokeManifestError, match="record count does not match"):
        build_diagnosis_smoke_manifests(
            manifest_dir=source,
            output_dir=tmp_path / "output",
            maximum_records=8,
        )


def test_smoke_cli_supports_module_and_direct_script_entrypoints() -> None:
    package_root = Path(__file__).resolve().parents[1]
    direct = subprocess.run(
        [sys.executable, str(package_root / "self_evolving_v1" / "smoke.py"), "--help"],
        cwd=package_root,
        capture_output=True,
        text=True,
        check=False,
    )
    module = subprocess.run(
        [sys.executable, "-m", "self_evolving_v1.smoke", "--help"],
        cwd=package_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert direct.returncode == 0, direct.stderr
    assert module.returncode == 0, module.stderr
    assert "diagnosis" in direct.stdout and "rules" in direct.stdout
    assert "diagnosis" in module.stdout and "rules" in module.stdout
