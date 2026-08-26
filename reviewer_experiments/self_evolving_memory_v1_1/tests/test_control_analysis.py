from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from self_evolving_v1.control_analysis import (
    ControlAnalysisError,
    analyze_paired_controls,
    file_sha256,
)
from self_evolving_v1.controls import V11_ARM, build_control_manifests
from self_evolving_v1.core import canonical_hash, inject_rule, rule_key


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _prompt(labels: tuple[str, str, str], finding: str) -> str:
    return f"""[1. Frozen Current Visual Findings]:
{finding}

[2. Frozen Model Prediction Probabilities]:
- {labels[0]}: 60.0%
- {labels[1]}: 30.0%
- {labels[2]}: 10.0%

[3. Evolved Diagnostic Guidelines from Verified Training Cases]:
old frozen guideline text

[4. Precedent Cases from the Same Frozen Training Checkpoint]:
unchanged precedent cases"""


def _rule(dataset: str, disease_a: str, disease_b: str, rule_id: str) -> dict:
    return {
        "schema_version": "v1.1",
        "rule_id": rule_id,
        "rule_key": rule_key(dataset, disease_a, disease_b),
        "dataset": dataset,
        "disease_a": disease_a,
        "disease_b": disease_b,
        "supports_a": [f"pattern supporting {disease_a}", "asymmetric radial streaks"],
        "supports_b": [f"pattern supporting {disease_b}", "uniform honey crust"],
        "nondiscriminative": ["visible superficial change"],
        "trigger_terms_a": [f"support {disease_a}", "radial streaks"],
        "trigger_terms_b": [f"support {disease_b}", "honey crust"],
        "required_context_terms": [],
        "abstain_terms": [],
        "required_context": [],
        "limitations": ["Apply only to this frozen disease pair."],
        "source_case_ids_a": [f"{rule_id}-a1", f"{rule_id}-a2", f"{rule_id}-a3"],
        "source_case_ids_b": [f"{rule_id}-b1", f"{rule_id}-b2", f"{rule_id}-b3"],
        "source_case_count": 6,
        "status": "active",
        "validation_exposures": 24,
        "validation_changes": 6,
        "validation_gains": 5,
        "validation_regressions": 1,
        "validation_effect": 4 / 24,
        "reliability": 0.25,
    }


def _response(label: str) -> str:
    return json.dumps({"PrimaryDiagnosis": label})


def _result_file(
    path: Path,
    manifest_path: Path,
    responses: dict[str, str],
) -> None:
    manifests = _read_jsonl(manifest_path)
    header = {
        "record_type": "run",
        "run_label": f"test-{path.stem}",
        "model": "/frozen/model",
        "model_revision": "model-revision",
        "model_config_sha256": "config-sha",
        "model_index_sha256": "index-sha",
        "code_revision": "code-revision",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "records": len(manifests),
        "batch_size": 2,
        "max_new_tokens": 512,
        "do_sample": False,
        "gpu": "Synthetic GPU",
        "torch_version": "2.5.1",
        "transformers_version": "5.8.1",
        "cuda_version": "12.4",
        "python_version": "3.11.0",
    }
    rows = [header]
    for manifest in manifests:
        inference = {key: value for key, value in manifest.items() if key != "prompt"}
        inference.update(
            {
                "record_type": "inference",
                "response": responses[manifest["sample_id"]],
                "input_tokens": 120,
                "output_tokens": 12,
                "batch_seconds": 0.5,
                "peak_memory_gib": 2.0,
            }
        )
        rows.append(inference)
    _write_jsonl(path, rows)


def _fixture(tmp_path: Path) -> dict[str, object]:
    rules = [
        _rule("DatasetOne", "Disease Alpha", "Disease Beta", "rule-one"),
        _rule("DatasetTwo", "Disease Charlie", "Disease Delta", "rule-two"),
    ]
    release = {
        "release_id": "v1_1_active",
        "parent_release_id": "v1_1_candidate",
        "source_stream_hashes": {},
        "rules": rules,
        "status": "active",
        "protocol": "synthetic paired-control release",
        "validation_policy": {},
    }
    release["content_sha256"] = canonical_hash(release)
    release_path = tmp_path / "v1_1_active.json"
    _write_json(release_path, release)

    specs = [
        (
            "DatasetOne::one.jpg",
            "DatasetOne",
            "one.jpg",
            ("Disease Alpha", "Disease Beta", "Disease Gamma"),
            "Disease Beta",
            "Disease Alpha",
            "rule-one",
            "b",
        ),
        (
            "DatasetOne::two.jpg",
            "DatasetOne",
            "two.jpg",
            ("Disease Alpha", "Disease Beta", "Disease Gamma"),
            "Disease Alpha",
            "Disease Alpha",
            "rule-one",
            "b",
        ),
        (
            "DatasetTwo::three.jpg",
            "DatasetTwo",
            "three.jpg",
            ("Disease Charlie", "Disease Delta", "Disease Echo"),
            "Disease Delta",
            "Disease Charlie",
            "rule-two",
            "b",
        ),
        (
            "DatasetTwo::four.jpg",
            "DatasetTwo",
            "four.jpg",
            ("Disease Charlie", "Disease Delta", "Disease Echo"),
            "Disease Echo",
            "Disease Charlie",
            "rule-two",
            "b",
        ),
    ]
    rules_by_id = {rule["rule_id"]: rule for rule in rules}
    base_rows: list[dict] = []
    true_rows: list[dict] = []
    for position, (
        sid,
        dataset,
        relative_path,
        labels,
        truth,
        anchor,
        rule_id,
        side,
    ) in enumerate(specs):
        base = {
            "sample_id": sid,
            "global_index": 100 + position,
            "dataset": dataset,
            "relative_path": relative_path,
            "ground_truth": truth,
            "split_role": "gate_development",
            "candidate_probabilities": [
                [labels[0], 0.60],
                [labels[1], 0.30],
                [labels[2], 0.10],
            ],
            "prompt": _prompt(labels, f"visible finding for {sid}"),
        }
        base_rows.append(base)
        frozen_rule = {**rules_by_id[rule_id], "release_id": release["release_id"]}
        true_prompt = inject_rule(base["prompt"], frozen_rule, baseline_prediction=anchor)
        true_rows.append(
            {
                **{key: value for key, value in base.items() if key != "prompt"},
                "global_index": 400000 + position,
                "source_control_global_index": base["global_index"],
                "condition": "V1_1TrueRule",
                "arm": V11_ARM,
                "guideline_diseases": [labels[0], labels[1]],
                "memory_release_id": release["release_id"],
                "memory_release_sha256": release["content_sha256"],
                "rule_id": rule_id,
                "rule_key": frozen_rule["rule_key"],
                "rule_disease_a": frozen_rule["disease_a"],
                "rule_disease_b": frozen_rule["disease_b"],
                "rule_reliability": frozen_rule["reliability"],
                "c100_anchor_prediction": anchor,
                "applicability_reason": f"trigger_supports_{side}",
                "trigger_matches": {"a": 0, "b": 2},
                "trigger_supported_side": side,
                "trigger_supported_disease": frozen_rule["disease_b"],
                "matching_protocol": "literal_v1",
                "prompt_sha256": canonical_hash(true_prompt),
                "prompt": true_prompt,
            }
        )

    base_path = tmp_path / "base_q100_records.jsonl"
    _write_jsonl(base_path, base_rows)
    true_dir = tmp_path / "true"
    true_manifest = true_dir / "gate_development_shard_0.jsonl"
    _write_jsonl(true_manifest, true_rows)
    _write_json(
        true_dir / "manifest_audit.json",
        {
            "cohort": "gate_development",
            "cohort_records": len(true_rows),
            "included_with_rule": len(true_rows),
            "release_sha256": release["content_sha256"],
            "status": "PASS",
        },
    )
    control_root = tmp_path / "controls"
    build_control_manifests(
        base_q100=base_path,
        release_path=release_path,
        true_manifest_dir=true_dir,
        output_dir=control_root,
        seed=73,
        global_index_offset=810000,
    )
    neutral_manifest = next((control_root / "neutral").glob("*.jsonl"))
    shuffled_manifest = next((control_root / "shuffled").glob("*.jsonl"))

    true_responses = {
        specs[0][0]: _response("Disease Beta"),
        specs[1][0]: _response("Disease Beta"),
        specs[2][0]: _response("Disease Delta"),
        specs[3][0]: _response("Disease Delta"),
    }
    neutral_responses = {
        specs[0][0]: _response("Disease Beta"),
        specs[1][0]: _response("Disease Alpha"),
        specs[2][0]: _response("Disease Charlie"),
        specs[3][0]: _response("Disease Delta"),
    }
    shuffled_responses = {
        specs[0][0]: _response("Disease Beta"),
        specs[1][0]: _response("Disease Alpha"),
        specs[2][0]: _response("Disease Charlie"),
        specs[3][0]: _response("Disease Charlie"),
    }
    true_result = tmp_path / "true_results.jsonl"
    neutral_result = tmp_path / "neutral_results.jsonl"
    shuffled_result = tmp_path / "shuffled_results.jsonl"
    _result_file(true_result, true_manifest, true_responses)
    _result_file(neutral_result, neutral_manifest, neutral_responses)
    _result_file(shuffled_result, shuffled_manifest, shuffled_responses)
    return {
        "base_path": base_path,
        "release_path": release_path,
        "true_manifest": true_manifest,
        "true_result": true_result,
        "neutral_manifest": neutral_manifest,
        "neutral_result": neutral_result,
        "shuffled_manifest": shuffled_manifest,
        "shuffled_result": shuffled_result,
        "control_root": control_root,
        "specs": specs,
    }


def _analyze(fixture: dict[str, object], output: Path, bootstraps: int = 250) -> dict:
    return analyze_paired_controls(
        base_q100=fixture["base_path"],
        release_path=fixture["release_path"],
        true_manifests=[fixture["true_manifest"]],
        true_results=[fixture["true_result"]],
        control_root=fixture["control_root"],
        neutral_results=[fixture["neutral_result"]],
        shuffled_results=[fixture["shuffled_result"]],
        output=output,
        bootstraps=bootstraps,
        seed=99,
    )


def test_strict_three_arm_summary_and_paired_statistics(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "analysis.json"
    payload = _analyze(fixture, output)

    assert payload["status"] == "PASS"
    assert payload["samples_per_arm"] == 4
    true = payload["arms"]["true"]["overall"]
    neutral = payload["arms"]["neutral"]["overall"]
    shuffled = payload["arms"]["shuffled"]["overall"]
    assert {
        key: true[key]
        for key in ("valid", "unchanged", "changed_wrong", "gain", "regression", "net")
    } == {
        "valid": 4,
        "unchanged": 0,
        "changed_wrong": 1,
        "gain": 2,
        "regression": 1,
        "net": 1,
    }
    assert {
        key: neutral[key]
        for key in ("valid", "unchanged", "changed_wrong", "gain", "regression", "net")
    } == {
        "valid": 4,
        "unchanged": 2,
        "changed_wrong": 1,
        "gain": 1,
        "regression": 0,
        "net": 1,
    }
    assert shuffled["valid"] == 4
    assert shuffled["unchanged"] == 4
    assert shuffled["net"] == 0
    assert payload["arms"]["true"]["by_dataset"]["DatasetOne"]["net"] == 0
    assert payload["arms"]["true"]["by_dataset"]["DatasetTwo"]["net"] == 1
    assert set(payload["arms"]["true"]["by_rule"]) == {"rule-one", "rule-two"}

    neutral_test = payload["comparisons"]["true_vs_neutral"]
    assert neutral_test["true_only_correct"] == 1
    assert neutral_test["control_only_correct"] == 1
    assert neutral_test["exact_mcnemar_two_sided_p"] == 1.0
    assert len(neutral_test["sample_level_differences"]) == 4
    assert neutral_test["bootstrap_95_ci"]["bootstraps"] == 250
    assert neutral_test["bootstrap_95_ci"]["ci_low"] <= 0.0
    assert neutral_test["bootstrap_95_ci"]["ci_high"] >= 0.0

    shuffled_test = payload["comparisons"]["true_vs_shuffled"]
    assert shuffled_test["true_only_correct"] == 2
    assert shuffled_test["control_only_correct"] == 1
    assert shuffled_test["exact_mcnemar_two_sided_p"] == 1.0
    assert canonical_hash(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    ) == payload["content_sha256"]
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_neutral_has_no_direction_but_shuffled_uses_swapped_direction(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    payload = _analyze(fixture, tmp_path / "analysis.json", bootstraps=25)

    sid = fixture["specs"][0][0]
    neutral_rows = payload["comparisons"]["true_vs_neutral"]["sample_level_differences"]
    shuffled_rows = payload["comparisons"]["true_vs_shuffled"]["sample_level_differences"]
    neutral = next(row for row in neutral_rows if row["sample_id"] == sid)
    shuffled = next(row for row in shuffled_rows if row["sample_id"] == sid)
    assert neutral["neutral_prediction"] == "Disease Beta"
    assert neutral["neutral_constraint_reason"] == "pair_constrained_output"
    assert shuffled["shuffled_prediction"] == "Disease Alpha"
    assert shuffled["shuffled_constraint_reason"] == "trigger_direction_fallback"


def test_invalid_and_third_class_outputs_fall_back_without_becoming_regressions(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    rows = _read_jsonl(fixture["true_result"])
    rows[1]["response"] = "not valid JSON"
    rows[2]["response"] = _response("Disease Gamma")
    _write_jsonl(fixture["true_result"], rows)

    payload = _analyze(fixture, tmp_path / "analysis.json", bootstraps=25)
    true = payload["arms"]["true"]["overall"]
    assert true["valid"] == 3
    assert true["invalid"] == 1
    assert true["constraint_reasons"]["invalid_output_fallback"] == 1
    assert true["constraint_reasons"]["third_class_fallback"] == 1
    assert true["regression"] == 0


def test_rejects_pairing_sha_tamper_even_with_recomputed_outer_hash(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    audit_path = fixture["control_root"] / "control_manifest_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["pairing_sha256"] = "0" * 64
    audit["audit_sha256"] = canonical_hash(
        {key: value for key, value in audit.items() if key != "audit_sha256"}
    )
    _write_json(audit_path, audit)

    with pytest.raises(ControlAnalysisError, match="pairing SHA mismatch"):
        _analyze(fixture, tmp_path / "analysis.json", bootstraps=10)
    assert not (tmp_path / "analysis.json").exists()


def test_rejects_result_metadata_that_disagrees_with_manifest_release(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    rows = _read_jsonl(fixture["neutral_result"])
    rows[1]["memory_release_sha256"] = "tampered"
    _write_jsonl(fixture["neutral_result"], rows)

    with pytest.raises(ControlAnalysisError, match="result/manifest mismatch.*memory_release_sha256"):
        _analyze(fixture, tmp_path / "analysis.json", bootstraps=10)


def test_reconstructs_controls_instead_of_trusting_rehashed_audit(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    shuffled_path = fixture["shuffled_manifest"]
    shuffled_rows = _read_jsonl(shuffled_path)
    shuffled_rows[0]["trigger_supported_side"] = "b"
    shuffled_rows[0]["trigger_supported_disease"] = "Disease Beta"
    _write_jsonl(shuffled_path, shuffled_rows)

    audit_path = fixture["control_root"] / "control_manifest_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    relative = str(shuffled_path.relative_to(fixture["control_root"]))
    audit["controls"]["shuffled"]["files"][relative] = file_sha256(shuffled_path)
    audit["controls"]["shuffled"]["content_sha256"] = canonical_hash(shuffled_rows)
    audit["audit_sha256"] = canonical_hash(
        {key: value for key, value in audit.items() if key != "audit_sha256"}
    )
    _write_json(audit_path, audit)

    with pytest.raises(ControlAnalysisError, match="shuffled manifest reconstruction mismatch"):
        _analyze(fixture, tmp_path / "analysis.json", bootstraps=10)


def test_cli_failure_is_nonzero_and_does_not_write_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    package_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "analysis.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "self_evolving_v1.control_analysis",
            "--base-q100",
            str(fixture["base_path"]),
            "--release",
            str(fixture["release_path"]),
            "--true-manifests",
            str(fixture["true_manifest"]),
            "--true-results",
            str(fixture["true_result"]),
            "--control-root",
            str(fixture["control_root"]),
            "--neutral-results",
            str(fixture["neutral_result"]),
            "--shuffled-results",
            str(fixture["shuffled_result"]),
            "--output",
            str(output),
            "--bootstraps",
            "0",
        ],
        cwd=package_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "bootstraps must be a positive integer" in completed.stderr
    assert not output.exists()


def test_cli_help_supports_module_and_direct_script_entrypoints() -> None:
    package_root = Path(__file__).resolve().parents[1]
    direct = subprocess.run(
        [sys.executable, str(package_root / "self_evolving_v1" / "control_analysis.py"), "--help"],
        cwd=package_root,
        capture_output=True,
        text=True,
        check=False,
    )
    module = subprocess.run(
        [sys.executable, "-m", "self_evolving_v1.control_analysis", "--help"],
        cwd=package_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert direct.returncode == 0, direct.stderr
    assert module.returncode == 0, module.stderr
    for flag in ("--true-manifests", "--neutral-results", "--shuffled-results"):
        assert flag in direct.stdout
        assert flag in module.stdout
