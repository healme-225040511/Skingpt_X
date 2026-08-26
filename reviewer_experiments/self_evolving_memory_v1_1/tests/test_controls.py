from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from self_evolving_v1.controls import (
    ControlManifestError,
    SAFETY_MARKER,
    V11_ARM,
    build_control_manifests,
    file_sha256,
    shuffled_rule,
)
from self_evolving_v1.core import canonical_hash, inject_rule, render_rule, rule_key


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


def _base_prompt() -> str:
    return """[1. Frozen Current Visual Findings]:
Blue radial streaks frame a dark asymmetric center.

[2. Frozen Model Prediction Probabilities]:
- Disease Alpha: 60.0%
- Disease Beta: 30.0%
- Disease Gamma: 10.0%

[3. Evolved Diagnostic Guidelines from Verified Training Cases]:
old frozen guideline text

[4. Precedent Cases from the Same Frozen Training Checkpoint]:
unchanged precedent cases"""


def _fixture(tmp_path: Path) -> dict[str, Path | dict]:
    pair_key = rule_key("ToyDataset", "Disease Alpha", "Disease Beta")
    rule = {
        "schema_version": "v1.1",
        "rule_id": "cr11_test",
        "rule_key": pair_key,
        "dataset": "ToyDataset",
        "disease_a": "Disease Alpha",
        "disease_b": "Disease Beta",
        "supports_a": ["Blue radial streaks", "Dark asymmetric center"],
        "supports_b": ["Honey colored crust", "Uniform symmetric plaque"],
        "nondiscriminative": ["Visible superficial change"],
        "trigger_terms_a": ["blue radial streaks", "dark asymmetric center"],
        "trigger_terms_b": ["honey colored crust", "uniform symmetric plaque"],
        "required_context_terms": [],
        "abstain_terms": [],
        "required_context": [],
        "limitations": ["Apply only when visible findings support one side."],
        "source_case_ids_a": ["a1", "a2", "a3"],
        "source_case_ids_b": ["b1", "b2", "b3"],
        "source_case_count": 6,
        "status": "active",
        "validation_exposures": 24,
        "validation_changes": 6,
        "validation_gains": 5,
        "validation_regressions": 1,
        "validation_effect": 4 / 24,
        "reliability": 0.23,
    }
    release = {
        "release_id": "v1_1_active",
        "parent_release_id": "v1_1_candidate",
        "source_stream_hashes": {"ToyDataset": "frozen-source"},
        "rules": [rule],
        "status": "active",
        "protocol": "synthetic frozen V1.1 release",
        "validation_policy": {},
    }
    release["content_sha256"] = canonical_hash(release)
    release_path = tmp_path / "v1_1_active.json"
    _write_json(release_path, release)

    base = {
        "sample_id": "ToyDataset::case-001.jpg",
        "global_index": 101,
        "dataset": "ToyDataset",
        "relative_path": "case-001.jpg",
        "ground_truth": "Disease Beta",
        "split_role": "gate_development",
        "candidate_probabilities": [
            ["Disease Alpha", 0.60],
            ["Disease Beta", 0.30],
            ["Disease Gamma", 0.10],
        ],
        "prompt": _base_prompt(),
    }
    base_path = tmp_path / "base_q100_records.jsonl"
    _write_jsonl(base_path, [base])

    rendered_rule = {**rule, "release_id": release["release_id"]}
    true_prompt = inject_rule(
        base["prompt"], rendered_rule, baseline_prediction="Disease Alpha"
    )
    true_row = {
        **{key: value for key, value in base.items() if key != "prompt"},
        "global_index": 400000,
        "source_control_global_index": base["global_index"],
        "condition": "V1_1TrueRule",
        "arm": V11_ARM,
        "guideline_diseases": ["Disease Alpha", "Disease Beta"],
        "memory_release_id": release["release_id"],
        "memory_release_sha256": release["content_sha256"],
        "rule_id": rule["rule_id"],
        "rule_key": pair_key,
        "rule_disease_a": rule["disease_a"],
        "rule_disease_b": rule["disease_b"],
        "rule_reliability": rule["reliability"],
        "c100_anchor_prediction": "Disease Alpha",
        "applicability_reason": "trigger_supports_a",
        "trigger_matches": {"a": 2, "b": 0},
        "prompt_sha256": canonical_hash(true_prompt),
        "prompt": true_prompt,
    }
    manifest_dir = tmp_path / "true_manifest"
    shard_path = manifest_dir / "gate_development_shard_0.jsonl"
    _write_jsonl(shard_path, [true_row])
    _write_json(
        manifest_dir / "manifest_audit.json",
        {
            "cohort": "gate_development",
            "cohort_records": 1,
            "included_with_rule": 1,
            "release_sha256": release["content_sha256"],
            "status": "PASS",
        },
    )
    return {
        "base_path": base_path,
        "release_path": release_path,
        "manifest_dir": manifest_dir,
        "shard_path": shard_path,
        "base": base,
        "rule": rule,
        "true_row": true_row,
    }


def _rule_block(prompt: str) -> str:
    return prompt.split(
        "[3. Evolved Diagnostic Guidelines from Verified Training Cases]:\n", 1
    )[1].split(SAFETY_MARKER, 1)[0]


def test_builds_paired_neutral_and_within_pair_shuffled_controls(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "controls"

    audit = build_control_manifests(
        base_q100=fixture["base_path"],
        release_path=fixture["release_path"],
        true_manifest_dir=fixture["manifest_dir"],
        output_dir=output,
        seed=73,
        global_index_offset=810000,
    )

    neutral_path = output / "neutral" / "gate_development_neutral_shard_0.jsonl"
    shuffled_path = output / "shuffled" / "gate_development_shuffled_shard_0.jsonl"
    neutral = _read_jsonl(neutral_path)[0]
    shuffled = _read_jsonl(shuffled_path)[0]
    true_row = fixture["true_row"]

    for row in (neutral, shuffled):
        assert row["sample_id"] == true_row["sample_id"]
        assert row["candidate_probabilities"] == true_row["candidate_probabilities"]
        assert row["c100_anchor_prediction"] == true_row["c100_anchor_prediction"]
        assert row["source_control_global_index"] == true_row["source_control_global_index"]
        assert row["source_true_global_index"] == true_row["global_index"]
        assert row["rule_disease_a"] == "Disease Alpha"
        assert row["rule_disease_b"] == "Disease Beta"
        assert row["arm"] == V11_ARM
        assert row["prompt"].split(SAFETY_MARKER, 1)[1] == true_row["prompt"].split(
            SAFETY_MARKER, 1
        )[1]

    neutral_block = _rule_block(neutral["prompt"])
    assert "Blue radial streaks" not in neutral_block
    assert "Honey colored crust" not in neutral_block
    assert "xxxx xxxxxx xxxxxxx" in neutral_block
    assert len(neutral_block.strip()) == len(_rule_block(true_row["prompt"]).strip())
    assert neutral["control_rule_block_length_ratio"] == 1.0

    shuffled_block = _rule_block(shuffled["prompt"])
    supports_a = shuffled_block.split("- Supports Disease Alpha: ", 1)[1].split("\n", 1)[0]
    supports_b = shuffled_block.split("- Supports Disease Beta: ", 1)[1].split("\n", 1)[0]
    triggers_a = shuffled_block.split("- Executable triggers for Disease Alpha: ", 1)[1].split("\n", 1)[0]
    triggers_b = shuffled_block.split("- Executable triggers for Disease Beta: ", 1)[1].split("\n", 1)[0]
    assert set(supports_a.split("; ")) == {"Honey colored crust", "Uniform symmetric plaque"}
    assert set(supports_b.split("; ")) == {"Blue radial streaks", "Dark asymmetric center"}
    assert set(triggers_a.split("; ")) == {"honey colored crust", "uniform symmetric plaque"}
    assert set(triggers_b.split("; ")) == {"blue radial streaks", "dark asymmetric center"}
    assert "This memory may compare only Disease Alpha and Disease Beta" in shuffled["prompt"]
    assert shuffled["trigger_matches"] == {"a": 0, "b": 2}

    assert neutral["global_index"] == 810000
    assert shuffled["global_index"] == 810001
    assert audit["records_per_arm"] == 1
    assert audit["invariants"]["sample_pairing_complete"] is True
    assert audit["rule_block_length"]["neutral_max_absolute_character_delta"] == 0
    assert audit["controls"]["neutral"]["files"][str(neutral_path.relative_to(output))] == file_sha256(neutral_path)
    stored_audit_hash = audit["audit_sha256"]
    assert canonical_hash({key: value for key, value in audit.items() if key != "audit_sha256"}) == stored_audit_hash

    second_output = tmp_path / "controls_repeat"
    second = build_control_manifests(
        base_q100=fixture["base_path"],
        release_path=fixture["release_path"],
        true_manifest_dir=fixture["manifest_dir"],
        output_dir=second_output,
        seed=73,
        global_index_offset=810000,
    )
    assert second["controls"]["neutral"]["content_sha256"] == audit["controls"]["neutral"]["content_sha256"]
    assert second["controls"]["shuffled"]["content_sha256"] == audit["controls"]["shuffled"]["content_sha256"]
    assert second["pairing_sha256"] == audit["pairing_sha256"]


def test_rejects_duplicate_true_case_before_writing(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    duplicate = dict(fixture["true_row"])
    duplicate["global_index"] = 400001
    _write_jsonl(fixture["shard_path"], [fixture["true_row"], duplicate])
    audit_path = fixture["manifest_dir"] / "manifest_audit.json"
    manifest_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest_audit["included_with_rule"] = 2
    _write_json(audit_path, manifest_audit)

    with pytest.raises(ControlManifestError, match="duplicate sample_id"):
        build_control_manifests(
            base_q100=fixture["base_path"],
            release_path=fixture["release_path"],
            true_manifest_dir=fixture["manifest_dir"],
            output_dir=tmp_path / "controls",
        )
    assert not (tmp_path / "controls").exists()


def test_shuffled_side_swap_is_seeded_and_pair_preserving() -> None:
    rule = {
        "rule_id": "seeded-rule",
        "disease_a": "A",
        "disease_b": "B",
        "supports_a": ["a1", "a2", "a3", "a4"],
        "supports_b": ["b1", "b2", "b3", "b4"],
        "trigger_terms_a": ["ta1", "ta2", "ta3", "ta4"],
        "trigger_terms_b": ["tb1", "tb2", "tb3", "tb4"],
    }

    first = shuffled_rule(rule, seed=1)
    repeated = shuffled_rule(rule, seed=1)
    second_seed = shuffled_rule(rule, seed=2)

    assert repeated == first
    assert second_seed != first
    assert first["disease_a"] == "A"
    assert first["disease_b"] == "B"
    assert set(first["supports_a"]) == set(rule["supports_b"])
    assert set(first["supports_b"]) == set(rule["supports_a"])
    assert set(first["trigger_terms_a"]) == set(rule["trigger_terms_b"])
    assert set(first["trigger_terms_b"]) == set(rule["trigger_terms_a"])


def test_rejects_true_case_missing_from_frozen_base(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    orphan = dict(fixture["true_row"])
    orphan["sample_id"] = "ToyDataset::missing.jpg"
    _write_jsonl(fixture["shard_path"], [orphan])

    with pytest.raises(ControlManifestError, match="missing from base_q100"):
        build_control_manifests(
            base_q100=fixture["base_path"],
            release_path=fixture["release_path"],
            true_manifest_dir=fixture["manifest_dir"],
            output_dir=tmp_path / "controls",
        )
    assert not (tmp_path / "controls").exists()


def test_rejects_release_hash_mismatch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    release = json.loads(fixture["release_path"].read_text(encoding="utf-8"))
    release["rules"][0]["supports_a"] = ["tampered feature"]
    _write_json(fixture["release_path"], release)

    with pytest.raises(ControlManifestError, match="release content hash mismatch"):
        build_control_manifests(
            base_q100=fixture["base_path"],
            release_path=fixture["release_path"],
            true_manifest_dir=fixture["manifest_dir"],
            output_dir=tmp_path / "controls",
        )


def test_rejects_non_list_rule_fields_before_shuffling(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    release = json.loads(fixture["release_path"].read_text(encoding="utf-8"))
    release["rules"][0]["supports_a"] = "not-a-list"
    release["content_sha256"] = canonical_hash(
        {key: value for key, value in release.items() if key != "content_sha256"}
    )
    _write_json(fixture["release_path"], release)
    manifest_audit_path = fixture["manifest_dir"] / "manifest_audit.json"
    manifest_audit = json.loads(manifest_audit_path.read_text(encoding="utf-8"))
    manifest_audit["release_sha256"] = release["content_sha256"]
    _write_json(manifest_audit_path, manifest_audit)

    with pytest.raises(ControlManifestError, match="supports_a must be a list"):
        build_control_manifests(
            base_q100=fixture["base_path"],
            release_path=fixture["release_path"],
            true_manifest_dir=fixture["manifest_dir"],
            output_dir=tmp_path / "controls",
        )


def test_rejects_string_alias_of_frozen_global_index(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    alias = dict(fixture["base"])
    alias["sample_id"] = "ToyDataset::case-002.jpg"
    alias["relative_path"] = "case-002.jpg"
    alias["global_index"] = "101"
    _write_jsonl(fixture["base_path"], [fixture["base"], alias])

    with pytest.raises(ControlManifestError, match="global_index must be a JSON integer"):
        build_control_manifests(
            base_q100=fixture["base_path"],
            release_path=fixture["release_path"],
            true_manifest_dir=fixture["manifest_dir"],
            output_dir=tmp_path / "controls",
        )


def test_rejects_tampered_safety_constraints(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    tampered = dict(fixture["true_row"])
    tampered["prompt"] = tampered["prompt"].replace(
        "It must not justify changing to any third diagnosis.",
        "A third diagnosis is permitted by this tampered control.",
    )
    tampered["prompt_sha256"] = canonical_hash(tampered["prompt"])
    _write_jsonl(fixture["shard_path"], [tampered])

    with pytest.raises(ControlManifestError, match="safety/rule injection"):
        build_control_manifests(
            base_q100=fixture["base_path"],
            release_path=fixture["release_path"],
            true_manifest_dir=fixture["manifest_dir"],
            output_dir=tmp_path / "controls",
        )


def test_rejects_split_role_that_disagrees_with_manifest_cohort(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    wrong_split = dict(fixture["true_row"])
    wrong_split["split_role"] = "heldout_evaluation"
    _write_jsonl(fixture["shard_path"], [wrong_split])

    with pytest.raises(ControlManifestError, match="split/cohort mismatch"):
        build_control_manifests(
            base_q100=fixture["base_path"],
            release_path=fixture["release_path"],
            true_manifest_dir=fixture["manifest_dir"],
            output_dir=tmp_path / "controls",
        )


def test_rejects_rule_pair_not_equal_to_frozen_candidate_top2(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    base = dict(fixture["base"])
    base["candidate_probabilities"] = [
        ["Disease Alpha", 0.60],
        ["Disease Gamma", 0.30],
        ["Disease Beta", 0.10],
    ]
    base["prompt"] = base["prompt"].replace(
        "- Disease Beta: 30.0%\n- Disease Gamma: 10.0%",
        "- Disease Gamma: 30.0%\n- Disease Beta: 10.0%",
    )
    _write_jsonl(fixture["base_path"], [base])
    true_row = dict(fixture["true_row"])
    true_row["candidate_probabilities"] = base["candidate_probabilities"]
    release = json.loads(fixture["release_path"].read_text(encoding="utf-8"))
    release_rule = {**release["rules"][0], "release_id": release["release_id"]}
    true_row["prompt"] = inject_rule(
        base["prompt"], release_rule, baseline_prediction=true_row["c100_anchor_prediction"]
    )
    true_row["prompt_sha256"] = canonical_hash(true_row["prompt"])
    _write_jsonl(fixture["shard_path"], [true_row])

    with pytest.raises(ControlManifestError, match="rule pair is not the frozen candidate top-2"):
        build_control_manifests(
            base_q100=fixture["base_path"],
            release_path=fixture["release_path"],
            true_manifest_dir=fixture["manifest_dir"],
            output_dir=tmp_path / "controls",
        )


def test_rejects_c100_anchor_outside_frozen_pair(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    wrong_anchor = dict(fixture["true_row"])
    wrong_anchor["c100_anchor_prediction"] = "Disease Gamma"
    release = json.loads(fixture["release_path"].read_text(encoding="utf-8"))
    release_rule = {**release["rules"][0], "release_id": release["release_id"]}
    wrong_anchor["prompt"] = inject_rule(
        fixture["base"]["prompt"],
        release_rule,
        baseline_prediction=wrong_anchor["c100_anchor_prediction"],
    )
    wrong_anchor["prompt_sha256"] = canonical_hash(wrong_anchor["prompt"])
    _write_jsonl(fixture["shard_path"], [wrong_anchor])

    with pytest.raises(ControlManifestError, match="C100 anchor is outside"):
        build_control_manifests(
            base_q100=fixture["base_path"],
            release_path=fixture["release_path"],
            true_manifest_dir=fixture["manifest_dir"],
            output_dir=tmp_path / "controls",
        )


def test_cli_help_supports_module_and_direct_script_entrypoints() -> None:
    package_root = Path(__file__).resolve().parents[1]
    direct = subprocess.run(
        [sys.executable, str(package_root / "self_evolving_v1" / "controls.py"), "--help"],
        cwd=package_root,
        capture_output=True,
        text=True,
        check=False,
    )
    module = subprocess.run(
        [sys.executable, "-m", "self_evolving_v1.controls", "--help"],
        cwd=package_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert direct.returncode == 0, direct.stderr
    assert module.returncode == 0, module.stderr
    assert "--true-manifest-dir" in direct.stdout
    assert "--true-manifest-dir" in module.stdout


@pytest.mark.parametrize("value", [None, 0.99])
def test_rejects_missing_or_mismatched_rule_reliability(
    tmp_path: Path, value: float | None
) -> None:
    fixture = _fixture(tmp_path)
    row = dict(fixture["true_row"])
    if value is None:
        row.pop("rule_reliability")
    else:
        row["rule_reliability"] = value
    _write_jsonl(fixture["shard_path"], [row])

    expected = "missing fields: rule_reliability" if value is None else "rule_reliability mismatch"
    with pytest.raises(ControlManifestError, match=expected):
        build_control_manifests(
            base_q100=fixture["base_path"],
            release_path=fixture["release_path"],
            true_manifest_dir=fixture["manifest_dir"],
            output_dir=tmp_path / "controls",
        )


def test_rejects_candidate_metadata_that_disagrees_with_prompt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    mismatched = [
        ["Disease Alpha", 0.99],
        ["Disease Beta", 0.001],
        ["Disease Gamma", 0.009],
    ]
    base = dict(fixture["base"])
    base["candidate_probabilities"] = mismatched
    source = dict(fixture["true_row"])
    source["candidate_probabilities"] = mismatched
    _write_jsonl(fixture["base_path"], [base])
    _write_jsonl(fixture["shard_path"], [source])

    with pytest.raises(ControlManifestError, match="prompt/metadata candidate probability mismatch"):
        build_control_manifests(
            base_q100=fixture["base_path"],
            release_path=fixture["release_path"],
            true_manifest_dir=fixture["manifest_dir"],
            output_dir=tmp_path / "controls",
        )
