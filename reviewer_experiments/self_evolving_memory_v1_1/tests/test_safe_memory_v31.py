from __future__ import annotations

import argparse
import json

from self_evolving_v1.safe_memory_v31 import (
    _match_rule,
    _prompt,
    compile_release,
    nested_role,
)


def _row(sample_id: str = "DermNet23::x", split_role: str = "rule_acceptance") -> dict:
    cases = [
        ("a", "annular plaque with central clearing"),
        ("a", "annular plaque with central clearing"),
        ("b", "asymmetric nodule with ulceration"),
        ("b", "asymmetric nodule with ulceration"),
    ]
    rendered = "\n".join(
        f"- Verified precedent [{label}] | similarity={0.90 - index * 0.01:.5f}\n  Findings: {findings}."
        for index, (label, findings) in enumerate(cases)
    )
    return {
        "sample_id": sample_id,
        "dataset": "DermNet23",
        "split_role": split_role,
        "retrieved_case_ids": [f"{sample_id}-case-{index}" for index in range(len(cases))],
        "candidate_probabilities": [["a", 0.45], ["b", 0.35]],
        "prompt": (
            "[Clinical Task]: Final Diagnosis Review.\n\n"
            "[1. Frozen Current Visual Findings]:\nAnnular plaque with central clearing.\n\n"
            "[2. Frozen Model Prediction Probabilities]:\n- a: 45%\n- b: 35%\n\n"
            "[3. Evolved Diagnostic Guidelines from Verified Training Cases]:\nOld guideline.\n\n"
            "[4. Precedent Cases from the Same Frozen Training Checkpoint]:\n"
            + rendered
            + "\n\n[5. Frozen Static Handbook Knowledge]:\nDistracting handbook.\n\n"
            "[Instructions]:\n- Choose one.\n"
        ),
    }


def _rule() -> dict:
    return {
        "rule_id": "v31r_test",
        "disease_a": "a",
        "disease_b": "b",
        "diagnostic_evidence_a": [
            {"concept_id": "annular_pattern", "feature": "Annular pattern", "own_rate": 1.0, "other_rate": 0.0, "prevalence_margin": 1.0},
            {"concept_id": "central_clearing", "feature": "Central clearing", "own_rate": 1.0, "other_rate": 0.0, "prevalence_margin": 1.0},
        ],
        "diagnostic_evidence_b": [
            {"concept_id": "asymmetry", "feature": "Asymmetry", "own_rate": 1.0, "other_rate": 0.0, "prevalence_margin": 1.0},
            {"concept_id": "nodule", "feature": "Nodule", "own_rate": 1.0, "other_rate": 0.0, "prevalence_margin": 1.0},
        ],
        "not_sufficient": ["generic biopsy advice"],
        "safety_red_flags": ["rapid evolution"],
    }


def test_nested_role_never_reuses_acceptance_rows() -> None:
    assert nested_role(_row(split_role="rule_acceptance")) == "excluded"
    assert nested_role(_row(split_role="gate_development")) == "excluded"
    assert nested_role(_row(split_role="heldout_evaluation")) in {"v31_gate", "v31_evaluation"}


def test_match_requires_two_contrastive_current_features() -> None:
    row = _row()
    match = _match_rule(row, _rule(), minimum_matches=2)
    assert match is not None
    assert match["recommendation"] == "a"
    row["prompt"] = row["prompt"].replace("Annular plaque with central clearing", "Annular lesion", 1)
    assert _match_rule(row, _rule(), minimum_matches=2) is None


def test_prompt_removes_handbook_and_separates_safety_from_diagnosis() -> None:
    row = _row()
    match = _match_rule(row, _rule(), minimum_matches=2)
    assert match is not None
    prompt = _prompt(row, "b", _rule(), match, [])
    assert "Frozen Static Handbook Knowledge" not in prompt
    assert "Distracting handbook" not in prompt
    assert "clinical action only, never diagnostic override" in prompt
    assert '"MemoryAction": "KEEP_BASELINE or OVERRIDE"' in prompt


def test_compile_release_uses_only_rule_acceptance_precedents(tmp_path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    acceptance = _row()
    heldout = _row("DermNet23::heldout", "heldout_evaluation")
    heldout["prompt"] = heldout["prompt"].replace("[a]", "[c]").replace("[b]", "[d]")
    with (root / "base_q100_records.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(acceptance) + "\n")
        handle.write(json.dumps(heldout) + "\n")
    output = tmp_path / "release.json"
    compile_release(
        argparse.Namespace(
            root=root,
            output=output,
            datasets=["DermNet23"],
            minimum_class_cases=1,
            minimum_feature_support=1,
            minimum_prevalence_margin=0.1,
            minimum_features_per_side=2,
            maximum_features=5,
        )
    )
    release = json.loads(output.read_text())
    assert len(release["rules"]) == 1
    assert {release["rules"][0]["disease_a"], release["rules"][0]["disease_b"]} == {"a", "b"}
    assert release["compile_policy"]["content_source_role"] == "rule_acceptance"
