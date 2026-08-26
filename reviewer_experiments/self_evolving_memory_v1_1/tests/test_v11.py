from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np

from self_evolving_v1.core import rule_applicability, strict_rule_from_response
from self_evolving_v1.pipeline import (
    density_proxy,
    finalize_evaluation,
    normalize_rows,
    pair_constrained_prediction,
    rule_prompt,
    select_pair_sources,
    tune_gate,
    validate_rules,
)


def _prepared_from_angles(angles: list[float], order: list[int] | None = None) -> dict:
    radians = np.deg2rad(np.asarray(angles, dtype=np.float32))
    original = normalize_rows(np.column_stack((np.cos(radians), np.sin(radians))))
    order = list(range(len(original))) if order is None else order
    matrix = original[order]
    rows = [
        {
            "case_id": f"case_{original_index:02d}",
            "findings": "Complete synthetic dermatology findings for deterministic selection testing. " * 3,
            "image_path": f"/train/class/case_{original_index:02d}.jpg",
            "feature_index": original_index,
        }
        for original_index in order
    ]
    centroid = matrix.mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
    return {
        "rows": rows,
        "matrix": matrix,
        "centroid": centroid,
        "density": density_proxy(matrix),
        "own_similarity": matrix @ centroid,
        "clean_indices": list(range(len(matrix))),
    }


def _unit_vector(angle: float) -> np.ndarray:
    radians = np.deg2rad(angle)
    return normalize_rows(
        np.asarray([[np.cos(radians), np.sin(radians)]], dtype=np.float32)
    )[0]


def _v11_record_and_payload() -> tuple[dict, dict]:
    findings_a = {
        "a1": "Blue radial streaks frame a dark asymmetric center on this lesion. A smooth peripheral rim is visible.",
        "a2": "Blue radial streaks frame a dark asymmetric center on this lesion. A faint peripheral rim is visible.",
        "a3": "Blue radial streaks frame a dark asymmetric center on this lesion. A broad peripheral rim is visible.",
    }
    findings_b = {
        "b1": "Honey colored crust covers a uniform symmetric plaque on this lesion. Mild redness is visible.",
        "b2": "Honey colored crust covers a uniform symmetric plaque on this lesion. Focal redness is visible.",
        "b3": "Honey colored crust covers a uniform symmetric plaque on this lesion. Diffuse redness is visible.",
    }
    record = {
        "schema_version": "v1.1",
        "rule_id": "rule_v11",
        "dataset": "ToyDataset",
        "disease_a": "Disease Alpha",
        "disease_b": "Disease Beta",
        "source_case_ids_a": list(findings_a),
        "source_case_ids_b": list(findings_b),
        "source_cases_a": [
            {"case_id": case_id, "findings": findings}
            for case_id, findings in findings_a.items()
        ],
        "source_cases_b": [
            {"case_id": case_id, "findings": findings}
            for case_id, findings in findings_b.items()
        ],
        "minimum_feature_support": 3,
    }
    payload = {
        "schema_version": "v1.1",
        "publishable": True,
        "disease_a": "Disease Alpha",
        "disease_b": "Disease Beta",
        "supports_a": ["Blue radial streaks around an asymmetric center"],
        "supports_b": ["Honey colored crust over a uniform plaque"],
        "nondiscriminative": ["Visible superficial change"],
        "trigger_terms_a": ["blue radial streaks", "dark asymmetric center"],
        "trigger_terms_b": ["honey colored crust", "uniform symmetric plaque"],
        "required_context_terms": [],
        "abstain_terms": [],
        "required_context": [],
        "limitations": ["Abstain when the observable morphology is absent."],
        "evidence_a": [
            {
                "feature": "Blue radial streaks around an asymmetric center",
                "cases": [
                    {
                        "case_id": case_id,
                        "quote": "Blue radial streaks frame a dark asymmetric center",
                    }
                    for case_id in findings_a
                ],
            }
        ],
        "evidence_b": [
            {
                "feature": "Honey colored crust over a uniform plaque",
                "cases": [
                    {
                        "case_id": case_id,
                        "quote": "Honey colored crust covers a uniform symmetric plaque",
                    }
                    for case_id in findings_b
                ],
            }
        ],
        "source_case_ids_a": list(findings_a),
        "source_case_ids_b": list(findings_b),
    }
    return record, payload


def _diagnosis_prompt(findings: str) -> str:
    return f"""[1. Frozen Current Visual Findings]:
{findings}

[2. Frozen Model Prediction Probabilities]:
- Disease Alpha: 60.0%
- Disease Beta: 30.0%
- Disease Gamma: 10.0%

[3. Evolved Diagnostic Guidelines from Verified Training Cases]:
old

[4. Precedent Cases from the Same Frozen Training Checkpoint]:
cases"""


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_predictions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_normalize_rows_normalizes_each_nonzero_row_without_nan() -> None:
    matrix = np.asarray([[3.0, 4.0], [0.0, 0.0], [-6.0, 8.0]], dtype=np.float32)

    normalized = normalize_rows(matrix)

    np.testing.assert_allclose(normalized[0], [0.6, 0.8], atol=1e-6)
    np.testing.assert_allclose(normalized[1], [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(normalized[2], [-0.6, 0.8], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(normalized[[0, 2]], axis=1), 1.0, atol=1e-6)
    assert np.isfinite(normalized).all()


def test_source_roles_are_disjoint_and_near_duplicates_are_not_selected() -> None:
    # case_05 and case_06 are exact embedding duplicates; enough distinct
    # alternatives remain to fill the requested cohort without selecting both.
    prepared = _prepared_from_angles(
        [-75, -60, -45, -30, -15, 0, 0, 15, 30, 45, 60, 75, 90, 105]
    )

    selected = select_pair_sources(prepared, _unit_vector(90), limit=12)

    selected_ids = [row["case_id"] for row in selected]
    assert len(selected) == 12
    assert len(selected_ids) == len(set(selected_ids))
    assert not {"case_05", "case_06"} <= set(selected_ids)

    role_sets: dict[str, set[str]] = {}
    for row in selected:
        role_sets.setdefault(row["selection_role"], set()).add(row["case_id"])
    assert sum(len(case_ids) for case_ids in role_sets.values()) == len(selected_ids)

    positions = {row["case_id"]: index for index, row in enumerate(prepared["rows"])}
    selected_positions = [positions[case_id] for case_id in selected_ids]
    similarities = prepared["matrix"][selected_positions] @ prepared["matrix"][selected_positions].T
    upper_triangle = similarities[np.triu_indices(len(selected_positions), k=1)]
    assert float(upper_triangle.max()) < 0.9995


def test_embedding_hard_sources_have_strictly_positive_pair_margin() -> None:
    prepared = _prepared_from_angles(
        [-70, -55, -40, -25, -10, 0, 10, 25, 40, 55, 70, 85, 100]
    )

    selected = select_pair_sources(prepared, _unit_vector(45), limit=9)
    hard = [
        row
        for row in selected
        if row["selection_role"] == "quality_screened_embedding_hard"
    ]

    assert hard, "the fixture must exercise the embedding-hard branch"
    assert all(float(row["pair_margin"]) > 0.0 for row in hard)


def test_source_selection_is_invariant_to_input_row_order() -> None:
    angles = [-73, -59, -42, -27, -13, -2, 8, 19, 33, 48, 64, 81, 103]
    natural = _prepared_from_angles(angles)
    permuted = _prepared_from_angles(
        angles,
        order=[5, 1, 10, 3, 8, 0, 12, 2, 9, 4, 7, 6, 11],
    )
    comparator = _unit_vector(67)

    natural_selection = select_pair_sources(natural, comparator, limit=9)
    permuted_selection = select_pair_sources(permuted, comparator, limit=9)

    natural_roles = {
        row["case_id"]: row["selection_role"] for row in natural_selection
    }
    permuted_roles = {
        row["case_id"]: row["selection_role"] for row in permuted_selection
    }
    assert permuted_roles == natural_roles


def test_rule_prompt_keeps_findings_beyond_the_old_900_character_cutoff() -> None:
    prefix = "regular morphology " * 55
    findings = prefix + "MARKER_AFTER_CHARACTER_900 remains clinically relevant."
    assert findings.index("MARKER_AFTER_CHARACTER_900") > 900
    case = {
        "case_id": "long_case",
        "selection_role": "representative",
        "findings": findings,
    }

    prompt = rule_prompt("ToyDataset", "Disease Alpha", "Disease Beta", [case], [case])

    assert "MARKER_AFTER_CHARACTER_900" in prompt


def test_v11_evidence_and_complete_source_references_are_accepted() -> None:
    record, payload = _v11_record_and_payload()

    rule, error = strict_rule_from_response(record, json.dumps(payload))

    assert error is None
    assert rule is not None
    assert rule.schema_version == "v1.1"
    assert len(rule.evidence_a[0]["cases"]) == 3
    assert len(rule.evidence_b[0]["cases"]) == 3


def test_v11_rejects_incomplete_sources_and_unverifiable_quotes() -> None:
    record, payload = _v11_record_and_payload()
    incomplete = copy.deepcopy(payload)
    incomplete["source_case_ids_a"] = incomplete["source_case_ids_a"][:-1]

    rule, error = strict_rule_from_response(record, json.dumps(incomplete))

    assert rule is None
    assert error == "incomplete_source_coverage"

    unverifiable = copy.deepcopy(payload)
    unverifiable["evidence_a"][0]["cases"][0]["quote"] = "fabricated morphology absent from source"

    rule, error = strict_rule_from_response(record, json.dumps(unverifiable))

    assert rule is None
    assert error == "invalid_feature_evidence"


def test_rule_applicability_executes_context_triggers_and_abstention() -> None:
    rule = {
        "trigger_terms_a": ["blue radial streaks", "dark asymmetric center"],
        "trigger_terms_b": ["honey colored crust", "uniform symmetric plaque"],
        "required_context_terms": ["on the face"],
        "abstain_terms": ["dense hair artifact"],
    }
    applicable_prompt = _diagnosis_prompt(
        "On the face, blue radial streaks frame a dark asymmetric center."
    )

    applicable, reason, matches = rule_applicability(applicable_prompt, rule)

    assert applicable
    assert reason == "trigger_supports_a"
    assert matches == {"a": 2, "b": 0}

    missing_context = _diagnosis_prompt(
        "Blue radial streaks frame a dark asymmetric center on the trunk."
    )
    assert rule_applicability(missing_context, rule)[1] == "required_context_missing"

    abstained = _diagnosis_prompt(
        "On the face, blue radial streaks frame a dark asymmetric center under dense hair artifact."
    )
    assert rule_applicability(abstained, rule)[1] == "abstention_term_present"


def test_v11_third_class_output_falls_back_to_c100_anchor() -> None:
    row = {
        "arm": "structured_evolving_memory_v1_1",
        "dataset": "ToyDataset",
        "candidate_probabilities": [
            ["Disease Alpha", 0.60],
            ["Disease Beta", 0.30],
            ["Disease Gamma", 0.10],
        ],
        "rule_disease_a": "Disease Alpha",
        "rule_disease_b": "Disease Beta",
        "response": json.dumps({"PrimaryDiagnosis": "Disease Gamma"}),
    }

    prediction, reason = pair_constrained_prediction(row, "Disease Alpha")

    assert prediction == "Disease Alpha"
    assert reason == "third_class_fallback"

    row["response"] = json.dumps({"PrimaryDiagnosis": "Disease Beta"})
    prediction, reason = pair_constrained_prediction(row, "Disease Alpha")
    assert prediction == "Disease Beta"
    assert reason == "pair_constrained_output"


def test_unsafe_v11_rule_is_rejected_even_when_exposure_is_insufficient(
    tmp_path: Path,
) -> None:
    root = tmp_path / "experiment"
    releases = root / "releases"
    candidate_path = releases / "v1_1_candidate.json"
    output_path = releases / "v1_1_active.json"
    predictions_path = root / "predictions.csv"
    results_path = root / "acceptance_results.jsonl"
    sample_id = "DDI::/test/alpha/case.jpg"

    _write_jsonl(
        root / "base_q100_records.jsonl",
        [{"sample_id": sample_id, "split_role": "rule_acceptance"}],
    )
    _write_json(
        candidate_path,
        {
            "release_id": "v1_1_candidate",
            "source_stream_hashes": {"DDI": "stream-hash"},
            "rules": [
                {
                    "rule_id": "unsafe_rule",
                    "schema_version": "v1.1",
                    "dataset": "DDI",
                    "disease_a": "Disease Alpha",
                    "disease_b": "Disease Beta",
                }
            ],
        },
    )
    _write_json(
        releases / "v0_c100.json",
        {"release_id": "v0_c100", "content_sha256": "baseline-hash"},
    )
    _write_predictions(
        predictions_path,
        [
            {
                "sample_id": sample_id,
                "condition": "C100",
                "prediction": "Disease Alpha",
                "correct": 1,
            }
        ],
    )
    _write_jsonl(
        results_path,
        [
            {
                "record_type": "inference",
                "sample_id": sample_id,
                "dataset": "DDI",
                "ground_truth": "Disease Alpha",
                "candidate_probabilities": [
                    ["Disease Alpha", 0.90],
                    ["Disease Beta", 0.10],
                ],
                "response": json.dumps({"PrimaryDiagnosis": "Disease Beta"}),
                "rule_id": "unsafe_rule",
                "arm": "structured_evolving_memory_v1_1",
                "rule_disease_a": "Disease Alpha",
                "rule_disease_b": "Disease Beta",
            }
        ],
    )
    args = argparse.Namespace(
        root=root,
        candidate_release=candidate_path,
        predictions=predictions_path,
        results=[str(results_path)],
        minimum_exposures=20,
        minimum_changes=5,
        minimum_gains=5,
        minimum_net_gain=4,
        maximum_regressions=1,
        maximum_one_sided_p=0.10,
        output=output_path,
    )

    validate_rules(args)

    release = json.loads(output_path.read_text(encoding="utf-8"))
    rule = release["rules"][0]
    assert rule["validation_exposures"] == 1
    assert rule["validation_high_confidence_regressions"] == 1
    assert rule["status"] == "rejected"
    assert release["status"] == "rolled_back"


def test_empty_v11_gate_writes_v11_fallback_policy(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    sample_id = "DDI::/test/alpha/gate.jpg"
    predictions_path = root / "predictions.csv"
    results_path = root / "empty_gate_results.jsonl"
    release_path = root / "releases" / "v1_1_active.json"
    output_path = root / "releases" / "gate_v1_1.json"

    _write_jsonl(
        root / "base_q100_records.jsonl",
        [{"sample_id": sample_id, "split_role": "gate_development"}],
    )
    _write_predictions(
        predictions_path,
        [
            {
                "sample_id": sample_id,
                "condition": "C100",
                "prediction": "Disease Alpha",
                "correct": 1,
            }
        ],
    )
    _write_json(release_path, {"release_id": "v1_1_active"})
    _write_jsonl(results_path, [])
    args = argparse.Namespace(
        root=root,
        predictions=predictions_path,
        results=[str(results_path)],
        release=release_path,
        minimum_applicable=100,
        minimum_overrides=30,
        minimum_discordances=10,
        minimum_net_gain=3,
        maximum_high_confidence_regressions=0,
        minimum_beyond_top1_gains=3,
        output=output_path,
    )

    tune_gate(args)

    gate = json.loads(output_path.read_text(encoding="utf-8"))
    audit = gate["development_audit"]
    assert gate["policy_id"] == "gate_v1_1_fallback"
    assert gate["minimum_rule_reliability"] == 1.01
    assert audit["fallback"] is True
    assert audit["status"] == "NO_APPLICABLE_RESULTS"
    assert audit["applicable_cases"] == 0


def test_finalize_evaluation_third_class_falls_back_without_override(
    tmp_path: Path,
) -> None:
    root = tmp_path / "experiment"
    sample_id = "DDI::/test/alpha/heldout.jpg"
    predictions_path = root / "predictions.csv"
    results_path = root / "heldout_results.jsonl"
    gate_path = root / "releases" / "gate_v1_1.json"
    output_dir = root / "final_evaluation"
    release_sha256 = "frozen-release-hash"

    _write_jsonl(
        root / "base_q100_records.jsonl",
        [
            {
                "sample_id": sample_id,
                "split_role": "heldout_evaluation",
                "dataset": "DDI",
                "relative_path": "/test/alpha/heldout.jpg",
                "ground_truth": "Disease Alpha",
                "candidate_probabilities": [
                    ["Disease Alpha", 0.45],
                    ["Disease Beta", 0.35],
                    ["Disease Gamma", 0.20],
                ],
            }
        ],
    )
    _write_predictions(
        predictions_path,
        [
            {
                "sample_id": sample_id,
                "condition": "C100",
                "prediction": "Disease Alpha",
                "correct": 1,
            },
            {
                "sample_id": sample_id,
                "condition": "E100",
                "prediction": "Disease Beta",
                "correct": 0,
            },
        ],
    )
    _write_jsonl(
        results_path,
        [
            {
                "record_type": "inference",
                "sample_id": sample_id,
                "dataset": "DDI",
                "ground_truth": "Disease Alpha",
                "candidate_probabilities": [
                    ["Disease Alpha", 0.45],
                    ["Disease Beta", 0.35],
                    ["Disease Gamma", 0.20],
                ],
                "response": json.dumps({"PrimaryDiagnosis": "Disease Gamma"}),
                "rule_id": "rule_v11",
                "rule_key": "DDI::disease alpha::disease beta",
                "arm": "structured_evolving_memory_v1_1",
                "rule_disease_a": "Disease Alpha",
                "rule_disease_b": "Disease Beta",
                "rule_reliability": 1.0,
                "memory_release_id": "v1_1_active",
                "memory_release_sha256": release_sha256,
                "applicability_reason": "trigger_supports_b",
                "trigger_matches": {"a": 0, "b": 2},
            }
        ],
    )
    # Deliberately permissive: the pair constraint, rather than the statistical
    # gate threshold, must prevent the third-class proposal from becoming an override.
    _write_json(
        gate_path,
        {
            "policy_id": "gate_v1_1_permissive_test",
            "allow_when_c_not_top1": True,
            "top1_probability_cap": 1.0,
            "top1_margin_cap": 1.0,
            "minimum_rule_reliability": 0.0,
            "content_sha256": "gate-hash",
        },
    )
    args = argparse.Namespace(
        root=root,
        predictions=predictions_path,
        results=[str(results_path)],
        gate=gate_path,
        release_sha256=release_sha256,
        output=output_dir,
        bootstraps=10,
        seed=20260825,
    )

    finalize_evaluation(args)

    with (output_dir / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["raw_proposal"] == "Disease Gamma"
    assert row["constraint_reason"] == "third_class_fallback"
    assert row["prediction_V1_1RawPairConstrained"] == "Disease Alpha"
    assert row["prediction_V1_1FinalGated"] == "Disease Alpha"
    assert row["override_allowed"] == "0"
    assert row["gate_reason"] == "no_change"

    audit = json.loads((output_dir / "audit.json").read_text(encoding="utf-8"))
    assert audit["third_class_fallbacks"] == 1
    assert audit["overrides"] == 0
