from __future__ import annotations

import json

from self_evolving_v1.core import (
    GatePolicy,
    inject_rule,
    normalize_overlapping_features,
    rule_key,
    strict_rule_from_response,
)


def test_rule_key_is_order_invariant() -> None:
    assert rule_key("D", "Beta", "Alpha") == rule_key("D", "Alpha", "Beta")


def test_gate_protects_high_confidence_top1() -> None:
    policy = GatePolicy("g", True, 0.7, 0.4, 0.6)
    allowed, reason = policy.allow(
        c_prediction="A",
        evolved_prediction="B",
        candidate_probabilities=[("A", 0.9), ("B", 0.05)],
        rule_reliability=0.8,
    )
    assert not allowed
    assert reason == "protected_frozen_top1"


def test_gate_can_act_when_c_disagrees_with_top1() -> None:
    policy = GatePolicy("g", True, 0.4, 0.2, 0.6)
    allowed, reason = policy.allow(
        c_prediction="B",
        evolved_prediction="A",
        candidate_probabilities=[("A", 0.8), ("B", 0.1)],
        rule_reliability=0.8,
    )
    assert allowed
    assert reason == "c_disagrees_with_frozen_top1"


def test_strict_rule_requires_grounded_sources() -> None:
    record = {
        "rule_id": "r1",
        "dataset": "D",
        "disease_a": "A",
        "disease_b": "B",
        "source_case_ids_a": ["a1", "a2"],
        "source_case_ids_b": ["b1", "b2"],
    }
    payload = {
        "disease_a": "A",
        "disease_b": "B",
        "supports_a": ["feature a"],
        "supports_b": ["feature b"],
        "nondiscriminative": ["generic"],
        "required_context": [],
        "limitations": ["limited"],
        "source_case_ids_a": ["a1", "a2"],
        "source_case_ids_b": ["b1", "b2"],
    }
    rule, error = strict_rule_from_response(record, json.dumps(payload))
    assert error is None
    assert rule is not None
    payload["source_case_ids_b"] = ["invented"]
    rule, error = strict_rule_from_response(record, json.dumps(payload))
    assert rule is None
    assert error == "source_id_mismatch"


def test_strict_rule_rejects_overlapping_feature_lists() -> None:
    record = {
        "rule_id": "r1",
        "dataset": "D",
        "disease_a": "A",
        "disease_b": "B",
        "source_case_ids_a": ["a1", "a2"],
        "source_case_ids_b": ["b1", "b2"],
    }
    payload = {
        "disease_a": "A",
        "disease_b": "B",
        "supports_a": ["smooth surface", "feature a"],
        "supports_b": ["rough surface"],
        "nondiscriminative": ["Smooth surface."],
        "required_context": [],
        "limitations": ["limited"],
        "source_case_ids_a": ["a1", "a2"],
        "source_case_ids_b": ["b1", "b2"],
    }
    rule, error = strict_rule_from_response(record, json.dumps(payload))
    assert rule is None
    assert error == "overlapping_rule_fields"

    repaired, overlaps = normalize_overlapping_features(payload)
    assert overlaps == ["smooth surface"]
    assert repaired["supports_a"] == ["feature a"]
    assert repaired["nondiscriminative"] == ["Smooth surface."]
    rule, error = strict_rule_from_response(record, json.dumps(repaired))
    assert error is None
    assert rule is not None


def test_strict_rule_requires_the_entire_response_to_be_json() -> None:
    record = {
        "rule_id": "r1",
        "dataset": "D",
        "disease_a": "A",
        "disease_b": "B",
        "source_case_ids_a": ["a1", "a2"],
        "source_case_ids_b": ["b1", "b2"],
    }
    payload = {
        "disease_a": "A",
        "disease_b": "B",
        "supports_a": ["feature a"],
        "supports_b": ["feature b"],
        "nondiscriminative": ["generic"],
        "required_context": [],
        "limitations": ["limited"],
        "source_case_ids_a": ["a1", "a2"],
        "source_case_ids_b": ["b1", "b2"],
    }
    rule, error = strict_rule_from_response(record, "```json\n" + json.dumps(payload) + "\n```")
    assert rule is None
    assert error == "invalid_json"
    rule, error = strict_rule_from_response(record, json.dumps(payload)[:-1])
    assert rule is None
    assert error == "invalid_json"

    payload["source_case_ids_b"] = ["b1", "b2", "invented"]
    rule, error = strict_rule_from_response(record, json.dumps(payload))
    assert rule is None
    assert error == "source_id_mismatch"


def test_only_rule_section_is_replaced() -> None:
    prompt = """[2. Frozen Model Prediction Probabilities]:
- A: 60.0%
- B: 40.0%

[3. Evolved Diagnostic Guidelines from Verified Training Cases]:
old

[4. Precedent Cases from the Same Frozen Training Checkpoint]:
cases"""
    updated = inject_rule(prompt, None)
    assert "old" not in updated
    assert "No active validated contrastive rule" in updated
    assert "[4. Precedent Cases" in updated
