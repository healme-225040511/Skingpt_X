from __future__ import annotations

from self_evolving_v1.clinical_router import (
    build_prompt,
    context_key,
    parse_precedents,
    select_cases,
)


def _row() -> dict:
    precedents = []
    ids = []
    labels = ["a", "a", "b", "b", "c", "c"]
    for index, label in enumerate(labels):
        ids.append(f"case-{index}")
        precedents.append(
            f"- Verified precedent [{label}] | similarity={0.95 - index * 0.01:.5f}\n"
            f"  Findings: finding {index}."
        )
    return {
        "sample_id": "DermNet23::x",
        "dataset": "DermNet23",
        "retrieved_case_ids": ids,
        "candidate_probabilities": [["a", 0.40], ["b", 0.30], ["c", 0.20]],
        "prompt": (
            "[Clinical Task]: Final Diagnosis Review.\n\n"
            "[1. Frozen Current Visual Findings]:\nlesion\n\n"
            "[2. Frozen Model Prediction Probabilities]:\n- a: 40.0%\n- b: 30.0%\n- c: 20.0%\n\n"
            "[3. Evolved Diagnostic Guidelines from Verified Training Cases]:\nNone.\n\n"
            "[4. Precedent Cases from the Same Frozen Training Checkpoint]:\n"
            + "\n".join(precedents)
            + "\n\n[5. Frozen Static Handbook Knowledge]:\nDistracting handbook.\n\n"
            "[Instructions]:\n- Choose one.\n"
        ),
    }


def _policy(status: str = "active") -> dict:
    row = _row()
    context = context_key(row, parse_precedents(row))
    return {
        "contexts": [{"context": context, "status": status}],
        "dataset_fallbacks": [{"dataset": "DermNet23", "status": "active"}],
        "items": [
            {"case_id": f"case-{index}", "posterior_lower_score": score}
            for index, score in enumerate([0.1, 0.2, 0.9, 0.8, 0.7, 0.6])
        ],
    }


def test_parse_precedents_binds_frozen_ids() -> None:
    cases = parse_precedents(_row())
    assert len(cases) == 6
    assert cases[0]["case_id"] == "case-0"
    assert cases[-1]["label"] == "c"


def test_s2_utility_changes_similarity_order() -> None:
    row = _row()
    cases = parse_precedents(row)
    selected, action = select_cases(row, cases, _policy(), "S2", limit=3)
    assert action == "utility_reranked"
    assert selected[0]["case_id"] == "case-2"


def test_s3_guarantees_candidate_label_contrast() -> None:
    row = _row()
    selected, action = select_cases(row, parse_precedents(row), _policy(), "S3", limit=3)
    assert action == "contrastive_balanced"
    assert {case["label"] for case in selected} == {"a", "b", "c"}


def test_quarantined_context_routes_to_no_memory() -> None:
    row = _row()
    selected, action = select_cases(row, parse_precedents(row), _policy("quarantine"), "S2")
    assert selected == []
    assert action == "router_no_memory"


def test_prompt_removes_handbook_and_adds_safety_instruction() -> None:
    row = _row()
    selected, _ = select_cases(row, parse_precedents(row), _policy(), "S3", limit=3)
    prompt = build_prompt(row, selected, "S3")
    assert "Frozen Static Handbook Knowledge" not in prompt
    assert "Distracting handbook" not in prompt
    assert "do not diagnose by majority vote" in prompt
    assert prompt.count("Verified precedent") == 3
