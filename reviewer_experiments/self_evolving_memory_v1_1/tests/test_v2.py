from __future__ import annotations

from self_evolving_v1.v2 import (
    _beta_probability_above_half,
    _confidence_bin,
    _policy_prediction,
    _retrieval_score,
    _select_ledger_concepts,
)


def test_beta_probability_is_exact_and_monotonic() -> None:
    assert _beta_probability_above_half(1, 1) == 0.5
    assert _beta_probability_above_half(2, 1) == 0.75
    assert _beta_probability_above_half(1, 2) == 0.25
    assert _beta_probability_above_half(4, 1) > _beta_probability_above_half(2, 1)


def test_confidence_bins_use_probability_and_margin() -> None:
    assert _confidence_bin([]) == "missing"
    assert _confidence_bin([("a", 0.80), ("b", 0.10)]) == "high"
    assert _confidence_bin([("a", 0.55), ("b", 0.45)]) == "medium"
    assert _confidence_bin([("a", 0.40), ("b", 0.35), ("c", 0.25)]) == "low"


def test_retrieval_score_rewards_utility_and_penalizes_regression() -> None:
    base = {
        "recommendation": "b",
        "trigger_matches": {"a": 0, "b": 3},
        "candidate_probability": 0.4,
        "rule": {
            "utility": {
                "posterior_mean": 0.8,
                "probability_positive": 0.9,
                "net_correct": 3,
                "exposures": 10,
                "regressions": 0,
            }
        },
    }
    harmful = {
        **base,
        "rule": {
            "utility": {
                "posterior_mean": 0.3,
                "probability_positive": 0.2,
                "net_correct": -2,
                "exposures": 10,
                "regressions": 4,
            }
        },
    }
    assert _retrieval_score(base, 3) > _retrieval_score(harmful, 3)


def test_controller_protects_strong_prior_and_supports_two_modes() -> None:
    row = {
        "dataset": "DDI",
        "candidate_probabilities": [["a", 0.8], ["b", 0.2]],
        "retrieval_score": 0.9,
    }
    outcome = {
        "c_prediction": "a",
        "llm_prediction": "b",
        "direct_prediction": "b",
    }
    policy = {
        "minimum_score": 0.7,
        "probability_cap": 0.7,
        "margin_cap": 0.2,
        "execution_mode": "consensus",
    }
    prediction, reason = _policy_prediction(row, outcome, policy)
    assert prediction == "a"
    assert reason == "protected_strong_prior"

    row["candidate_probabilities"] = [["a", 0.5], ["b", 0.45]]
    prediction, reason = _policy_prediction(row, outcome, policy)
    assert prediction == "b"
    assert reason == "consensus_override"

    outcome["llm_prediction"] = "a"
    prediction, reason = _policy_prediction(row, outcome, policy)
    assert prediction == "a"
    assert reason == "reviewer_disagreement"


def test_ledger_selection_uses_lift_and_distinct_families() -> None:
    rows = [
        {"concept_id": "silvery_scale", "prevalence_margin": 0.5, "own_rate": 0.8, "own_count": 4},
        {"concept_id": "scale", "prevalence_margin": 0.4, "own_rate": 0.9, "own_count": 5},
        {"concept_id": "erythema", "prevalence_margin": 0.3, "own_rate": 0.7, "own_count": 3},
    ]
    selected = _select_ledger_concepts(rows)
    assert [row["concept_id"] for row in selected] == ["silvery_scale", "erythema"]
