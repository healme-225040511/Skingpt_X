from __future__ import annotations

import copy
import hashlib
import json

import pytest

from self_evolving_v1.concepts import (
    CONCEPT_LEXICON_SHA256,
    CONCEPT_MATCHER_ALGORITHM_REVISION,
    CONCEPT_MATCHER_SHA256,
    CONCEPT_PROTOCOL,
    build_concept_ledger,
    concept_display,
    concept_ledger_hash,
    extract_concept_ids,
    extract_concept_text,
    extract_observable_concepts,
    recompute_concept_ledger,
    validate_concept_ledger,
    validate_observable_concepts,
)


def _case(case_id: str, findings: str, role: str) -> dict[str, str]:
    return {
        "case_id": case_id,
        "findings": findings,
        "selection_role": role,
    }


def test_fixed_protocol_and_hashes_are_canonical_sha256_values() -> None:
    assert CONCEPT_PROTOCOL == "canonical_observable_concepts_v1"
    assert CONCEPT_MATCHER_ALGORITHM_REVISION == "canonical_matcher_r2_20260826"
    assert len(CONCEPT_LEXICON_SHA256) == 64
    assert len(CONCEPT_MATCHER_SHA256) == 64
    int(CONCEPT_LEXICON_SHA256, 16)
    int(CONCEPT_MATCHER_SHA256, 16)
    assert CONCEPT_LEXICON_SHA256 != CONCEPT_MATCHER_SHA256
    assert concept_display("scale") == "Scale"
    with pytest.raises(KeyError):
        concept_display("a_disease_name")


def test_nfkc_casefold_and_fixed_synonyms_capture_semantic_rephrasing() -> None:
    text = "The lesion has an ＵＮＥＶＥＮ ＯＵＴＬＩＮＥ with greasy scaling."

    extracted = extract_observable_concepts(text)

    assert {"irregular_border", "greasy_scale"} <= set(extracted)
    assert "scale" not in extracted
    assert extracted["irregular_border"]["matched_term"] == "uneven outline"
    assert extracted["irregular_border"]["matched_text"] == "ＵＮＥＶＥＮ ＯＵＴＬＩＮＥ"
    assert extract_concept_text(text)["irregular_border"] in text
    valid, issues = validate_observable_concepts(text, extracted)
    assert valid, issues


@pytest.mark.parametrize(
    ("text", "absent", "present"),
    [
        (
            "No scale, crust, or erosion is seen; papules remain visible.",
            {"scale", "crust", "erosion"},
            {"papule"},
        ),
        (
            "Without ulceration or fissuring, but crusting is present.",
            {"ulceration", "fissure"},
            {"crust"},
        ),
        (
            "Vesicles are absent. Pustules are clearly visible.",
            {"vesicle"},
            {"pustule"},
        ),
        (
            "Neither erythema nor edema is identified, yet scale is prominent.",
            {"erythema", "edema"},
            {"scale"},
        ),
        (
            "There is no evidence of erosion, although an ulcerated center is present.",
            {"erosion"},
            {"ulceration"},
        ),
        (
            "No scale, crust, erosion, ulceration, fissures, vesicles, bullae, or pustules are seen.",
            {"scale", "crust", "erosion", "ulceration", "fissure", "vesicle", "bulla", "pustule"},
            set(),
        ),
        (
            "Scale, crust, erosion, ulceration, fissures, vesicles, bullae, and pustules are not seen.",
            {"scale", "crust", "erosion", "ulceration", "fissure", "vesicle", "bulla", "pustule"},
            set(),
        ),
        (
            "The surface is non-scaly and non ulcerated, but papules are visible.",
            {"scale", "ulceration"},
            {"papule"},
        ),
        (
            "The surface is scale-free and crust-free, with an ulcer-free center and visible papules.",
            {"scale", "crust", "ulceration"},
            {"papule"},
        ),
        (
            "No scale is seen, and a plaque is present.",
            {"scale"},
            {"plaque"},
        ),
    ],
)
def test_conservative_negation_scope_handles_lists_and_contrasts(
    text: str,
    absent: set[str],
    present: set[str],
) -> None:
    concepts = extract_concept_ids(text)
    assert not (concepts & absent)
    assert present <= concepts


def test_token_boundaries_reject_substring_traps() -> None:
    text = "A crustacean icon, grayscale card, scalene marker, and erosionally altered stone."
    assert extract_concept_ids(text) == set()


def test_query_presence_is_independent_of_two_word_evidence_quote_rule() -> None:
    assert extract_concept_ids("Scale.") == {"scale"}
    assert extract_observable_concepts("Scale.") == {}


def test_every_fixed_synonym_matches_itself_including_apostrophes() -> None:
    assert "targetoid_pattern" in extract_concept_ids("A bull's eye pattern.")


def test_per_case_mutually_exclusive_concepts_are_all_removed() -> None:
    text = (
        "The lesion is symmetric in one view but asymmetric in another. "
        "Its border is described as regular and also irregular. "
        "A crusted surface remains visible."
    )
    concepts = extract_concept_ids(text)
    assert not ({"symmetry", "asymmetry", "regular_border", "irregular_border"} & concepts)
    assert "crust" in concepts


@pytest.mark.parametrize(
    ("text", "specific", "generic"),
    [
        ("Honey-colored crust covers the lesion.", "honey_colored_crust", "crust"),
        ("Atypical pigment network is visible.", "atypical_pigment_network", "pigment_network"),
        ("Silvery scale covers the plaque.", "silvery_scale", "scale"),
        ("A broad raised lesion forms a plaque.", "plaque", "raised_surface"),
    ],
)
def test_specific_concept_suppresses_its_generic_parent(
    text: str,
    specific: str,
    generic: str,
) -> None:
    concepts = extract_concept_ids(text)
    assert specific in concepts
    assert generic not in concepts


def _unequal_prevalence_cases() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    roles_a = ["representative", "diverse", "representative", "quality_screened_embedding_hard"]
    cases_a = [
        _case(
            f"a{index}",
            "An uneven outline is visible with a scaly surface.",
            role,
        )
        for index, role in enumerate(roles_a, start=1)
    ]
    roles_b = [
        "representative",
        "diverse",
        "representative",
        "diverse",
        "representative",
        "diverse",
        "representative",
        "quality_screened_embedding_hard",
    ]
    cases_b = [
        _case(
            f"b{index}",
            (
                "An uneven outline is visible with a scaly surface."
                if index <= 4
                else "A smooth surface is visible with scaly change."
            ),
            role,
        )
        for index, role in enumerate(roles_b, start=1)
    ]
    return cases_a, cases_b


def test_ledger_uses_per_side_prevalence_when_sample_sizes_differ() -> None:
    cases_a, cases_b = _unequal_prevalence_cases()

    ledger = build_concept_ledger(cases_a, cases_b, minimum_support=3)
    irregular = next(row for row in ledger["candidate_a"] if row["concept_id"] == "irregular_border")

    # Raw support is tied (4 versus 4), but normalized prevalence is 1.0 versus
    # 0.5, so this is correctly an A-side candidate with a 0.5 margin.
    assert irregular["own_count"] == irregular["opposite_count"] == 4
    assert irregular["own_rate"] == 1.0
    assert irregular["opposite_rate"] == 0.5
    assert irregular["prevalence_margin"] == 0.5
    assert irregular["case_ids_a"] == ["a1", "a2", "a3", "a4"]
    assert irregular["case_ids_b"] == ["b1", "b2", "b3", "b4"]

    scale = next(row for row in ledger["nondiscriminative"] if row["concept_id"] == "scale")
    assert scale["count_a"] == 4
    assert scale["count_b"] == 8
    assert scale["rate_a"] == scale["rate_b"] == 1.0


def test_candidate_requires_two_roles_and_cannot_be_hard_only() -> None:
    cases_a = [
        _case("a1", "A nodular lesion has visible scale.", "quality_screened_embedding_hard"),
        _case("a2", "A nodular lesion has visible scale.", "quality_screened_embedding_hard"),
        _case("a3", "A nodular lesion has visible scale.", "quality_screened_embedding_hard"),
        _case("a4", "A papular lesion has visible crust.", "representative"),
        _case("a5", "A papular lesion has visible crust.", "representative"),
        _case("a6", "A papular lesion has visible crust.", "representative"),
        _case("a7", "A fissured surface has visible erosion.", "representative"),
        _case("a8", "A fissured surface has visible erosion.", "diverse"),
        _case("a9", "A fissured surface has visible erosion.", "quality_screened_embedding_hard"),
    ]
    cases_b = [
        _case("b1", "A smooth surface remains visible.", "representative"),
        _case("b2", "A smooth surface remains visible.", "diverse"),
        _case("b3", "A smooth surface remains visible.", "representative"),
    ]

    ledger = build_concept_ledger(cases_a, cases_b)
    ids = {entry["concept_id"] for entry in ledger["candidate_a"]}

    assert "nodule" not in ids  # embedding-hard only
    assert "papule" not in ids  # one role only
    assert {"fissure", "erosion"} <= ids
    fissure = next(entry for entry in ledger["candidate_a"] if entry["concept_id"] == "fissure")
    assert fissure["roles"] == ["diverse", "quality_screened_embedding_hard", "representative"]


def test_ledger_is_order_invariant_and_every_evidence_quote_is_exact() -> None:
    cases_a, cases_b = _unequal_prevalence_cases()

    natural = build_concept_ledger(cases_a, cases_b, min_support=3)
    reversed_input = build_concept_ledger(reversed(cases_a), reversed(cases_b), min_support=3)

    assert reversed_input == natural
    assert natural["content_sha256"] == concept_ledger_hash(natural)
    source_by_id = {row["case_id"]: row["findings"] for row in cases_a + cases_b}
    for category in ("candidate_a", "candidate_b", "nondiscriminative"):
        for entry in natural[category]:
            for evidence in entry["evidence"]:
                source = source_by_id[evidence["case_id"]]
                quote = evidence["quote"]
                assert source[evidence["quote_start"] : evidence["quote_end"]] == quote
                assert hashlib.sha256(quote.encode("utf-8")).hexdigest() == evidence["quote_sha256"]
                assert 2 <= len(quote.split()) <= 24

    valid, issues = validate_concept_ledger(natural, cases_a, cases_b)
    assert valid, issues
    assert recompute_concept_ledger(cases_a, cases_b) == natural


def test_validation_detects_quote_and_ledger_tampering() -> None:
    cases_a, cases_b = _unequal_prevalence_cases()
    ledger = build_concept_ledger(cases_a, cases_b)
    tampered = copy.deepcopy(ledger)
    tampered["candidate_a"][0]["evidence"][0]["quote"] = "fabricated exact quote"

    valid, issues = validate_concept_ledger(tampered, cases_a, cases_b)

    assert not valid
    assert "content_hash_mismatch" in issues
    assert any(issue.startswith("entry_hash_mismatch:") for issue in issues)
    assert any(issue.startswith("quote_hash_mismatch:") for issue in issues)
    assert "ledger_recomputation_mismatch" in issues


def test_hash_is_stable_through_json_round_trip() -> None:
    cases_a, cases_b = _unequal_prevalence_cases()
    ledger = build_concept_ledger(cases_a, cases_b)
    round_trip = json.loads(json.dumps(ledger, ensure_ascii=False, sort_keys=True))
    assert concept_ledger_hash(round_trip) == ledger["content_sha256"]
    assert round_trip == ledger


def test_invalid_sources_and_support_parameters_fail_closed() -> None:
    cases_a, cases_b = _unequal_prevalence_cases()
    duplicate = cases_a + [dict(cases_a[0])]
    with pytest.raises(ValueError, match="duplicate A case_id"):
        build_concept_ledger(duplicate, cases_b)
    with pytest.raises(ValueError, match="minimum_support"):
        build_concept_ledger(cases_a, cases_b, minimum_support=0)
    with pytest.raises(ValueError, match="disagree"):
        build_concept_ledger(cases_a, cases_b, minimum_support=4, min_support=3)
