#!/usr/bin/env python3
"""Deterministic, disease-independent observable morphology concepts.

The matcher in this module deliberately uses a small, versioned surface lexicon
instead of an embedding model or a generative model.  A concept match therefore
has an exact, replayable source span.  The ledger builder uses per-side
prevalence (not raw count), requires support across source-selection roles, and
never lets embedding-hard cases establish a candidate by themselves.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any


CONCEPT_PROTOCOL = "canonical_observable_concepts_v1"
CONCEPT_MATCHER_ALGORITHM_REVISION = "canonical_matcher_r2_20260826"
MINIMUM_PREVALENCE_MARGIN = 0.20
HARD_SELECTION_ROLE = "quality_screened_embedding_hard"


# These are observable morphology or distribution concepts.  No diagnosis,
# disease name, demographic attribute, or treatment response is part of the
# vocabulary.  Terms are literals; regular expressions are generated below.
_LEXICON_ROWS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "annular_pattern",
        "Annular pattern",
        ("annular", "ring shaped", "ring-like", "ringlike configuration"),
    ),
    (
        "asymmetry",
        "Asymmetry",
        ("asymmetry", "asymmetric", "asymmetrical", "unequal halves", "uneven shape"),
    ),
    (
        "atrophy",
        "Atrophy",
        ("atrophy", "atrophic", "skin thinning", "thinned skin"),
    ),
    (
        "atypical_pigment_network",
        "Atypical pigment network",
        ("atypical pigment network", "irregular pigment network", "disrupted pigment network"),
    ),
    (
        "blue_white_veil",
        "Blue-white veil",
        ("blue white veil", "blue-whitish veil", "blue-white veil"),
    ),
    (
        "bulla",
        "Bulla",
        ("bulla", "bullae", "bullous", "large blister", "large blisters"),
    ),
    (
        "central_clearing",
        "Central clearing",
        ("central clearing", "clearing in the center", "cleared center", "central sparing"),
    ),
    (
        "color_variegation",
        "Color variegation",
        (
            "color variegation",
            "colour variegation",
            "variegated color",
            "variegated pigmentation",
            "multiple colors",
            "multiple colours",
            "multicolored",
            "multicoloured",
            "heterogeneous pigmentation",
        ),
    ),
    (
        "comedone",
        "Comedone",
        ("comedone", "comedones", "comedonal", "open comedo", "closed comedo", "blackheads", "whiteheads"),
    ),
    (
        "crust",
        "Crust",
        ("crust", "crusts", "crusted", "crusting", "crusty"),
    ),
    (
        "diffuse_distribution",
        "Diffuse distribution",
        ("diffuse distribution", "diffusely distributed", "widespread distribution", "generalized distribution"),
    ),
    (
        "dots_globules",
        "Dots or globules",
        ("dots and globules", "dots or globules", "pigmented dots", "pigmented globules", "brown globules"),
    ),
    (
        "edema",
        "Edema",
        ("edema", "oedema", "edematous", "oedematous", "tissue swelling"),
    ),
    (
        "erosion",
        "Erosion",
        ("erosion", "erosions", "eroded surface", "superficial erosion"),
    ),
    (
        "erythema",
        "Erythema",
        ("erythema", "erythematous", "redness", "reddened skin"),
    ),
    (
        "excoriation",
        "Excoriation",
        ("excoriation", "excoriations", "excoriated", "scratch marks", "scratch-marked"),
    ),
    (
        "fissure",
        "Fissure",
        ("fissure", "fissures", "fissured", "fissuring", "skin crack", "skin cracks"),
    ),
    (
        "flat_surface",
        "Flat surface",
        ("flat surface", "nonraised", "non-raised", "not elevated"),
    ),
    (
        "follicular_pattern",
        "Follicular pattern",
        ("follicular pattern", "follicular distribution", "folliculocentric", "perifollicular"),
    ),
    (
        "greasy_scale",
        "Greasy scale",
        ("greasy scale", "greasy scales", "greasy scaling", "oily scale", "oily scaling"),
    ),
    (
        "grouped_arrangement",
        "Grouped arrangement",
        ("grouped arrangement", "grouped lesions", "clustered lesions", "herpetiform grouping"),
    ),
    (
        "hair_loss",
        "Hair loss",
        ("hair loss", "loss of hair", "alopecia", "reduced hair density"),
    ),
    (
        "honey_colored_crust",
        "Honey-colored crust",
        (
            "honey colored crust",
            "honey-coloured crust",
            "honey-colored crust",
            "golden crust",
            "yellow brown crust",
        ),
    ),
    (
        "hyperkeratosis",
        "Hyperkeratosis",
        ("hyperkeratosis", "hyperkeratotic", "thick keratin", "thickened keratin"),
    ),
    (
        "ill_defined_border",
        "Ill-defined border",
        (
            "ill defined border",
            "ill-defined margin",
            "poorly defined border",
            "indistinct margin",
            "blurred border",
            "fading border",
        ),
    ),
    (
        "induration",
        "Induration",
        ("induration", "indurated", "firm thickening", "hardened skin"),
    ),
    (
        "irregular_border",
        "Irregular border",
        (
            "irregular border",
            "irregular borders",
            "irregular margin",
            "uneven border",
            "uneven outline",
            "jagged edge",
            "notched border",
        ),
    ),
    (
        "lichenification",
        "Lichenification",
        ("lichenification", "lichenified", "accentuated skin lines", "thickened skin markings"),
    ),
    (
        "linear_arrangement",
        "Linear arrangement",
        ("linear arrangement", "linear distribution", "linearly arranged", "in a line", "along a line"),
    ),
    (
        "macule_or_patch",
        "Macule or patch",
        ("macule", "macules", "macular", "patch", "patches", "flat patch"),
    ),
    (
        "nail_pitting",
        "Nail pitting",
        ("nail pitting", "pitted nails", "nail pits"),
    ),
    (
        "nodule",
        "Nodule",
        ("nodule", "nodules", "nodular", "deep lump", "firm lump"),
    ),
    (
        "papule",
        "Papule",
        ("papule", "papules", "papular", "small raised bumps", "small raised bump"),
    ),
    (
        "pigment_network",
        "Pigment network",
        ("pigment network", "pigmented network", "reticular pigment network", "melanocytic network"),
    ),
    (
        "plaque",
        "Plaque",
        ("plaque", "plaques", "plaque-like", "broad raised lesion"),
    ),
    (
        "pustule",
        "Pustule",
        ("pustule", "pustules", "pustular", "pus filled bump", "pus-filled bumps"),
    ),
    (
        "radial_streaks",
        "Radial streaks",
        ("radial streaks", "radial streaming", "peripheral streaks", "pseudopods", "radial projections"),
    ),
    (
        "raised_surface",
        "Raised surface",
        ("raised surface", "elevated lesion", "raised lesion", "surface elevation"),
    ),
    (
        "regular_border",
        "Regular border",
        ("regular border", "regular margin", "even border", "smooth outline", "uniform outline"),
    ),
    (
        "reticular_pattern",
        "Reticular pattern",
        ("reticular pattern", "netlike pattern", "net-like pattern", "lacy network", "latticelike pattern"),
    ),
    (
        "rough_surface",
        "Rough surface",
        ("rough surface", "coarse surface", "rough texture", "uneven texture"),
    ),
    (
        "scale",
        "Scale",
        ("scale", "scales", "scaling", "scaly", "desquamation", "desquamative"),
    ),
    (
        "silvery_scale",
        "Silvery scale",
        ("silvery scale", "silvery scales", "silver white scale", "micaceous scale"),
    ),
    (
        "smooth_surface",
        "Smooth surface",
        ("smooth surface", "smooth texture", "glossy surface"),
    ),
    (
        "symmetry",
        "Symmetry",
        ("symmetry", "symmetric", "symmetrical", "bilateral symmetry", "equal halves"),
    ),
    (
        "targetoid_pattern",
        "Targetoid pattern",
        ("targetoid", "target-like", "target lesion", "concentric rings", "bull's eye pattern", "bulls eye pattern"),
    ),
    (
        "telangiectasia",
        "Telangiectasia",
        ("telangiectasia", "telangiectatic vessels", "fine surface vessels", "dilated surface vessels"),
    ),
    (
        "ulceration",
        "Ulceration",
        ("ulceration", "ulcerated", "ulcer", "ulcers", "open sore", "open sores"),
    ),
    (
        "uniform_color",
        "Uniform color",
        (
            "uniform color",
            "uniform colour",
            "homogeneous color",
            "homogeneous colour",
            "uniform pigmentation",
            "single color",
            "single colour",
        ),
    ),
    (
        "verrucous_surface",
        "Verrucous surface",
        ("verrucous", "warty surface", "wart-like surface", "papillomatous surface"),
    ),
    (
        "vesicle",
        "Vesicle",
        ("vesicle", "vesicles", "vesicular", "small blister", "small blisters"),
    ),
    (
        "well_defined_border",
        "Well-defined border",
        (
            "well defined border",
            "well-defined margin",
            "well circumscribed",
            "well-circumscribed",
            "sharply demarcated",
            "sharp border",
            "clear margin",
        ),
    ),
)


_EXCLUSIVE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("asymmetry", "symmetry"),
    ("irregular_border", "regular_border"),
    ("ill_defined_border", "well_defined_border"),
    ("color_variegation", "uniform_color"),
    ("flat_surface", "raised_surface"),
    ("rough_surface", "smooth_surface"),
)

# A more specific observation subsumes its generic lexical parent.  Without
# this rule, one phrase such as "honey-colored crust" would count as two
# supposedly independent concepts (honey_colored_crust + crust).
_SUBSUMPTION_RELATIONS: tuple[tuple[str, str], ...] = (
    ("atypical_pigment_network", "pigment_network"),
    ("greasy_scale", "scale"),
    ("honey_colored_crust", "crust"),
    ("silvery_scale", "scale"),
    ("macule_or_patch", "flat_surface"),
    ("bulla", "raised_surface"),
    ("nodule", "raised_surface"),
    ("papule", "raised_surface"),
    ("plaque", "raised_surface"),
    ("pustule", "raised_surface"),
    ("vesicle", "raised_surface"),
)

# Families are used as an additional fail-closed independence check when a
# generated rule selects multiple IDs.  Sibling specializations of one generic
# morphology (for example greasy and silvery scale) cannot satisfy a two-signal
# trigger by themselves.
_CONCEPT_FAMILIES: dict[str, str] = {
    "atypical_pigment_network": "pigment_network",
    "pigment_network": "pigment_network",
    "greasy_scale": "scale",
    "scale": "scale",
    "silvery_scale": "scale",
    "crust": "crust",
    "honey_colored_crust": "crust",
}

_NEGATION_PRE_CUES = (
    "no",
    "without",
    "neither",
    "lack",
    "lacks",
    "lacking",
    "absence",
    "absent",
    "denies",
    "denied",
    "negative",
    "free",
    "not",
)
_NEGATION_BREAKERS = (
    "although",
    "but",
    "contain",
    "containing",
    "contains",
    "demonstrate",
    "demonstrates",
    "display",
    "displays",
    "has",
    "have",
    "however",
    "include",
    "includes",
    "including",
    "nevertheless",
    "present",
    "reveal",
    "reveals",
    "show",
    "showing",
    "shows",
    "whereas",
    "with",
    "yet",
)
_NEGATION_SCOPE_TERMINATORS = (
    "appreciated",
    "identified",
    "noted",
    "observed",
    "seen",
    "visible",
)
_PSEUDO_NOT_FOLLOWERS = ("merely", "necessarily", "only", "uncommon")
_PSEUDO_NO_FOLLOWERS = ("doubt", "question")
_HYPHENS = frozenset("\u058a\u05be\u1400\u1806\u2010\u2011\u2012\u2013\u2014\u2015\u2e17\u2e1a\u2e3a\u2e3b\u2e40\u301c\u3030\u30a0\ufe31\ufe32\ufe58\ufe63\uff0d−")
_WORD_RE = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*", flags=re.UNICODE)
_NORMALIZED_WORD_RE = re.compile(r"[a-z0-9]+")
_CLAUSE_BOUNDARY_RE = re.compile(r"[.;:!?\n\r]")


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


_LEXICON_PAYLOAD = [
    {"concept_id": concept_id, "display": display, "synonyms": list(synonyms)}
    for concept_id, display, synonyms in _LEXICON_ROWS
]
CONCEPT_LEXICON_SHA256 = _canonical_hash(_LEXICON_PAYLOAD)

_MATCHER_CONFIG = {
    "protocol": CONCEPT_PROTOCOL,
    "algorithm_revision": CONCEPT_MATCHER_ALGORITHM_REVISION,
    "lexicon_sha256": CONCEPT_LEXICON_SHA256,
    "exclusive_groups": [list(group) for group in _EXCLUSIVE_GROUPS],
    "subsumption_relations": [list(relation) for relation in _SUBSUMPTION_RELATIONS],
    "concept_families": dict(sorted(_CONCEPT_FAMILIES.items())),
    "negation_pre_cues": list(_NEGATION_PRE_CUES),
    "negation_breakers": list(_NEGATION_BREAKERS),
    "negation_scope_terminators": list(_NEGATION_SCOPE_TERMINATORS),
    "negation_suffixes": ["free"],
    "pseudo_not_followers": list(_PSEUDO_NOT_FOLLOWERS),
    "pseudo_no_followers": list(_PSEUDO_NO_FOLLOWERS),
    "negation_scope": "complete_clause_with_explicit_breakers",
    "literal_separator_revision": "space_hyphen_underscore_apostrophe_v1",
    "minimum_quote_words": 2,
    "maximum_quote_words": 24,
    "minimum_prevalence_margin": MINIMUM_PREVALENCE_MARGIN,
    "hard_selection_role": HARD_SELECTION_ROLE,
}
CONCEPT_MATCHER_SHA256 = _canonical_hash(_MATCHER_CONFIG)
CONCEPT_CONFIG_SHA256 = CONCEPT_MATCHER_SHA256

# Public read-only view.  Tuples make synonyms stable even if a caller stores a
# reference to the nested value.
CONCEPT_LEXICON: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        concept_id: MappingProxyType({"display": display, "synonyms": synonyms})
        for concept_id, display, synonyms in _LEXICON_ROWS
    }
)


def _canonicalize_char(value: str) -> str:
    if value.isspace():
        return " "
    if value in _HYPHENS:
        return "-"
    return value


def _normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(_canonicalize_char(char) for char in normalized)


def _normalized_with_alignment(value: str) -> tuple[str, list[int], list[int]]:
    """Return normalized text and an exact mapping back to original spans."""
    source = str(value or "")
    normalized_chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    index = 0
    while index < len(source):
        end = index + 1
        while end < len(source) and unicodedata.combining(source[end]):
            end += 1
        cluster = unicodedata.normalize("NFKC", source[index:end]).casefold()
        for char in cluster:
            normalized_chars.append(_canonicalize_char(char))
            starts.append(index)
            ends.append(end)
        index = end
    return "".join(normalized_chars), starts, ends


def _literal_pattern(term: str) -> re.Pattern[str]:
    tokens = _NORMALIZED_WORD_RE.findall(_normalized_text(term))
    if not tokens:
        raise ValueError(f"concept term has no matchable tokens: {term!r}")
    interior = r"[\s\-_'’]+".join(re.escape(token) for token in tokens)
    return re.compile(rf"(?<![a-z0-9]){interior}(?![a-z0-9])")


_TERM_PATTERNS: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {}
for _concept_id, _display, _synonyms in _LEXICON_ROWS:
    ordered = sorted(
        set(_synonyms),
        key=lambda value: (-len(_NORMALIZED_WORD_RE.findall(_normalized_text(value))), -len(value), value),
    )
    _TERM_PATTERNS[_concept_id] = tuple((term, _literal_pattern(term)) for term in ordered)


def concept_display(concept_id: str) -> str:
    """Return the fixed display string for a canonical concept identifier."""
    try:
        return str(CONCEPT_LEXICON[str(concept_id)]["display"])
    except KeyError as error:
        raise KeyError(f"unknown observable concept: {concept_id}") from error


def concept_family(concept_id: str) -> str:
    """Return the fixed independence family for a canonical concept."""
    identifier = str(concept_id)
    if identifier not in CONCEPT_LEXICON:
        raise KeyError(f"unknown observable concept: {identifier}")
    return _CONCEPT_FAMILIES.get(identifier, identifier)


def _last_clause_boundary(text: str, before: int) -> int:
    last = -1
    for match in _CLAUSE_BOUNDARY_RE.finditer(text, 0, max(before, 0)):
        last = match.end() - 1
    return last


def _word_matches(text: str) -> list[re.Match[str]]:
    return list(_NORMALIZED_WORD_RE.finditer(text))


def _preceding_negation(normalized: str, start: int) -> bool:
    boundary = _last_clause_boundary(normalized, start)
    tokens = [match for match in _word_matches(normalized[:start]) if match.start() > boundary]
    if not tokens:
        return False

    # A contrast/presence predicate ends a preceding negation list.  Thus
    # "no scale, crust, or erosion" propagates, while "no scale, but crust"
    # does not.
    breaker = -1
    for index, token in enumerate(tokens):
        if token.group() in _NEGATION_BREAKERS or token.group() in _NEGATION_SCOPE_TERMINATORS:
            breaker = index
    tokens = tokens[breaker + 1 :]
    words = [token.group() for token in tokens]

    # Handle productive negation prefixes that are separated by whitespace or
    # a hyphen ("non-scaly", "non ulcerated").  Fused forms do not pass the
    # literal token boundary in the first place.
    immediate_prefix = normalized[max(boundary + 1, start - 8) : start]
    if re.search(r"(?:^|[^a-z0-9])non[\s-]*$", immediate_prefix):
        return True

    for index, word in enumerate(words):
        if word not in _NEGATION_PRE_CUES:
            continue
        follower = words[index + 1] if index + 1 < len(words) else ""
        if word == "not" and follower in _PSEUDO_NOT_FOLLOWERS:
            continue
        if word == "no" and follower in _PSEUDO_NO_FOLLOWERS:
            continue
        if word == "negative" and follower not in {"for", "to"}:
            continue
        if word == "free" and follower != "of":
            continue
        if word == "absence" and follower not in {"of", ""}:
            continue
        return True
    return False


def _following_negation(normalized: str, end: int) -> bool:
    segment = normalized[end:]
    if re.match(r"^[\s-]*free(?![a-z0-9])", segment):
        return True
    boundary_match = _CLAUSE_BOUNDARY_RE.search(segment)
    if boundary_match:
        segment = segment[: boundary_match.start()]
    words: list[str] = []
    for token in _word_matches(segment):
        value = token.group()
        if value in _NEGATION_BREAKERS:
            break
        words.append(value)

    if not words:
        return False
    denied_observations = {
        "appreciated",
        "identified",
        "noted",
        "observed",
        "present",
        "seen",
        "visible",
    }
    auxiliaries = {"appear", "appears", "are", "is", "remain", "remains", "was", "were"}
    for index, word in enumerate(words):
        if word == "absent" and (index == 0 or words[index - 1] in auxiliaries):
            return True
        if word != "not" or index + 1 >= len(words):
            continue
        if words[index + 1] == "only":
            continue
        if words[index + 1] in denied_observations and (
            index == 0 or words[index - 1] in auxiliaries
        ):
            return True
    return False


def _is_negated(normalized: str, start: int, end: int) -> bool:
    return _preceding_negation(normalized, start) or _following_negation(normalized, end)


def _quote_word_count(value: str) -> int:
    return len(_NORMALIZED_WORD_RE.findall(_normalized_text(value)))


def _exact_quote(source: str, match_start: int, match_end: int) -> tuple[str, int, int] | None:
    words = list(_WORD_RE.finditer(source))
    overlapping = [
        index
        for index, word in enumerate(words)
        if word.start() < match_end and word.end() > match_start
    ]
    if not overlapping:
        return None
    first, last = overlapping[0], overlapping[-1]

    def crosses_boundary(left_end: int, right_start: int) -> bool:
        return bool(_CLAUSE_BOUNDARY_RE.search(source[left_end:right_start]))

    # Literal one-word concepts need one adjacent source word so every evidence
    # quote meets the pre-registered 2--24 word rule.  Never cross a sentence or
    # semicolon solely to manufacture the second word.
    while _quote_word_count(source[words[first].start() : words[last].end()]) < 2:
        if last + 1 < len(words) and not crosses_boundary(words[last].end(), words[last + 1].start()):
            last += 1
        elif first > 0 and not crosses_boundary(words[first - 1].end(), words[first].start()):
            first -= 1
        else:
            return None

    quote_start, quote_end = words[first].start(), words[last].end()
    quote = source[quote_start:quote_end]
    count = _quote_word_count(quote)
    if not (2 <= count <= 24):
        return None
    return quote, quote_start, quote_end


def _extract_concepts(
    text: str,
    *,
    require_grounded_quote: bool,
) -> dict[str, dict[str, Any]]:
    source = str(text or "")
    normalized, starts, ends = _normalized_with_alignment(source)
    extracted: dict[str, dict[str, Any]] = {}

    for concept_id in sorted(_TERM_PATTERNS):
        matches: list[tuple[int, int, str]] = []
        for term, pattern in _TERM_PATTERNS[concept_id]:
            for match in pattern.finditer(normalized):
                matches.append((match.start(), match.end(), term))
        matches.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))

        for normalized_start, normalized_end, term in matches:
            if _is_negated(normalized, normalized_start, normalized_end):
                continue
            if not starts or normalized_end <= normalized_start:
                continue
            match_start = starts[normalized_start]
            match_end = ends[normalized_end - 1]
            quote_record = _exact_quote(source, match_start, match_end)
            if quote_record is None:
                if require_grounded_quote:
                    continue
                # Query-side presence detection must not depend on being able
                # to expand a one-word finding into a 2--24-word evidence
                # quote. Training ledgers still use the strict branch above.
                quote = source[match_start:match_end]
                quote_start, quote_end = match_start, match_end
            else:
                quote, quote_start, quote_end = quote_record
            extracted[concept_id] = {
                "concept_id": concept_id,
                "display": concept_display(concept_id),
                "quote": quote,
                "quote_start": quote_start,
                "quote_end": quote_end,
                "quote_sha256": hashlib.sha256(quote.encode("utf-8")).hexdigest(),
                "matched_text": source[match_start:match_end],
                "matched_term": term,
                "match_start": match_start,
                "match_end": match_end,
            }
            break

    conflicted: set[str] = set()
    for group in _EXCLUSIVE_GROUPS:
        present = set(group) & set(extracted)
        if len(present) > 1:
            conflicted.update(present)
    for concept_id in conflicted:
        extracted.pop(concept_id, None)
    for child, parent in _SUBSUMPTION_RELATIONS:
        if child in extracted:
            extracted.pop(parent, None)
    return {concept_id: extracted[concept_id] for concept_id in sorted(extracted)}


def extract_observable_concepts(text: str) -> dict[str, dict[str, Any]]:
    """Extract positive concepts with strict, auditable source quotations.

    Returned quotes are exact 2--24-word contiguous slices of ``text``. A
    negated match is ignored. If mutually exclusive positive concepts occur in
    the same case, every member of that conflict group is removed rather than
    guessed between.
    """
    return _extract_concepts(text, require_grounded_quote=True)


def extract_concept_ids(text: str) -> set[str]:
    """Return query-side canonical identifiers independent of quote length."""
    return set(_extract_concepts(text, require_grounded_quote=False))


def extract_concept_text(text: str) -> dict[str, str]:
    """Return canonical identifiers mapped to their exact evidence quotes."""
    return {
        concept_id: record["quote"]
        for concept_id, record in extract_observable_concepts(text).items()
    }


def extract_concept_texts(text: str) -> dict[str, str]:
    """Plural alias retained for callers that treat the result as a mapping."""
    return extract_concept_text(text)


def validate_observable_concepts(
    text: str,
    concepts: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, tuple[str, ...]]:
    """Recompute a case extraction and validate every exact quote and span."""
    issues: list[str] = []
    expected = extract_observable_concepts(text)
    if dict(concepts) != expected:
        issues.append("extraction_recomputation_mismatch")
    source = str(text or "")
    for concept_id, record in concepts.items():
        if concept_id not in CONCEPT_LEXICON:
            issues.append(f"unknown_concept:{concept_id}")
            continue
        try:
            start = int(record.get("quote_start", -1))
            end = int(record.get("quote_end", -1))
            quote = str(record.get("quote", ""))
        except (TypeError, ValueError):
            issues.append(f"invalid_quote_span:{concept_id}")
            continue
        if start < 0 or end <= start or source[start:end] != quote:
            issues.append(f"unverifiable_quote:{concept_id}")
        if not (2 <= _quote_word_count(quote) <= 24):
            issues.append(f"invalid_quote_word_count:{concept_id}")
        expected_sha = hashlib.sha256(quote.encode("utf-8")).hexdigest()
        if record.get("quote_sha256") != expected_sha:
            issues.append(f"quote_hash_mismatch:{concept_id}")
    return not issues, tuple(sorted(set(issues)))


def _canonical_role(value: Any) -> str:
    role = re.sub(r"[^a-z0-9]+", "_", _normalized_text(str(value or ""))).strip("_")
    if not role:
        return "unspecified"
    if "hard" in role:
        return HARD_SELECTION_ROLE
    if role.startswith("representative"):
        return "representative"
    if "diverse" in role:
        return "diverse"
    return role


def _case_findings(row: Mapping[str, Any]) -> str:
    for key in ("findings", "visual_findings", "finding", "text"):
        if key in row:
            return str(row.get(key) or "")
    return ""


def _prepare_cases(cases: Iterable[Mapping[str, Any]], side: str) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in cases:
        if not isinstance(raw, Mapping):
            raise TypeError(f"{side} case must be a mapping")
        case_id = str(raw.get("case_id", "")).strip()
        if not case_id:
            raise ValueError(f"{side} case is missing case_id")
        if case_id in seen:
            raise ValueError(f"duplicate {side} case_id: {case_id}")
        seen.add(case_id)
        findings = _case_findings(raw)
        prepared.append(
            {
                "case_id": case_id,
                "findings": findings,
                "selection_role": _canonical_role(raw.get("selection_role")),
                "concepts": extract_observable_concepts(findings),
            }
        )
    return sorted(prepared, key=lambda row: row["case_id"])


def _resolve_minimum_support(minimum_support: int, min_support: int | None) -> int:
    if min_support is not None:
        if minimum_support != 3 and int(minimum_support) != int(min_support):
            raise ValueError("minimum_support and min_support disagree")
        minimum_support = min_support
    if isinstance(minimum_support, bool):
        raise ValueError("minimum_support must be an integer of at least 1")
    try:
        value = int(minimum_support)
    except (TypeError, ValueError) as error:
        raise ValueError("minimum_support must be an integer of at least 1") from error
    if value < 1 or value != minimum_support:
        raise ValueError("minimum_support must be an integer of at least 1")
    return value


def _eligible_roles(rows: Sequence[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                row["selection_role"]
                for row in rows
                if row["selection_role"] != "unspecified"
            }
        )
    )


def _roles_support_candidate(rows: Sequence[dict[str, Any]]) -> bool:
    roles = _eligible_roles(rows)
    return len(roles) >= 2 and any(role != HARD_SELECTION_ROLE for role in roles)


def _evidence(rows: Sequence[dict[str, Any]], concept_id: str, side: str) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for row in rows:
        match = row["concepts"].get(concept_id)
        if match is None:
            continue
        evidence.append(
            {
                "case_id": row["case_id"],
                "side": side,
                "role": row["selection_role"],
                "quote": match["quote"],
                "quote_start": match["quote_start"],
                "quote_end": match["quote_end"],
                "quote_sha256": match["quote_sha256"],
            }
        )
    return sorted(evidence, key=lambda item: (item["role"], item["case_id"], item["quote_start"]))


def _ledger_entry(
    concept_id: str,
    category: str,
    hits_a: Sequence[dict[str, Any]],
    hits_b: Sequence[dict[str, Any]],
    total_a: int,
    total_b: int,
) -> dict[str, Any]:
    count_a, count_b = len(hits_a), len(hits_b)
    rate_a = count_a / total_a if total_a else 0.0
    rate_b = count_b / total_b if total_b else 0.0
    if category == "candidate_a":
        own_side = "a"
        own_hits, opposite_hits = hits_a, hits_b
        own_count, opposite_count = count_a, count_b
        own_rate, opposite_rate = rate_a, rate_b
        evidence = _evidence(hits_a, concept_id, "a")
        roles = _eligible_roles(hits_a)
        margin = rate_a - rate_b
    elif category == "candidate_b":
        own_side = "b"
        own_hits, opposite_hits = hits_b, hits_a
        own_count, opposite_count = count_b, count_a
        own_rate, opposite_rate = rate_b, rate_a
        evidence = _evidence(hits_b, concept_id, "b")
        roles = _eligible_roles(hits_b)
        margin = rate_b - rate_a
    else:
        own_side = "shared"
        own_hits, opposite_hits = hits_a, hits_b
        own_count, opposite_count = count_a, count_b
        own_rate, opposite_rate = rate_a, rate_b
        evidence = _evidence(hits_a, concept_id, "a") + _evidence(hits_b, concept_id, "b")
        evidence.sort(key=lambda item: (item["side"], item["role"], item["case_id"]))
        roles = _eligible_roles(tuple(hits_a) + tuple(hits_b))
        margin = abs(rate_a - rate_b)

    entry: dict[str, Any] = {
        "concept_id": concept_id,
        "display": concept_display(concept_id),
        "own_side": own_side,
        "own_count": own_count,
        "opposite_count": opposite_count,
        "own_rate": round(own_rate, 8),
        "opposite_rate": round(opposite_rate, 8),
        "prevalence_margin": round(margin, 8),
        "count_a": count_a,
        "count_b": count_b,
        "rate_a": round(rate_a, 8),
        "rate_b": round(rate_b, 8),
        "case_ids_a": [row["case_id"] for row in hits_a],
        "case_ids_b": [row["case_id"] for row in hits_b],
        "own_case_ids": [row["case_id"] for row in own_hits],
        "opposite_case_ids": [row["case_id"] for row in opposite_hits],
        "roles": list(roles),
        "roles_a": list(_eligible_roles(hits_a)),
        "roles_b": list(_eligible_roles(hits_b)),
        "evidence": evidence,
    }
    entry["entry_sha256"] = _canonical_hash(entry)
    return entry


def build_concept_ledger(
    cases_a: Iterable[Mapping[str, Any]],
    cases_b: Iterable[Mapping[str, Any]],
    minimum_support: int = 3,
    *,
    min_support: int | None = None,
) -> dict[str, Any]:
    """Build a deterministic prevalence-normalized contrastive concept ledger."""
    support = _resolve_minimum_support(minimum_support, min_support)
    prepared_a = _prepare_cases(cases_a, "A")
    prepared_b = _prepare_cases(cases_b, "B")
    if not prepared_a or not prepared_b:
        raise ValueError("both sides require at least one source case")

    candidate_a: list[dict[str, Any]] = []
    candidate_b: list[dict[str, Any]] = []
    nondiscriminative: list[dict[str, Any]] = []
    for concept_id in sorted(CONCEPT_LEXICON):
        hits_a = [row for row in prepared_a if concept_id in row["concepts"]]
        hits_b = [row for row in prepared_b if concept_id in row["concepts"]]
        count_a, count_b = len(hits_a), len(hits_b)
        rate_a = count_a / len(prepared_a)
        rate_b = count_b / len(prepared_b)
        margin = rate_a - rate_b
        if (
            count_a >= support
            and margin >= MINIMUM_PREVALENCE_MARGIN - 1e-12
            and _roles_support_candidate(hits_a)
        ):
            candidate_a.append(
                _ledger_entry(concept_id, "candidate_a", hits_a, hits_b, len(prepared_a), len(prepared_b))
            )
        elif (
            count_b >= support
            and -margin >= MINIMUM_PREVALENCE_MARGIN - 1e-12
            and _roles_support_candidate(hits_b)
        ):
            candidate_b.append(
                _ledger_entry(concept_id, "candidate_b", hits_a, hits_b, len(prepared_a), len(prepared_b))
            )
        elif (
            count_a >= support
            and count_b >= support
            and abs(margin) < MINIMUM_PREVALENCE_MARGIN - 1e-12
            and _roles_support_candidate(tuple(hits_a) + tuple(hits_b))
        ):
            nondiscriminative.append(
                _ledger_entry(
                    concept_id,
                    "nondiscriminative",
                    hits_a,
                    hits_b,
                    len(prepared_a),
                    len(prepared_b),
                )
            )

    def rank(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            entries,
            key=lambda entry: (-float(entry["prevalence_margin"]), entry["concept_id"]),
        )

    source_payload = {
        "a": [
            {
                "case_id": row["case_id"],
                "findings": row["findings"],
                "selection_role": row["selection_role"],
            }
            for row in prepared_a
        ],
        "b": [
            {
                "case_id": row["case_id"],
                "findings": row["findings"],
                "selection_role": row["selection_role"],
            }
            for row in prepared_b
        ],
    }
    ledger: dict[str, Any] = {
        "schema_version": "1.0",
        "protocol": CONCEPT_PROTOCOL,
        "matcher_algorithm_revision": CONCEPT_MATCHER_ALGORITHM_REVISION,
        "lexicon_sha256": CONCEPT_LEXICON_SHA256,
        "matcher_sha256": CONCEPT_MATCHER_SHA256,
        "minimum_support": support,
        "minimum_prevalence_margin": MINIMUM_PREVALENCE_MARGIN,
        "case_counts": {"a": len(prepared_a), "b": len(prepared_b)},
        "source_cases_sha256": _canonical_hash(source_payload),
        "candidate_a": rank(candidate_a),
        "candidate_b": rank(candidate_b),
        "nondiscriminative": rank(nondiscriminative),
    }
    ledger["content_sha256"] = _canonical_hash(ledger)
    return ledger


def concept_ledger_hash(ledger: Mapping[str, Any]) -> str:
    """Recompute the ledger hash without trusting its stored hash field."""
    payload = dict(ledger)
    payload.pop("content_sha256", None)
    return _canonical_hash(payload)


def recompute_concept_ledger(
    cases_a: Iterable[Mapping[str, Any]],
    cases_b: Iterable[Mapping[str, Any]],
    minimum_support: int = 3,
    *,
    min_support: int | None = None,
) -> dict[str, Any]:
    """Explicit replay alias used by full-audit code."""
    return build_concept_ledger(
        cases_a,
        cases_b,
        minimum_support=minimum_support,
        min_support=min_support,
    )


def validate_concept_ledger(
    ledger: Mapping[str, Any],
    cases_a: Iterable[Mapping[str, Any]] | None = None,
    cases_b: Iterable[Mapping[str, Any]] | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Validate hashes, structure, evidence, and optionally full source replay."""
    issues: list[str] = []
    if ledger.get("protocol") != CONCEPT_PROTOCOL:
        issues.append("protocol_mismatch")
    if ledger.get("matcher_algorithm_revision") != CONCEPT_MATCHER_ALGORITHM_REVISION:
        issues.append("matcher_algorithm_revision_mismatch")
    if ledger.get("lexicon_sha256") != CONCEPT_LEXICON_SHA256:
        issues.append("lexicon_hash_mismatch")
    if ledger.get("matcher_sha256") != CONCEPT_MATCHER_SHA256:
        issues.append("matcher_hash_mismatch")
    if ledger.get("content_sha256") != concept_ledger_hash(ledger):
        issues.append("content_hash_mismatch")

    seen: set[str] = set()
    for category in ("candidate_a", "candidate_b", "nondiscriminative"):
        entries = ledger.get(category)
        if not isinstance(entries, list):
            issues.append(f"invalid_category:{category}")
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                issues.append(f"invalid_entry:{category}")
                continue
            concept_id = str(entry.get("concept_id", ""))
            if concept_id not in CONCEPT_LEXICON:
                issues.append(f"unknown_concept:{concept_id}")
            if concept_id in seen:
                issues.append(f"duplicate_concept:{concept_id}")
            seen.add(concept_id)
            if concept_id in CONCEPT_LEXICON and entry.get("display") != concept_display(concept_id):
                issues.append(f"display_mismatch:{concept_id}")
            entry_payload = dict(entry)
            stored_entry_hash = entry_payload.pop("entry_sha256", None)
            if stored_entry_hash != _canonical_hash(entry_payload):
                issues.append(f"entry_hash_mismatch:{concept_id}")
            for evidence in entry.get("evidence", []):
                if not isinstance(evidence, Mapping):
                    issues.append(f"invalid_evidence:{concept_id}")
                    continue
                quote = str(evidence.get("quote", ""))
                if not (2 <= _quote_word_count(quote) <= 24):
                    issues.append(f"invalid_quote_word_count:{concept_id}")
                expected_sha = hashlib.sha256(quote.encode("utf-8")).hexdigest()
                if evidence.get("quote_sha256") != expected_sha:
                    issues.append(f"quote_hash_mismatch:{concept_id}")

    if (cases_a is None) != (cases_b is None):
        issues.append("incomplete_recomputation_sources")
    elif cases_a is not None and cases_b is not None:
        try:
            rebuilt = build_concept_ledger(
                cases_a,
                cases_b,
                minimum_support=int(ledger.get("minimum_support", 3)),
            )
        except (TypeError, ValueError) as error:
            issues.append(f"recomputation_error:{type(error).__name__}")
        else:
            if rebuilt != dict(ledger):
                issues.append("ledger_recomputation_mismatch")
    return not issues, tuple(sorted(set(issues)))


__all__ = [
    "CONCEPT_CONFIG_SHA256",
    "CONCEPT_LEXICON",
    "CONCEPT_LEXICON_SHA256",
    "CONCEPT_MATCHER_ALGORITHM_REVISION",
    "CONCEPT_MATCHER_SHA256",
    "CONCEPT_PROTOCOL",
    "HARD_SELECTION_ROLE",
    "MINIMUM_PREVALENCE_MARGIN",
    "build_concept_ledger",
    "concept_family",
    "concept_display",
    "concept_ledger_hash",
    "extract_concept_ids",
    "extract_concept_text",
    "extract_concept_texts",
    "extract_observable_concepts",
    "recompute_concept_ledger",
    "validate_concept_ledger",
    "validate_observable_concepts",
]
