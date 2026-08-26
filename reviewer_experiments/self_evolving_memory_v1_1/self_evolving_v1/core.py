#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


RULE_SECTION_RE = re.compile(
    r"(?P<head>\[3\. Evolved Diagnostic Guidelines from Verified Training Cases\]:\n)"
    r".*?"
    r"(?P<tail>\n\n\[4\. Precedent Cases from the Same Frozen Training Checkpoint\]:)",
    flags=re.S,
)
PROBABILITY_RE = re.compile(r"^\s*-\s*(.+?):\s*([-+]?\d+(?:\.\d+)?)%\s*$", flags=re.M)
VISUAL_FINDINGS_RE = re.compile(
    r"\[1\. Frozen Current Visual Findings\]:\s*(?P<findings>.*?)\s*"
    r"\[2\. Frozen Model Prediction Probabilities\]:",
    flags=re.S,
)
FEATURE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "can",
    "for",
    "from",
    "in",
    "is",
    "lesion",
    "lesions",
    "may",
    "no",
    "not",
    "of",
    "or",
    "skin",
    "the",
    "to",
    "with",
    "without",
}


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def label_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def rule_key(dataset: str, disease_a: str, disease_b: str) -> str:
    labels = sorted((str(disease_a).strip(), str(disease_b).strip()), key=label_key)
    return f"{dataset}::{label_key(labels[0])}::{label_key(labels[1])}"


def stable_fraction(value: str, seed: int = 20260825) -> float:
    digest = hashlib.sha256(f"{seed}::{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_role(sample_id: str, seed: int = 20260825) -> str:
    value = stable_fraction(sample_id, seed)
    if value < 0.25:
        return "rule_acceptance"
    if value < 0.50:
        return "gate_development"
    return "heldout_evaluation"


def probabilities_from_prompt(prompt: str) -> list[tuple[str, float]]:
    try:
        block = prompt.split("[2. Frozen Model Prediction Probabilities]:", 1)[1].split("[3.", 1)[0]
    except IndexError:
        return []
    return [(match.group(1).strip(), float(match.group(2)) / 100.0) for match in PROBABILITY_RE.finditer(block)]


def visual_findings_from_prompt(prompt: str) -> str:
    match = VISUAL_FINDINGS_RE.search(str(prompt or ""))
    return match.group("findings").strip() if match else ""


def normalized_phrase(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def phrase_in_text(phrase: str, text: str) -> bool:
    needle = normalized_phrase(phrase)
    return bool(needle) and needle in normalized_phrase(text)


def feature_tokens(value: str) -> set[str]:
    return {
        token
        for token in normalized_phrase(value).split()
        if token not in FEATURE_STOPWORDS and len(token) > 2
    }


def features_semantically_overlap(left: str, right: str) -> bool:
    """Conservative lexical-semantic overlap check for short morphology rules."""
    left_tokens = feature_tokens(left)
    right_tokens = feature_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    shared = left_tokens & right_tokens
    union = left_tokens | right_tokens
    containment = len(shared) / min(len(left_tokens), len(right_tokens))
    return len(shared) >= 2 and (len(shared) / len(union) >= 0.45 or containment >= 0.67)


def parse_json_object(text: str) -> dict[str, Any] | None:
    value = str(text or "").strip()
    value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.I)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(value[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def parse_primary_diagnosis(text: str, allowed_labels: Iterable[str]) -> str:
    parsed = parse_json_object(text)
    raw = str(parsed.get("PrimaryDiagnosis") or "") if parsed else ""
    exact = {label_key(label): label for label in allowed_labels}
    return exact.get(label_key(raw), "")


def normalized_entropy(probabilities: list[float]) -> float:
    values = [max(float(value), 1e-12) for value in probabilities]
    total = sum(values)
    if not values or total <= 0:
        return 0.0
    values = [value / total for value in values]
    entropy = -sum(value * math.log(value) for value in values)
    return entropy / math.log(len(values)) if len(values) > 1 else 0.0


@dataclass(frozen=True)
class StructuredRule:
    rule_id: str
    dataset: str
    disease_a: str
    disease_b: str
    supports_a: tuple[str, ...]
    supports_b: tuple[str, ...]
    nondiscriminative: tuple[str, ...]
    required_context: tuple[str, ...]
    limitations: tuple[str, ...]
    source_case_ids_a: tuple[str, ...]
    source_case_ids_b: tuple[str, ...]
    source_case_count: int
    trigger_terms_a: tuple[str, ...] = ()
    trigger_terms_b: tuple[str, ...] = ()
    required_context_terms: tuple[str, ...] = ()
    abstain_terms: tuple[str, ...] = ()
    evidence_a: tuple[dict[str, Any], ...] = ()
    evidence_b: tuple[dict[str, Any], ...] = ()
    source_selection_a: tuple[dict[str, Any], ...] = ()
    source_selection_b: tuple[dict[str, Any], ...] = ()
    schema_version: str = "v1"
    status: str = "candidate"
    validation_exposures: int = 0
    validation_changes: int = 0
    validation_gains: int = 0
    validation_regressions: int = 0
    validation_effect: float = 0.0
    reliability: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key, item in list(value.items()):
            if isinstance(item, tuple):
                value[key] = list(item)
        value["rule_key"] = rule_key(self.dataset, self.disease_a, self.disease_b)
        return value


@dataclass(frozen=True)
class GatePolicy:
    policy_id: str
    allow_when_c_not_top1: bool
    top1_probability_cap: float
    top1_margin_cap: float
    minimum_rule_reliability: float
    source_split: str = "gate_development"

    def allow(
        self,
        *,
        c_prediction: str,
        evolved_prediction: str,
        candidate_probabilities: list[tuple[str, float]],
        rule_reliability: float,
    ) -> tuple[bool, str]:
        if label_key(c_prediction) == label_key(evolved_prediction):
            return False, "no_change"
        if rule_reliability < self.minimum_rule_reliability:
            return False, "rule_below_reliability"
        if len(candidate_probabilities) < 2:
            return False, "missing_candidate_probabilities"
        top1, probability = candidate_probabilities[0]
        margin = probability - candidate_probabilities[1][1]
        if label_key(c_prediction) != label_key(top1) and self.allow_when_c_not_top1:
            return True, "c_disagrees_with_frozen_top1"
        if probability <= self.top1_probability_cap and margin <= self.top1_margin_cap:
            return True, "low_confidence_or_small_margin"
        return False, "protected_frozen_top1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryRelease:
    release_id: str
    parent_release_id: str | None
    source_stream_hashes: dict[str, str]
    rules: tuple[StructuredRule, ...]
    status: str
    protocol: str

    def payload(self) -> dict[str, Any]:
        value = {
            "release_id": self.release_id,
            "parent_release_id": self.parent_release_id,
            "source_stream_hashes": self.source_stream_hashes,
            "rules": [rule.to_dict() for rule in self.rules],
            "status": self.status,
            "protocol": self.protocol,
        }
        value["content_sha256"] = canonical_hash(value)
        return value


def render_rule(rule: dict[str, Any]) -> str:
    def joined(key: str) -> str:
        return "; ".join(str(item).strip() for item in rule.get(key, []) if str(item).strip()) or "None validated."

    return (
        f"[Validated Contrastive Rule {rule['rule_id']} | release={rule.get('release_id', 'candidate')} | "
        f"reliability={float(rule.get('reliability', 0.5)):.2f}]\n"
        f"- Distinguish: {rule['disease_a']} versus {rule['disease_b']}\n"
        f"- Supports {rule['disease_a']}: {joined('supports_a')}\n"
        f"- Supports {rule['disease_b']}: {joined('supports_b')}\n"
        f"- Non-discriminative (must not drive an override alone): {joined('nondiscriminative')}\n"
        f"- Executable triggers for {rule['disease_a']}: {joined('trigger_terms_a')}\n"
        f"- Executable triggers for {rule['disease_b']}: {joined('trigger_terms_b')}\n"
        f"- Required context terms: {joined('required_context_terms')}\n"
        f"- Executable abstention terms: {joined('abstain_terms')}\n"
        f"- Required context: {joined('required_context')}\n"
        f"- Limitations: {joined('limitations')}\n"
        f"- Evidence: {int(rule.get('source_case_count', 0))} immutable training cases; "
        f"validation gains={int(rule.get('validation_gains', 0))}, "
        f"regressions={int(rule.get('validation_regressions', 0))}."
    )


def inject_rule(prompt: str, rule: dict[str, Any] | None, baseline_prediction: str | None = None) -> str:
    block = render_rule(rule) if rule else "No active validated contrastive rule is available for this candidate pair."
    baseline = str(baseline_prediction or "").strip() or "not supplied"
    pair_contract = ""
    if rule:
        pair_contract = (
            f"- This memory may compare only {rule['disease_a']} and {rule['disease_b']}.\n"
            "- It must not justify changing to any third diagnosis.\n"
        )
    replacement = (
        "\\g<head>"
        + block
        + "\n\n[Memory Safety Constraints]:\n"
        + f"- Frozen C100 final diagnosis: {baseline}.\n"
        + pair_contract
        + "- Treat model probabilities as numerical priors.\n"
        + "- A non-discriminative feature cannot justify changing the leading diagnosis.\n"
        + "- If the supplied rule lacks its required context, retain the better-supported prior diagnosis.\n"
        + "- Any override must cite at least two executable trigger terms found in the frozen findings.\n"
        + "- If these conditions are not satisfied, retain the frozen C100 final diagnosis."
        + "\\g<tail>"
    )
    updated, count = RULE_SECTION_RE.subn(replacement, prompt, count=1)
    if count != 1:
        raise ValueError("Unable to replace evolved-guideline section")
    return updated


def rule_applicability(
    prompt: str,
    rule: dict[str, Any],
    *,
    minimum_trigger_matches: int = 2,
) -> tuple[bool, str, dict[str, int]]:
    """Execute query-side trigger, context, and abstention checks before injection."""
    findings = visual_findings_from_prompt(prompt)
    if not findings:
        return False, "missing_visual_findings", {"a": 0, "b": 0}
    missing_context = [
        term for term in rule.get("required_context_terms", []) if not phrase_in_text(str(term), findings)
    ]
    if missing_context:
        return False, "required_context_missing", {"a": 0, "b": 0}
    if any(phrase_in_text(str(term), findings) for term in rule.get("abstain_terms", [])):
        return False, "abstention_term_present", {"a": 0, "b": 0}
    matches_a = sum(phrase_in_text(str(term), findings) for term in rule.get("trigger_terms_a", []))
    matches_b = sum(phrase_in_text(str(term), findings) for term in rule.get("trigger_terms_b", []))
    counts = {"a": int(matches_a), "b": int(matches_b)}
    if max(matches_a, matches_b) < minimum_trigger_matches:
        return False, "insufficient_trigger_match", counts
    if matches_a == matches_b:
        return False, "ambiguous_trigger_match", counts
    side = "a" if matches_a > matches_b else "b"
    return True, f"trigger_supports_{side}", counts


def strict_rule_from_response(record: dict[str, Any], response: str) -> tuple[StructuredRule | None, str | None]:
    try:
        parsed = json.loads(str(response or "").strip())
    except json.JSONDecodeError:
        parsed = None
    if not isinstance(parsed, dict):
        return None, "invalid_json"
    if record.get("schema_version") == "v1.1" and parsed.get("publishable") is False:
        return None, "evidence_abstained"
    expected_a = str(record["disease_a"])
    expected_b = str(record["disease_b"])
    if label_key(parsed.get("disease_a", "")) != label_key(expected_a):
        return None, "disease_a_mismatch"
    if label_key(parsed.get("disease_b", "")) != label_key(expected_b):
        return None, "disease_b_mismatch"

    def strings(key: str, minimum: int = 1, maximum: int = 8) -> tuple[str, ...] | None:
        value = parsed.get(key)
        if not isinstance(value, list):
            return None
        cleaned = tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
        if len(cleaned) < minimum or len(cleaned) > maximum:
            return None
        return cleaned

    supports_a = strings("supports_a")
    supports_b = strings("supports_b")
    nondiscriminative = strings("nondiscriminative")
    required_context = strings("required_context", minimum=0)
    limitations = strings("limitations", minimum=1)
    if None in (supports_a, supports_b, nondiscriminative, required_context, limitations):
        return None, "invalid_rule_fields"
    feature_sets = [
        {label_key(item) for item in supports_a or ()},
        {label_key(item) for item in supports_b or ()},
        {label_key(item) for item in nondiscriminative or ()},
    ]
    if any(feature_sets[left] & feature_sets[right] for left in range(3) for right in range(left + 1, 3)):
        return None, "overlapping_rule_fields"
    if record.get("schema_version") == "v1.1":
        feature_lists = (supports_a or (), supports_b or (), nondiscriminative or ())
        if any(
            features_semantically_overlap(left, right)
            for left_index, left_values in enumerate(feature_lists)
            for right_values in feature_lists[left_index + 1 :]
            for left in left_values
            for right in right_values
        ):
            return None, "semantic_feature_overlap"
    allowed_a = set(record["source_case_ids_a"])
    allowed_b = set(record["source_case_ids_b"])

    def grounded_ids(key: str, allowed: set[str]) -> tuple[str, ...] | None:
        value = parsed.get(key)
        if not isinstance(value, list):
            return None
        ids = tuple(str(item).strip() for item in value if str(item).strip())
        if any(item not in allowed for item in ids):
            return None
        return tuple(dict.fromkeys(ids))

    ids_a = grounded_ids("source_case_ids_a", allowed_a)
    ids_b = grounded_ids("source_case_ids_b", allowed_b)
    if ids_a is None or ids_b is None:
        return None, "source_id_mismatch"
    if len(set(ids_a)) < 2 or len(set(ids_b)) < 2:
        return None, "missing_grounded_source_ids"
    if record.get("schema_version") == "v1.1":
        if set(ids_a) != allowed_a or set(ids_b) != allowed_b:
            return None, "incomplete_source_coverage"

        trigger_terms_a = strings("trigger_terms_a", minimum=2, maximum=8)
        trigger_terms_b = strings("trigger_terms_b", minimum=2, maximum=8)
        required_context_terms = strings("required_context_terms", minimum=0, maximum=6)
        abstain_terms = strings("abstain_terms", minimum=0, maximum=6)
        if None in (trigger_terms_a, trigger_terms_b, required_context_terms, abstain_terms):
            return None, "invalid_executable_terms"

        source_cases_a = {str(row["case_id"]): str(row["findings"]) for row in record.get("source_cases_a", [])}
        source_cases_b = {str(row["case_id"]): str(row["findings"]) for row in record.get("source_cases_b", [])}
        minimum_support = int(record.get("minimum_feature_support", 3))

        def grounded_evidence(
            key: str,
            features: tuple[str, ...],
            sources: dict[str, str],
        ) -> tuple[dict[str, Any], ...] | None:
            value = parsed.get(key)
            if not isinstance(value, list) or len(value) != len(features):
                return None
            expected = {label_key(feature): feature for feature in features}
            observed: dict[str, dict[str, Any]] = {}
            for item in value:
                if not isinstance(item, dict):
                    return None
                normalized_feature = label_key(item.get("feature", ""))
                if normalized_feature not in expected or normalized_feature in observed:
                    return None
                cases = item.get("cases")
                if not isinstance(cases, list):
                    return None
                validated_cases = []
                seen_ids = set()
                for citation in cases:
                    if not isinstance(citation, dict):
                        return None
                    case_id = str(citation.get("case_id", "")).strip()
                    quote = str(citation.get("quote", "")).strip()
                    quote_words = normalized_phrase(quote).split()
                    if case_id not in sources or case_id in seen_ids or not (2 <= len(quote_words) <= 24):
                        return None
                    if not phrase_in_text(quote, sources[case_id]):
                        return None
                    seen_ids.add(case_id)
                    validated_cases.append({"case_id": case_id, "quote": quote})
                if len(validated_cases) < minimum_support:
                    return None
                observed[normalized_feature] = {
                    "feature": expected[normalized_feature],
                    "cases": validated_cases,
                }
            if set(observed) != set(expected):
                return None
            return tuple(observed[label_key(feature)] for feature in features)

        evidence_a = grounded_evidence("evidence_a", supports_a or (), source_cases_a)
        evidence_b = grounded_evidence("evidence_b", supports_b or (), source_cases_b)
        if evidence_a is None or evidence_b is None:
            return None, "invalid_feature_evidence"

        def triggers_grounded(
            terms: tuple[str, ...],
            own_sources: dict[str, str],
            opposite_sources: dict[str, str],
        ) -> bool:
            for term in terms:
                own_count = sum(phrase_in_text(term, findings) for findings in own_sources.values())
                opposite_count = sum(phrase_in_text(term, findings) for findings in opposite_sources.values())
                if own_count < minimum_support or own_count < opposite_count + 2:
                    return False
            return True

        if not triggers_grounded(trigger_terms_a or (), source_cases_a, source_cases_b):
            return None, "ungrounded_trigger_terms_a"
        if not triggers_grounded(trigger_terms_b or (), source_cases_b, source_cases_a):
            return None, "ungrounded_trigger_terms_b"
    else:
        trigger_terms_a = trigger_terms_b = required_context_terms = abstain_terms = ()
        evidence_a = evidence_b = ()
    return (
        StructuredRule(
            rule_id=str(record["rule_id"]),
            dataset=str(record["dataset"]),
            disease_a=expected_a,
            disease_b=expected_b,
            supports_a=supports_a or (),
            supports_b=supports_b or (),
            nondiscriminative=nondiscriminative or (),
            required_context=required_context or (),
            limitations=limitations or (),
            source_case_ids_a=ids_a,
            source_case_ids_b=ids_b,
            source_case_count=len(set(ids_a + ids_b)),
            trigger_terms_a=trigger_terms_a or (),
            trigger_terms_b=trigger_terms_b or (),
            required_context_terms=required_context_terms or (),
            abstain_terms=abstain_terms or (),
            evidence_a=evidence_a or (),
            evidence_b=evidence_b or (),
            source_selection_a=tuple(record.get("source_selection_a", [])),
            source_selection_b=tuple(record.get("source_selection_b", [])),
            schema_version=str(record.get("schema_version", "v1")),
        ),
        None,
    )


def normalize_overlapping_features(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Move exact normalized cross-list overlaps into nondiscriminative only."""
    keys = ("supports_a", "supports_b", "nondiscriminative")
    occurrences: dict[str, list[tuple[str, str]]] = {}
    for key in keys:
        values = payload.get(key)
        if not isinstance(values, list):
            return payload, []
        for item in values:
            text = str(item).strip()
            if text:
                occurrences.setdefault(label_key(text), []).append((key, text))
    overlaps = {
        normalized
        for normalized, rows in occurrences.items()
        if len({key for key, _ in rows}) > 1
    }
    if not overlaps:
        return payload, []
    repaired = dict(payload)
    for key in keys:
        repaired[key] = [
            str(item).strip()
            for item in payload[key]
            if str(item).strip() and label_key(str(item)) not in overlaps
        ]
    for normalized in sorted(overlaps):
        rows = occurrences[normalized]
        preferred = next((text for key, text in rows if key == "nondiscriminative"), rows[0][1])
        repaired["nondiscriminative"].append(preferred)
    return repaired, sorted(overlaps)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows
