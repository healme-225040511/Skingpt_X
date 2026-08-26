# SkinGPT-X self-evolving memory v1.1

This package is the isolated, auditable implementation used to test whether
self-evolving memory improves SkinGPT-X recognition. It does not mutate the
legacy Neo4j graph or overwrite the original v1 experiment.

## What changed in v1.1

- Candidate disease pairs are frozen from the previous release, with obvious
  synonym and parent/child ontology conflicts removed.
- Each side of a pair uses balanced representative, diverse, and
  quality-screened positive-margin embedding-hard cases. Exact/near duplicates,
  repeated findings, low-quality findings, and cross-label near duplicates are
  excluded.
- Generated rules must be strict JSON, quote immutable source evidence, use all
  supplied source IDs, keep the three feature fields mutually exclusive, and
  provide literal query-side triggers. Unsupported pairs may abstain.
- A rule can act only when its disease pair matches the frozen top two, the
  frozen C100 diagnosis is inside that pair, and query-side trigger/context
  checks pass. Third-class or invalid generations fall back to C100.
- Rule acceptance, gate development, and evaluation use disjoint deterministic
  partitions. Acceptance and gate thresholds are fixed before evaluation.
- Neutral and within-pair shuffled controls preserve the exact case, prompt,
  inference budget, and rule pair while replacing only the learned rule block.

## Frozen lifecycle

1. `initialize`: build generator-only source sets and a frozen C100 snapshot.
2. `audit_initialization`: strictly parse every JSON record, verify hashes and
   source coverage, and exactly recompute every prompt and cross-label similarity.
3. Generate rule JSON with the frozen Qwen3-VL-8B-Instruct checkpoint.
4. `finalize-rules`: accept only fully grounded, strictly parsed candidates.
5. `build-cohort` and `validate-rules`: validate each rule on `rule_acceptance`.
6. Build/run paired controls on the same applicable cases.
7. `build-cohort` and `tune-gate`: select among three pre-registered gate policies
   on `gate_development`; otherwise roll back to C100.
8. `finalize-evaluation`: evaluate the frozen release/gate once on
   `heldout_evaluation`, with Top-1-matched and beyond-Top-1 decompositions.

The current 5,533-case split was inspected during earlier mechanism analysis,
so its final evaluation is explicitly exploratory reuse, not a fresh
confirmatory test. Only beyond-Top-1 gains can support improved recognition;
restoring the frozen Top-1 supports reduced retrieval-induced error only.

## Safety defaults

- Acceptance per rule: at least 20 exposures, 2 changes, 5 gains, net gain 4,
  no more than 1 regression, no high-confidence regression, no invalid or
  third-class output, and one-sided exact `p < 0.10`.
- Gate: at least 100 applicable cases, 30 overrides, 10 discordances, net gain
  3, 3 beyond-Top-1 gains, one-sided exact `p < 0.10`, no high-confidence
  regression, no adequately represented dataset with negative net change, at
  least 80% of raw gains retained, and at least 50% of raw regressions removed.
- If no rule or gate passes, the active pointer rolls back to immutable C100.

## Reproduction

```bash
python -m pip install -r requirements.txt
PYTHONPATH=. python -m pytest -q
PYTHONPATH=. python -m self_evolving_v1.pipeline --help
PYTHONPATH=. python -m self_evolving_v1.audit_initialization --help
PYTHONPATH=. python -m self_evolving_v1.controls --help
```

`self_evolving_v1.run_inference` is append-only and resumable by `global_index`.
Run one manifest shard per GPU. Keep model ID, checkpoint files, input manifests,
outputs, release pointers, and all SHA-256 audit artifacts frozen together.

`RESULTS_20260825.md` records the historical v1 result and must not be presented
as a v1.1 result.
