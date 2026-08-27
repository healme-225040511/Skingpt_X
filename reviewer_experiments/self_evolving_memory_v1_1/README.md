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
- A fixed disease-independent observable-morphology compiler normalizes a
  versioned synonym lexicon with token-boundary, negation, and mutual-exclusion
  handling. It builds prevalence-normalized candidates only when at least three
  cases and two source-selection roles support them; embedding-hard cases can
  never establish a candidate alone.
- The generator may select only canonical IDs from the frozen ledger. The
  parser recomputes the complete ledger and reconstructs all 2--24-word exact
  source quotes, spans, counts, triggers, and evidence hashes. Unsupported pairs
  may abstain; invented concepts, altered evidence, and unexecutable free-text
  context requirements fail closed.
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
   source coverage, and exactly recompute every prompt, concept ledger, exact
   evidence span, and cross-label similarity.
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
PYTHONPATH=. python -m self_evolving_v1.result_audit --help
PYTHONPATH=. python -m self_evolving_v1.control_analysis --help
PYTHONPATH=. python -m self_evolving_v1.smoke --help
```

Before a full inference run, `self_evolving_v1.smoke rules` selects two frozen
rule proposals per dataset by a deterministic stable hash. After
`build-cohort`, `self_evolving_v1.smoke diagnosis` selects at most 96 cases by
dataset and rule round-robin without using the ground-truth label. Both commands
write two unchanged-record shards and a `smoke_manifest_audit.json` containing
all input/output file, content, record, coverage, and self-audit hashes.

`self_evolving_v1.run_inference` is append-only and resumable by `global_index`.
Run one manifest shard per GPU. Keep model ID, checkpoint files, input manifests,
outputs, release pointers, and all SHA-256 audit artifacts frozen together.

The frozen canonical matcher protocol is
`canonical_observable_concepts_v1`; the current implementation revision is
`canonical_matcher_r2_20260826`. Its lexicon and matcher hashes are embedded
in every proposal, ledger, initialization audit, parsed rule, and formal bundle.
Changing any vocabulary, synonym, negation, prevalence, role, or evidence rule
therefore creates a new protocol revision rather than silently changing an
existing result.

`RESULTS_20260825.md` records the historical v1 result and must not be presented
as a v1.1 result.

## Utility-aware v2 experiment

`self_evolving_v1.v2` is an additive experiment over the immutable V1.1 rule
snapshot.  It does not overwrite V1/V1.1 releases.  V2 adds item-level beta
utility, staging/active/quarantine lifecycle states, top-k rule retrieval,
conflict abstention, and a pre-registered controller selected on the disjoint
`gate_development` split.  The controller supports direct structured-memory
recommendations and conservative Final-Reviewer consensus.  Held-out labels
are read only by `finalize-evaluation`.

```bash
PYTHONPATH=. python -m self_evolving_v1.v2 close-invalid-results --help
PYTHONPATH=. python -m self_evolving_v1.v2 compile-ledger-release --help
PYTHONPATH=. python -m self_evolving_v1.v2 compile-failure-memory --help
PYTHONPATH=. python -m self_evolving_v1.v2 build-cohort --help
PYTHONPATH=. python -m self_evolving_v1.v2 govern --help
PYTHONPATH=. python -m self_evolving_v1.v2 tune-controller --help
PYTHONPATH=. python -m self_evolving_v1.v2 finalize-evaluation --help
PYTHONPATH=. python -m self_evolving_v1.case_depth --help
```

`close-invalid-results` is fail-closed: it converts only residual malformed
generator outputs to schema-valid abstentions and never invents or repairs a
clinical rule. `case_depth` learns whether C0/C25/C50/C75/C100 is safer for a
dataset/predicted-class group on rule acceptance, governs that choice on the
development gate, and applies only active choices during evaluation.

## Clinical memory router v3 pilot

`self_evolving_v1.clinical_router` tests an additive two-dataset pilot without
SkinHandbook. It learns item utility and query routing on `rule_acceptance`,
compares relevance-only, utility-reranked, and contrastive label-balanced case
memory on `gate_development`, and permits only a gate-selected arm to proceed
to `heldout_evaluation`. It never treats similarity or memory popularity as
diagnostic proof and automatically falls back to the frozen S0 result when no
arm has positive pooled net benefit with nonnegative benefit in each dataset.

```bash
PYTHONPATH=. python -m self_evolving_v1.clinical_router fit --help
PYTHONPATH=. python -m self_evolving_v1.clinical_router build-manifest --help
PYTHONPATH=. python -m self_evolving_v1.clinical_router evaluate --help
```
