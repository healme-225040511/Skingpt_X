# Validation record — 2026-09-29

## Completed checks

- Python source compilation and command-line argument entry points passed.
- CPU unit tests passed: parser exact/fenced/substring/invalid cases; prompt, label and image tamper detection; synthetic paired scoring; incomplete-cohort rejection (3 test methods).
- Historical parser replay matched **all 9,873 archived predictions**, including validity flags and PanDerm fallback results. Invalid outputs: Curated-A 11, Curated-B 11, Curated-C 15 (37 total).
- Independent archive scan passed for all **44,912 image entries**: Curated-A 10,855; Curated-B 17,250; Curated-C 16,807. Each entry is a regular file whose SHA-256 equals its manifest record. These are per-cohort entries, not unique images across the three cohorts.
- Export asserts exact refined class and split counts, matches every job to its original manifest and historical candidates, preserves prompt bytes, and checks image SHA-256 identity. It rejects byte-identical train/test overlap within each cohort.
- Only the dedicated experiment folder is added to the Git repository. No credentials, model weights or image archives are committed to Git.
- All three Hugging Face folders were published and independently checked against local files: exact file sets, sizes, LFS SHA-256 values and Git blob hashes for non-LFS files all matched. Pinned dataset commit: `0590d59f07a94009f318fbdf51a86ad00110e176`.

## Not validated / not claimed

- No model weights were downloaded and no inference was launched for this delivery.
- The vLLM/FP8 GPU path has not been exercised on the destination hardware; an initial destination smoke test is still required before the full run.
- The archived historical run used Transformers; the new FP8 run uses vLLM. This is not an inference-engine-matched comparison unless the optional fixed-evidence 8B vLLM reference is also run.
- A MoE FP8 checkpoint differs from a dense BF16 checkpoint in architecture and precision as well as total parameter count.
- Refined labels are engineering candidates, not a completed dermatologist-reviewed ontology. Exact-byte deduplication does not establish patient-level, perceptual or pretraining independence.
- Cross-dataset source overlap prevents treating the three cohorts as independent observations in a pooled significance test.

The data commit is pinned in `release.json`. Dataset-side `audit.json` files contain file checksums and counts; `verify_archive.py` verifies the bytes of every archived image against its manifest.

CPU scoring tests ran on the source server with Python 3.10.19, NumPy 1.26.4, SciPy 1.15.2 and scikit-learn 1.6.1. This does not constitute a tested GPU runtime lock; the destination GPU environment is installed and recorded separately as described in the README.
