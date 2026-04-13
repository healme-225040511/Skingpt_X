# SkinGPT-X

**A Self-Evolving Collaborative Multi-Agent System for Transparent and Trustworthy Dermatological Diagnosis**

*Zhangtianyi Chen†, Yuhao Shen†, Florensia Widjaja†, Yan Xu, Liyuan Sun, Zijian Wang, Hongyi Chen, Wufei Dai, Ziwen Wang, Xinyuan Zhang, Juexiao Zhou\**

*† Equal contribution · \* Corresponding author: juexiao.zhou@gmail.com*

*School of Data Science, The Chinese University of Hong Kong, Shenzhen*

[![arXiv](https://img.shields.io/badge/arXiv-Preprint-b31b1b)](https://arxiv.org/abs/2304.10674)
[![GitHub](https://img.shields.io/badge/Code-GitHub-181717)](https://github.com/healme-225040511/Skingpt_X)
[![License](https://img.shields.io/badge/License-Non--commercial-blue)](#license)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-ee4c2c)](https://pytorch.org)

---

## Overview

SkinGPT-X addresses two fundamental limitations of standalone LLMs in clinical dermatology: (1) **bottlenecks in high-cardinality and rare disease tasks** — static fine-tuned models fail to generalize across vast, fine-grained pathology distributions; (2) **insufficient traceability** — monolithic LLMs lack the diagnostic evidence chain required for clinical accountability.

By simulating the cognitive workflow of expert dermatologists, SkinGPT-X combines visual reasoning, textbook knowledge retrieval, and a **self-evolving agent memory (EvoDerma-Mem)** that continuously refines diagnostic guidelines through accumulated clinical experience — without parameter retraining.

![Fig1](5_Figure_file/Fig1.pdf)

*Figure 1. (a) Clinicians formulate diagnosis reports by integrating three pillars: personal clinical experience, standard medical literature, and recall of similar historical cases. (b) SkinGPT-X architecture: the Vision Agent extracts visual findings, the Pre-Diagnosis Agent generates hypotheses, and the RAG module retrieves handbook knowledge. The Case-Review Agent synthesizes all evidence with EvoDerma-Mem (top-5 similar cases + evolved guidelines) to produce a validated report. The Summarize Agent continuously updates the dynamic repository.*

---

## Architecture

SkinGPT-X orchestrates six specialized agents across two phases.

**Diagnostic Phase** — Vision Agent (Qwen3-VL) extracts fine-grained morphological findings. Pre-Diagnosis Agent (fine-tuned PanDerm) outputs top-5 candidate diagnoses with confidence scores. RAG Module retrieves disease-specific standards from the Oxford Handbook of Medical Dermatology via LanceDB.

**Review Phase** — Case-Review Agent (Qwen3-30B-A3B) synthesizes visual findings, handbook standards, and EvoDerma-Mem evidence (evolved guidelines + top-5 similar historical cases) through a 5-stage protocol: visual feature validation → canonical guidelines cross-check → empirical evidence alignment → conflict resolution → final diagnostic determination.

**Evolutionary Phase** — Summarize Agent distills confirmed cases into iteratively updated diagnostic guidelines, persisted as versioned `PrototypeVersion` snapshots in Neo4j without any parameter retraining.

---

## Key Results

SkinGPT-X consistently outperforms MedGemma, Hulu-Med, Qwen3-VL, and fine-tuned PanDerm across all four evaluation metrics (ACC, Weighted F1, MCC, Cohen's Kappa) on all benchmarks (p < 0.001, two-sided t-test).

![Fig2](Fig2.png)

*Figure 2. (a) Performance comparison across four public benchmark datasets. SkinGPT-X achieves +13.0% Weighted F1 on DermNet and +9.6% ACC on DDI31 over the second-best model. (b) Case study: MedGemma and Hulu-Med both misdiagnose a porphyria case as psoriasis, while SkinGPT-X correctly identifies the rare condition by cross-referencing evolved guidelines with visual findings.*

---

## High-Dimensional Classification

To stress-test SkinGPT-X under realistic fine-grained diagnostic complexity, we constructed **DermNet498**, re-categorizing DermNet from 23 broad super-classes into 498 clinically distinct sub-classes. For example, *Psoriasis pictures Lichen Planus and related diseases* is decoupled into 27 distinct classes (17 Psoriasis subtypes + 6 Lichen Planus variants). Intermediate-scale variants (DermNet225, DermNet272, DermNet353) enable controlled granularity ablation.

![Fig3](Fig3.png)

*Figure 3. (a) Hierarchical Sankey diagram showing category expansion from DermNet (23 classes) through DermNet225/272/353 to DermNet498 (498 classes, 18,317 samples). (b) SkinGPT-X achieves +5.4% ACC and +8.2% MCC over PanDerm on DermNet498 (p < 0.001). (c) As diagnostic granularity scales from 23 to 498 categories, SkinGPT-X consistently maintains a robust margin over PanDerm, preserving ACC / Weighted F1 / Kappa above 60%.*

---

## Rare Skin Disease Diagnosis

We curated the **Rare Skin Disease Dataset (RSDD)** — the first benchmark specifically addressing dermatological rare disease scarcity — comprising 564 clinical samples across 8 distinct categories, aligned with the Rare Disease Diagnosis and Treatment Guidelines published by the National Health Commission of China (2025). Sources include Wiley Online Library, Stamford Skin Center, Wikidoc, Healthline, ResearchGate, and the National Psoriasis Foundation. Train-test split is 1:2 to simulate data-scarce clinical conditions.

The 8 categories: Cutaneous Neuroendocrine Carcinoma (n=115), Generalized Pustular Psoriasis (n=110), Behcet's Disease (n=81), Blue Rubber Bleb Nevus Syndrome (n=73), Dermatofibrosarcoma Protuberans (n=59), Gorlin Syndrome (n=58), Epithelioid Sarcoma (n=46), Cryopyrin-associated Periodic Syndrome (n=22).

![Fig4](Fig4.png)

*Figure 4. (a) RSDD construction pipeline integrating diverse medical sources. (b) SkinGPT-X achieves +9.8% ACC, +7.1% Weighted F1, +9.7% MCC, +10% Kappa over fine-tuned PanDerm on RSDD (p < 0.001). (c) Representative rare disease case studies: EvoDerma-Mem enables correct identification of Blue Rubber Bleb Nevus Syndrome and Behcet's Disease by cross-referencing evolved guidelines with past cases, where competing models fail.*

---

## EvoDerma-Mem: Self-Evolving Agent Memory

EvoDerma-Mem is a graph-native clinical memory stored in Neo4j. Each case is stored as a linked triplet `Mᵢ = ⟨zᵢ, Kᵢ, Dᵢ⟩` (embedding, key findings, diagnosis). At inference, top-5 similar cases are retrieved by cosine similarity. When `ΔN ≥ N_thresh` new cases arrive for disease class C, guidelines evolve as `Gᵗ⁺¹_C = A(Gᵗ_C ⊕ K_new)`, with each version persisted as an immutable snapshot with delta score `δ = 1 − cos(z_new, z_prev)`.

**Neo4j graph schema:**

```
(Case)
  ├── image_path, case_id, true_label, sub_label
  ├── key_findings, feature_vector, findings_embedding
  └──[BELONGS_TO]──► (Prototype)
                          ├── current_summary, current_embedding, version_counter
                          └──[HAS_VERSION]──► (PrototypeVersion)
                                                 ├── summary, embedding, delta
                                                 └──[USED_CASE]──► (Case)
```

![Fig5](Fig5.png)

*Figure 5. (a) Ablation study: EvoDerma-Mem yields +10.1% ACC and +11.0% Weighted F1 on DermNet498 over the memory-free baseline (p < 0.001). (b) Guidelines Evolution Timeline showing iterative refinement across 50+ versions for 100+ disease categories — deeper color intensity indicates larger knowledge updates. (c) Lichen Planus guideline evolution example: the current version extends beyond the classic "6 P's" to cover atypical anatomical sites and morphological mimics of psoriasis/eczema. (d) Physician validation (n=100 cases, 5-point scale): Rigorousness of Medical Logic 4.889, Completeness of Guidelines 4.904, Rationality of Manifestation Refinement 4.914.*

---

## Quick Start

### 1. Environment

```bash
# Python 3.10+, Ubuntu 18.04+, CUDA 12.8
# Recommended hardware: 8× RTX 4090 (24 GB) for full local deployment
pip install -r requirements.txt
```

Dependencies: Neo4j, LanceDB (local), Qwen3-VL, Qwen3-30B-A3B, BGE-small-en-v1.5, LlamaIndex.

### 2. Full Multi-Agent Inference

```bash
python agent_workflow.py \
  --model_name         "qwen3-30b-a3b" \
  --image_folder       "data/images/" \
  --markdown_file_path "skin_handbook.md" \
  --output_folder      "output/"
```

### 3. Build / Evolve Knowledge Base

```bash
# Initial build from labelled image split
python build_knowledge_base_evolve.py \
  --txt              "/path/to/split.txt" \
  --eval_dir         "/path/to/eval_dir" \
  --image_dir        "/path/to/image_root/" \
  --use_sub_label \
  --distill_recent_k 10

# Backfill sub_label for existing DB cases only
python build_knowledge_base_evolve.py \
  --eval_dir        "/path/to/eval_dir" \
  --image_dir       "/path/to/image_root/" \
  --update_existing \
  --use_sub_label
```

---

## Implementation Details

Fine-tuning Pre-Diagnosis Agent and PanDerm: `epochs=30`, `warmup=10`, `lr=5e-4`, `batch=128`. LLM agents: Qwen3 series, `max_tokens=4096`, `temperature=0.3`. Vision Agent: Qwen3-VL. Case-Review Agent: Qwen3-30B-A3B. Embedding model: BGE-small-en-v1.5 (local path). Vector store: LanceDB (read-only in CaseReviewAgent). Graph database: Neo4j. Stack: Python 3.10, PyTorch 2.9.1+cu128, CUDA 12.8.

**Data conventions:** `train_files.json` — list of relative image paths. `train_feats.npy` — feature vectors aligned with `train_files.json`. Disease label inferred from parent folder name. `sub_label` extracted from filename stem (digits removed, `-` → spaces).

---

## Known Limitations

**Computational overhead:** Multi-agent orchestration has higher latency than monolithic models; may challenge high-throughput real-time settings. Future work targets optimized agent communication protocols.

**Image acquisition heterogeneity:** Sensor calibration and lighting variations across medical centers can degrade EvoDerma-Mem retrieval accuracy via distribution shift in the latent feature space.

**Incomplete:** `skingpt_agent.py` currently simulates outputs via `ollama3.2-vision`; can be upgraded to a true SkinGPT-4 model integration. Move DB/API credentials to environment variables before deployment.

---

## Citation

```bibtex
@article{chen2025skingptx,
  title   = {SkinGPT-X: A Self-Evolving Collaborative Multi-Agent System
             for Transparent and Trustworthy Dermatological Diagnosis},
  author  = {Chen, Zhangtianyi and Shen, Yuhao and Widjaja, Florensia
             and Xu, Yan and Sun, Liyuan and Wang, Zijian and Chen, Hongyi
             and Dai, Wufei and Wang, Ziwen and Zhang, Xinyuan and Zhou, Juexiao},
  year    = {2025},
  note    = {CUHK-Shenzhen, Award No. UDF01004172}
}
```

Related work: [SkinGPT-R1](https://arxiv.org/abs/2511.15242) · [SkinGPT-4](https://arxiv.org/abs/2304.10674) · [PanDerm](https://www.nature.com/articles/s41591-025-03383-2)

---

## License

Non-commercial use only. Researchers may sign the license at the [code repository](https://github.com/healme-225040511/Skingpt_X) and contact J.Z. or Z.C. for access.

**Funding:** The Chinese University of Hong Kong, Shenzhen, Award No. UDF01004172.
