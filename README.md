# SkinGPT-4X

SkinGPT-4X is an explainable, evolving, multi-agent dermatology intelligence framework.
It combines visual reasoning, handbook retrieval, historical case memory, and prototype evolution
to continuously improve diagnostic consistency and interpretability.

![Workflow](workflow.png)

## Why SkinGPT-4X
- Multi-agent orchestration instead of single-shot diagnosis.
- Continually evolving disease prototypes in Neo4j.
- Hybrid decision support from visual findings, static handbook knowledge, and similar historical cases.
- JSON-structured outputs for downstream automation and auditability.

## Core Agents
- `RAG Agent`: Retrieves reliable textbook-style medical knowledge.
- `WebSearch Agent`: Gathers recent evidence (papers/guidelines/case reports).
- `SkinGPT Agent`: Produces structured visual findings from skin images.
- `Reasoning Agent`: Integrates multimodal evidence into candidate diagnoses.
- `CaseReview Agent`: Re-audits predictions using prototypes + handbook + historical cases.
- `TreatmentRecommend Agent`: Generates treatment and care suggestions.

## Deep Dive: Knowledge Evolution Engine
Main file: `build_knowledge_base_evolve.py`

This module builds and evolves a graph-native clinical memory in Neo4j.

### What it does
- Ingests image-level features from `train_feats.npy` and file paths from `train_files.json`.
- Calls `VisionAgent` to produce `key_findings` for each image (with optional cache reuse).
- Writes each case into Neo4j `Case` nodes and links to disease-level `Prototype` nodes.
- Extracts `sub_label` from filename patterns (optional flag control).
- Triggers prototype evolution every N new cases per disease.
- Saves each evolution as a versioned snapshot (`PrototypeVersion`) with:
  - summary text
  - embedding
  - delta from previous version
  - case lineage (`USED_CASE` links + case id list)

### Neo4j Conceptual Graph
- `Case`
  - `image_path`, `case_id`, `true_label`, `sub_label`
  - `key_findings`, `feature_vector`, `findings_embedding`
- `Prototype`
  - one per disease
  - current summary + current embedding + version counter
- `PrototypeVersion`
  - immutable snapshots of evolving diagnostic standards
  - linked to the concrete cases used for distillation

### Distillation Strategy
- Trigger condition: `case_count >= threshold` and divisible by threshold.
- Context window: last `K` verified recent cases (`--distill_recent_k`).
- LLM role: summarize cross-case discriminative dermatology patterns into one refined paragraph.
- Drift quantification: `delta = 1 - cosine_similarity(new_embedding, prev_embedding)`.

## Deep Dive: Case Review + RAG Audit Layer
Main file: `case_review_rag_agent.py`

This module is the final clinical auditor before output.

### What it does
- Loads feature banks (train/test) and maps image path -> embedding vector.
- Retrieves static medical knowledge from LanceDB via LlamaIndex retriever.
- Retrieves dynamic disease summaries from Neo4j `Prototype`.
- Finds similar historical cases using cosine similarity over stored feature vectors.
- Builds high-constraint prompts for:
  - main disease review (`review_case`)
  - fine-grained subclass review (`review_sub_class`)
- Enforces JSON-style outputs for structured reasoning and traceable decisions.

### Decision Evidence Sources
- Visual findings from image understanding.
- Top-K model candidates and probabilities.
- Handbook snippets (static knowledge).
- Similar confirmed historical cases.
- Evolved prototype summaries (dynamic knowledge memory).

## Quick Start
### 1) Environment
- Python `3.10+`
- Neo4j database
- LanceDB local store (for handbook/vector retrieval)
- Available local/remote LLM backends configured in utility files

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2) Run Full Multi-Agent Workflow
Put images in `data/images/` and run:

```bash
python agent_workflow.py \
  --model_name "gpt-4o-mini" \
  --image_folder "data/images/" \
  --markdown_file_path "skin_handbook.md" \
  --output_folder "output/"
```

### 3) Build / Evolve Knowledge Base
Use split txt + feature bank to write cases and evolve prototypes:

```bash
python build_knowledge_base_evolve.py \
  --txt "/path/to/split.txt" \
  --eval_dir "/path/to/eval_dir" \
  --image_dir "/path/to/image_root/" \
  --use_sub_label \
  --distill_recent_k 10
```

Only backfill `sub_label` for existing DB cases:

```bash
python build_knowledge_base_evolve.py \
  --eval_dir "/path/to/eval_dir" \
  --image_dir "/path/to/image_root/" \
  --update_existing \
  --use_sub_label
```

## Data Conventions
- `train_files.json`: list of relative image paths.
- `train_feats.npy`: feature vectors aligned with `train_files.json`.
- disease label is inferred from parent folder name in relative path.
- `sub_label` is extracted from filename stem (digits removed, `-` replaced by spaces).

## Engineering Notes
- The current project uses a local embedding model path for BGE (`bge-small-en-v1.5`).
- `CaseReviewAgent` opens LanceDB in read-only mode for stability.
- Prototype evolution quality depends heavily on the quality of `key_findings`.
- Prefer moving DB/API credentials to environment variables for safer deployment.

## Known Incomplete Part
- `skingpt_agent.py`: currently simulates outputs with `ollama3.2-vision`; can be upgraded to a true SkinGPT-4 model integration.
