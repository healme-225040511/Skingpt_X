#!/usr/bin/env python3
"""Download Hugging Face models used by the SkinGPT-X demo.

The current codebase uses fixed local paths under /225040511/project/hf_cache.
This helper downloads the public HF repos into those exact paths so the demo
can load them without additional path edits.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_CACHE_ROOT = Path("/225040511/project/hf_cache")

MODEL_PROFILES = {
    "minimal": [
        {
            "repo_id": "BAAI/bge-small-en-v1.5",
            "local_dir": DEFAULT_CACHE_ROOT / "bge-small-en-v1.5" / "BAAI" / "bge-small-en-v1___5",
            "note": "Embedding model used by RAGAgent and knowledge-base evolution.",
        },
        {
            "repo_id": "Qwen/Qwen2-VL-7B-Instruct",
            "local_dir": DEFAULT_CACHE_ROOT / "Qwen-VL-8B-Instruct",
            "note": "Vision-language model path currently loaded by local_llm_utils.py.",
        },
    ],
    "full": [
        {
            "repo_id": "BAAI/bge-small-en-v1.5",
            "local_dir": DEFAULT_CACHE_ROOT / "bge-small-en-v1.5" / "BAAI" / "bge-small-en-v1___5",
            "note": "Embedding model used by RAGAgent and knowledge-base evolution.",
        },
        {
            "repo_id": "Qwen/Qwen2-VL-7B-Instruct",
            "local_dir": DEFAULT_CACHE_ROOT / "Qwen-VL-8B-Instruct",
            "note": "Vision-language model path currently loaded by local_llm_utils.py.",
        },
        {
            "repo_id": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
            "local_dir": DEFAULT_CACHE_ROOT / "Qwen3-VL-30B",
            "note": "Qwen3-VL model described in the README; large download.",
        },
        {
            "repo_id": "Qwen/Qwen3-30B-A3B",
            "local_dir": DEFAULT_CACHE_ROOT / "Qwen3-30B-A3B",
            "note": "Text LLM described for case review; large download.",
        },
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(MODEL_PROFILES),
        default="minimal",
        help="minimal downloads only models referenced by current demo code; full also downloads README-level large Qwen3 models.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help="HF cache root. Defaults to the path hardcoded in the current project.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token. Defaults to HF_TOKEN from the environment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["HF_HOME"] = str(args.cache_root)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install project requirements first, "
            "or run `pip install huggingface-hub`."
        ) from exc

    for model in MODEL_PROFILES[args.profile]:
        local_dir = Path(str(model["local_dir"]).replace(str(DEFAULT_CACHE_ROOT), str(args.cache_root), 1))
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nDownloading {model['repo_id']} -> {local_dir}")
        print(f"Reason: {model['note']}")
        snapshot_download(
            repo_id=model["repo_id"],
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            token=args.token,
            resume_download=True,
        )

    print("\nAll requested Hugging Face models are ready.")


if __name__ == "__main__":
    main()
