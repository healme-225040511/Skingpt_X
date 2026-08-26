#!/usr/bin/env python3
"""Resumable deterministic Qwen3-VL text inference for v1 memory manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_completed(output: Path, expected_run: dict[str, Any]) -> set[int]:
    """Validate an existing output before resuming it."""
    if not output.exists() or output.stat().st_size == 0:
        return set()
    rows = []
    for line_number, line in enumerate(output.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank line in existing output at line {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in existing output at line {line_number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"non-object JSON in existing output at line {line_number}")
        rows.append(row)
    runs = [row for row in rows if row.get("record_type") == "run"]
    if len(runs) != 1:
        raise ValueError(f"existing output must contain exactly one run header, found {len(runs)}")
    mismatches = {
        key: {"existing": runs[0].get(key), "expected": value}
        for key, value in expected_run.items()
        if runs[0].get(key) != value
    }
    if mismatches:
        raise ValueError(f"refusing incompatible resume: {json.dumps(mismatches, sort_keys=True)}")
    indices = [int(row["global_index"]) for row in rows if row.get("record_type") == "inference"]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate inference global_index values in existing output")
    unknown_types = sorted({str(row.get("record_type")) for row in rows} - {"run", "inference"})
    if unknown_types:
        raise ValueError(f"unknown record types in existing output: {unknown_types}")
    return set(indices)


def main() -> None:
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--run-label", required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    indices = [int(row["global_index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("manifest contains duplicate global_index values")
    model_config = args.model / "config.json"
    model_index = args.model / "model.safetensors.index.json"
    expected_run = {
        "record_type": "run",
        "run_label": args.run_label,
        "model": str(args.model.resolve()),
        "model_revision": args.model_revision,
        "model_config_sha256": file_sha256(model_config),
        "model_index_sha256": file_sha256(model_index),
        "code_revision": args.code_revision,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "records": len(rows),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
    }
    completed = load_completed(args.output, expected_run)
    pending = [row for row in rows if int(row["global_index"]) not in completed]
    if not pending:
        print("No pending records", flush=True)
        return

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    ).eval()
    torch.cuda.synchronize()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.output.exists() or args.output.stat().st_size == 0
    with args.output.open("a", encoding="utf-8") as handle:
        if new_file:
            handle.write(json.dumps({
                **expected_run,
                "gpu": torch.cuda.get_device_name(0),
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "cuda_version": torch.version.cuda,
                "python_version": platform.python_version(),
            }, ensure_ascii=False) + "\n")
            handle.flush()
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            chats = [processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": row["prompt"]}]}],
                tokenize=False,
                add_generation_prompt=True,
            ) for row in batch]
            inputs = processor(text=chats, padding=True, return_tensors="pt").to(model.device)
            input_tokens = [int(mask.sum().item()) for mask in inputs.attention_mask]
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.inference_mode():
                generated_all = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            generated = generated_all[:, inputs.input_ids.shape[1] :]
            responses = processor.batch_decode(generated, skip_special_tokens=True)
            pad_id = processor.tokenizer.pad_token_id
            for index, (row, response) in enumerate(zip(batch, responses)):
                record = {key: value for key, value in row.items() if key != "prompt"}
                record.update({
                    "record_type": "inference",
                    "response": response,
                    "input_tokens": input_tokens[index],
                    "output_tokens": generated[index].numel() if pad_id is None else int((generated[index] != pad_id).sum()),
                    "batch_seconds": elapsed,
                    "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
                })
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"completed={min(offset + len(batch), len(pending))}/{len(pending)} seconds={elapsed:.2f}", flush=True)


if __name__ == "__main__":
    main()
