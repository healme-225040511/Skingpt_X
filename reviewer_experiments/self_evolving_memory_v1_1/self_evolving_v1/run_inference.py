#!/usr/bin/env python3
"""Resumable deterministic Qwen3-VL text inference for v1 memory manifests."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed: set[int] = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("record_type") == "inference":
                completed.add(int(row["global_index"]))
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
                "record_type": "run",
                "model": str(args.model),
                "manifest": str(args.manifest),
                "records": len(rows),
                "gpu": torch.cuda.get_device_name(0),
                "batch_size": args.batch_size,
                "max_new_tokens": args.max_new_tokens,
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
