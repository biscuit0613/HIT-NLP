#!/usr/bin/env python3
"""Merge a PEFT LoRA/DoRA adapter into its base model and export a standalone checkpoint."""
import argparse
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    t0 = time.time()

    print(f"[1/4] loading tokenizer from {args.base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    print(f"[2/4] loading base weights on {args.device} ({args.dtype})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map={"": args.device},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print(f"[3/4] merging adapter {args.adapter}", flush=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model = model.to(dtype)

    print(f"[4/4] saving merged checkpoint to {args.out}", flush=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"done in {time.time() - t0:.1f}s: {n_params:,} parameters -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
