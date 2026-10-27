#!/usr/bin/env python3
"""Single-turn interaction test on six fixed prompts.

Records answers, end-to-end response time and generation throughput for the
base model or a LoRA/DoRA adapter, using the same prompt and decoding settings
as the batch evaluation.
"""
import argparse
import json
import os
import statistics
import time

import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEMS = {
    "qwen": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    "official": "You are a helpful assistant.",
}
MAX_NEW_TOKENS = 256
REPETITION_PENALTY = 1.05

QUESTIONS = [
    "我最近总是头晕，可能是什么原因？",
    "荨麻疹不能吃什么？",
    "孩子发烧38.5度，需要马上吃退烧药吗？",
    "高血压患者的日常饮食需要注意什么？",
    "失眠多梦应该如何调理？",
    "中国的首都是哪里？",
]


def build_prompt(tokenizer, question, system):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_model(base_path, adapter_path, dtype, merge_adapter, attn):
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    major = int(transformers.__version__.split(".")[0])
    dtype_kw = "dtype" if major >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation=attn,
        **{dtype_kw: torch_dtype},
    )
    if adapter_path and adapter_path.lower() != "none":
        model = PeftModel.from_pretrained(model, adapter_path)
        if merge_adapter:
            model = model.merge_and_unload()
            model = model.to(torch_dtype)
    model.eval()
    return tokenizer, model


def generate_one(model, tokenizer, question, system):
    inputs = tokenizer(build_prompt(tokenizer, question, system), return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    t0 = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    latency = time.perf_counter() - t0
    generated = output[0, inputs["input_ids"].shape[1]:]
    n_tokens = int(generated.shape[0])
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, latency, n_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--system", default="qwen", choices=sorted(SYSTEMS))
    ap.add_argument("--merge-adapter", action="store_true")
    args = ap.parse_args()
    system = SYSTEMS[args.system]

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"loading base={args.base_model} adapter={args.adapter} dtype={args.dtype} attn={args.attn}", flush=True)
    tokenizer, model = load_model(args.base_model, args.adapter, args.dtype, args.merge_adapter, args.attn)

    results = []
    for i, question in enumerate(QUESTIONS, 1):
        answer, latency, n_tokens = generate_one(model, tokenizer, question, system)
        results.append({
            "idx": i - 1,
            "question": question,
            "answer": answer,
            "latency_s": latency,
            "new_tokens": n_tokens,
            "tokens_per_s": n_tokens / max(latency, 1e-6),
            "chars": len(answer),
        })
        print(f"[{i}/{len(QUESTIONS)}] {latency:.1f}s, {n_tokens} tok\n{answer}\n", flush=True)

    summary = {
        "tag": args.tag,
        "adapter": args.adapter,
        "base": args.base_model,
        "protocol": {
            "system": system,
            "decoding": "greedy",
            "repetition_penalty": REPETITION_PENALTY,
            "max_new_tokens": MAX_NEW_TOKENS,
            "dtype": args.dtype,
            "attn": args.attn,
            "merged_adapter": args.merge_adapter,
        },
        "avg_latency_s": statistics.mean(r["latency_s"] for r in results),
        "avg_tokens_per_s": statistics.mean(r["tokens_per_s"] for r in results),
        "results": results,
    }
    json_path = os.path.join(args.out_dir, f"gen_test_{args.tag}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    txt_path = os.path.join(args.out_dir, f"gen_test_{args.tag}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write("=" * 60 + "\n")
            f.write(f"问：{r['question']}\n答：{r['answer']}\n")
            f.write(f"（生成用时 {r['latency_s']:.1f} 秒，{r['new_tokens']} tokens）\n")
    print(f"avg latency {summary['avg_latency_s']:.1f}s, avg throughput {summary['avg_tokens_per_s']:.1f} tok/s")
    print("wrote", json_path, txt_path)


if __name__ == "__main__":
    main()
