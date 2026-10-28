#!/usr/bin/env python3
"""Evaluate Qwen2-7B-Instruct (base or LoRA/DoRA adapter) on medical dialogue and CMMLU.

Medical: Chinese-medical-dialogue answers, ROUGE-1/2/L F1.
CMMLU:   computer_science multiple choice, option accuracy.

Decoding and the prediction schema follow the course inference scripts: greedy
decoding, repetition penalty 1.05, at most 256 new tokens, and one JSON object
{"src": [...], "tgt": [...], "pred": ...} per line. The system prompt defaults to
the Qwen template used during SFT; --system official switches to the short prompt
of the course scripts. CMMLU is scored under the course protocol (no answer-format
instruction); --cmmlu-instruction adds one for a format-compliance ablation.
"""
import argparse
import json
import os
import re
import statistics
import time

import jieba
import torch
import transformers
from peft import PeftModel
from rouge_chinese import Rouge
from transformers import AutoModelForCausalLM, AutoTokenizer

_jieba_cache = os.path.join(os.path.expanduser("~"), ".cache", "jieba")
os.makedirs(_jieba_cache, exist_ok=True)
jieba.dt.tmp_dir = _jieba_cache

SYSTEMS = {
    "qwen": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    "official": "You are a helpful assistant.",
}
MAX_NEW_TOKENS = 256
REPETITION_PENALTY = 1.05
MAX_INPUT_TOKENS = 1024

RE_ALONE = re.compile(r"^\s*[（(]?\s*([ABCD])\s*[）)\.．、]?\s*$")
RE_EXPLICIT = re.compile(r"(?:答案|选项|选择|选)\s*(?:是|为)?\s*[:：]?\s*[（(]?\s*([ABCD])(?![A-Za-z])")
RE_LEADING = re.compile(r"^\s*[（(]?\s*([ABCD])(?![A-Za-z])")


def parse_choice(text):
    t = (text or "").strip()
    m = RE_ALONE.match(t)
    if m:
        return m.group(1)
    m = RE_EXPLICIT.search(t)
    if m:
        return m.group(1)
    m = RE_LEADING.match(t)
    return m.group(1) if m else None


def build_prompt(tokenizer, user, system):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def first(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else ""
    return value or ""


def load_samples(path, limit=None):
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            raw = [json.loads(line) for line in f if line.strip()]
        else:
            raw = json.load(f)
    if limit is not None:
        raw = raw[:limit]
    samples = []
    for i, item in enumerate(raw):
        src = first(item.get("src") or item.get("input") or "")
        tgt = first(item.get("tgt") or item.get("output") or "")
        samples.append({"idx": item.get("idx", i), "src": src, "tgt": tgt})
    return samples


def load_model(base_path, adapter_path, dtype, merge_adapter, attn):
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True, padding_side="left")
    # Over-length prompts must lose their head (the system turn), never the trailing
    # assistant\n cue that the chat template appends.
    tokenizer.truncation_side = "left"
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


def generate_batch(model, tokenizer, prompts, max_new_tokens):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=MAX_INPUT_TOKENS)
    # attention_mask row sums are the unpadded prompt lengths, so a row sitting
    # exactly on the cap is a prompt that left-truncation cut part of.
    truncated = int((inputs["attention_mask"].sum(dim=1) >= MAX_INPUT_TOKENS).sum())
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[:, inputs["input_ids"].shape[1]:]
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    lengths = (generated != tokenizer.pad_token_id).sum(dim=1).tolist()
    return [t.strip() for t in texts], lengths, truncated


def clean_answer(text):
    text = re.split(r"<\|im_end\|>|<\|endoftext\|>", text)[0]
    return text.strip()


def rouge_scores(pairs):
    """Official scoring: empty hypotheses count as zero, non-empty scores are rescaled."""
    valid = [(hyp, ref) for hyp, ref in pairs if hyp.strip()]
    scale = len(valid) / max(1, len(pairs))
    if not valid:
        return {k: 0.0 for k in ("rouge-1", "rouge-2", "rouge-l",
                                 "rouge-1_p", "rouge-1_r", "rouge-2_p",
                                 "rouge-2_r", "rouge-l_p", "rouge-l_r")}
    hyps = [" ".join(jieba.cut(hyp)) or "。" for hyp, _ in valid]
    refs = [" ".join(jieba.cut(ref)) or "。" for _, ref in valid]
    scores = Rouge().get_scores(hyps, refs, avg=True)
    out = {}
    for name in ("rouge-1", "rouge-2", "rouge-l"):
        for key in ("f", "p", "r"):
            suffix = "" if key == "f" else f"_{key}"
            out[name + suffix] = scores[name][key] * scale
    return out


def style_metrics(preds):
    hyps = [p["pred"] for p in preds if p["pred"].strip()]
    if not hyps:
        return {"n": 0}
    lengths = [len(h) for h in hyps]

    def is_list(h):
        return bool(re.search(r"(?m)^\s*(?:[-*•]|\d+[.、)])\s+", h))

    def has_question(h):
        return "？" in h or "?" in h

    def repeat_4gram(h):
        text = re.sub(r"\s+", "", h)
        grams = [text[i:i + 4] for i in range(max(0, len(text) - 3))]
        if not grams:
            return 0.0
        return 1.0 - len(set(grams)) / len(grams)

    def cjk_ratio(h):
        chars = [c for c in h if not c.isspace()]
        if not chars:
            return 0.0
        return sum(1 for c in chars if "一" <= c <= "鿿") / len(chars)

    caution = ("就医", "医生", "医院", "及时", "建议")
    return {
        "avg_chars": statistics.mean(lengths),
        "median_chars": statistics.median(lengths),
        "short_rate": sum(1 for n in lengths if n < 80) / len(lengths),
        "long_rate": sum(1 for n in lengths if n > 400) / len(lengths),
        "list_rate": sum(1 for h in hyps if is_list(h)) / len(hyps),
        "question_rate": sum(1 for h in hyps if has_question(h)) / len(hyps),
        "caution_rate": sum(1 for h in hyps if any(w in h for w in caution)) / len(hyps),
        "repeat_4gram": statistics.mean(repeat_4gram(h) for h in hyps),
        "cjk_ratio": statistics.mean(cjk_ratio(h) for h in hyps),
    }


def medical_metrics(preds):
    scores = rouge_scores([(p["pred"], first(p["tgt"])) for p in preds])
    out = {"n": len(preds)}
    out.update(scores)
    out["valid"] = sum(1 for p in preds if p["pred"].strip())
    out["empty"] = len(preds) - out["valid"]
    out.update(style_metrics(preds))
    return out


def evaluate_medical(model, tokenizer, samples, batch_size, system):
    preds = []
    truncated = 0
    t0 = time.time()
    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        prompts = [build_prompt(tokenizer, s["src"], system) for s in chunk]
        hyps, lengths, n_trunc = generate_batch(model, tokenizer, prompts, MAX_NEW_TOKENS)
        truncated += n_trunc
        for s, hyp, ntok in zip(chunk, hyps, lengths):
            preds.append({
                "idx": s["idx"],
                "src": [s["src"]],
                "tgt": [s["tgt"]],
                "pred": clean_answer(hyp),
                "new_tokens": ntok,
            })
        print(f"  medical {min(start + batch_size, len(samples))}/{len(samples)}", flush=True)
    metrics = medical_metrics(preds)
    if len(preds) > 100:
        metrics["first100"] = medical_metrics(preds[:100])
    metrics["truncated_prompts"] = truncated
    metrics["elapsed_s"] = time.time() - t0
    metrics["samples_per_s"] = len(preds) / max(metrics["elapsed_s"], 1e-6)
    metrics["avg_new_tokens"] = statistics.mean(p["new_tokens"] for p in preds)
    metrics["hit_limit"] = sum(1 for p in preds if p["new_tokens"] >= MAX_NEW_TOKENS)
    return preds, metrics


def evaluate_cmmlu(model, tokenizer, samples, batch_size, system, instruction=""):
    preds = []
    truncated = 0
    t0 = time.time()
    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        prompts = [
            build_prompt(tokenizer, s["src"] + (f"\n\n{instruction}" if instruction else ""), system)
            for s in chunk
        ]
        outs, lengths, n_trunc = generate_batch(model, tokenizer, prompts, MAX_NEW_TOKENS)
        truncated += n_trunc
        for s, raw, ntok in zip(chunk, outs, lengths):
            choice = parse_choice(raw)
            preds.append({
                "idx": s["idx"],
                "src": [s["src"]],
                "tgt": [s["tgt"]],
                "pred": raw,
                "choice": choice,
                "correct": choice == s["tgt"].strip().upper(),
                "new_tokens": ntok,
            })
        print(f"  cmmlu {min(start + batch_size, len(samples))}/{len(samples)}", flush=True)

    n = len(preds)
    correct = sum(1 for p in preds if p["correct"])
    unparsed = sum(1 for p in preds if p["choice"] is None)
    contains_gold = sum(1 for p in preds if first(p["tgt"]).strip().upper() in p["pred"].upper())
    dist = {c: sum(1 for p in preds if p["choice"] == c) for c in "ABCD"}
    metrics = {
        "n": n,
        "correct": correct,
        "wrong": n - correct - unparsed,
        "unparsed": unparsed,
        "accuracy": correct / max(1, n),
        "contains_gold": contains_gold,
        "contains_gold_rate": contains_gold / max(1, n),
        "pred_dist": dist,
        "hit_limit": sum(1 for p in preds if p["new_tokens"] >= MAX_NEW_TOKENS),
        "truncated_prompts": truncated,
        "avg_new_tokens": statistics.mean(p["new_tokens"] for p in preds),
        "elapsed_s": time.time() - t0,
        "samples_per_s": n / max(time.time() - t0, 1e-6),
    }
    return preds, metrics


def write_jsonl(path, preds):
    with open(path, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--datasets", default="medical,cmmlu")
    ap.add_argument("--medical-file", default="test_medical500.json")
    ap.add_argument("--medical-limit", type=int, default=100)
    ap.add_argument("--cmmlu-file", default="test_cmmlu.json")
    ap.add_argument("--cmmlu-limit", type=int, default=None)
    ap.add_argument("--cmmlu-instruction", default="",
                    help="optional answer-format instruction appended to CMMLU prompts")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--system", default="qwen", choices=sorted(SYSTEMS))
    ap.add_argument("--merge-adapter", action="store_true")
    args = ap.parse_args()
    system = SYSTEMS[args.system]

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"loading base={args.base_model} adapter={args.adapter} dtype={args.dtype} attn={args.attn}", flush=True)
    t0 = time.time()
    tokenizer, model = load_model(args.base_model, args.adapter, args.dtype, args.merge_adapter, args.attn)
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    all_metrics = {
        "tag": args.tag,
        "adapter": args.adapter,
        "base": args.base_model,
        "protocol": {
            "system": system,
            "decoding": "greedy",
            "repetition_penalty": REPETITION_PENALTY,
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "attn": args.attn,
            "merged_adapter": args.merge_adapter,
            "cmmlu_instruction": args.cmmlu_instruction,
        },
    }
    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        if dataset == "medical":
            samples = load_samples(os.path.join(args.data_dir, args.medical_file), args.medical_limit)
            print(f"  medical limit {len(samples)}", flush=True)
            preds, metrics = evaluate_medical(model, tokenizer, samples, args.batch_size, system)
        elif dataset == "cmmlu":
            samples = load_samples(os.path.join(args.data_dir, args.cmmlu_file), args.cmmlu_limit)
            print(f"  cmmlu {len(samples)}", flush=True)
            preds, metrics = evaluate_cmmlu(model, tokenizer, samples, args.batch_size, system,
                                            args.cmmlu_instruction)
        else:
            raise ValueError(dataset)
        all_metrics[dataset] = metrics
        write_jsonl(os.path.join(args.out_dir, f"pred_{args.tag}_{dataset}.jsonl"), preds)
        print(dataset, json.dumps(metrics, ensure_ascii=False), flush=True)

    metrics_path = os.path.join(args.out_dir, f"metrics_{args.tag}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print("wrote", metrics_path)


if __name__ == "__main__":
    main()
