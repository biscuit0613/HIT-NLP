#!/usr/bin/env python3
"""Compute BLEU-4, EM and token F1 on the medical predictions."""
import argparse
import collections
import csv
import json
import math
import os
import re
import string

import jieba

TAGS = ["base", "lora_r8_qv", "lora_r16_all_lr1e4", "lora_r16_all",
        "lora_r32_all", "dora_r16_all", "dora_r8_qv"]
PUNCT = set(string.punctuation + "，。！？；：“”‘’、（）【】《》…—·")


def norm(text):
    return "".join(ch for ch in text if not ch.isspace() and ch not in PUNCT)


def tokens(text):
    return [t for t in jieba.lcut(text) if t.strip()]


def ngrams(toks, n):
    return collections.Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def load(tag, results_dir):
    path = os.path.join(results_dir, f"pred_{tag}_medical.jsonl")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def corpus_bleu(hyps, refs, max_n=4):
    matches = [0] * (max_n + 1)
    totals = [0] * (max_n + 1)
    hyp_len = ref_len = 0
    for hyp, ref in zip(hyps, refs):
        h, r = tokens(hyp), tokens(ref)
        hyp_len += len(h)
        ref_len += len(r)
        for n in range(1, max_n + 1):
            hn, rn = ngrams(h, n), ngrams(r, n)
            matches[n] += sum(min(c, rn[g]) for g, c in hn.items())
            totals[n] += max(len(h) - n + 1, 0)
    if hyp_len == 0:
        return 0.0
    ps = [(matches[n] + 1) / (totals[n] + 1) for n in range(1, max_n + 1)]
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return bp * math.exp(sum(math.log(p) for p in ps) / max_n)


def f1(hyp, ref):
    h, r = ngrams(tokens(hyp), 1), ngrams(tokens(ref), 1)
    common = sum(min(c, r[g]) for g, c in h.items())
    if not h or not r:
        return 0.0
    p = common / sum(h.values())
    rec = common / sum(r.values())
    return 0.0 if p + rec == 0 else 2 * p * rec / (p + rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/official")
    ap.add_argument("--out", default="results/official/medical_extra.csv")
    ap.add_argument("--limit", type=int, default=100, choices=[100, 500])
    args = ap.parse_args()

    rows = []
    for tag in TAGS:
        preds = load(tag, args.results)
        if not preds or len(preds) < args.limit:
            continue
        subset = preds[:args.limit]
        hyps = [p.get("pred", "") for p in subset]
        refs = [p["tgt"][0] if isinstance(p["tgt"], list) else p["tgt"] for p in subset]
        rows.append({
            "tag": tag,
            "n": args.limit,
            "bleu4": round(corpus_bleu(hyps, refs), 4),
            "em": round(sum(norm(h) == norm(r) for h, r in zip(hyps, refs)) / args.limit, 4),
            "f1": round(sum(f1(h, r) for h, r in zip(hyps, refs)) / args.limit, 4),
        })
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "n", "bleu4", "em", "f1"])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
