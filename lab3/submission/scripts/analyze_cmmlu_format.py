#!/usr/bin/env python3
"""Diagnose CMMLU answer-format collapse.

The official parser only accepts an explicit option letter.  This script adds a
content-level diagnostic for otherwise unparsed outputs: the option whose text
has the largest character-trigram Jaccard overlap with the answer.
"""
import argparse
import csv
import json
import os
import re
import unicodedata

TAGS = ["base", "lora_r8_qv", "lora_r16_all_lr1e4", "lora_r16_all",
        "lora_r32_all", "dora_r16_all", "dora_r8_qv"]
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


def norm(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).lower()


def options(src):
    found = {}
    for line in src.splitlines():
        m = re.match(r"^([A-D])[.、)]\s*(.+)$", line.strip())
        if m:
            found[m.group(1)] = norm(m.group(2))
    return found


def trigrams(text):
    return {text[i:i + 3] for i in range(max(0, len(text) - 2))}


def overlap(a, b):
    ga, gb = trigrams(a), trigrams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/official")
    ap.add_argument("--out", default="results/official/cmmlu_format.csv")
    ap.add_argument("--mrr-out", default="results/official/cmmlu_mrr.csv")
    args = ap.parse_args()

    rows = []
    mrr_rows = []
    for tag in TAGS:
        preds = load(os.path.join(args.results, f"pred_{tag}_cmmlu.jsonl"))
        if not preds:
            continue
        strict = sum(r["correct"] for r in preds)
        unparsed = sum(r["choice"] is None for r in preds)
        relaxed = 0
        recovered = 0
        reciprocal_rank = 0.0
        for r in preds:
            gold = r["tgt"][0].strip().upper()
            choice = parse_choice(r["pred"])
            if choice is not None:
                reciprocal_rank += 1.0 if choice == gold else 0.0
            else:
                opts = options(r["src"][0])
                pred = norm(r["pred"])
                scores = {k: overlap(pred, v) for k, v in opts.items()}
                gold_score = scores.get(gold, 0.0)
                rank = 1 + sum(v > gold_score + 1e-12 for v in scores.values())
                reciprocal_rank += 1.0 / rank
            if r["correct"]:
                relaxed += 1
                continue
            opts = options(r["src"][0])
            pred = norm(r["pred"])
            pick = max(opts, key=lambda k: overlap(pred, opts[k])) if opts else None
            if pick == gold:
                relaxed += 1
                recovered += 1
        rows.append({
            "tag": tag,
            "n": len(preds),
            "strict_correct": strict,
            "strict_acc": round(strict / len(preds), 4),
            "unparsed": unparsed,
            "relaxed_correct": relaxed,
            "relaxed_acc": round(relaxed / len(preds), 4),
            "recovered_from_unparsed": recovered,
        })
        mrr_rows.append({
            "tag": tag,
            "n": len(preds),
            "mrr": round(reciprocal_rank / len(preds), 4),
        })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(args.mrr_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "n", "mrr"])
        writer.writeheader()
        writer.writerows(mrr_rows)
    for row, mrr_row in zip(rows, mrr_rows):
        print(row, mrr_row)


if __name__ == "__main__":
    main()
