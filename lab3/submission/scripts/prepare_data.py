#!/usr/bin/env python3
"""Convert DISC-Med-SFT multi-turn conversations into single-turn SFT pairs.

Each assistant reply is paired with the immediately preceding user utterance.
Test-set questions from Chinese-medical-dialogue are removed from the training
pool (decontamination) so that the ROUGE evaluation is not inflated by exact
train/test overlap; the overlap count is reported for the paper.
"""
import argparse
import json
import os
import random
import re
import unicodedata
from collections import Counter


PUNCT = re.compile(r"[\s　，。！？、；：“”‘’（）《》【】,.!?;:'\"()\[\]<>~…—\-·]+")


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return PUNCT.sub("", text)


def iter_pairs(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            conv = rec.get("conversation") or []
            for i, turn in enumerate(conv):
                if turn.get("role") != "assistant" or i == 0:
                    continue
                prev = conv[i - 1]
                if prev.get("role") != "user":
                    continue
                src = (prev.get("content") or "").strip()
                tgt = (turn.get("content") or "").strip()
                if src and tgt:
                    yield rec.get("source", ""), src, tgt


def pct(values, p):
    if not values:
        return 0
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * p))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--medical-test", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-train", type=int, default=100000)
    ap.add_argument("--max-val", type=int, default=2000)
    ap.add_argument("--val-ratio", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.medical_test, encoding="utf-8") as f:
        medical_test = json.load(f)
    banned = {norm(x["src"]) for x in medical_test}

    groups = []
    n_conv = n_pairs = n_empty = n_banned = 0
    source_counter = Counter()
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_conv += 1
            source = rec.get("source", "")
            conv = rec.get("conversation") or []
            pairs = []
            for i, turn in enumerate(conv):
                if turn.get("role") != "assistant" or i == 0:
                    continue
                prev = conv[i - 1]
                if prev.get("role") != "user":
                    continue
                src = (prev.get("content") or "").strip()
                tgt = (turn.get("content") or "").strip()
                if not src or not tgt:
                    n_empty += 1
                    continue
                n_pairs += 1
                if norm(src) in banned:
                    n_banned += 1
                    continue
                pairs.append((src, tgt))
            if pairs:
                groups.append((source, pairs))
                source_counter[source] += len(pairs)

    rng = random.Random(args.seed)
    rng.shuffle(groups)

    n_val_groups = max(1, int(len(groups) * args.val_ratio))
    val_groups, train_groups = groups[:n_val_groups], groups[n_val_groups:]

    def flatten(gs):
        out = []
        for _, ps in gs:
            out.extend(ps)
        return out

    train_pairs = flatten(train_groups)
    val_pairs = flatten(val_groups)
    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)
    if len(train_pairs) > args.max_train:
        train_pairs = train_pairs[: args.max_train]
    if len(val_pairs) > args.max_val:
        val_pairs = val_pairs[: args.max_val]

    seen = set()
    dedup = []
    for src, tgt in train_pairs:
        key = (norm(src), norm(tgt))
        if key in seen:
            continue
        seen.add(key)
        dedup.append((src, tgt))
    train_pairs = dedup

    os.makedirs(args.out_dir, exist_ok=True)

    def dump_alpaca(pairs, path):
        data = [{"instruction": s, "input": "", "output": t} for s, t in pairs]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)

    dump_alpaca(train_pairs, os.path.join(args.out_dir, "train.json"))
    dump_alpaca(val_pairs, os.path.join(args.out_dir, "val.json"))
    with open(os.path.join(args.out_dir, "train_srctgt.jsonl"), "w", encoding="utf-8") as f:
        for s, t in train_pairs:
            f.write(json.dumps({"src": s, "tgt": t}, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "val_srctgt.jsonl"), "w", encoding="utf-8") as f:
        for s, t in val_pairs:
            f.write(json.dumps({"src": s, "tgt": t}, ensure_ascii=False) + "\n")

    src_lens = [len(s) for s, _ in train_pairs]
    tgt_lens = [len(t) for _, t in train_pairs]
    stats = {
        "conversations": n_conv,
        "assistant_pairs_raw": n_pairs,
        "empty_pairs": n_empty,
        "removed_test_overlap": n_banned,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "src_chars": {"p50": pct(src_lens, 0.5), "p90": pct(src_lens, 0.9), "p99": pct(src_lens, 0.99), "max": max(src_lens)},
        "tgt_chars": {"p50": pct(tgt_lens, 0.5), "p90": pct(tgt_lens, 0.9), "p99": pct(tgt_lens, 0.99), "max": max(tgt_lens)},
        "sources_top": source_counter.most_common(15),
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
