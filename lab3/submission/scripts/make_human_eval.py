#!/usr/bin/env python3
"""Build a blind human-evaluation packet from medical predictions.

Compares the base model, the recommended minimal LoRA setting and DoRA on a
random subset. Answers are shuffled per item; the model identity is stored
separately in key.json.
"""
import argparse
import csv
import json
import os
import random

TAGS = ["base", "lora_r8_qv", "dora_r16_all"]
LABELS = {"base": "基座模型", "lora_r8_qv": "LoRA r=8 q,v", "dora_r16_all": "DoRA r=16 all"}
POSITIONS = ["A", "B", "C"]


def first(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else ""
    return value or ""


def read_preds(results_dir, tag):
    path = os.path.join(results_dir, f"pred_{tag}_medical.jsonl")
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                records[record["idx"]] = record
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out-dir", default="results/human_eval")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = {tag: read_preds(args.results, tag) for tag in TAGS}
    common = sorted(set.intersection(*(set(d) for d in data.values())))
    rng = random.Random(args.seed)
    items = rng.sample(common, min(args.n, len(common)))

    os.makedirs(args.out_dir, exist_ok=True)
    key = {}
    packet = [
        "# 医疗回答盲评",
        "",
        "请对每条回答按 1–5 分评价：相关性（是否回应问题）、准确性（医学内容是否合理）、流畅性（表达是否通顺）。",
        "参考答案仅用于了解语境，不必逐字比较；请独立判断每条回答的质量。",
        "",
    ]
    for item in items:
        record = data["base"][item]
        order = TAGS[:]
        rng.shuffle(order)
        key[str(item)] = {pos: tag for pos, tag in zip(POSITIONS, order)}
        packet += [
            f"## 题目 {item}",
            "",
            f"**患者提问：** {first(record.get('src'))}",
            "",
            f"**参考答案：** {first(record.get('tgt'))}",
            "",
        ]
        for pos, tag in zip(POSITIONS, order):
            packet += [f"**回答 {pos}：** {data[tag][item].get('pred', '').strip()}", ""]

    with open(os.path.join(args.out_dir, "key.json"), "w", encoding="utf-8") as f:
        json.dump(key, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "packet.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(packet))
    with open(os.path.join(args.out_dir, "ratings.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item", "answer", "relevance", "accuracy", "fluency", "comment"])
        for item in items:
            for pos in POSITIONS:
                writer.writerow([item, pos, "", "", "", ""])
    print(f"{len(items)} items x {len(TAGS)} answers -> {args.out_dir}")
    print("fill relevance/accuracy/fluency (1-5) in ratings.csv, then run evaluate_human.py")


if __name__ == "__main__":
    main()
