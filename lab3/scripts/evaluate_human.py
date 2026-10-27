#!/usr/bin/env python3
"""Summarize blind human ratings: per-system means, bootstrap CIs and pairwise wins."""
import argparse
import csv
import json
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

TAGS = ["base", "lora_r8_qv", "dora_r16_all"]
LABELS = {"base": "基座模型", "lora_r8_qv": "LoRA r=8 q,v", "dora_r16_all": "DoRA r=16 all"}
COLORS = ["#8a8a8a", "#4c72b0", "#c44e52"]
DIMS = ["relevance", "accuracy", "fluency"]


def load_ratings(path, key):
    per = {}
    for row in csv.DictReader(open(path, encoding="utf-8")):
        item = row["item"].strip()
        pos = row["answer"].strip()
        tag = key[item][pos]
        try:
            scores = [float(row[d]) for d in DIMS]
        except (TypeError, ValueError):
            continue
        per.setdefault(item, {}).setdefault(tag, []).append(scores)
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet-dir", default="results/human_eval")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--fig", default="report/figs/human_eval.png")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    key = json.load(open(os.path.join(args.packet_dir, "key.json"), encoding="utf-8"))
    per = load_ratings(os.path.join(args.packet_dir, "ratings.csv"), key)
    missing = sum(1 for item in key for pos in key[item]
                  if key[item][pos] not in per.get(item, {}))
    if missing:
        raise SystemExit(f"{missing} answers are not rated yet")
    n_items = len(per)
    item_means = {tag: [] for tag in TAGS}
    dim_means = {tag: {d: [] for d in DIMS} for tag in TAGS}
    for item, by_tag in per.items():
        for tag in TAGS:
            scores = np.array(by_tag[tag], dtype=float).mean(axis=0)
            dim_means[tag]["relevance"].append(scores[0])
            dim_means[tag]["accuracy"].append(scores[1])
            dim_means[tag]["fluency"].append(scores[2])
            item_means[tag].append(scores.mean())

    rng = random.Random(args.seed)
    idxs = [[rng.randrange(n_items) for _ in range(n_items)] for _ in range(args.bootstrap)]
    rows = []
    for tag in TAGS:
        overall = np.array(item_means[tag])
        boots = overall[np.array(idxs)].mean(axis=1)
        rec = {
            "tag": tag,
            "label": LABELS[tag],
            "n_items": n_items,
            "overall": overall.mean(),
            "ci_lo": np.percentile(boots, 2.5),
            "ci_hi": np.percentile(boots, 97.5),
        }
        for d in DIMS:
            rec[d] = float(np.mean(dim_means[tag][d]))
        rows.append(rec)

    for i, a in enumerate(TAGS):
        for b in TAGS[i + 1:]:
            wins = ties = losses = 0
            for x, y in zip(item_means[a], item_means[b]):
                if x > y:
                    wins += 1
                elif x < y:
                    losses += 1
                else:
                    ties += 1
            print(f"{LABELS[a]} vs {LABELS[b]}: {wins}W/{ties}T/{losses}L")

    path = os.path.join(args.out_dir, "human_eval_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['label']:14s} overall={r['overall']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] "
              f"rel={r['relevance']:.2f} acc={r['accuracy']:.2f} flu={r['fluency']:.2f}")

    os.makedirs(os.path.dirname(args.fig), exist_ok=True)
    x = np.arange(len(TAGS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    names = {"relevance": "相关性", "accuracy": "准确性", "fluency": "流畅性"}
    for i, d in enumerate(DIMS):
        vals = [r[d] for r in rows]
        ax.bar(x + (i - 1) * width, vals, width, label=names[d], color=["#4c72b0", "#dd8452", "#55a868"][i])
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], fontsize=8)
    ax.set_ylim(1, 5.5)
    ax.set_ylabel("mean score (1-5)")
    ax.legend(ncol=3, frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(args.fig, dpi=200)
    plt.close(fig)
    print("wrote", path, args.fig)


if __name__ == "__main__":
    main()
