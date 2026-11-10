#!/usr/bin/env python3
"""Compare CMMLU protocols and bootstrap base-adapter accuracy gaps.

Each --protocol takes LABEL=DIR, where DIR holds pred_<tag>_cmmlu.jsonl files
written by eval_models.py.  Strict accuracy follows the official parser; the
content diagnostic only reclassifies unparsed outputs, and contributes zero
reciprocal rank when no option text overlaps.
"""
import argparse
import csv
import json
import os
import random

from analyze_cmmlu_format import TAGS, load, norm, options, overlap, parse_choice

ORDER = ["base", "lora_r8_qv", "lora_r16_all_lr1e4", "lora_r16_all",
         "lora_r32_all", "dora_r16_all", "dora_r8_qv"]


def item_correct(r):
    gold = r["tgt"][0].strip().upper()
    choice = parse_choice(r["pred"])
    if choice is not None:
        return choice == gold, choice == gold
    opts = options(r["src"][0])
    pred = norm(r["pred"])
    scores = {k: overlap(pred, v) for k, v in opts.items()}
    if not opts or max(scores.values()) <= 0:
        return False, False
    return False, max(opts, key=lambda k: scores[k]) == gold


def reciprocal_rank(r):
    gold = r["tgt"][0].strip().upper()
    choice = parse_choice(r["pred"])
    if choice is not None:
        return 1.0 if choice == gold else 0.0
    opts = options(r["src"][0])
    pred = norm(r["pred"])
    scores = {k: overlap(pred, v) for k, v in opts.items()}
    gold_score = scores.get(gold, 0.0)
    if gold_score <= 0:
        return 0.0
    return 1.0 / (1 + sum(v > gold_score + 1e-12 for v in scores.values()))


def summarize(path):
    preds = load(path)
    if not preds:
        return None
    strict = [item_correct(r)[0] for r in preds]
    content = [item_correct(r)[1] for r in preds]
    unparsed = sum(parse_choice(r["pred"]) is None for r in preds)
    mrr = sum(reciprocal_rank(r) for r in preds) / len(preds)
    return {
        "n": len(preds),
        "strict_correct": sum(strict),
        "strict_acc": sum(strict) / len(preds),
        "content_correct": sum(content),
        "content_acc": sum(content) / len(preds),
        "unparsed": unparsed,
        "mrr": mrr,
        "_strict": strict,
        "_content": content,
    }


def paired_bootstrap(base, other, n_boot, seed=0):
    n = len(base["_strict"])
    rng = random.Random(seed)
    ds, dc = [], []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        ds.append(sum(base["_strict"][i] - other["_strict"][i] for i in idx) / n)
        dc.append(sum(base["_content"][i] - other["_content"][i] for i in idx) / n)
    ds.sort()
    dc.sort()
    lo, hi = int(0.025 * n_boot), int(0.975 * n_boot) - 1
    return ds[lo], ds[hi], dc[lo], dc[hi]


LABELS = {
    "base": "Base",
    "lora_r8_qv": "L8-QV",
    "lora_r16_all_lr1e4": "L16-1e4",
    "lora_r16_all": "L16-2e4",
    "lora_r32_all": "L32-ALL",
    "dora_r16_all": "DoRA-16",
    "dora_r8_qv": "DoRA-8",
}
COLORS = {"train": "#9e9e9e", "generic": "#4c72b0", "constraint": "#55a868"}
NAMES = {"train": "training prompt", "generic": "official short prompt",
         "constraint": "guide constraint prompt"}


def plot_protocols(summary, protocols, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [label for label, _ in protocols if label in summary]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9),
                             gridspec_kw={"width_ratios": [2.5, 1]})

    ax = axes[0]
    tags = [t for t in ORDER if t in summary.get("generic", {})]
    x = np.arange(len(tags))
    width = 0.38
    for i, label in enumerate([p for p in ("train", "generic") if p in summary]):
        vals = [summary[label].get(t, {}).get("strict_acc", 0) * 100 for t in tags]
        ax.bar(x + (i - 0.5) * width, vals, width, color=COLORS[label],
               label=NAMES[label])
        for xi, v in zip(x + (i - 0.5) * width, vals):
            ax.text(xi, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=6.2)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[t] for t in tags], fontsize=7, rotation=12, ha="right")
    ax.set_ylabel("CMMLU strict accuracy (%)", fontsize=8)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)

    ax = axes[1]
    tags = [t for t in ("base", "lora_r8_qv") if t in summary.get("generic", {})]
    x = np.arange(len(tags))
    width = 0.26
    for i, label in enumerate(labels):
        vals = [summary[label].get(t, {}).get("strict_acc", 0) * 100 for t in tags]
        ax.bar(x + (i - 1) * width, vals, width, color=COLORS[label])
        for xi, v in zip(x + (i - 1) * width, vals):
            ax.text(xi, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=6.2)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[t] for t in tags], fontsize=7)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[label])
               for label in labels]
    fig.legend(handles, [NAMES[label] for label in labels], fontsize=7,
               frameon=False, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", action="append", required=True,
                    help="LABEL=DIR, repeatable")
    ap.add_argument("--out", default="results/cmmlu_protocols.csv")
    ap.add_argument("--bootstrap-out", default="results/cmmlu_bootstrap.csv")
    ap.add_argument("--fig", default=None, help="optional grouped-bar figure path")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    protocols = []
    for spec in args.protocol:
        label, directory = spec.split("=", 1)
        protocols.append((label, directory))

    rows = []
    summary = {}
    for label, directory in protocols:
        summary[label] = {}
        for tag in ORDER:
            rec = summarize(os.path.join(directory, f"pred_{tag}_cmmlu.jsonl"))
            if rec is None:
                continue
            summary[label][tag] = rec
            rows.append({
                "protocol": label,
                "tag": tag,
                **{k: round(v, 4) if isinstance(v, float) else v
                   for k, v in rec.items() if not k.startswith("_")},
            })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    boot_rows = []
    for label, _ in protocols:
        if "base" not in summary[label]:
            continue
        base = summary[label]["base"]
        for tag in ORDER[1:]:
            if tag not in summary[label]:
                continue
            lo_s, hi_s, lo_c, hi_c = paired_bootstrap(
                base, summary[label][tag], args.n_boot)
            boot_rows.append({
                "protocol": label,
                "comparison": f"base-{tag}",
                "delta_strict": round(base["strict_acc"] - summary[label][tag]["strict_acc"], 4),
                "strict_lo": round(lo_s, 4),
                "strict_hi": round(hi_s, 4),
                "delta_content": round(base["content_acc"] - summary[label][tag]["content_acc"], 4),
                "content_lo": round(lo_c, 4),
                "content_hi": round(hi_c, 4),
            })

    with open(args.bootstrap_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(boot_rows[0]))
        writer.writeheader()
        writer.writerows(boot_rows)

    for row in rows:
        print(row)
    print()
    for row in boot_rows:
        print(row)

    if args.fig:
        plot_protocols(summary, protocols, args.fig)


if __name__ == "__main__":
    main()
