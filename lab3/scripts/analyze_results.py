#!/usr/bin/env python3
"""Aggregate evaluation metrics, bootstrap comparisons, tables and figures."""
import argparse
import csv
import json
import os
import random
import re
import statistics

import jieba
import matplotlib

_jieba_cache = os.path.join(os.path.expanduser("~"), ".cache", "jieba")
os.makedirs(_jieba_cache, exist_ok=True)
jieba.dt.tmp_dir = _jieba_cache

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rouge_chinese import Rouge

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ORDER = [
    "base",
    "lora_r8_qv",
    "lora_r16_all_lr1e4",
    "lora_r16_all",
    "lora_r32_all",
    "dora_r8_qv",
    "dora_r16_all",
]
LABELS = {
    "base": "基座模型",
    "lora_r8_qv": "LoRA r=8 q,v",
    "lora_r16_all_lr1e4": "LoRA r=16 all (1e-4)",
    "lora_r16_all": "LoRA r=16 all (2e-4)",
    "lora_r32_all": "LoRA r=32 all",
    "dora_r8_qv": "DoRA r=8 q,v",
    "dora_r16_all": "DoRA r=16 all",
}
COLORS = ["#8a8a8a", "#4c72b0", "#7aa6d8", "#2f5d8f", "#1b3a5c",
          "#55a868", "#c44e52"]


def first(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else ""
    return value or ""


def pred_text(item):
    return item.get("pred") or ""


def ref_text(item):
    return first(item.get("tgt"))


def src_text(item):
    return first(item.get("src"))


def load_med500(results500_dir, tag):
    """Medical metrics from the 500-sample run, or {} when it was not done."""
    if not results500_dir:
        return {}
    path = os.path.join(results500_dir, f"metrics_{tag}.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        med = json.load(f).get("medical", {})
    return med if med.get("n", 0) >= 500 else {}


def load_metrics(results_dir, results500_dir=None):
    rows = []
    for tag in ORDER:
        path = os.path.join(results_dir, f"metrics_{tag}.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        med = m.get("medical", {})
        first100 = med.get("first100", med)
        cm = m.get("cmmlu", {})
        med500 = med if med.get("n", 0) >= 500 else load_med500(results500_dir, tag)
        rows.append({
            "tag": tag,
            "label": LABELS.get(tag, tag),
            "r1_100": first100.get("rouge-1"),
            "r2_100": first100.get("rouge-2"),
            "rl_100": first100.get("rouge-l"),
            "r1p_100": first100.get("rouge-1_p"),
            "r1r_100": first100.get("rouge-1_r"),
            "r2p_100": first100.get("rouge-2_p"),
            "r2r_100": first100.get("rouge-2_r"),
            "rlp_100": first100.get("rouge-l_p"),
            "rlr_100": first100.get("rouge-l_r"),
            "r1_500": med500.get("rouge-1"),
            "r2_500": med500.get("rouge-2"),
            "rl_500": med500.get("rouge-l"),
            "acc": cm.get("accuracy"),
            "cmmlu_correct": cm.get("correct"),
            "cmmlu_n": cm.get("n"),
            "cmmlu_unparsed": cm.get("unparsed"),
            "contains_gold_rate": cm.get("contains_gold_rate"),
            "med_n": med.get("n"),
            "empty": med.get("empty"),
            "med_elapsed_s": med.get("elapsed_s"),
            "med_samples_per_s": med.get("samples_per_s"),
            "cm_elapsed_s": cm.get("elapsed_s"),
        })
    return rows


def read_preds(results_dir, tag, dataset):
    path = os.path.join(results_dir, f"pred_{tag}_{dataset}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _doc_scores(rouge, hyps, refs):
    per = rouge.get_scores(hyps, refs, avg=False)
    return {k: np.array([p[k]["f"] for p in per]) for k in ("rouge-1", "rouge-2", "rouge-l")}


def _bootstrap_indices(rng, n, n_boot):
    return np.array([[rng.randrange(n) for _ in range(n)] for _ in range(n_boot)])


def bootstrap_rouge(preds, n_boot=500, seed=0, subset=None):
    subset = list(range(len(preds))) if subset is None else subset
    refs = [" ".join(jieba.cut(ref_text(preds[i]))) or "。" for i in subset]
    hyps = [" ".join(jieba.cut(pred_text(preds[i]))) or "。" for i in subset]
    per = _doc_scores(Rouge(), hyps, refs)
    idxs = _bootstrap_indices(random.Random(seed), len(refs), n_boot)
    return {k: (float(np.percentile(v[idxs].mean(axis=1), 2.5)),
                float(np.percentile(v[idxs].mean(axis=1), 97.5)))
            for k, v in per.items()}


def bootstrap_delta(pred_a, pred_b, key, n_boot=500, seed=0):
    n = min(len(pred_a), len(pred_b))
    rng = random.Random(seed)
    if key == "correct":
        a = [1.0 if pred_a[i].get("correct") else 0.0 for i in range(n)]
        b = [1.0 if pred_b[i].get("correct") else 0.0 for i in range(n)]
        diffs = np.array([a[i] - b[i] for i in range(n)])
        idxs = _bootstrap_indices(rng, n, n_boot)
        deltas = diffs[idxs].mean(axis=1)
    else:
        refs = [" ".join(jieba.cut(ref_text(pred_a[i]))) or "。" for i in range(n)]
        hyps_a = [" ".join(jieba.cut(pred_text(pred_a[i]))) or "。" for i in range(n)]
        hyps_b = [" ".join(jieba.cut(pred_text(pred_b[i]))) or "。" for i in range(n)]
        rouge = Rouge()
        per_a = _doc_scores(rouge, hyps_a, refs)
        per_b = _doc_scores(rouge, hyps_b, refs)
        deltas = per_a[key] - per_b[key]
        idxs = _bootstrap_indices(rng, n, n_boot)
        deltas = deltas[idxs].mean(axis=1)
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def add_bootstrap(rows, results_dir, n_boot):
    for r in rows:
        med = read_preds(results_dir, r["tag"], "medical")
        cm = read_preds(results_dir, r["tag"], "cmmlu")
        if med and r["rl_100"] is not None:
            ci = bootstrap_rouge(med, n_boot=n_boot, subset=list(range(min(100, len(med)))))
            r["r1_ci"], r["rl_ci"] = ci["rouge-1"], ci["rouge-l"]
        if cm:
            rng = random.Random(1)
            acc = [1.0 if p.get("correct") else 0.0 for p in cm]
            vals = [sum(acc[rng.randrange(len(acc))] for _ in range(len(acc))) / len(acc)
                    for _ in range(n_boot)]
            r["acc_ci"] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    return rows


def paired_comparisons(rows, results_dir, n_boot):
    tags = {r["tag"] for r in rows}
    pairs = [
        ("lora_r8_qv", "base"),
        ("lora_r16_all_lr1e4", "lora_r8_qv"),
        ("lora_r16_all", "lora_r16_all_lr1e4"),
        ("lora_r32_all", "lora_r16_all"),
        ("lora_r32_all", "lora_r8_qv"),
        ("dora_r8_qv", "lora_r8_qv"),
        ("dora_r8_qv", "base"),
        ("dora_r16_all", "lora_r16_all"),
        ("dora_r16_all", "lora_r8_qv"),
        ("dora_r16_all", "base"),
    ]
    out = []
    for a, b in pairs:
        if a not in tags or b not in tags:
            continue
        pa_med, pb_med = read_preds(results_dir, a, "medical"), read_preds(results_dir, b, "medical")
        pa_cm, pb_cm = read_preds(results_dir, a, "cmmlu"), read_preds(results_dir, b, "cmmlu")
        rec = {"a": a, "b": b}
        if pa_med and pb_med:
            rec["d_r1_100"] = bootstrap_delta(pa_med, pb_med, "rouge-1", n_boot)
            rec["d_rl_100"] = bootstrap_delta(pa_med, pb_med, "rouge-l", n_boot)
        if pa_cm and pb_cm:
            rec["d_acc"] = bootstrap_delta(pa_cm, pb_cm, "correct", n_boot)
        out.append(rec)
    with open(os.path.join(results_dir, "paired_bootstrap.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "metric", "delta_mean", "ci_lo", "ci_hi"])
        for rec in out:
            for key in ("d_r1_100", "d_rl_100", "d_acc"):
                if key in rec:
                    m, lo, hi = rec[key]
                    w.writerow([rec["a"], rec["b"], key, f"{m:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
                    print(f"{rec['a']:22s} vs {rec['b']:22s} {key:8s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    return out


def style_stats(preds):
    hyps = [pred_text(p) for p in preds if pred_text(p).strip()]
    if not hyps:
        return {}
    lengths = [len(h) for h in hyps]

    def is_list(h):
        return bool(re.search(r"(?m)^\s*(?:[-*•]|\d+[.、)])\s+", h))

    def repeat_4gram(h):
        text = "".join(h.split())
        grams = [text[i:i + 4] for i in range(max(0, len(text) - 3))]
        return 1.0 - len(set(grams)) / len(grams) if grams else 0.0

    return {
        "avg_chars": statistics.mean(lengths),
        "list_rate": sum(is_list(h) for h in hyps) / len(hyps),
        "question_rate": sum(("？" in h or "?" in h) for h in hyps) / len(hyps),
        "caution_rate": sum(any(w in h for w in ("就医", "医生", "医院", "及时", "建议")) for h in hyps) / len(hyps),
        "repeat_4gram": statistics.mean(repeat_4gram(h) for h in hyps),
    }


def add_style(rows, results_dir):
    for r in rows:
        med = read_preds(results_dir, r["tag"], "medical")[:100]
        r.update(style_stats(med))
    return rows


def add_interaction(rows, results_dir):
    for r in rows:
        path = os.path.join(results_dir, f"gen_test_{r['tag']}.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            g = json.load(f)
        # main table reports the five medical prompts, excluding the short general-knowledge one
        results = [x for x in g.get("results", []) if x.get("idx", 99) < 5]
        lat = [x["latency_s"] for x in results]
        new_tokens = [x["new_tokens"] for x in results]
        r["latency_s"] = statistics.mean(lat) if lat else g.get("avg_latency_s")
        r["latency_median_s"] = statistics.median(lat) if lat else None
        r["latency_first_s"] = lat[0] if lat else None
        r["tokens_per_s"] = sum(new_tokens) / sum(lat) if lat else g.get("avg_tokens_per_s")
    return rows


def write_examples(rows, results_dir, path):
    tags = {r["tag"] for r in rows}
    if "base" not in tags or "lora_r16_all" not in tags:
        return
    base = read_preds(results_dir, "base", "medical")
    lora = read_preds(results_dir, "lora_r16_all", "medical")
    dora = read_preds(results_dir, "dora_r16_all", "medical")
    n = min(len(base), len(lora))
    if n == 0:
        return
    rouge = Rouge()

    def f1(text, ref):
        hyp = " ".join(jieba.cut(text)) or "。"
        ref_tok = " ".join(jieba.cut(ref)) or "。"
        return rouge.get_scores(hyp, ref_tok)[0]["rouge-l"]["f"]

    gains = sorted(range(n), key=lambda i: f1(pred_text(lora[i]), ref_text(base[i]))
                   - f1(pred_text(base[i]), ref_text(base[i])), reverse=True)
    losses = sorted(range(n), key=lambda i: f1(pred_text(lora[i]), ref_text(base[i]))
                    - f1(pred_text(base[i]), ref_text(base[i])))
    picks = gains[:3] + losses[:1]
    with open(path, "w", encoding="utf-8") as f:
        for i in picks:
            f.write(f"## idx {i}  base_rl={f1(pred_text(base[i]), ref_text(base[i])):.3f} "
                    f"lora_rl={f1(pred_text(lora[i]), ref_text(base[i])):.3f}"
                    + (f" dora_rl={f1(pred_text(dora[i]), ref_text(base[i])):.3f}" if len(dora) > i else "")
                    + "\n")
            f.write(f"- src: {src_text(base[i])}\n- ref: {ref_text(base[i])}\n")
            f.write(f"- base: {pred_text(base[i])}\n- lora: {pred_text(lora[i])}\n")
            if len(dora) > i:
                f.write(f"- dora: {pred_text(dora[i])}\n")
            f.write("\n")


def write_csv(rows, path):
    fields = [
        "tag", "label", "r1_100", "r2_100", "rl_100",
        "r1p_100", "r1r_100", "r2p_100", "r2r_100", "rlp_100", "rlr_100",
        "r1_500", "r2_500", "rl_500", "acc",
        "cmmlu_correct", "cmmlu_n", "cmmlu_unparsed", "contains_gold_rate",
        "med_n", "empty", "avg_chars", "list_rate", "question_rate", "caution_rate",
        "repeat_4gram", "latency_s", "tokens_per_s",
        "r1_ci_lo", "r1_ci_hi", "rl_ci_lo", "rl_ci_hi", "acc_ci_lo", "acc_ci_hi",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            rec = {k: r.get(k) for k in fields}
            if r.get("r1_ci"):
                rec["r1_ci_lo"], rec["r1_ci_hi"] = r["r1_ci"]
            if r.get("rl_ci"):
                rec["rl_ci_lo"], rec["rl_ci_hi"] = r["rl_ci"]
            if r.get("acc_ci"):
                rec["acc_ci_lo"], rec["acc_ci_hi"] = r["acc_ci"]
            w.writerow(rec)

    print(f"{'tag':22s} {'R1@100':>7s} {'R2@100':>7s} {'RL@100':>7s} "
          f"{'R1@500':>7s} {'RL@500':>7s} {'CMMLU':>7s} {'Acc':>4s}")
    for r in rows:
        def fmt(x, nd=4):
            return "—" if x is None else f"{x:.{nd}f}"
        print(f"{r['tag']:22s} {fmt(r['r1_100']):>7s} {fmt(r['r2_100']):>7s} {fmt(r['rl_100']):>7s} "
              f"{fmt(r['r1_500']):>7s} {fmt(r['rl_500']):>7s} {fmt(r['acc']):>7s} "
              f"{fmt(r.get('cmmlu_correct'), 0):>4s}")


def plot_rouge(rows, path):
    rows = [r for r in rows if r["r1_100"] is not None]
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    for i, (key, name) in enumerate([("r1_100", "ROUGE-1"), ("r2_100", "ROUGE-2"), ("rl_100", "ROUGE-L")]):
        vals = [r[key] for r in rows]
        ax.bar(x + (i - 1) * width, vals, width, label=name)
        for xi, v in zip(x + (i - 1) * width, vals):
            ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=12, ha="right")
    ax.set_ylabel("F1 (first 100)")
    ax.set_ylim(0, max(max(r[k] for r in rows for k in ("r1_100", "r2_100", "rl_100")) * 1.28, 0.1))
    ax.legend(ncol=3, frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_cmmlu(rows, path):
    rows = [r for r in rows if r["acc"] is not None]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(rows))
    vals = [r["acc"] for r in rows]
    ax.bar(x, vals, 0.55, color=[COLORS[ORDER.index(r["tag"])] for r in rows])
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], fontsize=8, rotation=12, ha="right")
    ax.set_ylabel("CMMLU computer_science accuracy")
    ax.set_ylim(0, min(1.0, max(vals) * 1.25))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_latency(rows, path):
    rows = [r for r in rows if r.get("latency_s")]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(rows))
    vals = [r["latency_s"] for r in rows]
    ax.bar(x, vals, 0.55, color=[COLORS[ORDER.index(r["tag"])] for r in rows])
    for xi, v, r in zip(x, vals, rows):
        ax.text(xi, v + 0.3, f"{v:.1f}s\n{r['tokens_per_s']:.1f} tok/s",
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], fontsize=8, rotation=12, ha="right")
    ax.set_ylabel("avg response time (s)")
    ax.set_ylim(0, max(vals) * 1.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_loss(results_dir, path):
    curves = {}
    for tag in ORDER:
        p = os.path.join(results_dir, f"state_{tag}.json")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            state = json.load(f)
        pts = [(h["step"], h["loss"]) for h in state.get("log_history", []) if "loss" in h]
        if pts:
            curves[tag] = pts
    if not curves:
        print("no trainer_state files, skip loss figure")
        return
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    for tag, pts in curves.items():
        xs, ys = zip(*pts)
        color = COLORS[ORDER.index(tag)]
        ax.plot(xs, ys, color=color, alpha=0.22, lw=0.8)
        if len(ys) >= 5:
            k = max(2, len(ys) // 30)
            smooth = np.convolve(ys, np.ones(k) / k, mode="valid")
            ax.plot(xs[k - 1:], smooth, color=color, lw=1.8, label=LABELS.get(tag, tag))
    ax.set_xlabel("step")
    ax.set_ylabel("training loss")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--results500", default="results/official500",
                    help="directory with the 500-sample medical metrics; skipped when absent")
    ap.add_argument("--figs", default="report/figs")
    ap.add_argument("--bootstrap", type=int, default=500, help="bootstrap resamples (0 disables)")
    args = ap.parse_args()
    os.makedirs(args.figs, exist_ok=True)
    rows = load_metrics(args.results, args.results500)
    if not rows:
        raise SystemExit("no metrics files found")
    rows = add_style(rows, args.results)
    rows = add_interaction(rows, args.results)
    if args.bootstrap > 0:
        rows = add_bootstrap(rows, args.results, args.bootstrap)
        paired_comparisons(rows, args.results, args.bootstrap)
        write_examples(rows, args.results, os.path.join(args.results, "examples.md"))
    write_csv(rows, os.path.join(args.results, "summary.csv"))
    plot_rouge(rows, os.path.join(args.figs, "rouge_compare.png"))
    plot_cmmlu(rows, os.path.join(args.figs, "cmmlu_compare.png"))
    plot_latency(rows, os.path.join(args.figs, "latency_compare.png"))
    plot_loss(args.results, os.path.join(args.figs, "loss_curves.png"))
    print("figures ->", args.figs)


if __name__ == "__main__":
    main()
