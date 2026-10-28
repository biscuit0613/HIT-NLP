"""生成实验二报告用图。

用法: python scripts/plot_report_figs.py
输入: results/curves.csv（parse_logs.py 生成）、results/bleu_eval.csv
输出: report/figs/{loss_curves,bleu_seed}.png
本地性分析图由 analyze_localness.py 生成后复制到 report/figs/。
"""
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "report" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

GROUPS = {
    "base": ("base_s", "#4c72b0"),
    "loconly6": ("loconly6_s", "#55a868"),
    "locrel6": ("locrel6_s", "#c44e52"),
}
GROUP_LABELS = {"base": "基线", "loconly6": "局部性×6",
                "locrel6": "相对位置 + 局部性×6"}
SEEDS = [23, 42, 7, 123]


def ema(values, alpha=0.06):
    out, v = [], values[0]
    for x in values:
        v = alpha * x + (1 - alpha) * v
        out.append(v)
    return out


def load_curves():
    data = {}
    with open(RESULTS / "curves.csv") as f:
        for row in csv.DictReader(f):
            data.setdefault(row["run"], {"step": [], "loss": []})
            data[row["run"]]["step"].append(int(row["step"]))
            data[row["run"]]["loss"].append(float(row["loss"]))
    return data


def group_runs(data, prefix):
    return sorted(r for r in data
                  if r.startswith(prefix) and r[len(prefix):].isdigit())


def seed_of(run):
    return int(run.rsplit("_s", 1)[1])


def plot_loss(data):
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for group, (prefix, color) in GROUPS.items():
        runs = group_runs(data, prefix)
        if not runs:
            continue
        n = min(len(data[r]["step"]) for r in runs)
        steps = data[runs[0]]["step"][:n]
        losses = np.array([data[r]["loss"][:n] for r in runs])
        mean = losses.mean(axis=0)
        ax.plot(steps, ema(mean), color=color, lw=1.4,
                label=f"{GROUP_LABELS[group]}（{len(runs)} 个种子均值）")
    ax.set_xlabel("训练步数 / iteration")
    ax.set_ylabel("平均交叉熵损失")
    ax.set_xlim(0, steps[-1])
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "loss_curves.png", dpi=200)
    plt.close(fig)
    print("wrote loss_curves.png")


def load_bleu():
    table = {}
    with open(RESULTS / "bleu_eval.csv") as f:
        for row in csv.DictReader(f):
            if row["protocol"] != "aligned":
                continue
            table.setdefault(row["run"], {})[int(row["year"])] = float(
                row["bleu"])
    return table


def plot_seed_bleu(table):
    groups = list(GROUPS)
    seeds = [s for s in SEEDS
             if all(f"{prefix}{s}" in table for prefix, _ in GROUPS.values())]
    if not seeds:
        print("no grid bleu data, skip bleu_seed.png")
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=False)
    rng = np.random.default_rng(0)
    width = 0.55
    for ax, year in zip(axes, (2017, 2018)):
        levels = [table[f"{prefix}{s}"][year]
                  for prefix, _ in GROUPS.values() for s in seeds]
        for gi, group in enumerate(groups):
            prefix, color = GROUPS[group]
            vals = np.array([table[f"{prefix}{s}"][year] for s in seeds])
            x = gi + rng.uniform(-0.10, 0.10, size=len(seeds))
            ax.bar(gi, vals.mean(), width, color=color, alpha=0.30,
                   edgecolor=color, linewidth=1.0)
            ax.errorbar(gi, vals.mean(), yerr=vals.std(ddof=1),
                        color=color, capsize=4, lw=1.2)
            ax.scatter(x, vals, s=16, color=color, zorder=3,
                       label="种子" if gi == 0 and year == 2017 else None)
            ax.text(gi, vals.mean() + vals.std(ddof=1) + 0.08,
                    f"{vals.mean():.2f}", ha="center", fontsize=8,
                    color=color)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([GROUP_LABELS[g] for g in groups], fontsize=8)
        ax.set_title(f"newstest{year}" + ("（对齐口径）" if year == 2018 else ""))
        ax.set_ylabel("BLEU")
        ax.set_ylim(min(levels) - 0.6, max(levels) + 0.8)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
    axes[0].scatter([], [], s=16, color="k", label=f"单次运行（{len(seeds)} 个种子）")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / "bleu_seed.png", dpi=200)
    plt.close(fig)
    print("wrote bleu_seed.png", seeds)


if __name__ == "__main__":
    plot_loss(load_curves())
    plot_seed_bleu(load_bleu())
