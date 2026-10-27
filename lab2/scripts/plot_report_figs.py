"""生成实验二报告用图。

用法: python scripts/plot_report_figs.py
输入: results/curves.csv（parse_logs.py 生成）、results/**/bleu_*.log、
      results/improve/relpos/rel_pos_bias.npy
输出: report/figs/{loss_curves,bleu_compare,rel_pos_bias}.png
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

RUN_COLORS = {
    "baseline": "#1f77b4", "relpos": "#d62728", "lr3": "#2ca02c",
    "warmup4k": "#9467bd", "ls05": "#ff7f0e", "combo": "#8c564b",
    "base_s23": "#17becf", "relpos_s23": "#bcbd22", "beam10": "#7f7f7f",
}


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


def plot_loss(data):
    # 仅绘制标签平滑同为 0.1 的 run，combo/ls05 的损失与基线不可直接比较。
    runs = [r for r in ["baseline", "relpos", "warmup4k"] if r in data]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for run in runs:
        if run not in data:
            continue
        d = data[run]
        ax.plot(d["step"], d["loss"], color=RUN_COLORS[run], alpha=0.18, lw=0.7)
        ax.plot(d["step"], ema(d["loss"]), color=RUN_COLORS[run], lw=1.4,
                label=run)
    ax.set_xlabel("训练步数 / iteration")
    ax.set_ylabel("平均交叉熵损失")
    ax.set_xlim(0, max(max(d["step"]) for d in data.values()))
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "loss_curves.png", dpi=200)
    plt.close(fig)
    print("wrote loss_curves.png")


BLEU_PAT = re.compile(r"BLEU = ([\d.]+)")


def load_aligned_2018():
    """读取 eval_aligned_bleu.py 的对齐口径结果，返回 {run: bleu}。"""
    p = RESULTS / "bleu_eval.csv"
    out = {}
    if not p.exists():
        return out
    with open(p) as f:
        for row in csv.DictReader(f):
            if row["year"] == "2018" and row["protocol"] == "aligned":
                out[row["run"]] = float(row["bleu"])
    return out


def collect_bleu():
    """扫描 results 下的 BLEU 日志，返回 {(run, year): bleu}。"""
    out = {}
    for log in RESULTS.rglob("*.log"):
        if not log.name.startswith("bleu_") or log.name.endswith("_aligned.log"):
            continue
        m = BLEU_PAT.search(log.read_text(errors="ignore"))
        if not m:
            continue
        year = re.search(r"(201[78])", log.name)
        year = year.group(1) if year else "?"
        # run 名：bleu_<run>_<year>.log；bleu_<year>.log 则取所属 run 目录名
        stem = log.stem
        if stem in (f"bleu_{year}", "bleu_2017", "bleu_2018"):
            run = log.parent.parent.name
        else:
            run = stem[len("bleu_"):].rsplit("_", 1)[0]
        out[(run, year)] = float(m.group(1))
    return out


def plot_bleu(bleu):
    bleu.update({(run, "2018"): v for run, v in load_aligned_2018().items()})
    order = ["baseline", "lr3", "warmup4k", "ls05", "combo",
             "base_s23", "beam10", "relpos", "relpos_s23"]
    order = [r for r in order if (r, "2018") in bleu]
    if not order:
        print("no bleu data, skip")
        return
    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    v17 = [bleu.get((r, "2017"), np.nan) for r in order]
    v18 = [bleu.get((r, "2018"), np.nan) for r in order]
    b1 = ax.bar(x - w / 2, v17, w, label="newstest2017", color="#4c72b0")
    b2 = ax.bar(x + w / 2, v18, w, label="newstest2018（对齐）", color="#dd8452")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            if not np.isnan(h):
                ax.text(b.get_x() + b.get_width() / 2, h + 0.06,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=8)
    ax.set_ylabel("BLEU")
    ax.set_ylim(0, 13.2)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.legend([b1, b2], ["newstest2017", "newstest2018（对齐）"],
              frameon=False, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    fig.savefig(FIGS / "bleu_compare.png", dpi=200)
    plt.close(fig)
    print("wrote bleu_compare.png:", {k: v for k, v in bleu.items()})


def plot_bias():
    p = RESULTS / "improve" / "relpos" / "rel_pos_bias.npy"
    if not p.exists():
        print("no bias npy, skip")
        return
    arr = np.load(p)  # [layers, buckets, heads]
    layers, buckets, heads = arr.shape
    # bucket 0..15 为非负距离（0..128），bucket 17..31 为负距离（-1..-128）；
    # bucket 16 不会被任何相对距离命中，绘图时剔除。按代表距离重排 x 轴。
    pos = [b if b < 8 else 8 * 16 ** ((b - 8) / 8) for b in range(16)]
    pos[-1] = 128.0
    signed = [(pos[b], b) for b in range(16)] + \
             [(-pos[b - 16], b) for b in range(17, 32)]
    signed.sort(key=lambda t: t[0])
    x = np.array([d for d, _ in signed])
    idx = [b for _, b in signed]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.6), sharex=True, sharey=True)
    for li, ax in enumerate(axes.flat):
        vals = arr[li][idx]
        for h in range(heads):
            ax.plot(x, vals[:, h], lw=0.8, alpha=0.75)
        ax.axvline(0, color="k", lw=0.6, ls=":")
        ax.set_title(f"第 {li} 层", fontsize=8)
        ax.grid(alpha=0.2, lw=0.4)
        ax.set_xscale("symlog", linthresh=8)
        ax.set_xticks([-128, -32, -8, -1, 0, 1, 8, 32, 128])
        ax.set_xticklabels(["-128", "-32", "-8", "-1", "0", "1", "8", "32", "128"],
                           fontsize=6.5)
    for ax in axes[1]:
        ax.set_xlabel("相对距离（← 反向 | 正向 →）", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("偏置值", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "rel_pos_bias.png", dpi=200)
    plt.close(fig)
    print("wrote rel_pos_bias.png", arr.shape)


if __name__ == "__main__":
    data = load_curves()
    plot_loss(data)
    plot_bias()
    plot_bleu(collect_bleu())
