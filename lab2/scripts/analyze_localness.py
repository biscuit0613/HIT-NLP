"""Inspect the learned localness centers/widths of a trained checkpoint.

Loads a checkpoint into the improved model, runs the encoder on a few
newstest sentences, and reports per-layer/per-head Gaussian statistics
together with a heatmap of the additive bias for one example.

Usage:
  python scripts/analyze_localness.py --params /tmp/transformer.pdparams \
      --src data/wmt171819_zhen_data_bpe3/newstest2017.tok.bpe.32000.en \
      --vocab data/wmt171819_zhen_data_bpe3/vocab_all.bpe.32000 \
      --out results/improve/localness_stats.json \
      --fig results/improve/localness_bias.png \
      --use-rel-pos True --use-std-attn-scale False --localness-layers 3
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import paddle
import paddle.nn.functional as F

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "improve"))
from transformer import TransformerModel  # noqa: E402


def load_vocab(path):
    vocab = {}
    with open(path, "rb") as f:
        for idx, line in enumerate(f):
            vocab[line.strip(b"\n")] = idx
    return vocab


def str2bool(value):
    return str(value).lower() in ("1", "true", "yes")


def build_model(args, vocab_size):
    return TransformerModel(
        src_vocab_size=vocab_size, trg_vocab_size=vocab_size,
        max_length=256,
        n_layer=6, n_head=8, d_key=64, d_value=64, d_model=512,
        d_inner_hid=2048, prepostprocess_dropout=0.1, attention_dropout=0.1,
        relu_dropout=0.1, preprocess_cmd="n", postprocess_cmd="da",
        weight_sharing=True, label_smooth_eps=0.1, bos_idx=0,
        use_rel_pos=args.use_rel_pos, use_localness=True,
        localness_layers=args.localness_layers,
        use_std_attn_scale=args.use_std_attn_scale)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    parser.add_argument("--src", default=str(
        ROOT / "data/wmt171819_zhen_data_bpe3/"
        "newstest2017.tok.bpe.32000.en"))
    parser.add_argument("--vocab", default=str(
        ROOT / "data/wmt171819_zhen_data_bpe3/vocab_all.bpe.32000"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--fig", required=True)
    parser.add_argument("--n-sentences", type=int, default=200)
    parser.add_argument("--example", type=int, default=10)
    parser.add_argument("--use-rel-pos", type=str2bool, default=True)
    parser.add_argument("--use-std-attn-scale", type=str2bool, default=False)
    parser.add_argument("--localness-layers", type=int, default=6)
    args = parser.parse_args()

    paddle.set_device("cpu")
    vocab = load_vocab(args.vocab)
    eos = vocab[b"<e>"]
    unk = vocab[b"<unk>"]
    lines = Path(args.src).read_text().splitlines()[:args.n_sentences]
    seqs = [[vocab.get(tok.encode("utf-8"), unk) for tok in line.split()]
            + [eos] for line in lines]
    max_len = max(len(s) for s in seqs)
    src_word = np.full((len(seqs), max_len), eos, dtype="int64")
    for i, seq in enumerate(seqs):
        src_word[i, :len(seq)] = seq
    src_pos = np.tile(np.arange(max_len, dtype="int64"), (len(seqs), 1))
    src_bias = np.zeros((len(seqs), 1, 1, max_len), dtype="float32")
    for i, seq in enumerate(seqs):
        src_bias[i, :, :, len(seq):] = -1e9

    model = build_model(args, len(vocab))
    state = paddle.load(args.params)
    model.set_state_dict(state)
    model.eval()

    captured = {}

    def make_hook(name):
        def hook(module, inputs, output):
            query, lengths = inputs
            hidden = paddle.tanh(module.w(query))
            length = lengths.reshape([-1, 1, 1])
            center = length * F.sigmoid(module.u_center(hidden))
            width = length * F.sigmoid(module.u_width(hidden))
            captured[name] = {
                "center": center.numpy(),
                "width": width.numpy(),
                "scale": module.scale.numpy(),
                "lengths": lengths.numpy(),
            }
        return hook

    for idx, layer in enumerate(model.encoder.layers):
        if layer.localness is not None:
            layer.localness.register_forward_post_hook(
                make_hook(f"layer{idx}"))

    with paddle.no_grad():
        model.encode(
            paddle.to_tensor(src_word), paddle.to_tensor(src_pos),
            paddle.to_tensor(src_bias))

    lengths = np.array([len(s) for s in seqs], dtype="float32")
    stats = {}
    for name, cap in captured.items():
        center = cap["center"]  # [B, T, H]
        width = cap["width"]
        lens = cap["lengths"]
        valid = np.arange(max_len)[None, :] < lens[:, None]
        pos = np.broadcast_to(
            np.arange(max_len)[None, :, None], center.shape)
        offset = (center - pos) / lens[:, None, None]
        valid3 = valid[:, :, None]
        denom = valid.sum()
        stats[name] = {
            "mean_abs_center_offset_ratio": float(
                np.abs(offset)[valid].mean()),
            "mean_width_ratio": float(
                (width / lens[:, None, None])[valid].mean()),
            "per_head_scale": [float(x) for x in cap["scale"]],
            "per_head_width_ratio": [
                float(x) for x in
                ((width / lens[:, None, None]) * valid3).sum(
                    axis=(0, 1)) / denom],
            "per_head_abs_center_offset_ratio": [
                float(x) for x in
                (np.abs(offset) * valid3).sum(axis=(0, 1)) / denom],
        }

    # Example Gaussian bias for three evenly spaced localness layers, using
    # the sentence closest to the median length and, per layer, the head
    # closest to that layer's median window width.
    ex = int(np.argmin(np.abs(lengths - np.median(lengths))))
    ex_len = int(lengths[ex])
    names = sorted(captured, key=lambda s: int(s[5:]))
    show = sorted({round(k * (len(names) - 1) / 2) for k in range(3)})
    fig, axes = plt.subplots(1, len(show), figsize=(4.2 * len(show), 4.0))
    if len(show) == 1:
        axes = [axes]
    for ax, layer_idx in zip(axes, show):
        name = names[layer_idx]
        cap = captured[name]
        width_ratio = cap["width"][ex, :ex_len, :] / lengths[ex]
        head = int(np.argmin(np.abs(
            width_ratio.mean(axis=0) - np.median(width_ratio.mean(axis=0)))))
        center = cap["center"][ex, :ex_len, head]
        width = cap["width"][ex, :ex_len, head]
        scale = cap["scale"][head]
        sigma = 0.5 * width + 1e-6
        pos = np.arange(ex_len)
        diff = pos[None, :] - center[:, None]
        bias = -(diff * diff) / (2 * sigma[:, None] ** 2) * scale
        im = ax.imshow(bias, origin="lower", cmap="viridis",
                       aspect="auto", vmin=-6, vmax=0)
        ax.plot(center, pos, color="#ff4d4d", lw=1.1, label="中心 $P_i$")
        ax.plot(center - sigma, pos, color="white", lw=0.8, ls="--")
        ax.plot(center + sigma, pos, color="white", lw=0.8, ls="--",
                label="中心 ± $\\sigma$")
        ax.set_xlim(-0.5, ex_len - 0.5)
        ax.set_ylim(-0.5, ex_len - 0.5)
        ax.set_title(f"第 {int(name[5:])} 层 · 第 {head} 头", fontsize=9)
        ax.set_xlabel("键位置 $j$")
        ax.set_ylabel("查询位置 $i$")
        if ax is axes[0]:
            ax.legend(frameon=False, fontsize=7, loc="upper left")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="加性偏置")
    caption = (f"句长 {ex_len} 的样本句；偏置低于 $-$6 处截断")
    fig.suptitle(f"学到的查询相关高斯窗，{caption}")
    fig.tight_layout()
    fig.savefig(args.fig, dpi=160)
    print("example", ex, "length", ex_len, "layers", show)

    # Per-layer/per-head statistics: window width, center offset, scale.
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, key, ylabel in zip(
            axes2,
            ("width", "offset"),
            ("平均窗口宽度 $D/I$", "平均中心偏移 $|P_i - i|/I$")):
        for idx, name in enumerate(names):
            cap = captured[name]
            lens = cap["lengths"]
            valid = (np.arange(max_len)[None, :] < lens[:, None]).astype(
                "float32")
            denom = valid.sum()
            if key == "width":
                vals = cap["width"] / lens[:, None, None]
            else:
                pos = np.broadcast_to(
                    np.arange(max_len)[None, :, None], cap["center"].shape)
                vals = np.abs(cap["center"] - pos) / lens[:, None, None]
            per_head = (vals * valid[:, :, None]).sum(axis=(0, 1)) / denom
            ax.scatter([idx] * len(per_head), per_head, s=14, alpha=0.55,
                       color=f"C{idx % 10}")
            ax.plot([idx - 0.18, idx + 0.18],
                    [per_head.mean(), per_head.mean()], color="black",
                    linewidth=1.4)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("layer", "L") for n in names])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    means = [np.mean(stats[name]["per_head_scale"]) for name in names]
    axes2[2].plot(range(len(names)), means, marker="o")
    axes2[2].set_xticks(range(len(names)))
    axes2[2].set_xticklabels([n.replace("layer", "L") for n in names])
    axes2[2].set_ylabel("平均每头尺度")
    axes2[2].grid(axis="y", alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(Path(args.fig).with_name("depth.png"), dpi=160)

    with open(args.out, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
