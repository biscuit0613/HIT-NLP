#!/usr/bin/env python3
"""Prepare Chinese-medical-dialogue test set (first 100 Alpaca samples)."""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--ext-n", type=int, default=500)
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    print("total samples:", len(data))

    samples = []
    for i, rec in enumerate(data[: args.ext_n]):
        src = (rec.get("input") or "").strip() or (rec.get("instruction") or "").strip()
        tgt = (rec.get("output") or "").strip()
        if not src or not tgt:
            continue
        samples.append(
            {
                "idx": i,
                "instruction": rec.get("instruction", ""),
                "src": src,
                "tgt": tgt,
            }
        )

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "test_medical500.json"), "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=1)
    first = samples[: args.n]
    with open(os.path.join(args.out_dir, "test_medical.json"), "w", encoding="utf-8") as f:
        json.dump(first, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out_dir, "test_medical_alpaca.json"), "w", encoding="utf-8") as f:
        json.dump([{"instruction": s["src"], "input": "", "output": s["tgt"]} for s in first], f, ensure_ascii=False, indent=1)

    print("prepared:", len(first), "extended:", len(samples))
    for s in samples[:3]:
        print("Q:", s["src"][:80])
        print("A:", s["tgt"][:80])


if __name__ == "__main__":
    main()
