#!/usr/bin/env python3
"""Prepare CMMLU computer_science multiple-choice test set."""
import argparse
import csv
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    samples = []
    with open(args.input, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            question = (row.get("Question") or "").strip()
            options = {k: (row.get(k) or "").strip() for k in ("A", "B", "C", "D")}
            answer = (row.get("Answer") or "").strip().upper()
            if not question or not all(options.values()) or answer not in options:
                continue
            src = question + "\n" + "\n".join(f"{k}. {v}" for k, v in options.items())
            samples.append({"idx": i, "src": src, "tgt": answer, "options": options})

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "test_cmmlu.json"), "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=1)

    from collections import Counter

    print("prepared:", len(samples), "answers:", dict(Counter(s["tgt"] for s in samples)))
    print(samples[0]["src"])
    print("answer:", samples[0]["tgt"])


if __name__ == "__main__":
    main()
