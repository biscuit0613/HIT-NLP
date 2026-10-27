#!/usr/bin/env python3
"""Interactively fill ratings.csv for the blind human-evaluation packet."""
import argparse
import csv
import json
import os

DIMS = ["relevance", "accuracy", "fluency"]
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
    ap.add_argument("--packet-dir", default="results/human_eval")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    key = json.load(open(os.path.join(args.packet_dir, "key.json"), encoding="utf-8"))
    tags = sorted({tag for by_pos in key.values() for tag in by_pos.values()})
    preds = {tag: read_preds(args.results, tag) for tag in tags}
    rating_path = os.path.join(args.packet_dir, "ratings.csv")

    rows = []
    done = {}
    if os.path.exists(rating_path):
        for row in csv.DictReader(open(rating_path, encoding="utf-8")):
            rows.append(row)
            if row["relevance"].strip() and row["accuracy"].strip() and row["fluency"].strip():
                done[(row["item"], row["answer"])] = row
    if not rows:
        raise SystemExit("ratings.csv not found; run make_human_eval.py first")

    for item, by_pos in key.items():
        record = preds[tags[0]][item]
        print("=" * 70)
        print(f"题目 {item}")
        print(f"患者：{first(record.get('src'))}")
        print(f"参考：{first(record.get('tgt'))}\n")
        for pos in POSITIONS:
            tag = by_pos[pos]
            answer = preds[tag][item].get("pred", "").strip()
            print(f"[{pos}] {answer}\n")
            if (item, pos) in done:
                print("  已评分，跳过\n")
                continue
            while True:
                raw = input(f"  评分 {pos}（相关性 准确性 流畅性，1-5；s 跳过）：").strip()
                if raw.lower() == "s":
                    break
                parts = raw.split()
                if len(parts) == 3 and all(p.isdigit() and 1 <= int(p) <= 5 for p in parts):
                    row = next(r for r in rows if r["item"] == item and r["answer"] == pos)
                    row["relevance"], row["accuracy"], row["fluency"] = parts
                    done[(item, pos)] = row
                    break
                print("  请输入三个 1-5 的整数，例如：4 5 4")
        with open(rating_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["item", "answer", "relevance", "accuracy", "fluency", "comment"])
            writer.writeheader()
            writer.writerows(rows)

    missing = [(r["item"], r["answer"]) for r in rows
               if not (r["relevance"].strip() and r["accuracy"].strip() and r["fluency"].strip())]
    print(f"\n已完成 {len(rows) - len(missing)}/{len(rows)} 条评分，写入 {rating_path}")
    if missing:
        print(f"仍有 {len(missing)} 条未评分，可重新运行本脚本继续。")
    else:
        print("运行 scripts/evaluate_human.py 生成汇总与图。")


if __name__ == "__main__":
    main()
