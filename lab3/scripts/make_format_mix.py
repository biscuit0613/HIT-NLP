#!/usr/bin/env python3
"""Build a format-alignment SFT file from the same DISC-Med-SFT training split.

The MCQ examples are derived only from training pairs: the correct option is the
paired doctor reply, the three distractors are replies to other questions.  They
teach the adapter to answer option-style questions with "B. <option text>"
under the official short system prompt.  No CMMLU item is used.
"""
import argparse
import json
import random
import re
from collections import Counter

OFFICIAL_SYSTEM = "You are a helpful assistant."
PLEASANTRY_OPEN = re.compile(
    r"^(非常)?(谢谢|感谢|您好|你好|好的|在吗|辛苦了|再见|不客气|收到|明白|知道了|嗯|哦|行|可以)")
CLOSING = re.compile(
    r"(谢谢|感谢|辛苦了|不客气|祝您|按照您的建议|按照医生|会按照|我按照|遵医嘱|明白了|知道了)")


def is_substantive(question, min_chars):
    text = norm(question, 10 ** 9)
    if len(text) < min_chars:
        return False
    has_question = bool(re.search(r"[?？]", text))
    if PLEASANTRY_OPEN.match(text) and not has_question:
        return False
    if len(text) < 50 and CLOSING.search(text) and not has_question:
        return False
    return True


def norm(text, limit):
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) > limit:
        text = text[:limit].rstrip()
    return text


def load_pairs(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    pairs = []
    for ex in raw:
        user = (ex.get("instruction") or "").strip()
        extra = (ex.get("input") or "").strip()
        if extra:
            user = user + "\n" + extra
        target = (ex.get("output") or "").strip()
        if user and target:
            pairs.append({"instruction": user, "input": "", "output": target})
    return pairs


def make_mcq(pair, pool, rng, max_question_chars, max_option_chars,
             min_question_chars, min_answer_chars):
    if not is_substantive(pair["instruction"], min_question_chars):
        return None
    if len(norm(pair["output"], 10 ** 9)) < min_answer_chars:
        return None
    question = norm(pair["instruction"], max_question_chars)
    correct = norm(pair["output"], max_option_chars)
    if not question or not correct:
        return None
    chosen = {correct}
    distractors = []
    for _ in range(20):
        other = pool[rng.randrange(len(pool))]
        text = norm(other["output"], max_option_chars)
        if not text or text in chosen:
            continue
        chosen.add(text)
        distractors.append(text)
        if len(distractors) == 3:
            break
    if len(distractors) < 3:
        return None
    options = distractors + [correct]
    rng.shuffle(options)
    letter = "ABCD"[options.index(correct)]
    prompt = question + "\n" + "\n".join(
        f"{k}. {v}" for k, v in zip("ABCD", options))
    return {
        "instruction": prompt,
        "input": "",
        "output": f"{letter}. {correct}",
        "system": OFFICIAL_SYSTEM,
        "source": "format_mcq",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-format", type=int, default=2000)
    ap.add_argument("--n-replay", type=int, default=1000)
    ap.add_argument("--max-question-chars", type=int, default=300)
    ap.add_argument("--max-option-chars", type=int, default=80)
    ap.add_argument("--min-question-chars", type=int, default=15)
    ap.add_argument("--min-answer-chars", type=int, default=20)
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()

    pool = load_pairs(args.train)
    rng = random.Random(args.seed)
    order = list(range(len(pool)))
    rng.shuffle(order)

    format_examples = []
    cursor = 0
    while len(format_examples) < args.n_format and cursor < len(order):
        mcq = make_mcq(pool[order[cursor]], pool, rng,
                       args.max_question_chars, args.max_option_chars,
                       args.min_question_chars, args.min_answer_chars)
        cursor += 1
        if mcq is not None:
            format_examples.append(mcq)
        if cursor % 2000 == 0:
            print(f"[format] scanned={cursor} kept={len(format_examples)}", flush=True)
    if len(format_examples) < args.n_format:
        raise SystemExit(f"only built {len(format_examples)} MCQs, need {args.n_format}")

    replay = []
    for i in range(args.n_replay):
        ex = dict(pool[order[cursor % len(order)]])
        ex["source"] = "medical_replay"
        replay.append(ex)
        cursor += 1

    mixed = format_examples + replay
    rng.shuffle(mixed)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(mixed, f, ensure_ascii=False, indent=1)

    letters = Counter(ex["output"][0] for ex in format_examples)
    print(f"[done] pool={len(pool)} format={len(format_examples)} replay={len(replay)} "
          f"total={len(mixed)} letters={dict(sorted(letters.items()))}")
    if format_examples:
        print(f"[sample] {format_examples[0]['instruction']}")
        print(f"[sample] -> {format_examples[0]['output']}")
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
