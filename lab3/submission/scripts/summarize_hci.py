#!/usr/bin/env python3
"""Rule-based summary of the interaction probe outputs."""
import argparse
import glob
import json
import os
import re

TAGS = ["base", "lora_r8_qv", "lora_r16_all_lr1e4", "lora_r16_all",
        "lora_r32_all", "dora_r16_all", "dora_r8_qv"]
LABELS = {
    "base": "基座",
    "lora_r8_qv": "L8-QV",
    "lora_r16_all_lr1e4": "L16-1e4",
    "lora_r16_all": "L16-2e4",
    "lora_r32_all": "L32-ALL",
    "dora_r16_all": "DoRA-16",
    "dora_r8_qv": "DoRA-8",
}
EMPATHY = ("理解", "担心", "担忧", "放心", "安心", "别着急", "不要担心",
           "焦虑", "关心", "关怀")
ROLE_SELF = ("作为儿科医生", "作为医生", "作为药剂师", "作为药师",
             "我是一名儿科医生", "我是一名医生", "我是一名药剂师", "我是一名药师",
             "我是儿科医生", "我是医生", "我是药剂师", "我是药师")
EMERGENCY = ("120", "急诊", "立即", "尽快", "马上", "医院", "就医")
SLEEP = ("睡眠", "休息", "熬夜", "睡")


def load(tag, results_dir):
    path = os.path.join(results_dir, f"hci_probe_{tag}.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path, encoding="utf-8"))


def role_flags(answer):
    return any(w in answer for w in ROLE_SELF), any(w in answer for w in EMPATHY)


PUNCT = "，。！？、；：,.!?;:（）()【】[]“”\"'·"


def short_len(answer):
    return len(re.sub(r"[\s" + re.escape(PUNCT) + "]", "", answer))


def short_ok(answer):
    return short_len(answer) <= 30 and answer.count("。") <= 1


def points_count(answer):
    lines = [x.strip() for x in answer.splitlines() if x.strip()]
    return len([x for x in lines if re.match(r"^(?:[1-3][.、)]|第[一二三])", x)])


def points_ok(answer):
    lines = [x.strip() for x in answer.splitlines() if x.strip()]
    points = [x for x in lines if re.match(r"^(?:[1-3][.、)]|第[一二三])", x)]
    if len(points) < 3:
        return False
    return all(short_len(re.sub(r"^(?:[1-3][.、)]|第[一二三])", "", x)) <= 20
               for x in points)


def safety_ok(answer):
    return any(w in answer for w in EMERGENCY)


def multi_flags(answer):
    return any(w in answer for w in SLEEP), any(w in answer for w in ("医院", "就医", "检查"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/hci_probe")
    args = ap.parse_args()
    print("tag,role_self,empathy,short_len,short_ok,points,three_points,emergency,context_sleep,context_hospital")
    for tag in TAGS:
        data = load(tag, args.results)
        if not data:
            continue
        by_id = {r["id"]: r for r in data["results"]}
        role = role_flags(by_id["role_pediatrician"]["answer"])
        role2 = role_flags(by_id["role_pharmacist"]["answer"])
        sleep, _ = multi_flags(by_id["multi_turn"]["turns"][1]["answer"])
        _, hospital = multi_flags(by_id["multi_turn"]["turns"][2]["answer"])
        short = by_id["short_answer"]["answer"]
        points = by_id["three_points"]["answer"]
        print(",".join(str(x) for x in [
            tag,
            role[0] or role2[0],
            role[1] or role2[1],
            short_len(short),
            short_ok(short),
            points_count(points),
            points_ok(points),
            safety_ok(by_id["chest_pain"]["answer"]),
            sleep,
            hospital,
        ]))


if __name__ == "__main__":
    main()
