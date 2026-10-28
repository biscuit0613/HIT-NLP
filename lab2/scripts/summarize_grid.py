"""Summarize paired multi-seed BLEU results from results/bleu_eval.csv."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

GROUPS = {
    "base": "base_s",
    "loconly6": "loconly6_s",
    "locrel6": "locrel6_s",
}

SINGLES = ["relpos", "relpos_s23", "loconly_s23", "locrel_s23",
           "stdscale_s23", "locrelstd_s23"]


def load():
    table = {}
    with open(RESULTS / "bleu_eval.csv") as f:
        for row in csv.DictReader(f):
            if row["protocol"] != "aligned":
                continue
            table.setdefault(row["run"], {})[int(row["year"])] = float(
                row["bleu"])
    return table


def discover(table):
    groups = {}
    for group, prefix in GROUPS.items():
        runs = sorted(r for r in table if r.startswith(prefix)
                      and r[len(prefix):].isdigit())
        assert runs, f"no runs for {group}"
        groups[group] = runs
    return groups


def main():
    table = load()
    groups = discover(table)
    out = {}
    for group, runs in groups.items():
        vals = {year: np.array([table[r][year] for r in runs])
                for year in (2017, 2018)}
        out[group] = {
            "runs": runs,
            "2017": [float(x) for x in vals[2017]],
            "2018": [float(x) for x in vals[2018]],
            "mean_2017": float(vals[2017].mean()),
            "std_2017": float(vals[2017].std(ddof=1)),
            "mean_2018": float(vals[2018].mean()),
            "std_2018": float(vals[2018].std(ddof=1)),
        }
    for group in ("loconly6", "locrel6"):
        for year in (2017, 2018):
            d = np.array(out[group][str(year)]) - np.array(out["base"][str(year)])
            out[group][f"delta_{year}"] = [float(x) for x in d]
            out[group][f"mean_delta_{year}"] = float(d.mean())
            out[group][f"std_delta_{year}"] = float(d.std(ddof=1))
    out["singles"] = {r: table[r] for r in SINGLES if r in table}

    for group in GROUPS:
        g = out[group]
        print(f"{group:10s} 2017 {g['mean_2017']:.2f}±{g['std_2017']:.2f} "
              f"{[round(x, 2) for x in g['2017']]} | "
              f"2018 {g['mean_2018']:.2f}±{g['std_2018']:.2f} "
              f"{[round(x, 2) for x in g['2018']]}")
        if group != "base":
            print(f"{'  delta':10s} 2017 "
                  f"{g['mean_delta_2017']:+.2f}±{g['std_delta_2017']:.2f} "
                  f"{[round(x, 2) for x in g['delta_2017']]} | 2018 "
                  f"{g['mean_delta_2018']:+.2f}±{g['std_delta_2018']:.2f} "
                  f"{[round(x, 2) for x in g['delta_2018']]}")
    for run, years in out["singles"].items():
        print(f"{run:14s} 2017={years.get(2017)} 2018={years.get(2018)}")

    with open(RESULTS / "grid_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", RESULTS / "grid_summary.json")


if __name__ == "__main__":
    main()
