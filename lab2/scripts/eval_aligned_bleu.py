"""Recompute BLEU with the overlong-sentence filter applied to the reference.

The default prediction filter (max_length=256) drops two interior sentences
from newstest2018. Scoring the resulting 3,979 hypotheses against the full
3,981-line reference shifts every later line. This script reproduces the
filter, builds an aligned reference, and writes aligned BLEU logs plus
results/bleu_eval.csv for all runs.

Usage: python scripts/eval_aligned_bleu.py
"""
import csv
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "wmt171819_zhen_data_bpe3"
MOSES = ROOT / "aistudio-project" / "mosesdecoder" / "scripts" / "generic" / "multi-bleu.perl"
RESULTS = ROOT / "results"
MAX_LEN = 256

RUNS = {
    "baseline": RESULTS / "baseline",
    "lr3": RESULTS / "tuning" / "lr3",
    "warmup4k": RESULTS / "tuning" / "warmup4k",
    "ls05": RESULTS / "tuning" / "ls05",
    "combo": RESULTS / "tuning" / "combo",
    "base_s23": RESULTS / "tuning" / "base_s23",
    "beam10": RESULTS / "tuning" / "beam10",
    "relpos": RESULTS / "improve" / "relpos",
    "relpos_s23": RESULTS / "improve" / "relpos_s23",
}

BLEU_RE = re.compile(
    r"BLEU = ([\d.]+), ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+) "
    r"\(BP=([\d.]+), ratio=([\d.]+), hyp_len=(\d+), ref_len=(\d+)\)")


def reference_filter(year):
    bitext = DATA / f"newstest{year}.tok.bpe.32000.en-zh"
    ref = DATA / f"newstest{year}.tok.zh"
    src_lines = bitext.read_text().splitlines()
    ref_lines = ref.read_text().splitlines()
    assert len(src_lines) == len(ref_lines), (len(src_lines), len(ref_lines))
    keep, dropped = [], []
    for i, line in enumerate(src_lines):
        src, trg = line.split("\t", 1)
        if max(len(src.split()) + 1, len(trg.split()) + 2) <= MAX_LEN:
            keep.append(i)
        else:
            dropped.append(i + 1)
    return keep, ref_lines, dropped


def hyp_path(run_dir, year):
    matches = sorted((run_dir / "output_file").glob(f"*{year}*.tok.txt"))
    assert len(matches) == 1, (run_dir, matches)
    return matches[0]


def run_bleu(hyp, ref):
    with open(hyp, "rb") as f:
        proc = subprocess.run(["perl", str(MOSES), str(ref)],
                              stdin=f, capture_output=True)
    text = (proc.stdout + proc.stderr).decode("utf-8", "replace")
    assert proc.returncode == 0, text
    m = BLEU_RE.search(text)
    assert m, text
    vals = [float(x) for x in m.groups()[:7]] + [int(x) for x in m.groups()[7:]]
    return dict(zip(
        ["bleu", "p1", "p2", "p3", "p4", "bp", "ratio", "hyp_len", "ref_len"],
        vals))


def main():
    rows = []
    for year in (2017, 2018):
        keep, ref_lines, dropped = reference_filter(year)
        with tempfile.NamedTemporaryFile("w", suffix=".zh", delete=False) as f:
            f.write("\n".join(ref_lines[i] for i in keep) + "\n")
            aligned_ref = Path(f.name)
        try:
            for run, run_dir in RUNS.items():
                hyp = hyp_path(run_dir, year)
                n_hyp = sum(1 for _ in open(hyp, "rb"))
                assert n_hyp == len(keep), (run, year, n_hyp, len(keep))
                for protocol, ref in (("aligned", aligned_ref),
                                      ("course", DATA / f"newstest{year}.tok.zh")):
                    stats = run_bleu(hyp, ref)
                    rows.append({"run": run, "year": year, "protocol": protocol,
                                 "dropped": ";".join(map(str, dropped)),
                                 "n_hyp": n_hyp, **stats})
                    if year == 2018 and protocol == "aligned":
                        log = run_dir / "logs" / f"bleu_{run}_{year}_aligned.log"
                        log.write_text(
                            f"# aligned reference: max_length={MAX_LEN}, "
                            f"dropped source lines={','.join(map(str, dropped))}\n"
                            f"BLEU = {stats['bleu']:.2f}, "
                            f"{stats['p1']:.1f}/{stats['p2']:.1f}/"
                            f"{stats['p3']:.1f}/{stats['p4']:.1f} "
                            f"(BP={stats['bp']:.3f}, ratio={stats['ratio']:.3f}, "
                            f"hyp_len={stats['hyp_len']}, "
                            f"ref_len={stats['ref_len']})\n")
        finally:
            aligned_ref.unlink()
        print(f"{year}: kept {len(keep)}, dropped lines {dropped}")

    out = RESULTS / "bleu_eval.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", out)
    for r in rows:
        if r["year"] == 2018 and r["protocol"] == "aligned":
            print(f"  {r['run']:<11} aligned 2018 BLEU = {r['bleu']:.2f}")


if __name__ == "__main__":
    main()
