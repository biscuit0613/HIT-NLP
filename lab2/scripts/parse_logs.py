"""从 HPC 训练日志解析 loss/ppl/speed 曲线，输出 results/curves.csv。

用法: python scripts/parse_logs.py
扫描 results/ 下所有 logs/train_*.log（每行含 step_idx/epoch/batch/avg loss/ppl）。
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "curves.csv"

PAT = re.compile(
    r"step_idx: (\d+), epoch: (\d+), batch: (\d+), avg loss: ([\d.]+), "
    r"normalized loss: ([\d.]+), ppl: ([\d.]+)(?:, speed: ([\d.]+))?")


def parse_log(path: Path):
    for line in path.read_text(errors="ignore").splitlines():
        m = PAT.search(line)
        if m:
            step, epoch, batch, loss, norm, ppl, speed = m.groups()
            yield int(step), int(epoch), int(batch), float(loss), float(norm), float(ppl), speed


def main():
    rows = []
    for log in sorted(RESULTS.rglob("logs/train_*.log")):
        run = log.stem.replace("train_", "")
        rows.extend((run, *rec) for rec in parse_log(log))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "step", "epoch", "batch", "loss", "norm_loss", "ppl", "speed"])
        w.writerows(rows)
    runs = sorted({r[0] for r in rows})
    print(f"wrote {OUT} ({len(rows)} rows) runs: {runs}")


if __name__ == "__main__":
    main()
