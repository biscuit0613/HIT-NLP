"""生成 AI Studio 后台任务的执行 ipynb。

任务在项目版本根目录（/home/aistudio/）执行，项目内自带数据目录 wmt18_zhen_data_bpe3/。
用法: python make_ipynb.py baseline  -> 生成 run_baseline.ipynb
"""

import json
import sys
import uuid


def code_cell(src: str) -> dict:
    src = "%%bash\n" + src.strip("\n")
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in src.split("\n")],
    }


def md_cell(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": [line + "\n" for line in src.strip("\n").split("\n")],
    }


DATA = "wmt18_zhen_data_bpe3"
COMMON = (
    f"--src_vocab_fpath {DATA}/vocab_all.bpe.32000 \\\n"
    f"  --trg_vocab_fpath {DATA}/vocab_all.bpe.32000 \\\n"
    "  --special_token '<s>' '<e>' '<unk>'"
)


def train_cell(extra_args: str, log_name: str) -> str:
    extra_line = f"  {extra_args} \\\n" if extra_args.strip() else ""
    return f"""
set -o pipefail
date
python -u main.py \\
  --do_train True \\
  --epoch 10 \\
  {COMMON} \\
  --training_file {DATA}/train.tok.clean.bpe.32000.en-zh \\
  --batch_size 4096 \\
  --save_step 5000 \\
  --print_step 100 \\
{extra_line}  2>&1 | tee run_logs/{log_name}
echo "TRAIN_EXIT=$?"
date
"""


def predict_cell(year: str) -> str:
    return f"""
set -o pipefail
python -u main.py \\
  --do_predict True \\
  {COMMON} \\
  --predict_file {DATA}/newstest{year}.tok.bpe.32000.en-zh \\
  --batch_size 64 \\
  --init_from_params saved_models/step_final/ \\
  --beam_size 5 \\
  --max_out_len 255 \\
  --output_file output_file/predict_{year}.txt \\
  2>&1 | tee run_logs/predict_{year}.log
echo "PREDICT_EXIT=$?"
"""


def bleu_cell(files: str, log_name: str) -> str:
    return f"""
for f in {files}; do
  sed -r 's/(@@ )|(@@ ?$)//g' output_file/$f.txt > output_file/$f.tok.txt
done
perl mosesdecoder/scripts/generic/multi-bleu.perl {DATA}/newstest2017.tok.zh < output_file/predict_2017.tok.txt 2>&1 | tee run_logs/bleu_2017.log
perl mosesdecoder/scripts/generic/multi-bleu.perl {DATA}/newstest2018.tok.zh < output_file/predict_2018.tok.txt 2>&1 | tee run_logs/bleu_2018.log
"""


def make_notebook(cells: list, title: str) -> dict:
    return {
        "cells": [md_cell(title)] + cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


ENV_CELL = """
set -x
date
nvidia-smi | head -12
python -c "import paddle; print('paddle', paddle.__version__, 'cuda', paddle.is_compiled_with_cuda())"
ls -la
mkdir -p run_logs output_file
"""


def baseline_notebook() -> dict:
    return make_notebook(
        [
            code_cell(ENV_CELL),
            code_cell(train_cell("", "train_baseline.log")),
            code_cell(predict_cell("2017")),
            code_cell(predict_cell("2018")),
            code_cell(bleu_cell("predict_2017 predict_2018", "bleu")),
            code_cell(
                "zip -qr -1 baseline_outputs.zip run_logs output_file\n"
                "zip -qr -1 saved_models.zip saved_models\n"
                "ls -la baseline_outputs.zip saved_models.zip\n"
                "du -sh saved_models/\n"
                "ls saved_models/"
            ),
        ],
        """# 实验二 · 基线训练任务（Transformer 英汉翻译）

一键流程：环境确认 → 训练 10 epoch（batch 4096，每 5000 iter 存 checkpoint）→
预测 newstest2017/2018 → BLEU 评测 → 打包产物。

预计耗时约 2.5 小时（V100 单卡）。训练日志实时重定向到 run_logs/train_baseline.log。""",
    )


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    if which == "baseline":
        nb = baseline_notebook()
        out = "run_baseline.ipynb"
    else:
        raise SystemExit(f"unknown target: {which}")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
