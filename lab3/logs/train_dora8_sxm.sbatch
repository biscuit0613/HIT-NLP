#!/bin/bash
#SBATCH -J lab3_train_dora8
#SBATCH -p compute
#SBATCH --gres=gpu:tesla_v100-sxm2-32gb:1
#SBATCH --nodelist=gpu13
#SBATCH --cpus-per-task=4
#SBATCH --mem=22G
#SBATCH --time=03:00:00
#SBATCH -o /home/jnguan/lab3/logs/slurm-%x-%j.out
set -euo pipefail
export PYTHONPATH=/home/jnguan/lab3/pylibs38
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jnguan/lab3
PY=/home/jnguan/.miniconda/envs/doggybot/bin/python
BASE=/home/jnguan/lab3/models/Qwen2-7B-Instruct
DATA=/home/jnguan/lab3/data/sft
TAG=dora_r8_qv
OUT=/home/jnguan/lab3/runs/$TAG
echo "=== dora8 train start $(date) ==="
$PY scripts/train_sft.py --config configs/dora_r8_qv.yaml --tag "$TAG" \
  --model "$BASE" --data-dir "$DATA" --out-dir "$OUT" --meta-dir results \
  --fp16 --gradient-checkpointing
echo "=== dora8 train end $(date) ==="
$PY scripts/eval_models.py \
  --base-model "$BASE" --adapter "$OUT" --tag "$TAG" \
  --data-dir "$DATA" --out-dir /home/jnguan/lab3/results/official \
  --datasets medical,cmmlu \
  --medical-file test_medical500.json --medical-limit 500 \
  --batch-size 8 --dtype float16 --attn sdpa --system qwen --merge-adapter
$PY scripts/gen_test.py \
  --base-model "$BASE" --adapter "$OUT" --tag "$TAG" \
  --out-dir /home/jnguan/lab3/results/official \
  --dtype float16 --attn sdpa --system qwen --merge-adapter
$PY scripts/hci_probe.py \
  --base-model "$BASE" --adapter "$OUT" --tag "$TAG" \
  --out-dir /home/jnguan/lab3/results/hci_probe \
  --dtype float16 --attn sdpa --system qwen --merge-adapter
$PY scripts/eval_models.py \
  --base-model "$BASE" --adapter "$OUT" --tag "$TAG" \
  --data-dir "$DATA" --out-dir /home/jnguan/lab3/results/cmmlu_scaffold --datasets cmmlu \
  --cmmlu-instruction "请回答上述单项选择题，只输出正确选项的字母（A、B、C、D）。" \
  --batch-size 8 --dtype float16 --attn sdpa --system qwen --merge-adapter
echo "=== ALL DONE $(date) ==="
