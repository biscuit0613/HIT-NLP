#!/bin/bash
#SBATCH -J lab3_cmmlu_scaffold
#SBATCH -p compute
#SBATCH --gres=gpu:tesla_v100s-pcie-32gb:1
#SBATCH --nodelist=gpu13
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o /home/jnguan/lab3/logs/slurm-%x-%j.out
set -o pipefail
export PYTHONPATH=/home/jnguan/lab3/pylibs38
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd /home/jnguan/lab3
PY=/home/jnguan/.miniconda/envs/doggybot/bin/python
BASE=/home/jnguan/lab3/models/Qwen2-7B-Instruct
DATA=/home/jnguan/lab3/data/sft
OUT=/home/jnguan/lab3/results/cmmlu_scaffold
mkdir -p "$OUT"
for TAG in base lora_r8_qv lora_r16_all_lr1e4 dora_r16_all; do
  if [ "$TAG" = "base" ]; then
    ADAPTER=none
    MERGE=""
  else
    ADAPTER=/home/jnguan/lab3/runs/$TAG
    MERGE="--merge-adapter"
  fi
  echo "=== scaffold $TAG start $(date) ==="
  $PY scripts/eval_models.py \
    --base-model "$BASE" --adapter "$ADAPTER" --tag "$TAG" \
    --data-dir "$DATA" --out-dir "$OUT" --datasets cmmlu \
    --cmmlu-instruction "请回答上述单项选择题，只输出正确选项的字母（A、B、C、D）。" \
    --batch-size 8 --dtype float16 --attn sdpa --system qwen $MERGE || exit 1
  echo "=== scaffold $TAG end $(date) ==="
done
echo "=== ALL DONE $(date) ==="
