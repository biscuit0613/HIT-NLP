#!/bin/bash
#SBATCH -J lab3_smoke_sdpa20
#SBATCH -p compute
#SBATCH --gres=gpu:tesla_v100s-pcie-32gb:1
#SBATCH --nodelist=gpu13
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:20:00
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
OUT=/home/jnguan/lab3/results/smoke_sdpa20
mkdir -p "$OUT"
echo "=== sdpa base $(date) ==="
$PY scripts/eval_models.py --base-model "$BASE" --adapter none --tag base --data-dir "$DATA" \
  --out-dir "$OUT" --datasets medical,cmmlu --medical-limit 2 --cmmlu-limit 3 \
  --batch-size 8 --dtype float16 --attn sdpa
echo "=== done $(date) ==="
