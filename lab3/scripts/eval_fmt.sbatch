#!/bin/bash
#SBATCH -J lab3_eval_fmt
#SBATCH -p compute
#SBATCH --gres=gpu:tesla_v100-sxm2-32gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:15:00
#SBATCH -o /home/jnguan/lab3/logs/slurm-%x-%j.out
set -o pipefail
TAGS=${TAGS:-"base lora_r8_qv lora_r8_qv_fmt lora_r8_qv_replay"}
OUT_DIR=${OUT_DIR:-/home/jnguan/lab3/results/official_fmt}
MED_LIMIT=${MED_LIMIT:-100}
DATASETS=${DATASETS:-medical,cmmlu}
export PYTHONPATH=/home/jnguan/lab3/pylibs38
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd /home/jnguan/lab3
echo "=== LAB3 FORMAT EVAL job $SLURM_JOB_ID tags=$TAGS out=$OUT_DIR on $(hostname) start $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p "$OUT_DIR"
for TAG in $TAGS; do
  if [ "$TAG" = "base" ]; then
    ADAPTER=none
    MERGE=""
  else
    ADAPTER=/home/jnguan/lab3/runs/$TAG
    MERGE="--merge-adapter"
    if [ ! -e "$ADAPTER" ]; then echo "skip $TAG (no adapter)"; continue; fi
  fi
  echo "=== EVAL $TAG start $(date) ==="
  /home/jnguan/.miniconda/envs/doggybot/bin/python scripts/eval_models.py \
    --base-model /home/jnguan/lab3/models/Qwen2-7B-Instruct \
    --adapter "$ADAPTER" --tag "$TAG" \
    --data-dir /home/jnguan/lab3/data/sft \
    --out-dir "$OUT_DIR" \
    --datasets "$DATASETS" \
    --medical-file test_medical500.json --medical-limit "$MED_LIMIT" \
    --cmmlu-file test_cmmlu.json \
    --batch-size 8 --dtype float16 --attn sdpa \
    --system qwen --cmmlu-system official \
    --cmmlu-max-new-tokens 256 --cmmlu-repetition-penalty 1.05 \
    $MERGE || echo "=== EVAL $TAG FAILED ==="
  echo "=== EVAL $TAG end $(date) ==="
done
echo "=== ALL DONE $(date) ==="
