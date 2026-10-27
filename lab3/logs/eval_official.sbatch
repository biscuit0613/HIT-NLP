#!/bin/bash
#SBATCH -J lab3_official_eval
#SBATCH -p compute
#SBATCH --gres=gpu:tesla_v100s-pcie-32gb:1
#SBATCH --nodelist=gpu13
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
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
TAGS=${TAGS:-"base lora_r8_qv lora_r16_all_lr1e4 lora_r16_all lora_r32_all dora_r16_all"}
MED_LIMIT=${MED_LIMIT:-100}
SYSTEM=${SYSTEM:-qwen}
OUTDIR=${OUTDIR:-/home/jnguan/lab3/results/official}
GEN=${GEN:-1}
mkdir -p "$OUTDIR"
echo "=== job $SLURM_JOB_ID tags=$TAGS system=$SYSTEM med=$MED_LIMIT gen=$GEN $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for TAG in $TAGS; do
  if [ "$TAG" = "base" ]; then
    ADAPTER=none
    MERGE=""
  else
    ADAPTER=/home/jnguan/lab3/runs/$TAG
    MERGE="--merge-adapter"
  fi
  if [ "$ADAPTER" != "none" ] && [ ! -e "$ADAPTER" ]; then
    echo "skip $TAG (no adapter)"; continue
  fi
  echo "=== eval $TAG start $(date) ==="
  $PY scripts/eval_models.py \
    --base-model "$BASE" --adapter "$ADAPTER" --tag "$TAG" \
    --data-dir "$DATA" --out-dir "$OUTDIR" \
    --medical-file test_medical500.json --medical-limit "$MED_LIMIT" \
    --batch-size 8 --dtype float16 --attn sdpa --system "$SYSTEM" $MERGE || exit 1
  if [ "$GEN" = "1" ]; then
    echo "=== gen_test $TAG start $(date) ==="
    $PY scripts/gen_test.py \
      --base-model "$BASE" --adapter "$ADAPTER" --tag "$TAG" \
      --out-dir "$OUTDIR" --dtype float16 --attn sdpa --system "$SYSTEM" $MERGE || exit 1
  fi
  echo "=== eval $TAG end $(date) ==="
done
echo "=== ALL DONE $(date) ==="
