#!/bin/bash
#SBATCH -J lab3_train_fmt
#SBATCH -p compute
#SBATCH --gres=gpu:tesla_v100-sxm2-32gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:15:00
#SBATCH -o /home/jnguan/lab3/logs/slurm-%x-%j.out
set -o pipefail
TAG=${TAG:-lora_r8_qv_fmt}
CONFIG=${CONFIG:-configs/$TAG.yaml}
EXTRA=${EXTRA:-}
export PYTHONPATH=/home/jnguan/lab3/pylibs38
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export DATALOADER_WORKERS=${DATALOADER_WORKERS:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jnguan/lab3
echo "=== LAB3 FMT TRAIN job $SLURM_JOB_ID tag=$TAG config=$CONFIG start $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p results
/home/jnguan/.miniconda/envs/doggybot/bin/python scripts/train_sft.py \
  --config "$CONFIG" --tag "$TAG" --meta-dir results --max-steps -1 --fp16 --gradient-checkpointing $EXTRA
cp runs/${TAG}/trainer_state.json results/state_${TAG}.json 2>/dev/null || true
echo "=== DONE $(date) ==="
