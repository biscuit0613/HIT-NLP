#!/bin/bash
# =====================================================================
# Sanity check for the refactored transformer: run 200 training steps at
# seed 23 for three flag settings and compare step 0/100/200 losses with
# the archived base_s23 / relpos_s23 logs. No prediction is run.
# =====================================================================
#SBATCH -J lab2_sanity
#SBATCH -p compute
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH -o /home/jnguan/lab2/logs/slurm-sanity-%j.out

set -eo pipefail
source ~/.miniconda/etc/profile.d/conda.sh
conda activate lab2-paddle

cd ${WORKDIR:-$HOME/lab2/code_locrel}
DATA=/home/jnguan/lab2/data/wmt171819_zhen_data_bpe3
OUT=$HOME/lab2/sanity
mkdir -p $OUT

COMMON=(--do_train True --epoch 1 --max_iter 200
  --src_vocab_fpath $DATA/vocab_all.bpe.32000
  --trg_vocab_fpath $DATA/vocab_all.bpe.32000
  --special_token '<s>' '<e>' '<unk>'
  --training_file $DATA/train.tok.clean.bpe.32000.en-zh
  --batch_size 4096 --save_step 100000 --print_step 100
  --save_model_path $OUT/models --random_seed 23)

echo "=== SANITY flags-off (should match base_s23) ==="
python -u main.py "${COMMON[@]}" --use_rel_pos False --use_localness False \
  2>&1 | tee $OUT/train_off.log

echo "=== SANITY relpos-only ==="
python -u main.py "${COMMON[@]}" --use_rel_pos True --use_localness False \
  2>&1 | tee $OUT/train_relpos.log

echo "=== SANITY localness+relpos ==="
python -u main.py "${COMMON[@]}" --use_rel_pos True --use_localness True \
  --localness_layers 3 \
  2>&1 | tee $OUT/train_locrel.log

echo "=== SANITY losses ==="
for TAG in off relpos locrel; do
  echo "-- $TAG"
  grep -E "step_idx: (0|100|200)," $OUT/train_$TAG.log || true
done
echo "=== SANITY DONE $(date) ==="
