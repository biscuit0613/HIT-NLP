#!/bin/bash
# =====================================================================
# 实验二 3.6 解码端调参：用基线模型以 beam_size=10 重新解码
# 提交: sbatch ~/lab2/jobs/decode_beam10.sbatch
# =====================================================================
#SBATCH -J lab2_beam10
#SBATCH -p compute
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH -o /home/jnguan/lab2/logs/slurm-beam10-%j.out

set -eo pipefail
source ~/.miniconda/etc/profile.d/conda.sh
conda activate lab2-paddle

cd ~/lab2/code
RUN=$HOME/lab2/runs/beam10
mkdir -p $RUN/run_logs $RUN/output_file
DATA=/home/jnguan/lab2/data/wmt171819_zhen_data_bpe3

echo "=== LAB2 beam10 job $SLURM_JOB_ID on $(hostname) start $(date) ==="
nvidia-smi | head -12 || true

for YEAR in 2017 2018; do
  echo "=== PREDICT newstest$YEAR (beam10) ==="
  # beam=10 at batch_size 64 OOMs on the 16GB V100; 32 keeps the decode within memory.
  python -u main.py --do_predict True \
    --src_vocab_fpath $DATA/vocab_all.bpe.32000 \
    --trg_vocab_fpath $DATA/vocab_all.bpe.32000 \
    --special_token '<s>' '<e>' '<unk>' \
    --predict_file $DATA/newstest${YEAR}.tok.bpe.32000.en-zh \
    --batch_size 32 \
    --init_from_params $HOME/lab2/code/saved_models/step_final/ \
    --beam_size 10 \
    --max_out_len 255 \
    --output_file $RUN/output_file/predict_${YEAR}.txt \
    2>&1 | tee $RUN/run_logs/predict_${YEAR}.log

  sed -r 's/(@@ )|(@@ ?$)//g' $RUN/output_file/predict_${YEAR}.txt > $RUN/output_file/predict_${YEAR}.tok.txt
  echo "=== BLEU newstest$YEAR (beam10) ==="
  perl $HOME/lab2/code/mosesdecoder/scripts/generic/multi-bleu.perl $DATA/newstest${YEAR}.tok.zh \
    < $RUN/output_file/predict_${YEAR}.tok.txt 2>&1 | tee $RUN/run_logs/bleu_${YEAR}.log
done

echo "=== DONE beam10 $(date) ==="
