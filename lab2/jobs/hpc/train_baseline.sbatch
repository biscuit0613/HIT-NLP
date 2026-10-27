#!/bin/bash
# =====================================================================
# 实验二 基线：训练 10 epoch + 预测 newstest2017/2018 + BLEU
# 提交: sbatch ~/lab2/jobs/train_baseline.sbatch
# 产物: ~/lab2/code/{run_logs,output_file,saved_models}
# stdout: ~/lab2/logs/slurm-baseline-<jobid>.out
# =====================================================================
#SBATCH -J lab2_baseline
#SBATCH -p compute
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:30:00
#SBATCH -o /home/jnguan/lab2/logs/slurm-baseline-%j.out

set -eo pipefail
source ~/.miniconda/etc/profile.d/conda.sh
conda activate lab2-paddle

cd ~/lab2/code
mkdir -p run_logs output_file

DATA=/home/jnguan/lab2/data/wmt171819_zhen_data_bpe3

echo "=== LAB2 BASELINE job $SLURM_JOB_ID on $(hostname) start $(date) ==="
nvidia-smi | head -12 || true
python -c "import paddle; print('paddle', paddle.__version__, 'cuda', paddle.is_compiled_with_cuda())"

echo "=== TRAIN 10 epochs ==="
python -u main.py --do_train True --epoch 10 \
  --src_vocab_fpath $DATA/vocab_all.bpe.32000 \
  --trg_vocab_fpath $DATA/vocab_all.bpe.32000 \
  --special_token '<s>' '<e>' '<unk>' \
  --training_file $DATA/train.tok.clean.bpe.32000.en-zh \
  --batch_size 4096 \
  --save_step 5000 \
  --print_step 100 \
  2>&1 | tee run_logs/train_baseline.log

for YEAR in 2017 2018; do
  echo "=== PREDICT newstest$YEAR ==="
  python -u main.py --do_predict True \
    --src_vocab_fpath $DATA/vocab_all.bpe.32000 \
    --trg_vocab_fpath $DATA/vocab_all.bpe.32000 \
    --special_token '<s>' '<e>' '<unk>' \
    --predict_file $DATA/newstest${YEAR}.tok.bpe.32000.en-zh \
    --batch_size 64 \
    --init_from_params saved_models/step_final/ \
    --beam_size 5 \
    --max_out_len 255 \
    --output_file output_file/predict_${YEAR}.txt \
    2>&1 | tee run_logs/predict_${YEAR}.log

  sed -r 's/(@@ )|(@@ ?$)//g' output_file/predict_${YEAR}.txt > output_file/predict_${YEAR}.tok.txt
  echo "=== BLEU newstest$YEAR ==="
  perl mosesdecoder/scripts/generic/multi-bleu.perl $DATA/newstest${YEAR}.tok.zh \
    < output_file/predict_${YEAR}.tok.txt 2>&1 | tee run_logs/bleu_${YEAR}.log
done

echo "=== DONE $(date) ==="
