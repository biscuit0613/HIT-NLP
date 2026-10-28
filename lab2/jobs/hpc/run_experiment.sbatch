#!/bin/bash
# =====================================================================
# 实验二 通用实验任务：训练 10 epoch + 预测 + BLEU（V100-SXM2-16GB 1卡）
# 每个实验的产物隔离在 ~/lab2/runs/$TAG/ 下，代码目录只读共享。
# 用法:
#   sbatch --job-name=lab2_lr3 \
#     --export=ALL,TAG=lr3,EXTRA="--learning_rate 3.0" \
#     ~/lab2/jobs/run_experiment.sbatch
# 变量:
#   TAG     实验标签（产物目录名/日志命名）
#   EXTRA   追加的超参覆盖（如 "--learning_rate 3.0"）
#   WORKDIR 代码目录（默认 ~/lab2/code；改进版实验指向独立副本）
# =====================================================================
#SBATCH -J lab2_exp
#SBATCH -p compute
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:30:00
#SBATCH -o /home/jnguan/lab2/logs/slurm-exp-%j.out

set -eo pipefail
source ~/.miniconda/etc/profile.d/conda.sh
conda activate lab2-paddle

TAG=${TAG:-exp}
EXTRA=${EXTRA:-}
cd ${WORKDIR:-$HOME/lab2/code}
RUN=$HOME/lab2/runs/$TAG
mkdir -p $RUN/run_logs $RUN/output_file $RUN/saved_models

DATA=/home/jnguan/lab2/data/wmt171819_zhen_data_bpe3

echo "=== LAB2 $TAG job $SLURM_JOB_ID on $(hostname) start $(date) ==="
echo "WORKDIR=$(pwd) RUN=$RUN EXTRA=[$EXTRA]"
nvidia-smi | head -12 || true
python -c "import paddle; print('paddle', paddle.__version__, 'cuda', paddle.is_compiled_with_cuda())"

echo "=== TRAIN 10 epochs ($TAG) ==="
python -u main.py --do_train True --epoch 10 \
  --src_vocab_fpath $DATA/vocab_all.bpe.32000 \
  --trg_vocab_fpath $DATA/vocab_all.bpe.32000 \
  --special_token '<s>' '<e>' '<unk>' \
  --training_file $DATA/train.tok.clean.bpe.32000.en-zh \
  --batch_size 4096 \
  --save_step 5000 \
  --print_step 100 \
  --save_model_path $RUN/saved_models \
  $EXTRA \
  2>&1 | tee $RUN/run_logs/train_${TAG}.log

for YEAR in 2017 2018; do
  echo "=== PREDICT newstest$YEAR ($TAG) ==="
  python -u main.py --do_predict True \
    --src_vocab_fpath $DATA/vocab_all.bpe.32000 \
    --trg_vocab_fpath $DATA/vocab_all.bpe.32000 \
    --special_token '<s>' '<e>' '<unk>' \
    --predict_file $DATA/newstest${YEAR}.tok.bpe.32000.en-zh \
    --batch_size 64 \
    --init_from_params $RUN/saved_models/step_final/ \
    --beam_size 5 \
    --max_out_len 255 \
    --output_file $RUN/output_file/predict_${TAG}_${YEAR}.txt \
    $EXTRA \
    2>&1 | tee $RUN/run_logs/predict_${TAG}_${YEAR}.log

  sed -r 's/(@@ )|(@@ ?$)//g' $RUN/output_file/predict_${TAG}_${YEAR}.txt > $RUN/output_file/predict_${TAG}_${YEAR}.tok.txt
  echo "=== BLEU newstest$YEAR course ($TAG), $(wc -l < $RUN/output_file/predict_${TAG}_${YEAR}.tok.txt) hypotheses ==="
  perl $HOME/lab2/code/mosesdecoder/scripts/generic/multi-bleu.perl $DATA/newstest${YEAR}.tok.zh \
    < $RUN/output_file/predict_${TAG}_${YEAR}.tok.txt 2>&1 | tee $RUN/run_logs/bleu_${TAG}_${YEAR}.log
  if [ "$YEAR" = "2018" ]; then
    echo "=== BLEU newstest2018 aligned ($TAG) ==="
    perl $HOME/lab2/code/mosesdecoder/scripts/generic/multi-bleu.perl $DATA/newstest2018.tok.aligned.zh \
      < $RUN/output_file/predict_${TAG}_${YEAR}.tok.txt 2>&1 | tee $RUN/run_logs/bleu_${TAG}_2018_aligned.log
  fi
done

echo "=== DONE $TAG $(date) ==="
