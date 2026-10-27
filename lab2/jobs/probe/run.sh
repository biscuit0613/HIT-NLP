#!/bin/bash
# =====================================================================
# 产线任务全流程小验证：
#   1) 探测任务环境（cwd、目录结构、paddle 版本、GPU）
#   2) 定位并解压挂载的数据集 60419
#   3) 小训练（--max_iter 20，全量训练数据，batch 4096，存 step_20）
#   4) 预测 100 句（beam 5）→ sed 去 @@ → multi-bleu.perl
#   5) 全部日志与结果收集到 output 目录，关键结果打到 stdout
# =====================================================================
exec > >(tee /tmp/probe_run.log) 2>&1
set -x

echo "================ PROBE START $(date) ================"
echo "--- pwd / 目录结构 ---"
pwd
ls -la
echo "--- /home/aistudio ---"
ls -la /home/aistudio 2>/dev/null
echo "--- 挂载的数据集位置 ---"
ls -la /home/aistudio/data 2>/dev/null
ls -la /home/aistudio/data/data60419 2>/dev/null
ls -la data 2>/dev/null
ls -la data/data60419 2>/dev/null
echo "--- 候选 output 目录 ---"
ls -la /home/aistudio/output 2>/dev/null
ls -la output 2>/dev/null

echo "--- python / paddle 环境 ---"
which python python3
python -c "import sys; print('python', sys.version)"
python -c "import paddle; print('paddle', paddle.__version__, 'cuda_compiled', paddle.is_compiled_with_cuda())"
python -c "import numpy; print('numpy', numpy.__version__)"
nvidia-smi | head -12

echo "--- 搜索数据 zip ---"
find /home/aistudio -maxdepth 4 -name "*.zip" 2>/dev/null
DATASET_ZIP=$(find /home/aistudio -maxdepth 4 -name "wmt*zhen*.zip" 2>/dev/null | head -1)
echo "DATASET_ZIP=$DATASET_ZIP"

DATADIR=""
if [ -n "$DATASET_ZIP" ]; then
    mkdir -p /home/aistudio/work/data
    unzip -q -o "$DATASET_ZIP" -d /home/aistudio/work/data
    DATADIR=$(dirname $(find /home/aistudio/work/data -name "train.tok.clean.bpe.32000.en-zh" | head -1))
fi
echo "DATADIR=$DATADIR"
ls -la "$DATADIR" 2>/dev/null

# --- 若数据未找到，尝试任务 cwd 附近 ---
if [ -z "$DATADIR" ]; then
    for p in data train_data wmt18_zhen_data_bpe3 wmt171819_zhen_data_bpe3; do
        if [ -f "$p/train.tok.clean.bpe.32000.en-zh" ]; then DATADIR="$p"; break; fi
    done
fi
# 本地干跑覆盖入口
if [ -n "$PROBE_DATADIR" ]; then DATADIR="$PROBE_DATADIR"; fi
echo "final DATADIR=$DATADIR"

CODE_DIR=$(cd "$(dirname "$0")" && pwd)
echo "CODE_DIR=$CODE_DIR"
cd "$CODE_DIR"

OUT=skip
if [ -n "$DATADIR" ] && [ -f "$DATADIR/train.tok.clean.bpe.32000.en-zh" ]; then
    echo "================ SMALL TRAINING ================"
    timeout 1200 python -u main.py \
        --do_train True \
        --epoch 1 \
        --src_vocab_fpath "$DATADIR/vocab_all.bpe.32000" \
        --trg_vocab_fpath "$DATADIR/vocab_all.bpe.32000" \
        --special_token '<s>' '<e>' '<unk>' \
        --training_file "$DATADIR/train.tok.clean.bpe.32000.en-zh" \
        --batch_size 4096 \
        --print_step 10 \
        --save_step 20 \
        --max_iter 25
    echo "train_exit=$?"
    ls -la saved_models/step_20/ 2>/dev/null

    echo "================ PREDICT 100 SENTENCES ================"
    mkdir -p output_file
    timeout 900 python -u main.py \
        --do_predict True \
        --src_vocab_fpath "$DATADIR/vocab_all.bpe.32000" \
        --trg_vocab_fpath "$DATADIR/vocab_all.bpe.32000" \
        --special_token '<s>' '<e>' '<unk>' \
        --predict_file test100.en-zh \
        --batch_size 32 \
        --init_from_params saved_models/step_20/ \
        --beam_size 5 \
        --max_out_len 255 \
        --output_file output_file/predict_smoke.txt
    echo "predict_exit=$?"
    wc -l output_file/predict_smoke.txt 2>/dev/null

    echo "================ BLEU ================"
    sed -r 's/(@@ )|(@@ ?$)//g' output_file/predict_smoke.txt > output_file/predict_smoke.tok.txt
    perl mosesdecoder/scripts/generic/multi-bleu.perl test100.tok.zh < output_file/predict_smoke.tok.txt
    echo "bleu_exit=$?"
else
    echo "!!!! DATADIR 未找到，跳过训练与预测 !!!!"
fi

echo "================ COLLECT OUTPUTS ================"
BASES="$CODE_DIR/output"
[ -d /home/aistudio ] && BASES="/home/aistudio/output /home/aistudio/work/output $BASES"
for base in $BASES; do
    if mkdir -p "$base" 2>/dev/null; then
        cp /tmp/probe_run.log "$base/probe_run.log" 2>/dev/null
        cp -r output_file "$base/" 2>/dev/null
        mkdir -p "$base/saved_models/step_20"
        cp saved_models/step_20/transformer.pdparams "$base/saved_models/step_20/" 2>/dev/null
        echo "wrote to $base"
    fi
done
ls -la /home/aistudio/output 2>/dev/null
ls -la /home/aistudio/work/output 2>/dev/null

echo "================ PROBE DONE $(date) ================"
