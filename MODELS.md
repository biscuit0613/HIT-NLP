# 模型产物与复现条件

大体积模型权重不入 git，统一存放于 Hugging Face：

**<https://huggingface.co/biscuitzzz/hit-nlp-course-models>**（Public）

## 存放内容

| HF 仓库路径 | 本地对应位置 | 大小 | 说明 |
| --- | --- | --- | --- |
| `lab1/teaching-package/` | `lab1/实验一情感分析教学包/` | ~971 MB | 实验一教学包全量：两本 notebook、数据、15 个经典模型、微调模型、UER 基座、指标与曲线 |
| `lab1/report/` | `lab1/report/` | ~684 KB | 实验一报告 PDF 与插图源码 |
| `lab2/weights/` | HPC `lab2/code/saved_models`、`lab2/runs/*/saved_models` | ~2.5 GiB | 8 组 Transformer 终稿参数 `transformer.pdparams`，另含 `rel_pos_bias.npy` |
| `lab2/` 其余 | `lab2/` | ~18 MB | 代码、结果、日志与报告 |
| `lab3/adapters/` | HPC `lab3/runs/*/` | ~949 MiB | 7 组 LoRA/DoRA adapter，含配置、tokenizer 与 `trainer_state.json` |
| `lab3/` 其余 | `lab3/` | ~12 MB | 配置、脚本、结果、日志与报告 |
| `SHA256SUMS` | — | — | 教学包全部文件的 SHA-256 校验和 |
| `SHA256SUMS.weights` | — | — | lab2 权重与 lab3 adapter 的 SHA-256 校验和 |

## 复现条件（lab1）

| 项目 | 条件 |
| --- | --- |
| 代码与数据版本 | 本仓库提交 `101b245`（模型文件从此版本起不入库，后续提交仅增加文档） |
| 数据 | MARC 中文评论；train 20,000 / validation 4,000；seed 42；剔除 3 星中性样本；标签 0=负面、1=正面（`data/manifest.json`） |
| 基座模型 | `uer/chinese_roberta_L-8_H-512`，revision `5684143aeca0f9f4ecc91ef249b8184525767eff`；8 层 / 512 隐层 / 8 头 / vocab 21,128 |
| 微调配方 | max_length 256，batch 32，eval batch 64，epochs 3，lr 2e-5，warmup_ratio 0.1，weight_decay 0.01，seed 2024；最优 epoch 2 |
| 微调指标（验证集） | macro-F1 0.8787 / accuracy 0.8788 |
| 经典最优 | TF-IDF char1-3（min_df=1）+ 卡方 top-50,000 + LR(C=10)，macro-F1 0.8495；15 模型对照最优为 RidgeClassifier char13，macro-F1 0.8467 |
| Python 环境 | Python 3.11；torch 2.5.1；transformers 4.44.2；tokenizers 0.19.1；scikit-learn 1.5.2；numpy 1.26.4；pandas 2.2.3；jieba 0.42.1（`lab1/pyproject.toml`、`uv.lock`） |
| 硬件与耗时 | NVIDIA RTX 4060 Laptop 8GB；微调 GPU 约 8 分钟，经典全流程 CPU 约 30 分钟 |
| 随机性 | 固定 seed 2024（微调）/ 42（数据划分）；耗时随机器浮动 |

### 复现步骤

```bash
cd lab1
uv sync

# 可选：把权重从 HF 拉回本地教学包目录
hf download biscuitzzz/hit-nlp-course-models --repo-type model \
  --include "lab1/teaching-package/*" --local-dir /tmp/hit-nlp-models
```

- 微调路线：全量运行 `实验一情感分析教学包/中文预训练模型情感分类微调.ipynb`（第 6—8 节，GPU 约 8 分钟）。
- 经典路线：全量运行 `实验一情感分析教学包/经典机器学习情感分类介绍.ipynb`（第 6—10 节，CPU 约 30 分钟）。
- 权重是可再生产物，不下载也可通过上述 notebook 重新生成。
- 训练记录：`finetuned/train_log.json`、`epoch_metrics.csv`、`steps.csv`、`training_curves.png`；经典对照记录：`classical/classical15_reports.json`、`refine_final_selection.csv`。

## 使用方式

微调模型（自包含 config 与词表）：

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo = "biscuitzzz/hit-nlp-course-models"
sub = "lab1/teaching-package/finetuned"
tok = AutoTokenizer.from_pretrained(repo, subfolder=sub)
model = AutoModelForSequenceClassification.from_pretrained(repo, subfolder=sub)
# id2label: 0=负面, 1=正面
```

经典最优模型（自包含 Pipeline：tfidf → chi2 → clf）：

```python
import joblib

pipe = joblib.load("best_refined.joblib")
pred = pipe.predict(["这家店味道很好，服务也不错"])
```

其余 14 个经典模型以 `vectorizer_char13.joblib` 产出的 TF-IDF 特征为输入（`mlp` 额外含 SVD 步骤），完整用法见经典 notebook。

## 备注（lab2 / lab3）

### lab2 权重

每组只归档 `step_final/transformer.pdparams`；`transformer.pdopt`（优化器状态）与 step_5000/10000/15000 中间检查点未上传，它们只影响断点续训，不影响推理复现。

| HF 路径 | 配置 | 2018 aligned BLEU |
| --- | --- | --- |
| `lab2/weights/baseline/` | lr 2.0, warmup 8000, ls 0.1 | 11.31 |
| `lab2/weights/tuning/base_s23/` | 基线 + seed 23 | 11.01 |
| `lab2/weights/tuning/combo/` | lr 3.0, warmup 4000, ls 0.05 | 11.00 |
| `lab2/weights/tuning/lr3/` | lr 3.0 | 11.11 |
| `lab2/weights/tuning/ls05/` | ls 0.05 | 10.86 |
| `lab2/weights/tuning/warmup4k/` | warmup 4000 | 11.32 |
| `lab2/weights/improve/relpos/` | 相对位置偏置（含 `rel_pos_bias.npy`） | 11.45 |
| `lab2/weights/improve/relpos_s23/` | 相对位置偏置 + seed 23 | 10.89 |

`tuning/beam10` 不训练，复用基线模型仅改解码 beam=10（BLEU 10.88），因此没有权重。

### lab3 adapter

| HF 路径 | 方法 | rank / alpha | target | lr | 大小 |
| --- | --- | --- | --- | --- | --- |
| `lab3/adapters/lora_r8_qv/` | LoRA | 8 / 16 | q_proj, v_proj | 1e-4 | 10.1 MB |
| `lab3/adapters/dora_r8_qv/` | DoRA | 8 / 16 | q_proj, v_proj | 1e-4 | 10.5 MB |
| `lab3/adapters/lora_r16_all/` | LoRA | 16 / 32 | 全部线性层 | 2e-4 | 161.5 MB |
| `lab3/adapters/lora_r16_all_lr1e4/` | LoRA | 16 / 32 | 全部线性层 | 1e-4 | 161.5 MB |
| `lab3/adapters/dora_r16_all/` | DoRA | 16 / 32 | 全部线性层 | 2e-4 | 167.1 MB |
| `lab3/adapters/lora_r32_all/` | LoRA | 32 / 64 | 全部线性层 | 2e-4 | 323.0 MB |
| `lab3/adapters/smoke_v100s/` | LoRA（smoke 验证，20 step） | 16 / 32 | 全部线性层 | 2e-4 | 161.5 MB |

每组含 `adapter_model.safetensors`、`adapter_config.json`、tokenizer 文件与 `trainer_state.json`。
未归档：checkpoint-375（与终稿相同）、`dora_r8_qv` 的 checkpoint-200（中间快照）、`exports/*_merged`（约 15 GB/个，可由 `lab3/scripts/export_merge.py` 重建）、Qwen2-7B-Instruct 基座与数据集。

### 下载与校验

```bash
hf download biscuitzzz/hit-nlp-course-models --repo-type model \
  --include "lab2/weights/**" --include "lab3/adapters/**" \
  --local-dir hit-nlp-models
cd hit-nlp-models && sha256sum -c SHA256SUMS.weights
```

两个 lab 的小体积结果与日志均已入库，无需额外下载。
