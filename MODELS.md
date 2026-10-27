# 模型产物与复现条件

大体积模型权重不入 git，统一存放于 Hugging Face：

**<https://huggingface.co/biscuitzzz/hit-nlp-course-models>**（Public）

## 存放内容

| HF 仓库路径 | 本地对应位置 | 大小 | 说明 |
| --- | --- | --- | --- |
| `lab1/teaching-package/` | `lab1/实验一情感分析教学包/` | ~971 MB | 实验一教学包全量：两本 notebook、数据、15 个经典模型、微调模型、UER 基座、指标与曲线 |
| `lab1/report/` | `lab1/report/` | ~684 KB | 实验一报告 PDF 与插图源码 |
| `lab2/README.md` | — | — | 说明：无保留权重，复现依赖本仓库脚本、配置、结果与日志 |
| `lab3/README.md` | — | — | 说明：LoRA/DoRA adapter 未留存，复现依赖本仓库脚本、配置、日志与报告 |
| `SHA256SUMS` | — | — | 教学包全部文件的 SHA-256 校验和 |

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

- lab2 的 `saved_models/`（Paddle 训练产物）本地未保留；复现依赖 `jobs/`、`scripts/`、`configs/` 与 `results/**/logs/`，全部在 git 中。
- lab3 的 LoRA/DoRA adapter 未保留；复现依赖 `configs/`、`scripts/`、`submission/logs/` 与 `submission/results/`，全部在 git 中。
- 两个 lab 的小体积结果与日志均已入库，无需额外下载。
