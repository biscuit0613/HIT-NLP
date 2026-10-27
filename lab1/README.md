# 自然语言处理 C 实验一：基于分类模型的情感分析

经典机器学习模型与预训练模型微调（UER 中文 RoBERTa-Medium）。
环境由 [uv](https://docs.astral.sh/uv/) 管理，报告由 [typst](https://typst.app/) 撰写。

## 目录结构

```
lab1/
├── pyproject.toml / uv.lock / .python-version   # uv 环境定义（Python 3.11）
├── 实验一情感分析教学包/                          # 最终交付范围（验收主线）
│   ├── 经典机器学习情感分类介绍.ipynb              # 经典路线全流程（第 6—10 节为补全部分）
│   ├── 中文预训练模型情感分类微调.ipynb            # 微调路线全流程（第 6—8 节为补全部分）
│   ├── data/{train,validation}.csv               # 训练 20,000 / 验证 4,000
│   ├── classical/                                # 经典 notebook 产物（结果/预测/模型/特征）
│   ├── finetuned/                                # 微调 notebook 产物（最优模型+逐轮记录+曲线）
│   └── pretrained/                               # UER 预训练权重与 WordPiece 词表
└── report/                                       # typst 实验报告（report.typ → report.pdf，figs/ 为插图）
```

教学包之外的分析脚本与中间产物（src/、results/、models/、artifacts/）属前期探索
过程，最终已按交付口径清理，可经 git 历史回溯（见下）。

## 环境准备

```bash
uv sync                # 按 pyproject.toml 安装全部依赖（已用清华镜像）
```

## 验收 notebook（主线）

教学包内两本 notebook 为最终验收物，均在教学包目录打开、全量运行：

- `经典机器学习情感分类介绍.ipynb`：第 6—10 节完成 TF-IDF 表示对照、15 种分类器
  训练与评价、四组改进对照与改进最优管线（CPU 约半小时），产物写入 `classical/`；
- `中文预训练模型情感分类微调.ipynb`：第 6—8 节完成微调训练、每轮验证与最优模型
  保存（GPU 约 8 分钟），产物写入 `finetuned/`，配方与报告微调最优单模型一致
  （len256 + 种子 2024）。

两本 notebook 的验证指标均与报告逐项核对一致（经典最优 0.8495、微调最优 0.8787）；
表格中的耗时列随机器环境浮动，属正常现象。

## 报告编译

```bash
cd report && typst compile report.typ        # 生成 report.pdf
```

## git 管理

仓库根为 `lab1/`。tag `lab1-complete` 与提交历史存档了实验一完成时的完整状态
（含 src/ 全套分析脚本与 results/ 全部中间结果），可随时回溯对照：

```bash
git log --oneline            # 提交历史
git diff                     # 查看未提交改动
git checkout lab1-complete   # 回溯到完整存档（只读对照）
git checkout main            # 回到主线
```

`.gitignore` 排除 `.venv/`、教学包内预训练权重与缓存，以及两本 notebook 产物中的
大体积模型文件（全量重跑 notebook 可再生）。

## 说明

- 训练设备：NVIDIA RTX 4060 Laptop（8GB），CUDA 版 PyTorch。
- 主要成绩（验证集 Macro-F1）：经典最优 0.8495；微调最优单模型 0.8787；
  三种子集成 0.8777。报告 `report/report.pdf` 中全部数值与实验留档逐项核对一致。
- 教学包目录仅补全两本 notebook 的实验节（验收要求），其余内容未改动
  （word 级表示所需的 jieba 补入 requirements-sklearn.txt）；
  `classical/`、`finetuned/` 为 notebook 全量运行可再生的产物。
