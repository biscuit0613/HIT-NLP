# 实验三提交材料

本目录按课程要求组织：数据准备、训练与导出、推理与评测、实验报告。所有命令均在仓库根目录执行；模型、数据和输出路径可按实际位置替换。

## 0. 材料索引

| 提交要求 | 对应文件 |
| --- | --- |
| 训练数据处理脚本 | `scripts/prepare_data.py` |
| 医疗测试集处理脚本 | `scripts/prepare_medical.py` |
| CMMLU 处理脚本 | `scripts/prepare_cmmlu.py` |
| 微调训练命令与配置 | 本文第 2 节；`configs/*.yaml` |
| LoRA 训练截图 | `screenshots/02_train_lora_r8_qv.png` |
| 训练进度截图（DoRA 对照） | `screenshots/03_train_dora_r8_qv.png` |
| 导出合并命令 | 本文第 2.3 节；`scripts/export_merge.py` |
| 合并过程截图 | `screenshots/04_merge_export.png` |
| 批量推理脚本 | `scripts/gen_test.py` |
| 推理过程截图 | `screenshots/05_gen_test_inference.png` |
| 医疗评测脚本 | `scripts/eval_models.py`、`scripts/analyze_medical_extra.py` |
| CMMLU 评测脚本 | `scripts/eval_models.py`、`scripts/analyze_cmmlu_format.py` |
| 评测结果图片 | `figures/rouge_compare.png`、`figures/cmmlu_compare.png`、`figures/latency_compare.png` |
| 实验报告 | `report/report.pdf`（共 9 页：封面 + 8 页正文，含 3 张结果图；`report/report.typ` 为源码） |

整理后的运行日志见 `logs/`；截图为真实日志的终端整理版（见第 5 节说明）。

## 1. 数据准备

### 1.1 脚本与命令

| 脚本 | 作用 | 命令 |
| --- | --- | --- |
| `prepare_medical.py` | 从中文医疗对话测试集构造 100/500 条评测集，并生成去重用的 Alpaca 测试文件 | `python scripts/prepare_medical.py --input data/cmd_500.json --out-dir data/sft --n 100 --ext-n 500` |
| `prepare_cmmlu.py` | 将 CMMLU `test/computer_science.csv` 转为 204 道四选一题 | `python scripts/prepare_cmmlu.py --input data/cmmlu_test/test/computer_science.csv --out-dir data/sft` |
| `prepare_data.py` | 将 DISC-Med-SFT 多轮对话拆成“用户提问—医生回复”单轮对，去重、按对话级切分，并移除与医疗测试集重复的提问 | `python scripts/prepare_data.py --input data/disc_med/DISC-Med-SFT_released.jsonl --medical-test data/sft/test_medical_alpaca.json --out-dir data/sft --max-train 100000 --max-val 2000 --seed 42` |

建议顺序：先运行 `prepare_medical.py` 生成 `test_medical_alpaca.json`，再运行 `prepare_data.py` 做去污染。

### 1.2 数据统计

`prepare_data.py` 输出的 `data/sft/stats.json`：

- 原始对话 464,898 段，医生回复 1,362,284 条；
- 空回复 3 条，与医疗测试集重复 0 条；
- 去重后训练对 96,293 条、验证对 2,000 条（按对话级切分，训练/验证对话组分别为 460,234/4,648）；
- 训练时从训练对中随机抽取 12,000 条、训练 1 个 epoch。

医疗测试集：`test_medical.json` 100 条、`test_medical500.json` 500 条；CMMLU：204 题（A/B/C/D 分别为 52/51/51/50）。

截图：`screenshots/01_prepare_data.png`。

## 2. 训练与导出

### 2.1 环境

- Python 3.8，PyTorch 2.4.1，Transformers 4.44.2，PEFT 0.12.0；
- 单卡 32 GB 显存，fp16 + PyTorch SDPA，开启梯度检查点；
- 基座：Qwen2-7B-Instruct（7.62B 参数）。

### 2.2 训练命令与配置

主配置 L8-QV（LoRA，$r=8$，只训练注意力 `q_proj`/`v_proj`）：

```bash
python scripts/train_sft.py --config configs/lora_r8_qv.yaml --tag lora_r8_qv \
    --model models/Qwen2-7B-Instruct --data-dir data/sft \
    --out-dir runs/lora_r8_qv --meta-dir results \
    --fp16 --gradient-checkpointing
```

其余配置只需替换 `--config`、`--tag` 与 `--out-dir`，例如 DoRA 对照：

```bash
python scripts/train_sft.py --config configs/dora_r8_qv.yaml --tag dora_r8_qv \
    --model models/Qwen2-7B-Instruct --data-dir data/sft \
    --out-dir runs/dora_r8_qv --meta-dir results \
    --fp16 --gradient-checkpointing
```

| 配置 | 方法 | r | 目标模块 | 学习率 | 可训练参数 | 训练时长 | 峰值显存 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `lora_r8_qv.yaml` | LoRA | 8 | q,v | 1e-4 | 2,523,136 (0.033%) | 31.3 min | 21.55 GB |
| `lora_r16_all_lr1e4.yaml` | LoRA | 16 | all | 1e-4 | 40,370,176 (0.530%) | 40.9 min | 22.19 GB |
| `lora_r16_all.yaml` | LoRA | 16 | all | 2e-4 | 40,370,176 (0.530%) | 41.0 min | 22.19 GB |
| `lora_r32_all.yaml` | LoRA | 32 | all | 2e-4 | 80,740,352 (1.060%) | 41.1 min | 22.86 GB |
| `dora_r8_qv.yaml` | DoRA | 8 | q,v | 1e-4 | 2,637,824 (0.0346%) | 42.9 min | 18.12 GB |
| `dora_r16_all.yaml` | DoRA | 16 | all | 2e-4 | 41,760,768 (0.548%) | 2 h 05 min | 22.21 GB |

所有配置有效批大小均为 32（除 DoRA-8 为降低显存峰值采用微批 2×累积 16 外，其余均为微批 4×累积 8），截断长度 1,024，AdamW，余弦调度，warmup 3%。

LoRA 训练截图：`screenshots/02_train_lora_r8_qv.png`；同预算 DoRA 对照：`screenshots/03_train_dora_r8_qv.png`。截图中的 loss、学习率、耗时与峰值显存均来自实际训练日志（`logs/train_lora_r8_qv.log`、`logs/train_dora_r8_qv.log`）。

### 2.3 导出合并

`export_merge.py` 用 PEFT 的 `merge_and_unload()` 把适配器权重合并进基座，导出可独立加载的 fp16 checkpoint：

```bash
python scripts/export_merge.py \
    --base-model models/Qwen2-7B-Instruct \
    --adapter runs/lora_r8_qv \
    --out exports/lora_r8_qv_merged \
    --dtype float16 --device cpu
```

实际运行 111.1 秒完成，导出 7,615,616,512 个参数（4 个 safetensors 分片，共 15 GB；适配器已并入基座权重，因此参数量回到基座本身）。过程截图：`screenshots/04_merge_export.png`；运行日志：`logs/merge_lora_r8_qv.log`。

## 3. 推理与评测

### 3.1 批量推理

`gen_test.py` 用 6 个固定问题逐条生成，记录端到端时延与新词元数，用于交互层面的时延/吞吐分析：

```bash
python scripts/gen_test.py \
    --base-model models/Qwen2-7B-Instruct --adapter runs/lora_r8_qv \
    --tag lora_r8_qv --out-dir results/official \
    --dtype float16 --attn sdpa --system qwen --merge-adapter
```

解码设置为贪心搜索、重复惩罚 1.05、最多 256 个新词元，系统提示与训练时一致。输出 `gen_test_<tag>.json` 与 `.txt`。截图：`screenshots/05_gen_test_inference.png`。

### 3.2 医疗评测

`eval_models.py` 统一完成生成、ROUGE-1/2/L、长度/形态统计与 CMMLU 严格解析；医疗 100 条主实验：

```bash
python scripts/eval_models.py \
    --base-model models/Qwen2-7B-Instruct --adapter runs/lora_r8_qv \
    --tag lora_r8_qv --data-dir data/sft --out-dir results/official \
    --datasets medical,cmmlu --medical-file test_medical500.json --medical-limit 100 \
    --batch-size 8 --dtype float16 --attn sdpa --system qwen --merge-adapter
```

500 条扩展集将 `--out-dir` 改为 `results/official500`、`--datasets medical`、`--medical-limit 500`。`analyze_medical_extra.py` 从预测文件计算 BLEU-4、EM 与 jieba 词元 F1：

```bash
python scripts/analyze_medical_extra.py --results results/official
```

截图：`screenshots/06_eval_medical.png`（500 条评测进度、指标 JSON 与 BLEU/EM/F1 表）。

### 3.3 CMMLU 评测

```bash
python scripts/eval_models.py \
    --base-model models/Qwen2-7B-Instruct --adapter runs/lora_r8_qv \
    --tag lora_r8_qv --data-dir data/sft --out-dir results/official \
    --datasets cmmlu --batch-size 8 --dtype float16 --attn sdpa \
    --system qwen --merge-adapter
```

`analyze_cmmlu_format.py` 在官方“只解析选项字母”的严格口径之外，补充内容三元组匹配诊断与代理 MRR：

```bash
python scripts/analyze_cmmlu_format.py --results results/official
```

输出 `cmmlu_format.csv` 与 `cmmlu_mrr.csv`。截图：`screenshots/07_eval_cmmlu.png`、`screenshots/08_cmmlu_format.png`。

此外，`results/cmmlu_scaffold/` 保存“只输出字母”格式指令消融，`results/official_prompt/` 保存官方短系统提示的对照；对应命令只需增加 `--cmmlu-instruction "请回答上述单项选择题，只输出正确选项的字母（A、B、C、D）。"` 或把 `--system qwen` 改为 `--system official`。

### 3.4 统计分析与交互评测

```bash
# 汇总、500 次配对 bootstrap 与三张结果图
python scripts/analyze_results.py --results results/official --figs report/figs --bootstrap 500

# 交互探针：响应时间、指令遵循、角色扮演、多轮与急症安全
python scripts/hci_probe.py --base-model models/Qwen2-7B-Instruct --adapter runs/lora_r8_qv \
    --tag lora_r8_qv --out-dir results/hci_probe \
    --dtype float16 --attn sdpa --system qwen --merge-adapter
python scripts/summarize_hci.py --results results/hci_probe > results/hci_summary.csv

# 人工盲评：生成 A/B/C 随机匿名包、交互打分、汇总
python scripts/make_human_eval.py --results results --out-dir results/human_eval --n 15 --seed 42
python scripts/rate_human.py --packet-dir results/human_eval --results results
python scripts/evaluate_human.py --packet-dir results/human_eval --out-dir results
```

人工评价使用单评审 LLM（Claude）作为代理、评分前隐藏模型身份，结果作为探索性证据；评分表与复现脚本均已保留。汇总表见 `results/hci_summary.csv`、`results/human_eval_summary.csv`。

### 3.5 评测结果图

- `figures/rouge_compare.png`：7 个模型在医疗 100 条上的 ROUGE-1/2/L 对比（同时嵌入报告图 1）；
- `figures/cmmlu_compare.png`：CMMLU 严格准确率与内容匹配诊断（同时嵌入报告图 2）；
- `figures/latency_compare.png`：5 个医学问题的平均端到端时延与解码吞吐（同时嵌入报告图 3）。

详细指标表见 `results/summary.csv`、`results/paired_bootstrap.csv`、`results/medical_extra.csv`、`results/cmmlu_format.csv`。

## 4. 实验报告

- `report/report.pdf`：共 9 页（封面 1 页 + 正文 8 页，其中第 7 页为结论结尾与参考文献、第 8 页为成绩页），正文不含引用约 7 页；ACL 风格，参考文献完整；
- `report/report.typ`：Typst 源码；
- 报告内已放入 3 张结果图：图 1 医疗 ROUGE 对比、图 2 CMMLU 严格准确率与内容匹配、图 3 5 个医学问题的时延与吞吐；交互结果以表格给出。

## 5. 截图与日志说明

- 所有截图由真实运行日志渲染：loss、学习率、步数、耗时、显存、评测指标均为实际值；
- 为便于阅读，终端截图把绝对路径改写为仓库相对路径，并省略了与实验无关的框架告警、环境标识与作业调度行；
- 整理后的日志保存在 `logs/`：LoRA 训练、DoRA 训练、导出合并各一份；数值与截图一致，已去除框架告警与机器标识、路径改为仓库相对路径；每个截图旁附有同名 `.txt` 文本版。

## 6. 主要结果（训练系统提示）

| 模型 | R-1 | R-2 | R-L | BLEU-4 | F1 | CMMLU 严格 | CMMLU 内容 | 平均长度 | 时延 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 基座 | 0.2150 | 0.0338 | 0.1359 | 0.0071 | 0.258 | 61.3 | 77.5 | 437 | 8.4 s |
| L8-QV | **0.2284** | **0.0476** | **0.1807** | 0.0166 | **0.303** | 27.0 | 70.1 | 257 | 3.7 s |
| L16-1e4 | 0.2323 | 0.0424 | 0.1813 | 0.0174 | 0.299 | 48.0 | 75.0 | 233 | 2.3 s |
| L16-2e4 | 0.2277 | 0.0400 | 0.1786 | 0.0159 | 0.299 | 47.5 | 71.6 | 254 | 2.2 s |
| L32-ALL | 0.2311 | 0.0423 | 0.1815 | 0.0158 | 0.299 | 51.0 | 72.6 | 230 | 2.1 s |
| DoRA-8 | 0.2254 | 0.0486 | 0.1802 | 0.0175 | 0.296 | 25.5 | 69.6 | 236 | 3.5 s |
| DoRA-16 | 0.2236 | 0.0387 | 0.1787 | 0.0151 | 0.297 | 49.0 | 71.6 | 252 | 2.2 s |

医疗列为前 100 条 F1；时延为 5 个医学问题的平均端到端时延；500 条上 ROUGE-L：基座 0.1446、L8-QV 0.1915、L32-ALL 0.1919、DoRA-8 0.1911、DoRA-16 0.1902。CMMLU “严格”为官方只解析选项字母的准确率，“内容”为字符三元组匹配诊断；严格准确率下降主要来自格式缺失而非知识遗忘，详见报告第 4 节与 `results/cmmlu_format.csv`。
