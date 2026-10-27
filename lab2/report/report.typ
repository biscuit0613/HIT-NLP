// ============================================================
// 实验二：基于 Transformer 神经机器翻译建模与优化 —— 实验报告（typst）
// 编译：typst compile report.typ
// 数值来源：results/ 下的训练日志与 BLEU 日志（逐项核对一致）
// ============================================================

#set page(
  paper: "a4",
  margin: (x: 2.4cm, top: 2.7cm, bottom: 2.4cm),
  header: align(center, text(size: 9pt)[《自然语言处理 C》实验报告]),
  footer: context align(center, text(size: 9pt, counter(page).display())),
)
#set text(font: ("Noto Serif CJK SC", "Liberation Serif"), size: 10pt, lang: "zh")
#set par(justify: true, leading: 0.72em, first-line-indent: 2em)
#set heading(numbering: "1.1")

#show heading.where(level: 1): it => {
  set text(size: 12pt, font: "Noto Sans CJK SC", weight: "bold")
  block(above: 1.1em, below: 0.55em, it)
}
#show heading.where(level: 2): it => {
  set text(size: 10.5pt, font: "Noto Sans CJK SC", weight: "bold")
  block(above: 0.8em, below: 0.4em, it)
}
#show figure.caption: set text(size: 9pt)
#show figure: set block(breakable: false)

// ---------- 封面（无页眉页码）----------
#page(header: none, footer: none, margin: (x: 2.4cm, top: 4.5cm, bottom: 2.4cm))[
  #align(center)[
    #text(size: 22pt, font: "Noto Sans CJK SC", weight: "bold")[哈尔滨工业大学] #v(1.6cm)
    #text(size: 18pt, font: "Noto Sans CJK SC")[自然语言处理] #v(0.4cm)
    #text(size: 18pt, font: "Noto Sans CJK SC")[实验报告] #v(1.2cm)
    #text(size: 13pt, font: "Noto Sans CJK SC")[（2026 年秋季学期）] #v(3.2cm)
  ]
  #align(center)[
    #text(size: 12pt, font: "Noto Sans CJK SC")[
    #grid(
      columns: (auto, auto),
      row-gutter: 14pt, column-gutter: 8pt,
      align: (right, left),
      [姓名：], [#box(width: 6.6cm, stroke: (bottom: 0.6pt))[#align(center)[邴海诺、高新来]]],
      [学号：], [#box(width: 6.6cm, stroke: (bottom: 0.6pt))[#align(center)[2024112353、2024112774]]],
      [学院：], [#box(width: 6.6cm, stroke: (bottom: 0.6pt))[#align(center)[计算学部]]],
      [教师：], [#box(width: 6.6cm, stroke: (bottom: 0.6pt))[#align(center)[杨沐昀]]],
    )
    ]
  ]
]

// ---------- 正文 ----------
#set page(header: align(center, text(size: 9pt)[《自然语言处理 C》实验报告]),
          footer: context align(center, text(size: 9pt, counter(page).display("1"))))
#counter(page).update(1)

#align(center)[
  #text(size: 14pt, font: "Noto Sans CJK SC", weight: "bold")[
    实验二：基于 Transformer 神经机器翻译建模与优化
  ]
  #text(size: 11pt, font: "Noto Sans CJK SC")[
    ——参数调优与相对位置偏置改进
  ]
]
#v(0.4em)

*摘要*　本文在 WMT18 英汉平行语料（252,385 个句对，共享 BPE 词表 32,000）上复现了 Transformer 基线翻译模型（约 6,050 万参数），并在统一的 10 epoch 训练预算下完成性能分析与代码级改进两部分工作。评测前我们发现，项目默认的 `max_length = 256` 过滤会在 newstest2018 中跳过 2 句超长句，若直接对未过滤参考按行号计算 BLEU，从第 1415 行起译文将整体错位。本文因此采用对齐口径（从参考中剔除同样两句，3,979 个句对），并同时报告课程口径以便与课程参考值比较。对齐口径下基线在 newstest2017/newstest2018 上的 BLEU 为 10.24/11.31；按未对齐的课程口径，2018 仅为 4.65，仍高于课程参考值 4.46。性能分析围绕 Noam 学习率系数、预热步数、标签平滑与解码束宽设计单因素对照：预热缩短至 4000 步在 2017 上取得最大单项增益（10.59，+0.35），但在 2018 上基本持平（+0.01）；标签平滑降至 0.05 与组合配置分别使 2018 下降 0.45 与 0.31，束宽增至 10 则使两个测试集同时下降（-0.07/-0.43）。随机种子对照显示，仅更换种子即可让基线在 2017 上变动 0.54、在 2018 上变动 0.30（相对位置偏置配置下为 0.56），单次运行的 0.2--0.4 调参增益因此不能视为可靠。模型改进实现了 T5 式分桶相对位置偏置：在编码器每层为每个注意力头引入 32 个距离桶的可学习偏置，并以零初始化保证与基线功能等价起步。该改动仅增加 1536 个参数，默认种子下 BLEU 为 10.73/11.45，提升 +0.49/+0.14，但同种子成对对照中相对基线为 -0.06/-0.12，说明其性能与基线相当、未能带来可复现的提升；偏置参数可视化显示各层均学到了邻近优先、且方向偏好随层深反转的注意力先验。本文同时讨论了单次 BLEU 评测的噪声、训练损失与翻译质量的不一致以及评测口径对结论的影响。

*关键词*　神经机器翻译；Transformer；相对位置编码；超参数调优；BLEU；PaddlePaddle

= 任务与数据

本实验在英汉方向上构建端到端的神经机器翻译系统。训练语料为 WMT18 英汉平行数据（Bojar 等，2018），共 252,385 个句对，已完成 tokenization 与清洗，并使用大小为 32,000 的共享 BPE 子词词表切分；源语言与目标语言共用词表，为后续的嵌入—输出权重共享提供了条件。评测使用 newstest2017（2,001 句）与 newstest2018（3,981 句）两个标准测试集。

翻译质量采用语料级 BLEU 评价，由 mosesdecoder 的 `multi-bleu.perl` 计算：统计 1--4 gram 精度并以几何平均汇总，再乘以长度惩罚因子 $op("BP")$。解码结果先合并 BPE 续接符（将 `@@ ` 删除）再与 `newstest20XX.tok.zh` 参考答案比对，与课程给定的评测流程一致。

评测流程中有一个必须处理的细节：项目默认的 `max_length` 为 256，预测时数据读取器按源文长度加 1、目标长度加 2 中的较大者过滤样本；newstest2018 中第 1415、3708 句超过该上限（源文分别 293、279 个 BPE 词元），被跳过，译文只有 3,979 行，而参考译文有 3,981 行。若直接按行号与未过滤参考比对，从第 1415 行起所有译文都会错位一行，第 3708 行后错位两行，BLEU 会被严重压低。本文采用对齐口径：从参考中剔除被过滤的同样两句，得到 3,979 个句对；同时报告课程原始口径（未过滤参考）以便与课程参考值比较。newstest2017 没有超长句，两种口径完全一致。课程参考值 4.46 的 `ref_len = 90019` 恰对应前 3,979 行参考（完整 3,981 行为 90,089 个词元），说明课程基线也处于同一错位口径；因此 4.46 只能与课程口径下的 4.65 对比，不能与对齐口径的绝对值直接比较。

= 方法

== 基线模型

基线模型为标准 Transformer（Vaswani 等，2017）编码器—解码器结构：编码器与解码器各 6 层，每层多头注意力 8 个头，模型维度 $d_(op("model")) = 512$，前馈网络隐层 2048，每个头的键值维度 $d_k = d_v = 64$（即 $d_(op("model")) \/ 8$）。模型采用 Pre-LN 结构（子层输入先做 LayerNorm），dropout 统一为 0.1，源/目标嵌入与输出投影三处共享同一权重矩阵，位置信息由正弦绝对位置编码提供。按上述配置统计，模型参数量约 6,050 万。

训练细节保持与实验项目默认值一致：批大小按词元数组织，每个 batch 最多 4096 个词元（含填充）；优化器为 Adam（$beta_1 = 0.9$，$beta_2 = 0.997$，$epsilon = 10^(-9)$）；学习率采用 Noam 调度，基础系数 2.0、预热 8000 步，峰值学习率约 $1.0 times 10^(-3)$ 并在其后按步数的平方根倒数衰减；标签平滑系数 0.1；共训练 10 个 epoch（约 18,400 步），每 5000 步保存一次检查点，最终以 `step_final` 解码。解码采用束搜索，基线束宽 5，最大输出长度 255 个 BPE 词元。

== 性能分析实验设计

本次实验的训练预算固定为 10 个 epoch。在这一约束下，18,400 步远小于原论文训练 base 模型的 10 万步，因此调度参数与正则化强度的相对作用会与标准设置不同。既有研究与训练实践也表明，学习率调度与正则化是 Transformer NMT 中最敏感的两类因素（Popel 与 Bojar，2018；Chen 等，2018）。基于这一观察，选取四个因素做单因素对照，每个因素的动机如下。

*（1）学习率系数（lr3）。* Noam 调度下峰值学习率与基础系数成正比。固定步数预算下适度提高学习率，可在衰减前覆盖更远的参数空间；若学习率过高则可能破坏收敛。将系数由 2.0 提高到 3.0（峰值约 $1.5 times 10^(-3)$）。

*（2）预热步数（warmup4k）。* 基线预热占总步数的 43%，衰减阶段仅约 10,400 步。将预热缩短至 4000 步，使模型更早进入高学习率区间并延长有效衰减段；需要注意，Noam 调度的峰值学习率与 $sqrt(op("warmup"))$ 成反比，缩短预热同时会把峰值由约 $1.0 times 10^(-3)$ 抬升至约 $1.4 times 10^(-3)$，因此该实验检验的是“更短的预热 + 更长的衰减”这一调度形状，而非单纯的预热长度。

*（3）标签平滑（ls05）。* 标签平滑（Szegedy 等，2016）通过把目标分布与均匀分布混合来抑制过度自信，是 NMT 的常用正则手段。基线取 0.1；在数据规模有限、训练轮数固定的条件下，降低至 0.05 可减轻正则化对拟合能力的压制，但同时可能削弱模型的泛化性，因此需要实验判定。

*（4）解码束宽（beam10）。* 束搜索宽度不改变模型参数，只影响推断阶段的搜索质量。基线束宽为 5，将其提高到 10，考察在低 BLEU 的英中任务上扩大搜索空间的边际收益。

在上述单因素实验之后，将训练端的有效改动组合为 combo 配置，检验收益是否叠加。此外，鉴于 BLEU 是离散的语料级指标，单次训练的差异可能来自初始化与数据顺序的随机性，本文以 `--random_seed 23` 重训基线与改进模型 base_s23、relpos_s23，并以同种子的成对比较判断改进是否真实。

== 相对位置偏置改进

Transformer 的注意力打分只由内容向量的点积决定，绝对位置编码在进入第一层后仅以加性方式影响内容表示，注意力本身没有显式的距离先验。相对位置表示（Shaw 等，2018）直接把键与查询的相对距离注入注意力打分，使模型对平移更稳定；T5（Raffel 等，2020）进一步用分桶方式把任意距离压缩到有限个桶中，兼顾短距离的精确性与长距离的外推能力。

本文在编码器自注意力中实现 T5 式分桶相对位置偏置。对位置 $i$ 的查询与位置 $j$ 的键，令相对距离 $d = j - i$，将其映射为桶编号 $op("bucket") (d) in {0, dots, 31}$：正负方向各占 16 个桶，绝对距离 0--7 逐距离精确分桶，8 及以上按对数间隔映射到剩余桶，超过 128 的距离并入最后一桶。每一层、每一个注意力头拥有独立的可学习偏置表 $bold(B)_l in RR^(32 times 8)$，注意力打分修正为

$ a_(i j)^h = (bold(q)_i^h dot bold(k)_j^h) / sqrt(d_(op("model"))) + B_(l, op("bucket") (j - i), h). $

项目实现沿用原始 static-graph 版本的缩放方式，以 $d_(op("model"))$ 而非每头维度 $d_k$ 作分母；本文保持该设置，使改进模型与基线处于同一缩放尺度。

偏置表以全零初始化，因此改进模型在训练起点与基线完全等价，偏置只在数据确有需要时才增长；新增参数仅 $6 times 32 times 8 = 1536$ 个，相对 6,050 万参数量可以忽略。实现上，编码器层内由位置编号构造距离矩阵、分桶查表并把偏置广播到注意力 logits（代码见下）。改进只施加于编码器：解码器增量解码时只保留历史键值的缓存、逐词生成，查询在序列中的绝对位置需要额外跟踪，若在缓存路径中引入相对偏置，则要同时改写束搜索的缓存与打分逻辑，风险与工作量都明显更高；编码器一侧已足以检验“显式距离先验是否有益”这一假设，解码器侧扩展留作后续工作。

#figure(
  raw(
    block: true,
    lang: "python",
    "# 每层每头独立的距离桶偏置表，零初始化保证与基线等价\nself.rel_pos_bias = nn.Embedding(\n    REL_POS_NUM_BUCKETS, n_head,\n    weight_attr=nn.initializer.Constant(0.0))\n\nseq_len = x.shape[1]\nposition_ids = paddle.arange(seq_len, dtype=\"int64\")\nrelative_position = position_ids[None, :] - position_ids[:, None]\nbuckets = relative_position_bucket(relative_position)\nbias = self.rel_pos_bias(buckets).transpose([2, 0, 1]).unsqueeze(0)\nattn_output = self.self_attn(self.self_attn_ln(x), None, None,\n                             attn_bias + bias)",
  ),
  caption: [编码器中相对位置偏置的关键实现（完整实现见 `improve/transformer.py`）],
)

= 实验设置

实验代码以课程提供的 Paddle Transformer 参考实现（AI Studio 项目 10747476）为基础。全部训练与解码在实验室 HPC 集群上完成（Slurm 调度，单卡 NVIDIA Tesla V100-SXM2-16GB，16 GB 显存，计算能力 7.0），软件环境为 Python 3.11、PaddlePaddle 3.3.1（CUDA 12.6；Ma 等，2019）。每个任务的训练阶段约 51 分钟（约 6.05 step/s），加上两个测试集的解码与 BLEU 评测后总耗时约 1 小时。除 2.3 节的模型改进外，所有对照实验共享同一份训练数据、词表、模型结构与训练轮数，仅改变 2.2 节所述的因素；改进实验与对应的种子对照共享除偏置模块以外的全部配置，以保证比较的可控性。基线训练超参数汇总于表 1。各实验的训练与 BLEU 原始日志、对齐评测脚本（`scripts/eval_aligned_bleu.py`）、日志解析与绘图脚本均随代码提交，报告中的数值可由日志复算。

#figure(
  kind: table,
  supplement: none,
  caption: [表 1　基线模型结构与训练超参数],
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    stroke: (x, y) => (
      top: if y == 0 { 0.9pt } else { 0pt },
      bottom: if y == 1 { 0.5pt } else if y == 11 { 0.9pt } else { 0pt },
    ),
    inset: 3.2pt,
    text(size: 8.2pt)[*项目*], text(size: 8.2pt)[*取值*], text(size: 8.2pt)[*说明*],
    [编码器 / 解码器层数], [6 / 6], [Pre-LN，dropout 0.1],
    [注意力头数], [8], [$d_k = d_v = 64$],
    [模型维度 / 前馈维度], [512 / 2048], [嵌入、输出与源端共享权重],
    [批大小], [4096], [按词元数组织（含填充）],
    [训练轮数], [10], [约 18,400 步],
    [优化器], [Adam], [$beta_1 = 0.9$，$beta_2 = 0.997$，$epsilon = 10^(-9)$],
    [学习率调度], [Noam], [系数 2.0，预热 8000 步，峰值约 $1.0 times 10^(-3)$],
    [标签平滑], [0.1], [混合均匀分布的强度],
    [检查点间隔], [5000 步], [最终模型为 `step_final`],
    [解码], [束宽 5], [最大输出长度 255],
  ),
)

= 结果与分析

== 基线复现

基线模型训练 10 epoch（含两个测试集的解码与评测共约 58 分钟）后完成 BLEU 计算。表 2 给出对齐口径的 BLEU 及其分解指标。newstest2017 上 BLEU = 10.24；newstest2018 上为 11.31（42.2/15.7/7.1/3.5）。两个测试集上长度惩罚因子均为 1.000，说明译文整体不短于参考译文；ratio 分别为 1.035 与 1.054，译文略长，但未触发长度惩罚。2018 的 hyp_len 为 94,571，对齐后参考长度为 89,721，多出的约 5% 主要来自中文译文中重复或冗余的表达，这也是后文改进希望缓解的现象之一。若改用课程口径（未过滤参考），同一批译文在 2018 上只有 4.65（23.8/5.8/2.6/1.3，ratio 1.051，hyp/ref 为 94,571/90,019），比课程参考值 4.46 高 0.19；但两个数都受前述错位影响，不能作为翻译质量的绝对度量。

#figure(
  kind: table,
  supplement: none,
  caption: [表 2　基线模型在两个测试集上的 BLEU 及分解指标（2018 同时给出对齐口径与课程口径）],
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    stroke: (x, y) => (
      top: if y == 0 { 0.9pt } else { 0pt },
      bottom: if y == 1 { 0.5pt } else if y == 4 { 0.9pt } else { 0pt },
    ),
    inset: 3.2pt,
    text(size: 8.2pt)[*测试集*], text(size: 8.2pt)[*BLEU*],
    text(size: 8.2pt)[*1--4 gram 精度*], text(size: 8.2pt)[*BP*],
    text(size: 8.2pt)[*ratio*], text(size: 8.2pt)[*hyp/ref 长度*],
    [newstest2017（对齐）], [10.24], [40.7/14.5/6.3/3.0], [1.000], [1.035], [48778/47144],
    [newstest2018（对齐）], [11.31], [42.2/15.7/7.1/3.5], [1.000], [1.054], [94571/89721],
    [newstest2018（课程口径）], [4.65], [23.8/5.8/2.6/1.3], [1.000], [1.051], [94571/90019],
    [2018 参考值（课程口径）], [4.46], [23.3/5.7/2.5/1.2], [1.000], [1.087], [97832/90019],
  ),
)

图 1 给出基线、warmup4k 与改进模型 relpos 的训练损失曲线（每 100 步的原始记录与指数滑动平均）。损失从约 11 降至 3.0 附近，前 5000 步下降最快，随后进入缓慢改善阶段，末段出现明显波动。预热 4000 步的曲线在大部分训练区间低于基线，符合“更早进入高学习率、衰减段更长”的预期；relpos 的曲线与基线大部分重合，末段滑动平均略低于基线但仍高于 warmup4k，三者的末段差距不足 0.1，远小于它们的 BLEU 差异——训练损失更低并不必然意味着翻译质量更好，二者关系将在 4.2 节结合测试结果讨论。

#figure(
  image("figs/loss_curves.png", width: 100%),
  caption: [基线、warmup4k 与 relpos 的训练损失曲线（细线为原始值，粗线为滑动平均）],
)

== 调参实验结果

各实验的 BLEU 汇总于表 3，图 2 以柱状图对照两个测试集。基线一行为参照，其余各行给出相对基线的差值。

#figure(
  kind: table,
  supplement: none,
  caption: [表 3　性能分析与改进实验的 BLEU 汇总（2018 为对齐口径；差值相对基线）],
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, left, right, right, right, right),
    stroke: (x, y) => (
      top: if y == 0 { 0.9pt } else { 0pt },
      bottom: if y == 1 { 0.5pt } else if y == 10 { 0.9pt } else { 0pt },
    ),
    inset: 3.2pt,
    text(size: 8.2pt)[*实验*], text(size: 8.2pt)[*改动*],
    text(size: 8.2pt)[*BLEU 2017*], text(size: 8.2pt)[*BLEU 2018*],
    text(size: 8.2pt)[*Δ2017*], text(size: 8.2pt)[*Δ2018*],
    [baseline], [—], [10.24], [11.31], [—], [—],
    [lr3], [学习率系数 2.0 → 3.0], [10.47], [11.11], [+0.23], [-0.20],
    [warmup4k], [预热 8000 → 4000 步], [10.59], [11.32], [+0.35], [+0.01],
    [ls05], [标签平滑 0.1 → 0.05], [10.53], [10.86], [+0.29], [-0.45],
    [combo], [lr3 + warmup4k + ls05], [10.63], [11.00], [+0.39], [-0.31],
    [base_s23], [基线，随机种子 23], [10.78], [11.01], [+0.54], [-0.30],
    [beam10], [束宽 5 → 10], [10.17], [10.88], [-0.07], [-0.43],
    [relpos], [编码器分桶相对位置偏置], [10.73], [11.45], [+0.49], [+0.14],
    [relpos_s23], [相对位置偏置，种子 23], [10.72], [10.89], [+0.48], [-0.42],
  ),
)

单因素实验呈现出清晰而一致的模式：三项训练端改动（lr3、warmup4k、ls05）都提升了 2017 的成绩，增幅分别为 +0.23、+0.35 与 +0.29；在 2018 上，warmup4k 基本持平（+0.01），lr3 下降 0.20，标签平滑降至 0.05 则下降 0.45。这一分化说明，缩短预热与提高学习率主要改变了模型在 2017 上的表现，而 2018 对这类训练端改动不敏感；降低标签平滑则在 2017 上有效、在 2018 上明显削弱泛化。后者与“标签平滑是重要的泛化正则”的判断一致，但 0.45 的降幅仍与后文估计的种子方差（0.30--0.56）部分重叠，需谨慎解读。

#figure(
  image("figs/bleu_compare.png", width: 100%),
  caption: [各实验在 newstest2017 与对齐口径 newstest2018 上的 BLEU],
)

需要强调 BLEU 的统计稳定性。两个测试集分别只有 2,001 与 3,979 个可评句对，评测本身存在不可忽略的抽样波动；更主要的不确定性来自训练：不同随机种子会改变初始化与数据顺序，使同一配置的 BLEU 出现远大于评测噪声的差异。因此表中 0.0x--0.x 量级的差异不能直接解释为方法优劣，必须结合种子对照（base_s23 与 relpos_s23）判断。本文在改进实验中采用同种子成对比较，正是为了把“方法带来的差异”与“训练随机性”分开。

组合配置与种子对照进一步修正了上述判断。combo 在 2017 上达到 10.63，较基线提升 +0.39，为调参组中的最高值；但在 2018 上为 11.00（-0.31），低于 lr3（11.11）与 warmup4k（11.32），仅高于 ls05（10.86），说明提高学习率与减弱标签平滑叠加后没有带来跨测试集的收益。更关键的是种子对照：仅把随机种子固定为 23，基线在 2017 上就从 10.24 升至 10.78（+0.54），高于本文所有调参单项的增益；同一改动在 2018 上则使基线由 11.31 降至 11.01（-0.30），而相对位置偏置配置换种子后 2018 由 11.45 降至 10.89（-0.56）。也就是说，在 10 epoch 预算与小测试集下，0.2--0.5 量级的差异与种子方差同量级，单次运行的调参结论不能作为可靠证据；ls05（-0.45）与 beam10（-0.43）的下降虽接近或略超过 0.30，但仍落在另一配置 0.56 的种子波动范围内，不能排除随机性。解码端的 beam10 也支持谨慎解读：束宽由 5 增至 10 后两个测试集分别下降 0.07 与 0.43，扩大搜索空间并未带来收益，反而放大了模型在长度与校准上的偏差。

== 模型改进结果

相对位置偏置模型（relpos）在相同数据与训练预算下达到 BLEU 10.73/11.45，较基线提升 +0.49/+0.14；参数量仅增加 1536 个，训练耗时与基线基本一致。仅看这一组对照，2017 上的提升约为调参实验单项最优（warmup4k，+0.35）的 1.4 倍；2018 上的提升为 +0.14，与 warmup4k 的 +0.01 一样没有降低 2018，但幅度远小于种子方差。也就是说，relpos 与 warmup4k 是仅有的未降低 2018 BLEU 的配置，但这不足以证明瓶颈不在位置建模。结合 4.2 节揭示的种子方差，该比较尚不足以证明显式距离先验带来了真实增益；下一段的同种子对照将给出更严格的判断。从 n-gram 分解看，2017 上 1--4 gram 精度由 40.7/14.5/6.3/3.0 全面提升到 41.5/14.9/6.6/3.2，译文长度比由 1.035 降至 1.023，句内重复的 4-gram 比例由 2.58% 降至 1.45%，译文更简洁；2018 上 1--4 gram 精度由 42.2/15.7/7.1/3.5 变为 41.9/15.8/7.3/3.6，1-gram 略降而 2--4 gram 略升，译文长度比由 1.054 升至 1.056，句内重复比例由 2.03% 升至 2.43%，表明两版译文的差异集中在局部搭配而非词汇覆盖，与 2018 增益有限的现象一致。

实例层面也能看到两版译文的差异。例 1：源句 “Forget the pressure.”，参考译文 “忘掉压力。”，基线译为 “忘却了这一压力。”，多出“了/这/一”三个词元，而改进模型给出 “忘记压力。”，更贴合参考的简洁表达。例 2：源句 “It is in itself a great shame.”，参考译文使用“耻辱”，基线选择了近义的“羞耻”，改进模型则与参考用词一致。例 3（较长句）：源句描述“回国后在华人与热爱中国文化的俄罗斯人中传播中华传统文化”，基线将谓语误译为“在……文化中占据一席之地”，丢失了“传播”这一核心动作，改进模型正确译出“在……中间传播传统的中国文化”。这些例子仅作定性说明：relpos 的译文在部分句子上更简洁或更贴近参考，但个例差异不能替代语料级指标，且其中同样可能混有种子差异。

同种子对照给出了更保守的结论。base_s23 与 relpos_s23 除偏置模块外配置完全相同，且零初始化使二者在训练起点逐位一致（第 100 步损失分别为 9.356938 与 9.356940）。在此条件下，relpos_s23 的 BLEU 为 10.72/10.89，相对 base_s23 的 10.78/11.01 分别下降 0.06 与 0.12。也就是说，相对默认基线的 +0.49/+0.14 在同种子对照中消失，不能归因于偏置模块；2017 上基础模型仅由种子带来的波动（+0.54）甚至大于改进的表面增益，2018 上两次基线运行相差 0.30，同样大于 +0.14。由于每个配置仅有两个种子，本文不对均值差作更强推断，但可以确认：在 10 epoch 预算下，编码器相对位置偏置没有带来可复现的 BLEU 提升。同时，它以极小的参数代价保持了与基线相当的翻译性能，并且（如下文所示）学到了可解释的距离先验。值得注意的是，两次 relpos 运行在 2017 上几乎重合（10.73 与 10.72），但在 2018 上相差 0.56（11.45 与 10.89）；因此不能据此断言偏置降低了训练随机性，2017 上的稳定性更可能是该测试集对此改动不敏感。这一负结果也说明，在中小规模数据与短训练预算下，位置建模可能不是当前 BLEU 的主要瓶颈，词汇覆盖与解码行为的影响更大。

为确认偏置并非退化为零或仅拟合噪声，导出训练完成的 `step_final` 中各层偏置表（每层 32 桶 × 8 头）。统计显示 6 层的偏置绝对值最大为 2.5--3.4，标准差约 0.8--1.1，远非零初始化状态，说明模型确实在使用这一自由度。图 3 按代表距离绘制各层各头的偏置曲线（横轴为对称对数坐标，0 左侧为反向距离、右侧为正向距离）。曲线呈现出三点规律：其一，偏置在相邻位置最强（$|d| = 1$ 处平均 $|B|$ 约 1.5），随后随精确分桶距离迅速衰减，到 $|d| = 7$ 时约为 0.7；在对数分桶的中长距离上幅度维持在 0.8--1.1，直到距离超过约 64 后才回落，末端桶（$|d| > 90$）接近零，说明模型主要学到了“邻近优先”，而远端桶可能因样本稀少未形成稳定偏好。其二，方向偏好随层深反转：第 0--3 层的正反向平均偏置之差为 +0.06 至 +0.31，整体略偏好正向（关注后文），第 4--5 层转为 -0.13 与 -0.21，略偏好反向（关注前文）。其三，同一层内不同头的方向差异随层深增大，第 3--5 层各头正反向平均偏置之差的绝对值平均为 0.5--0.6；深层对近距离的偏置幅度也明显大于远距离（近/远比 1.3--1.7），而第 0 层相反（0.62），表明距离先验的尺度随表示层级而变化。

#figure(
  image("figs/rel_pos_bias.png", width: 100%),
  caption: [相对位置偏置随距离的变化（每层 8 个头，横轴负值为关注前文、正值为关注后文）],
)

== 讨论

综合以上结果，可以得到三点认识。第一，在固定 10 epoch 的预算下，Transformer 的训练损失与翻译质量并不单调对应：warmup4k 的训练损失最低且 2017 单次最好，但 2018 仅与基线持平；relpos 的末段损失略低于基线，而其 BLEU 优势在同种子对照中并不成立；标签平滑模型的损失因目标分布改变而不可与基线直接比较。因此调参与改进的判据必须是独立的测试集 BLEU 并辅以种子对照，而不是训练损失或单次运行。第二，评测口径会显著改变 2018 的绝对分数：默认过滤跳过 2 句超长句后，3,979 行译文对 3,981 行参考会从第 1415 行起整体错位，使基线 BLEU 从对齐口径的 11.31 降到课程口径的 4.65；课程参考值 4.46 的 `ref_len` 也显示其处于同一错位口径。对齐后 2018 并不比 2017 难，两个测试集上的种子方差（0.54 与 0.30--0.56）才是解释小差异的主要因素；ls05（-0.45）与 beam10（-0.43）的下降幅度较大，但仍需更多种子确认。改进工作的价值应结合两个测试集与种子对照综合判断：本文的 relpos 在默认种子下两项指标均有小幅提升（+0.49/+0.14），但同种子对照中该差异消失（-0.06/-0.12），因此只能认为其性能与基线相当。第三，单次 BLEU 的噪声远大于通常预期：仅更换种子即可让基线在 2017 上变动 0.54、在 2018 上变动 0.30，relpos 配置在 2018 上甚至变动 0.56，超过本文多数调参增益。本文的成对实验表明，只有同种子比较才能把方法效应与种子方差分开；受算力额度限制，每个配置仅覆盖 1--2 个种子，更严格的结论需要 3 个以上种子的均值与置信区间。

= 结论

本文在 WMT18 英汉数据上复现了约 6,050 万参数的 Transformer 基线。评测时发现默认 `max_length = 256` 过滤会跳过 newstest2018 中的 2 句超长句，导致按未过滤参考评分时后续行整体错位；采用对齐口径后，基线在 newstest2017/2018 上取得 BLEU 10.24/11.31，而课程口径下 2018 为 4.65，仍高于课程参考值 4.46（二者均受同一错位影响）。性能分析表明：缩短预热至 4000 步与提高学习率系数在单次运行中改善了 2017 的 BLEU（最高 +0.35），但在 2018 上分别为 +0.01 与 -0.20；降低标签平滑、组合配置与增大束宽则损害 2018 的表现（-0.45、-0.31、-0.43）。种子对照显示 2017 的种子方差为 0.54、2018 为 0.30--0.56，大于或接近上述单次差异，因此调参结论需要更严格的重复实验。模型改进方面，本文实现了 T5 式分桶相对位置偏置，以零初始化嵌入把每层每头的距离先验注入编码器自注意力，仅增加 1536 个参数；默认种子下 BLEU 为 10.73/11.45，提升 +0.49/+0.14，但同种子对照为 -0.06/-0.12，说明该改动保持了与基线相当的翻译性能但未带来可复现的提升。偏置可视化确认模型学到了邻近优先与方向分工的结构。后续工作可将相对位置偏置扩展到解码器与增量解码路径，并以多种子平均、更大测试集和更长训练预算重新评估其收益。

= 参考文献

#set par(first-line-indent: 0em)
#text(size: 9pt)[
  [1] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[C]\/\/Advances in Neural Information Processing Systems 30 (NIPS 2017). Long Beach: Curran Associates, 2017: 5998--6008.

  [2] Shaw P, Uszkoreit J, Vaswani A. Self-attention with relative position representations[C]\/\/Proceedings of NAACL-HLT 2018. New Orleans: ACL, 2018: 464--468.

  [3] Raffel C, Shazeer N, Roberts A, et al. Exploring the limits of transfer learning with a unified text-to-text transformer[J]. Journal of Machine Learning Research, 2020, 21(140): 1--67.

  [4] Popel M, Bojar O. Training tips for the Transformer model[J]. The Prague Bulletin of Mathematical Linguistics, 2018, 110(1): 43--70.

  [5] Szegedy C, Vanhoucke V, Ioffe S, et al. Rethinking the Inception architecture for computer vision[C]\/\/Proceedings of CVPR 2016. Las Vegas: IEEE, 2016: 2818--2826.

  [6] Chen M X, Firat O, Bapna A, et al. The best of both worlds: Combining recent advances in neural machine translation[C]\/\/Proceedings of ACL 2018. Melbourne: ACL, 2018: 76--86.

  [7] Bojar O, Federmann C, Fishel M, et al. Findings of the 2018 conference on machine translation (WMT18)[C]\/\/Proceedings of the Third Conference on Machine Translation, Volume 2: Shared Task Papers. Brussels: ACL, 2018: 272--307.

  [8] Ma Y, Yu D, Wu T, et al. PaddlePaddle: An open-source deep learning platform from industrial practice[J]. Frontiers of Data and Computing, 2019, 1(1): 105--115.
]
#set par(first-line-indent: 2em)

= 附录 A　实验工具清单

#set par(first-line-indent: 0em)
#block(breakable: false, table(
  columns: (auto, auto, auto),
  align: (left, left, left),
  stroke: (x, y) => (
    top: if y == 0 { 0.9pt } else { 0pt },
    bottom: if y == 1 { 0.5pt } else if y == 9 { 0.9pt } else { 0pt },
  ),
  inset: 3pt,
  text(size: 8.5pt)[*工具*], text(size: 8.5pt)[*版本*], text(size: 8.5pt)[*用途*],
  [PaddlePaddle], [3.3.1 (CUDA 12.6)], [Transformer 模型的训练、解码与参数管理],
  [Python], [3.11], [实验主语言],
  [实验室 HPC 集群（Slurm）], [单卡 V100-SXM2-16GB], [模型训练与批量解码],
  [NumPy], [1.26.4 / 2.5.3], [训练环境依赖、日志解析与结果汇总],
  [Moses multi-bleu.perl], [—], [语料级 BLEU 评测],
  [Matplotlib], [3.11.2], [训练曲线、BLEU 对照与偏置可视化],
  [Typst], [0.15.1], [本报告的排版与编译],
))
#set par(first-line-indent: 2em)

= 附录 B　实验过程记录（终端日志摘录）

以下摘录取自本实验在实验室 HPC 集群（Slurm 调度，单卡 Tesla V100-SXM2-16GB，计算能力 7.0）上的真实运行记录，来源为 `results/*/logs/` 与 `jobs/hpc/` 下的原始文件，逐字抄录、未作改写，仅将制表符展开为空格，并省略 `multi-bleu.perl` 每次固定输出的风险提示行。括号内「行 N」为原始日志中的行号，便于核对。正文表 2、表 3 的数值与下列记录逐项对应；模型检查点按提交要求不随压缩包上传。

#set par(first-line-indent: 0em)
#show raw: set text(size: 7.4pt, font: "Liberation Mono")

*B.1　任务提交（`jobs/hpc/run_experiment.sbatch`）*

#block(inset: 5pt, fill: rgb("#f4f4f4"), radius: 2pt)[
```bash
$ sbatch --job-name=lab2_lr3 \
    --export=ALL,TAG=lr3,EXTRA="--learning_rate 3.0" \
    ~/lab2/jobs/run_experiment.sbatch

#SBATCH -J lab2_exp
#SBATCH -p compute
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:30:00
#SBATCH -o /home/jnguan/lab2/logs/slurm-exp-%j.out
```
]

*B.2　基线训练（`results/baseline/logs/train_baseline.log`，行 10–41、63、69、119、252）*

#block(inset: 5pt, fill: rgb("#f4f4f4"), radius: 2pt)[
```text
save_step:              5000
print_step:             100
save_model_path:        saved_models
random_seed:            None
training_file:          /home/jnguan/lab2/data/wmt171819_zhen_data_bpe3/train.tok.clean.bpe.32000.en-zh
batch_size:             4096
epoch:                  10
learning_rate:          2.0
warmup_steps:           8000
label_smooth_eps:       0.1
beam_size:              5
W1015 19:01:59.793642 817533 gpu_resources.cc:116] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 13.0, Runtime API Version: 12.6
[2026-10-15 19:02:03,352] [    INFO] train.py:138 - step_idx: 0, epoch: 0, batch: 0, avg loss: 11.034841, normalized loss: 9.653107, ppl: 61996.992837
[2026-10-15 19:15:52,905] [    INFO] train.py:146 - step_idx: 5000, epoch: 2, batch: 1329, avg loss: 4.163615, normalized loss: 2.781881, ppl: 64.303549, speed: 6.04 step/s
[2026-10-15 19:52:40,137] [    INFO] train.py:146 - step_idx: 18300, epoch: 9, batch: 1779, avg loss: 3.157074, normalized loss: 1.775340, ppl: 23.501739, speed: 6.03 step/s
```
]

*B.3　各调参实验中被改动的参数（对应日志首部参数表的行号）*

#block(inset: 5pt, fill: rgb("#f4f4f4"), radius: 2pt)[
```text
results/tuning/lr3/logs/train_lr3.log:35                learning_rate:      3.0
results/tuning/ls05/logs/train_ls05.log:42              label_smooth_eps:   0.05
results/tuning/warmup4k/logs/train_warmup4k.log:41      warmup_steps:       4000
results/tuning/combo/logs/train_combo.log:35,39,40      learning_rate: 3.0
                                                        warmup_steps: 4000
                                                        label_smooth_eps: 0.05
results/tuning/base_s23/logs/train_base_s23.log:19      random_seed:        23
results/improve/relpos_s23/logs/train_relpos_s23.log:19 random_seed:        23
results/tuning/beam10/logs/predict_2018.log:41,14       beam_size: 10
                          init_from_params: /home/jnguan/lab2/code/saved_models/step_final/
```
]

除上述条目与各实验各自的 `save_model_path` 外，七个重新训练的变体（lr3、ls05、warmup4k、combo、base_s23、relpos、relpos_s23）参数表与基线逐行一致；`beam10` 不重新训练，仅以基线 `step_final` 权重束宽 10 重新解码。

*B.4　改进模型 relpos 的运行证据（`results/improve/relpos/logs/train_relpos.log`，行 17、65、66、71、254）*

#block(inset: 5pt, fill: rgb("#f4f4f4"), radius: 2pt)[
```text
save_model_path:        /home/jnguan/lab2/runs/relpos/saved_models
W1015 19:11:07.696030 193665 gpu_resources.cc:116] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 13.0, Runtime API Version: 12.6
/home/jnguan/lab2/code_improve/transformer.py:541: VisibleDeprecationWarning:
[2026-10-15 19:11:10,601] [    INFO] train.py:138 - step_idx: 0, epoch: 0, batch: 0, avg loss: 11.123986, normalized loss: 9.742252, ppl: 67777.507418
[2026-10-15 20:04:07,976] [    INFO] train.py:146 - step_idx: 18300, epoch: 9, batch: 1779, avg loss: 3.135651, normalized loss: 1.753917, ppl: 23.003606, speed: 5.78 step/s
```
]

该日志与基线参数表逐行比对仅 `save_model_path` 一项不同，即编码器分桶相对位置偏置为唯一改动，与 2.3 节的代码说明一致；`code_improve/transformer.py` 一行的堆栈路径也表明实际执行的是改写后的模型文件。

*B.5　BLEU 评测原始输出（`results/*/logs/bleu_*.log`）*

#block(inset: 5pt, fill: rgb("#f4f4f4"), radius: 2pt)[
```text
# results/baseline/logs/bleu_baseline_2018_aligned.log
# aligned reference: max_length=256, dropped source lines=1415,3708
BLEU = 11.31, 42.2/15.7/7.1/3.5 (BP=1.000, ratio=1.054, hyp_len=94571, ref_len=89721)

# results/baseline/logs/bleu_2017.log
BLEU = 10.24, 40.7/14.5/6.3/3.0 (BP=1.000, ratio=1.035, hyp_len=48778, ref_len=47144)

# results/improve/relpos/logs/bleu_relpos_2018_aligned.log
# aligned reference: max_length=256, dropped source lines=1415,3708
BLEU = 11.45, 41.9/15.8/7.3/3.6 (BP=1.000, ratio=1.056, hyp_len=94774, ref_len=89721)

# results/improve/relpos/logs/bleu_relpos_2017.log
BLEU = 10.73, 41.5/14.9/6.6/3.2 (BP=1.000, ratio=1.023, hyp_len=48221, ref_len=47144)
```
]

其余七组实验的 BLEU 原始输出格式与上述完全一致，数值见正文表 3；对应的 `.log` 文件与解码译文（`output_file/`）均随 `code/` 一并提交。

#set par(first-line-indent: 2em)

// ---------- 末页：实验成绩与教师签名 ----------
#pagebreak()
#v(1.5cm)
#align(left)[
  #text(size: 11pt, font: "Noto Sans CJK SC")[
    实验成绩：#box(width: 3cm, stroke: (bottom: 0.6pt))[]
  ]
  #v(2.2cm)
  #text(size: 11pt, font: "Noto Sans CJK SC")[
    教师签名：#box(width: 3cm, stroke: (bottom: 0.6pt))[]
  ]
]
