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
  #linebreak()
  #text(size: 11pt, font: "Noto Sans CJK SC")[
    ——参数调优与相对位置—局部性联合偏置
  ]
]
#v(0.4em)

*摘要*　本文在 WMT18 英汉平行语料（252,385 个句对，BPE 合并规模 32,000，实际共享词表 38,814 项）上复现了 Transformer 基线翻译模型（63,976,448 个参数），并在固定的 10 epoch 训练预算下完成参数调优与代码级改进。评测前我们发现，项目默认的 `max_length = 256` 过滤会在 newstest2018 中跳过第 1415、3708 两句超长句；若直接与未过滤参考按行号比对，3,979 行译文会从第 1415 行起整体错位。本文采用对齐口径（从参考中剔除同样两句）并同时报告课程口径：基线在对齐口径下的 BLEU 为 10.24/11.31，课程口径下 2018 为 4.65，高于课程参考值 4.46（二者同处错位口径）。参数调优围绕 Noam 学习率系数、预热步数、标签平滑与束宽展开：预热缩短至 4000 步在 2017 上取得最大单项增益（+0.35），但 2018 仅 +0.01；降低标签平滑、组合配置与增大束宽均损害 2018。四个随机种子的基线重训在 2017/2018 上的极差分别为 0.93 与 1.07，远大于多数单次调参增益，因此改进实验一律采用同种子成对比较。改进方法将 T5 式分桶相对位置偏置与 Yang 等（2018）的查询相关高斯局部性偏置结合，施加于全部 6 层编码器自注意力；改动仅增加 1,626,768 个参数（+2.5%）。四个种子下，该配置相对同种子基线的对齐口径 2018 BLEU 平均提升 +0.26（逐种子 -0.30 至 +0.53，3/4 为正，样本标准差 0.38），2017 平均提升 +0.11（3/4 为正，标准差 0.52）；它是唯一在两个测试集上平均增益均为正的配置，但单种子波动较大，尚不能认定每个种子都稳定增益。对训练后偏置的分析显示，模型在深层学到更窄、中心更贴近查询的窗口，同一层内各头分化出不同宽度与尺度；在种子 23 上，改进模型的句内四元重复率由 1.83% 降至 1.01%，译文长度比由 1.071 降至 1.039，译文更紧凑。

*关键词*　神经机器翻译；Transformer；局部性建模；相对位置偏置；BLEU；PaddlePaddle

= 任务与数据

本实验在英汉方向上构建端到端的神经机器翻译系统。训练语料为 WMT18 英汉平行数据（Bojar 等，2018），共 252,385 个句对，已完成 tokenization 与清洗。BPE 模型以 32,000 为合并规模，但词表文件在合并单元之外还收录了特殊符号与未参与合并的字符，最终共享词表为 38,814 项；源语言与目标语言共用该词表，为嵌入—输出权重共享提供了条件。评测使用 newstest2017（2,001 句）与 newstest2018（3,981 句）两个标准测试集。

翻译质量采用语料级 BLEU 评价，由 mosesdecoder 的 `multi-bleu.perl` 计算：统计 1--4 gram 精度并以几何平均汇总，再乘以长度惩罚因子 $op("BP")$。解码结果先合并 BPE 续接符（将 `@@ ` 删除）再与 `newstest20XX.tok.zh` 参考答案比对，与课程给定的评测流程一致。

评测流程中有一个必须处理的细节：项目默认的 `max_length` 为 256，预测时数据读取器按源文长度加 1、目标长度加 2 中的较大者过滤样本；newstest2018 中第 1415、3708 句超过该上限（源文分别 293、279 个 BPE 词元），被跳过，译文只有 3,979 行。`multi-bleu.perl` 只按假设的行数读取参考，因此课程口径实际使用前 3,979 行参考（`ref_len = 90019`），但不会剔除内部的两句：从第 1415 行起，每条译文都对应到后一句的参考，第 3708 行后错位两句，BLEU 被严重压低。本文采用对齐口径，从参考中剔除被过滤的同样两句，得到 3,979 个一一对应的句对（`ref_len = 89721`）；同时报告课程原始口径以便与课程参考值比较。newstest2017 没有超长句，两种口径完全一致。课程参考值 4.46 也处于同一错位口径，因此 4.46 只能与课程口径下的分数对比，不能与对齐口径的绝对值直接比较。我们另用课程修正版 `reader.py` 生成对齐参考作对照，结果与本文脚本逐行一致（均为 3,979 行、剔除第 1415/3708 句）。

= 方法

== 基线模型

基线模型为标准 Transformer（Vaswani 等，2017）编码器—解码器结构：编码器与解码器各 6 层，每层多头注意力 8 个头，模型维度 $d_(op("model")) = 512$，前馈网络隐层 2048，每个头的键值维度 $d_k = d_v = 64$。模型采用 Pre-LN 结构（子层输入先做 LayerNorm），dropout 统一为 0.1，源/目标嵌入与输出投影三处共享同一权重矩阵，位置信息由正弦绝对位置编码提供。按实际词表 38,814 统计，模型参数量为 63,976,448（约 6,398 万）。

训练细节保持与实验项目默认值一致：批大小按词元数组织，每个 batch 最多 4096 个词元（含填充）；优化器为 Adam（$beta_1 = 0.9$，$beta_2 = 0.997$，$epsilon = 10^(-9)$）；学习率采用 Noam 调度，基础系数 2.0、预热 8000 步，峰值学习率约 $1.0 times 10^(-3)$ 并在其后按步数的平方根倒数衰减；标签平滑系数 0.1；共训练 10 个 epoch（约 18,400 步），每 5000 步保存一次检查点，最终以 `step_final` 解码。解码采用束搜索，基线束宽 5，最大输出长度 255 个 BPE 词元。

== 性能分析实验设计

本次实验的训练预算固定为 10 个 epoch。在这一约束下，18,400 步远小于原论文训练 base 模型的 10 万步，调度参数与正则化强度的相对作用会与标准设置不同（Popel 与 Bojar，2018；Chen 等，2018）。据此选取四个因素做单因素对照。

*（1）学习率系数（lr3）。* Noam 调度下峰值学习率与基础系数成正比。将系数由 2.0 提高到 3.0（峰值约 $1.5 times 10^(-3)$），检验固定步数下提高学习率能否在衰减前覆盖更远的参数空间。

*（2）预热步数（warmup4k）。* 基线预热占总步数的 43%，衰减阶段仅约 10,400 步。将预热缩短至 4000 步；由于 Noam 峰值与 $sqrt(op("warmup"))$ 成反比，峰值同时由约 $1.0 times 10^(-3)$ 抬升至约 $1.4 times 10^(-3)$，因此该实验检验的是“更短预热 + 更长衰减”的整体调度形状。

*（3）标签平滑（ls05）。* 标签平滑（Szegedy 等，2016）把目标分布与均匀分布混合以抑制过度自信。基线取 0.1；在固定训练轮数下降低至 0.05，检验减弱正则是否更有利于拟合。

*（4）解码束宽（beam10）。* 束搜索不改变模型参数。将基线束宽 5 提高到 10，考察扩大搜索空间的边际收益。

在上述单因素实验之后，将训练端的三项改动组合为 combo 配置，检验收益是否叠加。鉴于 BLEU 是离散的语料级指标，单次训练的差异可能来自初始化与数据顺序，本文另以 4 个随机种子（23、42、7、123）重训基线，并以同种子成对比较判断改进是否真实。

== 查询相关局部性偏置改进

Transformer 的注意力打分只由内容向量的点积决定，绝对位置编码在第一层之后只以加性方式影响内容表示，注意力本身没有显式的距离先验。相对位置表示（Shaw 等，2018）直接把键与查询的距离注入打分；T5（Raffel 等，2020）用分桶把任意距离压缩到有限个桶。Yang 等（2018）进一步指出，自然语言的局部依赖不仅与距离有关，也与当前查询的内容有关：他们让每个查询预测一个高斯窗的中心与宽度，在 WMT17 英汉任务上把 base 模型由 24.13 提升到 24.77，而“局部性 + 相对位置”的组合达到 24.96，是该文的最佳配置。本文沿用这一组合，在编码器自注意力中同时加入分桶相对位置偏置与查询相关高斯局部性偏置。

*相对位置偏置。* 对位置 $i$ 的查询与位置 $j$ 的键，令距离 $d = j - i$，用 32 个桶编码（正负方向各 16 个，绝对距离 0--7 逐距离精确分桶，8 及以上按对数间隔映射，超过 128 并入最后一桶）。每层、每个头拥有独立的零初始化偏置表 $B_l^m (op("bucket") (d))$。编码器第 $l$ 层第 $m$ 个头的注意力打分修正为

$ e_(i j)^m = (bold(q)_i^m dot bold(k)_j^m) / sqrt(d_(op("model"))) + B_l^m (op("bucket") (j - i)) + G_(i j)^m, $

其中 $G_(i j)^m$ 为下文的局部性偏置。项目沿用原始实现的缩放方式，以 $d_(op("model"))$ 而非每头维度 $d_k$ 作分母；本文保持这一设置，使所有模型处于同一缩放尺度。

*查询相关高斯窗。* 设源句长度为 $I$，编码器第 $l$ 层的查询表示为 $bold(h)_i$。先做共享变换 $bold(g)_i = tanh(bold(W)_p bold(h)_i)$，再由每个头自己的投影预测窗口中心 $P_i^m$ 与宽度 $D_i^m$：

$ P_i^m = I dot op("sigmoid")(bold(u)_p^m dot bold(g)_i), quad
  D_i^m = I dot op("sigmoid")(bold(u)_d^m dot bold(g)_i). $

取标准差 $sigma_i^m = D_i^m / 2$，该头对键位置 $j$ 的对数打分修正为

$ G_(i j)^m = -((j - P_i^m)^2) / (2 (sigma_i^m)^2) dot s_l^m, $

其中 $s_l^m$ 是每头可学习的尺度。长度 $I$ 取掩码统计出的真实源句长度，而不是批次填充后的长度，使高斯窗在不同句长的样本中处于同一相对坐标系。

初始化与尺度校准需要单独说明。中心与宽度投影 $bold(u)_p^m, bold(u)_d^m$ 零初始化，因此训练起点处 $P_i^m = D_i^m = I / 2$，高斯以句中央为中心、标准差 $I / 4$，是一个覆盖整句的宽窗而非零偏置。基线以 $d_(op("model"))^(-0.5)$ 缩放注意力 logits，Yang 等采用标准的 $d_k^(-0.5)$，同样的高斯偏置在前者中相对更强。为使初始偏置与基线 logits 处于同一量级，本文把每头尺度初始化为 $sqrt(d_k / d_(op("model"))) approx 0.354$（若改用标准缩放则初始化为 1.0）。该初始化下，4 个种子的平均步 0 损失为 11.094，与基线 11.102 接近，第 100 步均为 9.383，训练动态未被破坏。

*层数与位置。* Yang 等只在编码器前 3 层施加局部性偏置；其开发集消融中前 3 层为 23.29，全部 6 层为 23.13，因此选择前 3 层。本文发现该结论在小数据、短预算设置下并不成立：对最终采用的“相对位置 + 局部性”组合，同种子下 6 层在两个测试集上均优于 3 层（10.95/11.54 对 10.67/11.26，见 4.3 节），故把局部性偏置施加于全部 6 层编码器自注意力。改进只作用于编码器：解码器增量解码只保留历史键值的缓存，若在缓存路径中引入逐查询窗口，需要额外跟踪绝对位置并改写束搜索的打分逻辑，风险与工作量都明显更高；编码器一侧已足以检验“显式距离先验是否有益”这一假设，解码器侧扩展留作后续工作。

整个改动只增加编码器参数：每层局部性模块含 $512 times 512 + 512$ 的共享变换、2 组 $512 times 8 + 8$ 的投影和 8 个尺度，共 270,872 个参数；6 层合计 1,625,232 个，加上每层 $32 times 8$ 的相对位置表（6 层共 1,536 个），总计 1,626,768 个，相对基线增加 2.5%。解码端与束搜索的实现完全不变，因此改进前后的推理超参数严格一致。关键实现如下。

#figure(
  raw(
    block: true,
    lang: "python",
    "# 编码器层内：相对位置偏置 + 查询相关高斯偏置\nlengths = paddle.sum(attn_bias[:, 0, 0, :] > -1e8, axis=-1)\nrel_bias = self.rel_pos_bias(buckets).transpose([2, 0, 1])\nlocal_bias = self.localness(query, lengths)\nattn_output = self.self_attn(query, None, None,\n                             attn_bias + rel_bias + local_bias)\n\n# LocalnessBias.forward：由查询预测每头高斯窗的中心与宽度\nhidden = paddle.tanh(self.w(query))\ncenter = lengths[:, None, None] * F.sigmoid(self.u_center(hidden))\nwidth = lengths[:, None, None] * F.sigmoid(self.u_width(hidden))\nsigma = 0.5 * width + 1e-6\nbias = -(positions - center) ** 2 / (2 * sigma ** 2) * self.scale",
  ),
  caption: [编码器中相对位置偏置与局部性偏置的关键实现（完整实现见 `improve/transformer.py`）],
)

= 实验设置

全部训练与解码在百度 AI Studio 的免费算力上完成（单卡 NVIDIA Tesla V100-SXM2-16GB，16 GB 显存），软件环境为 Python 3.11、PaddlePaddle 3.3.1（CUDA 12.6；Ma 等，2019）。每个任务的训练阶段约 51--54 分钟（约 5.6--6.0 step/s），加上两个测试集的解码与 BLEU 评测后总耗时约 1 小时。除 2.3 节的模型改进外，所有对照实验共享同一份训练数据、词表、模型结构与训练轮数，仅改变 2.2 节所述的因素；改进实验与对应的种子对照共享除偏置模块以外的全部配置。所有实验的解码批大小为 64、束宽 5、最大输出长度 255，改进前后的推理超参数保持一致。基线训练超参数汇总于表 1。各实验的训练与 BLEU 原始日志、对齐评测脚本（`scripts/eval_aligned_bleu.py`）、日志解析与绘图脚本均随代码提交，报告中的数值可由日志复算。

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
    [解码], [束宽 5], [批大小 64，最大输出长度 255],
  ),
)

= 结果与分析

== 基线复现与评测口径

基线模型训练 10 epoch 后完成 BLEU 计算。表 2 给出对齐口径的 BLEU 及其分解指标。newstest2017 上 BLEU = 10.24；newstest2018 上为 11.31（42.2/15.7/7.1/3.5）。两个测试集上长度惩罚因子均为 1.000，ratio 分别为 1.035 与 1.054，译文整体略长于参考但未触发长度惩罚。2018 的 hyp_len 为 94,571，对齐后参考长度为 89,721，多出的约 5% 主要来自中文译文中重复或冗余的表达。若改用课程口径，同一批译文在 2018 上只有 4.65（23.8/5.8/2.6/1.3），比课程参考值 4.46 高 0.19；但两个数都受前述错位影响，不能作为翻译质量的绝对度量。

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

== 调参实验与随机种子

各调参实验的 BLEU 汇总于表 3，差值相对默认基线。三项训练端改动都提升了 2017 的成绩：warmup4k 增益最大（+0.35），ls05 与 lr3 分别为 +0.29 与 +0.23；但在 2018 上，只有 warmup4k 基本持平（+0.01），lr3 下降 0.20，ls05 下降 0.45。组合配置在 2017 上达到 10.63（+0.39），2018 却为 11.00（-0.31），说明提高学习率与减弱标签平滑叠加后没有带来跨测试集的收益。束宽由 5 增至 10 使两个测试集同时下降（-0.07/-0.43），扩大搜索空间并未改善低 BLEU 英中任务的结果。

#figure(
  kind: table,
  supplement: none,
  caption: [表 3　参数调优实验的 BLEU（2018 为对齐口径；差值为相对默认基线）],
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, left, right, right, right, right),
    stroke: (x, y) => (
      top: if y == 0 { 0.9pt } else { 0pt },
      bottom: if y == 1 { 0.5pt } else if y == 7 { 0.9pt } else { 0pt },
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
    [beam10], [束宽 5 → 10], [10.17], [10.88], [-0.07], [-0.43],
  ),
)

这些 0.2--0.5 量级的差异必须放在随机种子的背景下解读。4 个种子的基线重训在 2017 上的 BLEU 为 10.23--11.16（极差 0.93），在 2018 上为 10.24--11.31（极差 1.07）；仅更换种子就能造成比全部调参增益更大的变化。因此，表 3 中任何单次运行的结果都不能单独作为因素优劣的证据，ls05 与 beam10 的下降也不能排除随机性。本文在改进部分采用 4 个种子的同种子成对比较，正是为了把方法效应与训练随机性分开。

== 改进模型结果

表 4 给出同一种子（23）下的结构消融，所有行使用相同的训练与解码配置，只改变偏置模块。单独加入相对位置偏置在两个测试集上均略低于基线（2017 -0.06，2018 -0.12）；单独在 3 层加入局部性偏置对 2018 有 +0.10、对 2017 为 -0.16；把局部性扩展到 6 层后 2018 提升到 +0.40，但 2017 仍为 -0.20。两者组合后收益最大：3 层组合为 2018 +0.25、2017 -0.11，6 层组合达到 2018 +0.53、2017 +0.17，是唯一在两个测试集上同时为正的配置。作为负对照，把注意力缩放改为标准的 $d_k^(-0.5)$（不加任何偏置）时 BLEU 为 10.37/10.78，两个测试集分别下降 0.41 与 0.23，说明基线缩放不应在改进中一并更改，也说明本文对局部性偏置的尺度校准是必要的。

#figure(
  kind: table,
  supplement: none,
  caption: [表 4　同种子（23）下的结构消融（2018 为对齐口径；差值为相对同种子基线）],
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    stroke: (x, y) => (
      top: if y == 0 { 0.9pt } else { 0pt },
      bottom: if y == 1 { 0.5pt } else if y == 7 { 0.9pt } else { 0pt },
    ),
    inset: 3.2pt,
    text(size: 8.2pt)[*配置*], text(size: 8.2pt)[*2017*], text(size: 8.2pt)[*2018*],
    text(size: 8.2pt)[*Δ2017*], text(size: 8.2pt)[*Δ2018*],
    [基线（无偏置）], [10.78], [11.01], [—], [—],
    [\+ 相对位置偏置（6 层）], [10.72], [10.89], [-0.06], [-0.12],
    [\+ 局部性（前 3 层）], [10.62], [11.11], [-0.16], [+0.10],
    [\+ 相对位置 + 局部性（前 3 层）], [10.67], [11.26], [-0.11], [+0.25],
    [\+ 局部性（全部 6 层）], [10.58], [11.41], [-0.20], [+0.40],
    [\+ 相对位置 + 局部性（全部 6 层）], [10.95], [11.54], [+0.17], [+0.53],
  ),
)

单种子的消融只能指导结构选择，最终结论以 4 个种子的成对比较为准。表 5 中，每个种子的改进模型与同一随机种子的基线相减，均值与标准差均基于 4 个成对差值。组合配置（相对位置 + 局部性×6）在 2018 上的平均增益为 +0.26，样本标准差 0.38，4 个种子中 3 个为正（逐种子 -0.30、+0.53、+0.36、+0.44）；在 2017 上平均 +0.11、标准差 0.52，同样 3 个为正。它是唯一在两个测试集上平均增益均为正的配置：仅加局部性（全部 6 层）在 2018 上平均 +0.07、标准差 0.60（2/4 为正），在 2017 上平均 -0.30、标准差 0.50（1/4 为正）。相对位置偏置因此不是可有可无的附加项：它提供了一个与内容无关、跨句共享的距离先验，使局部性模块的查询相关窗口不至于在短训练预算下过拟合到少数模式；但即便加上它，单种子增益仍可落在负区间（种子 123 的 2018 为 -0.30），说明 0.2--0.4 量级的效应需要多种子平均才能可靠观测。图 1 将同一比较画成散点叠加：2018 上组合组 4 个种子中有 3 个高于对应基线，仅加局部性的组只有 2 个；2017 上两组改进的点分布在基线两侧，组间差异与种子波动同量级。

#figure(
  image("figs/bleu_seed.png", width: 100%),
  caption: [三组配置在各随机种子下的 BLEU（柱为均值，误差线为样本标准差，圆点为单次运行）],
)

#figure(
  kind: table,
  supplement: none,
  caption: [表 5　4 个随机种子的成对比较（2018 为对齐口径；Δ 为同种子改进减基线）],
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    stroke: (x, y) => (
      top: if y == 0 { 0.9pt } else { 0pt },
      bottom: if y == 1 { 0.5pt } else if y == 4 { 0.9pt } else { 0pt },
    ),
    inset: 3.2pt,
    text(size: 8.2pt)[*配置*],
    text(size: 8.2pt)[*2017 均值±std*], text(size: 8.2pt)[*2018 均值±std*],
    text(size: 8.2pt)[*Δ2017*], text(size: 8.2pt)[*Δ2018*],
    [基线], [10.74±0.38], [10.96±0.50], [—], [—],
    [\+ 局部性（全部 6 层）], [10.44±0.16], [11.03±0.34], [-0.30±0.50], [+0.07±0.60],
    [\+ 相对位置 + 局部性（全部 6 层）], [10.85±0.16], [11.21±0.45], [+0.11±0.52], [+0.26±0.38],
  ),
)

从 n-gram 分解看，改进带来的变化分布在所有阶数上而非集中在词汇覆盖。以种子 23 的 2018 为例，1--4 gram 精度由基线的 41.4/15.2/6.9/3.4 提升到 42.5/15.8/7.3/3.6，译文长度比由 1.071 降至 1.039，句内四元重复率由 1.83% 降至 1.01%；2017 上前三项精度均上升、4-gram 持平，长度比由 1.038 降至 1.022，重复率由 0.96% 降至 0.86%。局部性单独使用时并未产生这一效果：loconly6 在 2018 上的长度比为 1.058、重复率 1.87%，与基线相当，说明“相对位置 + 查询相关窗口”的组合才同时改善了精确率与译文紧凑度。2018 的平均源句长度为 32.8 个 BPE 词元，且超过 50 个词元的句子有 550 句（2017 为 181 句），长句更多，显式距离先验在 2018 上的收益也更大，这与两个测试集增益幅度的差异一致。一个直观的例子是 2018 测试集中的一个短句（源文 “Kelly is trying to do the same thing, three months later.”）：参考译文为「凯利打算做同样的事，在三个月后。」，基线输出「凯利也在做同样的事情，三个月后又开始做同样的事情。」，把「做同样的事情」重复了一遍；改进模型输出「凯利正在试图做同样的事情，三个月后。」，冗余消失，长度也更接近参考。

2017 上的结论要谨慎得多。组合配置的逐种子增益为 +0.10、+0.17、-0.55 与 +0.73，均值 +0.11，标准差 0.52；种子波动大于 2018，且不能排除个别种子上的负增益。因此本文的结论是：显式距离先验在 2018 上平均带来 +0.26 的正收益且 3/4 种子为正，但并非每个种子都优于基线；在 2017 上平均增益仅 +0.11、种子波动更大，只能视为与基线相当。这一不对称与 2017 的句长分布、以及该测试集本身对位置建模不如 2018 敏感有关，但不能由现有实验完全解释。

训练动态方面，图 2 给出三组配置按种子平均后的训练损失（指数滑动平均）。三条曲线在整个训练过程中几乎重合，终值均在 3.1 附近，说明改进没有改变优化的收敛行为，BLEU 差异来自注意力结构本身而非训练是否充分。

#figure(
  image("figs/loss_curves.png", width: 100%),
  caption: [三组配置的平均训练损失曲线（各种子均值，指数滑动平均平滑）],
)

== 学到的局部窗口

为确认局部性模块没有退化为常数或噪声，本文把种子 23 的 `step_final` 载入改进模型，在 newstest2017 的 200 个句子上逐层、逐头导出窗口参数（`scripts/analyze_localness.py`）。图 3 给出每层 8 个头的平均窗口宽度（按句长归一化）、中心偏移与平均尺度。三个量都呈现出与层深相关的结构：第 0 层的窗口最宽（平均宽度为句长的 0.92，8 个头均在 0.84 以上）、中心偏移最大（0.34）、尺度最小（0.25）；第 3、5 层窗口收窄到 0.75 与 0.74，中心偏移降至 0.19，各头宽度可低至 0.62--0.65。第 4 层出现回升（宽度 0.91、偏移 0.27），说明这种变化并非严格单调，但总体趋势是“浅层宽而弱、深层窄而强”。尺度从第 0 层的 0.25 上升到第 1 层的 0.54，此后稳定在 0.42--0.48，表明局部性约束主要在深层发挥作用，第 0 层仍以内容与相对位置信息为主。

#figure(
  image("figs/localness_depth.png", width: 100%),
  caption: [训练后各层各头的局部性窗口统计（每点为一个头，横线为该层均值）],
)

同一层内各头的分化比层间差异更明显。以宽度为例，第 2 层的 8 个头从 0.62 到 0.99 不等，尺度从 0.19 到 0.61；第 4 层尺度跨越 0.16--0.78。这说明模型没有把局部性模块学成所有头共享的固定邻域，而是让不同头承担不同宽度的上下文：部分头关注较窄的局部片段，部分头保留接近整句的宽窗。图 4 给出一个句长 23 的样本中三个代表性层的加性偏置热图（颜色越深表示惩罚越强），红线为学到的中心 $P_i$，白虚线为 $P_i plus.minus sigma_i$。第 0 层的偏置整体平坦，窗口几乎覆盖全句；第 2、5 层的零偏置带明显收窄并沿对角线附近移动，且中心轨迹随查询位置变化，而不是固定的对角线，说明窗口确实由查询内容预测。

#figure(
  image("figs/localness_bias.png", width: 100%),
  caption: [样本句上的局部性加性偏置（第 0/2/5 层的代表头；红线为中心，白虚线为中心 ± 标准差）],
)

== 讨论

综合以上结果，可以得到三点认识。第一，在 10 epoch 的固定预算下，Transformer 的种子方差远大于多数调参效应：4 个种子的基线在 2018 上跨越 1.07 个 BLEU，而表 3 中最大的单次调参差异（绝对值）为 0.45。若不采用同种子成对比较，任何 0.2--0.5 量级的“改进”都可能只是重新抽了一次随机种子。第二，评测口径会显著改变 2018 的绝对分数，但对同一口径下的成对比较没有影响；对齐参考是得到可解释 BLEU 的前提，而课程参考值只能与课程口径比较。第三，模型改进的价值应看跨种子的均值与一致性，而非单次最好结果。本文的相对位置 + 局部性组合在 2018 上取得 +0.26±0.38 的平均增益（3/4 种子为正），在 2017 上为 +0.11±0.52（3/4 种子为正）；它是唯一在两个测试集上平均增益均为正的改动，但单种子波动仍大，优势幅度有限。它不改变解码逻辑，只增加 2.5% 的参数，训练与推理开销几乎不变。局限同样明显：每个配置只有 4 个种子，2017 的估计仍不精确；改进只在编码器侧，解码器的局部依赖尚未建模；10 epoch 的预算也限制了对更长训练下结论外推的能力。

= 结论

本文在 WMT18 英汉数据上复现了 63,976,448 个参数的 Transformer 基线。评测时发现默认 `max_length = 256` 过滤会跳过 newstest2018 中的 2 句超长句，导致按未过滤参考评分时后续行整体错位；采用对齐口径后，基线在 newstest2017/2018 上取得 BLEU 10.24/11.31，课程口径下 2018 为 4.65，高于课程参考值 4.46（二者均受同一错位影响）。参数调优表明：缩短预热至 4000 步在单次运行中改善了 2017（+0.35），但在 2018 上基本持平；降低标签平滑、组合配置与增大束宽均损害 2018（-0.45、-0.31、-0.43）。4 个种子的基线在 2017/2018 上的极差为 0.93/1.07，大于上述单次差异，说明调参结论必须建立在多种子对照上。模型改进方面，本文将 T5 式分桶相对位置偏置与 Yang 等（2018）的查询相关高斯局部性偏置结合，施加于全部 6 层编码器自注意力，仅增加 1,626,768 个参数；4 个种子的同种子成对比较显示，对齐口径 2018 BLEU 平均提升 +0.26（3/4 为正，标准差 0.38），2017 平均提升 +0.11（3/4 为正，标准差 0.52）；该配置是唯一在两个测试集上平均增益均为正的改动，但种子波动较大，尚不能认定每个种子都稳定增益。偏置分析确认模型学到了随层深收窄、随查询移动、且在各头间分化的局部窗口。后续工作可将局部性扩展到解码器增量解码路径，并以更多种子、更长训练预算和更大测试集重新评估其收益。

= 参考文献

#set par(first-line-indent: 0em)
#text(size: 9pt)[
  [1] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[C]\/\/Advances in Neural Information Processing Systems 30 (NIPS 2017). Long Beach: Curran Associates, 2017: 5998--6008.

  [2] Shaw P, Uszkoreit J, Vaswani A. Self-attention with relative position representations[C]\/\/Proceedings of NAACL-HLT 2018. New Orleans: ACL, 2018: 464--468.

  [3] Raffel C, Shazeer N, Roberts A, et al. Exploring the limits of transfer learning with a unified text-to-text transformer[J]. Journal of Machine Learning Research, 2020, 21(140): 1--67.

  [4] Yang B, Tu Z, Wong D F, et al. Modeling localness for self-attention networks[C]\/\/Proceedings of EMNLP 2018. Brussels: ACL, 2018: 4449--4458.

  [5] Popel M, Bojar O. Training tips for the Transformer model[J]. The Prague Bulletin of Mathematical Linguistics, 2018, 110(1): 43--70.

  [6] Szegedy C, Vanhoucke V, Ioffe S, et al. Rethinking the Inception architecture for computer vision[C]\/\/Proceedings of CVPR 2016. Las Vegas: IEEE, 2016: 2818--2826.

  [7] Chen M X, Firat O, Bapna A, et al. The best of both worlds: Combining recent advances in neural machine translation[C]\/\/Proceedings of ACL 2018. Melbourne: ACL, 2018: 76--86.

  [8] Bojar O, Federmann C, Fishel M, et al. Findings of the 2018 conference on machine translation (WMT18)[C]\/\/Proceedings of the Third Conference on Machine Translation, Volume 2: Shared Task Papers. Brussels: ACL, 2018: 272--307.

  [9] Ma Y, Yu D, Wu T, et al. PaddlePaddle: An open-source deep learning platform from industrial practice[J]. Frontiers of Data and Computing, 2019, 1(1): 105--115.
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
  [百度 AI Studio], [免费算力，单卡 V100-SXM2-16GB], [模型训练与批量解码],
  [NumPy], [1.26.4 / 2.5.3], [训练环境依赖、日志解析与结果汇总],
  [Moses multi-bleu.perl], [—], [语料级 BLEU 评测],
  [Matplotlib], [3.11.2], [训练曲线、BLEU 对照与局部性可视化],
  [Typst], [0.15.1], [本报告的排版与编译],
))
#set par(first-line-indent: 2em)

= 附录 B　实验过程记录（运行日志摘录）

以下摘录取自本实验的真实运行日志（百度 AI Studio 免费算力，单卡 Tesla V100-SXM2-16GB），来源为 `results/*/logs/` 下的原始文件，逐字抄录、未作改写，仅将制表符展开为空格、把绝对路径改写为仓库相对路径，并省略 `multi-bleu.perl` 每次固定输出的风险提示行。括号内「行 N」为原始日志中的行号，便于核对。正文表 2--表 5 的数值与下列记录逐项对应；模型检查点按提交要求不随压缩包上传。

#set par(first-line-indent: 0em)
#show raw: set text(size: 7.4pt, font: "Liberation Mono")

*B.1　基线训练（`results/baseline/logs/train_baseline.log`，行 10–41、63、69、119、252）*

#block(inset: 5pt, fill: rgb("#f4f4f4"), radius: 2pt)[
```text
save_step:              5000
print_step:             100
save_model_path:        saved_models
random_seed:            None
training_file:          data/wmt171819_zhen_data_bpe3/train.tok.clean.bpe.32000.en-zh
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

*B.2　调参与种子实验中被改动的参数（对应日志首部参数表的行号）*

#block(inset: 5pt, fill: rgb("#f4f4f4"), radius: 2pt)[
```text
results/tuning/lr3/logs/train_lr3.log:35                learning_rate:      3.0
results/tuning/ls05/logs/train_ls05.log:42              label_smooth_eps:   0.05
results/tuning/warmup4k/logs/train_warmup4k.log:41      warmup_steps:       4000
results/tuning/combo/logs/train_combo.log:35,39,40      learning_rate: 3.0
                                                        warmup_steps: 4000
                                                        label_smooth_eps: 0.05
results/tuning/base_s23/logs/train_base_s23.log:19      random_seed:        23
results/tuning/base_s42/logs/train_base_s42.log:19      random_seed:        42
results/tuning/base_s7/logs/train_base_s7.log:19        random_seed:        7
results/tuning/base_s123/logs/train_base_s123.log:21    random_seed:        123
results/tuning/beam10/logs/predict_2018.log:41,14       beam_size: 10
                          init_from_params: saved_models/step_final/
```
]

除上述条目与各实验各自的 `save_model_path` 外，重新训练的变体参数表与基线逐行一致；`beam10` 不重新训练，仅以基线 `step_final` 权重束宽 10 重新解码。改进实验只额外打开 `use_rel_pos` 与 `use_localness` 并把 `localness_layers` 设为 6，其余配置与同种子基线逐行一致。

*B.3　改进模型的运行证据（`results/improve/locrel6_s23/logs/train_locrel6_s23.log`）*

#block(inset: 5pt, fill: rgb("#f4f4f4"), radius: 2pt)[
```text
save_model_path:        runs/locrel6_s23/saved_models
random_seed:            23
use_rel_pos:            True
use_localness:          True
localness_layers:       6
W1027 22:27:36.385118 4073922 gpu_resources.cc:116] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 13.0, Runtime API Version: 12.6
[2026-10-27 22:27:38,871] [    INFO] train.py:138 - step_idx: 0, epoch: 0, batch: 0, avg loss: 11.092257, normalized loss: 9.710523, ppl: 65660.746540
[2026-10-27 23:22:01,823] [    INFO] train.py:146 - step_idx: 18300, epoch: 9, batch: 1779, avg loss: 3.136805, normalized loss: 1.755071, ppl: 23.030163, speed: 5.63 step/s
```
]

*B.4　BLEU 评测原始输出（`results/*/logs/bleu_*.log`）*

#block(inset: 5pt, fill: rgb("#f4f4f4"), radius: 2pt)[
```text
# results/tuning/base_s23/logs/bleu_base_s23_2018_aligned.log
# aligned reference: max_length=256, dropped source lines=1415,3708
BLEU = 11.01, 41.4/15.2/6.9/3.4 (BP=1.000, ratio=1.071, hyp_len=96133, ref_len=89721)

# results/tuning/base_s23/logs/bleu_base_s23_2017.log
BLEU = 10.78, 41.2/14.8/6.7/3.3 (BP=1.000, ratio=1.038, hyp_len=48948, ref_len=47144)

# results/improve/locrel6_s23/logs/bleu_locrel6_s23_2018_aligned.log
# aligned reference: max_length=256, dropped source lines=1415,3708
BLEU = 11.54, 42.5/15.8/7.3/3.6 (BP=1.000, ratio=1.039, hyp_len=93250, ref_len=89721)

# results/improve/locrel6_s23/logs/bleu_locrel6_s23_2017.log
BLEU = 10.95, 42.0/15.3/6.8/3.3 (BP=1.000, ratio=1.022, hyp_len=48195, ref_len=47144)
```
]

其余种子与配置的 BLEU 原始输出格式与上述完全一致，数值见正文表 3--表 5；对应的 `.log` 文件与解码译文（`output_file/`）均随代码一并提交。

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
