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
    ——参数调优与编码器局部性偏置
  ]
]
#v(0.4em)

*摘要*　我们在 WMT18 英汉平行语料（252,385 个句对，共享词表 38,814 项）上复现了 63,976,448 个参数的 Transformer 基线，并在 10 epoch 预算内完成参数调优和代码级改进。评测前发现默认的 `max_length = 256` 会过滤 newstest2018 的第 1415、3708 句，若直接与未过滤参考比较，译文从第 1415 行起整体错位。我们改用对齐参考，并同时保留课程口径：基线在对齐口径下 newstest2017/2018 BLEU 为 10.24/11.31，课程口径下 2018 为 4.65。调参实验中，把预热缩短到 4000 步对 2017 最有利（+0.35），但 2018 仅 +0.01；降低标签平滑、组合配置和增大束宽都使 2018 变差。四个种子的基线重训极差达到 0.93/1.07，因此改进采用同种子成对比较。改进方法在全部 6 层编码器自注意力上叠加 T5 分桶相对位置偏置和 Yang 等（2018）的查询相关高斯局部性偏置，仅增加 1,626,768 个参数（+2.5%）。四个种子下，相对同种子基线的 2018 BLEU 平均提升 +0.26（3/4 为正），2017 为 +0.11（3/4 为正）；训练后的窗口统计显示深层窗口更窄、中心随查询移动，改进模型的译文重复率和长度比也有下降。

*关键词*　神经机器翻译；Transformer；局部性建模；相对位置偏置；BLEU；PaddlePaddle

= 任务与数据

实验在英汉方向上构建端到端翻译系统。训练语料为 WMT18 英汉平行数据（Bojar 等，2018），共 252,385 个句对，已经过 tokenization 和清洗。BPE 合并规模为 32,000，词表文件还包含特殊符号与未参与合并的字符，最终共享词表为 38,814 项，源端和目标端共用同一词表，因此嵌入层与输出投影可以共享权重。测试使用 newstest2017（2,001 句）和 newstest2018（3,981 句）。

翻译质量用 mosesdecoder 的 `multi-bleu.perl` 计算：统计 1--4 gram 精度，取几何平均后乘以长度惩罚因子 $op("BP")$。解码结果先去掉 BPE 续接符 `@@`，再与 `newstest20XX.tok.zh` 参考答案比对。

评测时有一个会直接影响分数的问题。`transformer.yaml` 的 `max_length = 256` 会在预测阶段过滤超长句；newstest2018 的第 1415、3708 句源文分别为 293 和 279 个 BPE 词元，因此被跳过，输出只有 3,979 行。`multi-bleu.perl` 只读取与假设行数相同的参考，但前 3,979 行参考并没有剔除这两句，于是从第 1415 行开始，每条译文都对应到后一句的参考，BLEU 被严重压低。我们从参考中删掉同样的两句，得到 3,979 个一一对应的句对（`ref_len = 89721`），下称对齐口径；课程原始口径仍按未过滤参考计算，只用于和课程给出的 4.46 比较。newstest2017 没有超长句，两种口径一致。课程修正版 `reader.py` 的过滤结果与本文脚本逐行一致（同为 3,979 行）。

= 方法

== 基线模型

基线是标准的 Transformer base（Vaswani 等，2017）：编码器和解码器各 6 层，每层 8 个注意力头，$d_(op("model")) = 512$，前馈隐层 2048，$d_k = d_v = 64$。子层采用 Pre-LN（先对输入做 LayerNorm），dropout 为 0.1；源词嵌入、目标词嵌入和输出投影三处共享权重，位置信息由正弦绝对位置编码提供。按 38,814 的词表计算，参数量为 63,976,448。

训练参数沿用项目默认值：按词元数组批，每批最多 4096 个词元（含填充）；Adam 优化器（$beta_1 = 0.9$，$beta_2 = 0.997$，$epsilon = 10^(-9)$）；Noam 学习率，系数 2.0、预热 8000 步，峰值约 $1.0 times 10^(-3)$，之后按步数的平方根倒数衰减；标签平滑 0.1；训练 10 个 epoch（约 18,400 步），每 5000 步保存检查点，最后用 `step_final` 解码。解码用束搜索，束宽 5，最大输出长度 255 个 BPE 词元。

== 性能分析实验设计

10 个 epoch 只有约 18,400 步，远少于原论文训练 base 模型的 10 万步（Popel 与 Bojar，2018；Chen 等，2018）。训练预算受限时，学习率调度和正则化强度的作用会和标准设置不同，因此我们选了四个因素做单因素实验：

*（1）学习率系数（lr3）。* Noam 调度的峰值学习率与基础系数成正比。把系数从 2.0 提到 3.0（峰值约 $1.5 times 10^(-3)$），看固定步数下能否在衰减前走到更远的参数区域。

*（2）预热步数（warmup4k）。* 基线预热 8000 步，占总步数的 43%，衰减阶段只剩约 10,400 步。把预热缩短到 4000 步；由于 Noam 峰值与 $sqrt(op("warmup"))$ 成反比，峰值同时升到约 $1.4 times 10^(-3)$，所以这个实验实际检验的是“更短预热 + 更长衰减”的整体调度形状。

*（3）标签平滑（ls05）。* 标签平滑（Szegedy 等，2016）把目标分布和均匀分布混合，抑制模型过度自信。基线取 0.1，这里降到 0.05，看在固定轮数下减弱正则是否更有利于拟合。

*（4）解码束宽（beam10）。* 束搜索不改变参数，只把束宽从 5 提到 10，检查扩大搜索空间是否还有收益。

之后把三项训练端改动合成 combo，看收益能否叠加。BLEU 是离散的语料级指标，单次训练的差异可能来自初始化或数据顺序，因此另外用 23、42、7、123 四个种子重训基线，改进实验与同种子基线配对比较。

== 编码器局部性偏置改进

注意力打分只由内容向量的点积决定，绝对位置编码在第一层之后只以加性方式进入表示，注意力本身没有距离先验。相对位置表示（Shaw 等，2018）把键和查询的距离直接加入打分，T5（Raffel 等，2020）用分桶把任意距离压到有限个桶里。Yang 等（2018）进一步让每个查询预测一个高斯窗的中心和宽度，在 WMT17 英汉上把 base 模型从 24.13 提到 24.77，再加相对位置后达到 24.96。我们采用这一组合，加在编码器自注意力上。

*相对位置偏置。* 对位置 $i$ 的查询和位置 $j$ 的键，令距离 $d = j - i$，用 32 个桶编码：正负方向各 16 个，绝对距离 0--7 逐距离精确分桶，8 以上按对数间隔映射，超过 128 归入最后一个桶。每层、每个头有一张独立的零初始化偏置表 $B_l^m (op("bucket") (d))$。第 $l$ 层第 $m$ 个头的注意力打分改为

$ e_(i j)^m = (bold(q)_i^m dot bold(k)_j^m) / sqrt(d_(op("model"))) + B_l^m (op("bucket") (j - i)) + G_(i j)^m, $

其中 $G_(i j)^m$ 是下面的局部性偏置。项目沿用原始实现的缩放方式，分母用 $d_(op("model"))$ 而不是每头维度 $d_k$；我们保持这一设置，使所有模型处于同一缩放尺度。

*查询相关高斯窗。* 设源句长度为 $I$，第 $l$ 层的查询表示为 $bold(h)_i$。先做共享变换 $bold(g)_i = tanh(bold(W)_p bold(h)_i)$，再由每个头自己的投影预测窗口中心 $P_i^m$ 和宽度 $D_i^m$：

$ P_i^m = I dot op("sigmoid")(bold(u)_p^m dot bold(g)_i), quad
  D_i^m = I dot op("sigmoid")(bold(u)_d^m dot bold(g)_i). $

取标准差 $sigma_i^m = D_i^m / 2$，该头对键位置 $j$ 的打分修正为

$ G_(i j)^m = -((j - P_i^m)^2) / (2 (sigma_i^m)^2) dot s_l^m, $

其中 $s_l^m$ 是每个头可学习的尺度。长度 $I$ 取掩码统计出的真实源句长度，而不是填充后的批长度，这样不同长度的句子处在同一个相对坐标系里。

初始化需要说明一下。中心与宽度投影 $bold(u)_p^m, bold(u)_d^m$ 零初始化，所以训练起点处 $P_i^m = D_i^m = I / 2$，高斯窗以句中央为中心、标准差 $I / 4$，是一个覆盖整句的宽窗，并不是零偏置。基线用 $d_(op("model"))^(-0.5)$ 缩放 logits，Yang 等用的是标准 $d_k^(-0.5)$，同样的高斯偏置在前者中相对更强。为了让初始偏置与基线 logits 处于同一量级，我们把每头尺度初始化为 $sqrt(d_k / d_(op("model"))) approx 0.354$（若用标准缩放则初始化为 1.0）。该初始化下，4 个种子的平均步 0 损失为 11.094，基线为 11.102，第 100 步均为 9.383，训练动态没有被破坏。

Yang 等只在编码器前 3 层加局部性偏置，其消融中前 3 层（23.29）优于全部 6 层（23.13）。但在我们的数据与预算下结论相反：对最终采用的“相对位置 + 局部性”组合，同种子下 6 层在两个测试集上都优于 3 层（10.95/11.54 对 10.67/11.26，见 4.3 节），所以最终把局部性偏置加在全部 6 层编码器上。

改动只增加编码器参数。每层的局部性模块包含 $512 times 512 + 512$ 的共享变换、2 组 $512 times 8 + 8$ 的投影和 8 个尺度，共 270,872 个参数；6 层合计 1,625,232 个，加上每层 $32 times 8$ 的相对位置表（共 1,536 个），总计 1,626,768 个，相对基线增加 2.5%。解码端和束搜索没有改动，推理超参数与基线完全一致。

= 实验设置

训练和解码在百度 AI Studio 的免费算力上完成（单卡 NVIDIA Tesla V100-SXM2-16GB，16 GB 显存），软件环境为 Python 3.11 和 PaddlePaddle 3.3.1（CUDA 12.6；Ma 等，2019）。每个训练任务约 51--54 分钟（5.6--6.0 step/s），两个测试集的解码和 BLEU 评测另需约 5--10 分钟。除 2.2 节和 2.3 节的因素外，所有实验共用同一份数据、词表、模型结构和训练轮数；解码批大小为 64，束宽 5，最大输出长度 255。基线配置见表 1，原始训练日志、BLEU 日志和评测脚本随代码提交，报告中的数值可以逐项复算。

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

基线训练 10 epoch 后，对齐口径的 BLEU 在 newstest2017 上为 10.24，在 newstest2018 上为 11.31（1--4 gram 精度 42.2/15.7/7.1/3.5）。两个测试集的长度惩罚因子都是 1.000，ratio 分别为 1.035 和 1.054，译文整体略长于参考但没有触发惩罚。2018 的译文长度为 94,571，对齐参考长度为 89,721，多出的约 5% 主要是中文译文里的重复表达。同一批译文按课程口径只有 4.65（23.8/5.8/2.6/1.3），比课程参考值 4.46 高 0.19；两个数都受错位影响，只能互相比较。

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

表 3 列出四个单因素实验和 combo 的结果。三项训练端改动都提高了 2017：warmup4k 最多（+0.35），ls05 和 lr3 分别为 +0.29 和 +0.23。但 2018 上只有 warmup4k 基本持平（+0.01），lr3 下降 0.20，ls05 下降 0.45。combo 在 2017 上到 10.63（+0.39），2018 却只有 11.00（-0.31），说明提高学习率和减弱标签平滑叠加后没有带来跨测试集的收益。束宽从 5 增到 10 后两个测试集同时下降（-0.07/-0.43），在这个低 BLEU 的英中任务上，扩大搜索空间没有帮助。

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

这些 0.2--0.5 的差异要放到种子波动里看。四个种子的基线在 2017 上是 10.23--11.16（极差 0.93），在 2018 上是 10.24--11.31（极差 1.07），仅换种子造成的波动就超过表 3 里最大的单次调参差异。所以表 3 不能单独用来判断某个因素的好坏，ls05 和 beam10 的下降也不能完全排除随机性。这里能比较稳妥地说的规律有两条：一是 2017 和 2018 对同一改动的反应经常相反，例如 lr3 和 ls05 都是 2017 升、2018 降；二是把三项训练端改动叠加并不会让收益相加，combo 的 2017 最高，但 2018 仍比基线低 0.31。可能的原因是 2018 的句子更长（平均源句 32.8 个 BPE 词元，超过 50 词元的有 550 句，2017 只有 181 句），在短预算下提高学习率、减弱正则更容易破坏长句的建模。这个解释只是从现有结果推断，还没有单独验证。

== 改进模型结果

先看种子 23 下的结构消融（表 4）。单独加相对位置偏置在两个测试集上都略低于基线（2017 -0.06，2018 -0.12）；单独在 3 层加局部性对 2018 有 +0.10，但 2017 为 -0.16；把局部性扩到 6 层后 2018 到 +0.40，2017 仍是 -0.20。两者组合后收益最大：3 层组合为 2018 +0.25、2017 -0.11，6 层组合为 2018 +0.53、2017 +0.17，是表中唯一在两个测试集上同时为正的配置。另一个负对照是把注意力缩放改成标准的 $d_k^(-0.5)$（不加偏置），BLEU 为 10.37/10.78，分别下降 0.41 和 0.23；这说明基线缩放不该在改进里顺手改掉，也说明局部性偏置的尺度校准是必要的。

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

单种子结果只能用来选结构，最终结论看四个种子的成对比较（表 5）。组合配置在 2018 上平均 +0.26，样本标准差 0.38，四个种子中三个为正（-0.30、+0.53、+0.36、+0.44）；2017 上平均 +0.11，标准差 0.52，也是三个为正。只加局部性（全部 6 层）在 2018 上平均 +0.07（标准差 0.60，2/4 为正），2017 上平均 -0.30（1/4 为正）。可见相对位置偏置不是可有可无的附加项：它给注意力一个与内容无关、全句共享的距离先验，局部性窗口在短预算下才不至于只拟合少数位置模式。但即使加上它，单个种子仍可能为负（种子 123 的 2018 为 -0.30），0.2--0.4 量级的效应需要多种子平均才看得出来。图 1 画出了同一组比较，2018 上组合组有 3 个点高于对应基线，只加局部性的组只有 2 个；2017 上两组点都落在基线两侧，组间差异和种子波动差不多大。

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

从 n-gram 分解看，改进带来的变化分散在各阶精度上，不只是词汇覆盖。以种子 23 的 2018 为例，1--4 gram 精度从基线的 41.4/15.2/6.9/3.4 升到 42.5/15.8/7.3/3.6，长度比从 1.071 降到 1.039，句内四元重复率从 1.83% 降到 1.01%；2017 上前三阶精度上升、4-gram 持平，长度比从 1.038 降到 1.022，重复率从 0.96% 降到 0.86%。只加局部性的 loconly6 没有这个效果：它在 2018 上的长度比 1.058、重复率 1.87%，和基线差不多。也就是说，相对位置和查询窗口一起使用时，精确率和译文紧凑度才同时改善。一个直观的例子是 2018 中的短句“Kelly is trying to do the same thing, three months later.”，参考为「凯利打算做同样的事，在三个月后。」；基线输出「凯利也在做同样的事情，三个月后又开始做同样的事情。」，把「做同样的事情」重复了一遍；改进模型输出「凯利正在试图做同样的事情，三个月后。」，冗余消失，长度也更接近参考。

2017 上的结果要谨慎。组合配置的逐种子增益是 +0.10、+0.17、-0.55、+0.73，平均 +0.11，标准差 0.52，种子波动比 2018 更大，不能排除个别种子上的负增益。能下的结论是：这种显式距离先验在 2018 上平均有 +0.26 的收益、3/4 种子为正；在 2017 上只能算与基线相当。两个测试集的差异与 2018 句长更大、长句更多是一致的，但现有实验还解释不了全部原因。

训练动态方面，图 2 是三组配置按种子平均后的训练损失（指数滑动平均）。三条曲线在训练过程中几乎重合，终值都在 3.1 附近，说明改进没有改变收敛行为，BLEU 差异来自注意力结构本身，而不是训练是否充分。

#figure(
  image("figs/loss_curves.png", width: 100%),
  caption: [三组配置的平均训练损失曲线（各种子均值，指数滑动平均平滑）],
)

== 学到的局部窗口

为了确认局部性模块没有退化成常数或噪声，我们把种子 23 的 `step_final` 载入改进模型，在 newstest2017 的 200 个句子上逐层、逐头导出窗口参数（`scripts/analyze_localness.py`）。图 3 给出每层 8 个头按句长归一化的平均窗口宽度、中心偏移和平均尺度。三个量都随层深变化：第 0 层窗口最宽（平均 0.92 个句长，8 个头都高于 0.84），中心偏移最大（0.34），尺度最小（0.25）；第 3、5 层窗口收窄到 0.75 和 0.74，中心偏移降到 0.19，部分头的宽度只有 0.63--0.65。第 4 层有回升（宽度 0.91、偏移 0.27），所以变化不是严格单调的，但总体上是浅层宽而弱、深层窄而强。尺度从第 0 层的 0.25 升到第 1 层的 0.54，之后稳定在 0.42--0.48，说明局部性约束主要在深层起作用，第 0 层仍以内容和相对位置信息为主。同一层内各头的差异比层间还大：第 2 层的 8 个头宽度从 0.62 到 0.99，尺度从 0.19 到 0.61；第 4 层尺度覆盖 0.16--0.78。模型没有学成所有头共用一个固定邻域，而是让不同头承担不同宽度的上下文。

#figure(
  image("figs/localness_depth.png", width: 100%),
  caption: [训练后各层各头的局部性窗口统计（每点为一个头，横线为该层均值）],
)

== 讨论与结论

综合调参和改进两组实验，有三点值得记录。第一，固定 10 epoch 时，种子方差比多数调参效应都大：基线在 2018 上跨 1.07 个 BLEU，而表 3 中最大的单次调参差异只有 0.45。不采用同种子成对比较，0.2--0.5 量级的改动很可能只是重新抽了一次种子。第二，评测口径对 2018 的绝对值影响很大，但对同一口径下的成对比较没有影响；对齐参考是让分数可解释的前提，课程参考值只能和课程口径比较。第三，改进是否有效要看跨种子的均值和一致性，而不是单次最好结果。我们的相对位置 + 局部性组合在 2018 上平均 +0.26±0.38（3/4 种子为正），2017 上 +0.11±0.52（3/4 为正），是本文尝试的配置中唯一在两个测试集上平均都为正的；但它只增加 2.5% 的参数，优势幅度有限，单个种子仍可能低于基线。局限也很明显：每个配置只有 4 个种子，2017 的估计不精确；改进只在编码器侧；10 epoch 的预算也限制了对更长训练的推断。

#set par(first-line-indent: 0em)
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
