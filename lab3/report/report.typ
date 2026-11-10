// ============================================================
// 实验三：Qwen2-7B-Instruct 的中文医疗对话微调与交互质量评估
// 编译：typst compile report.typ
// 数值来源：results/official/ 与 runs/*/trainer_state.json
// ============================================================

#set page(
  paper: "a4",
  margin: (x: 2.0cm, top: 2.5cm, bottom: 2.1cm),
  header: align(center, text(size: 9pt)[《自然语言处理 C》实验报告]),
  footer: context align(center, text(size: 9pt, counter(page).display())),
)
#set text(font: ("Noto Serif CJK SC", "Liberation Serif"), size: 10pt, lang: "zh")
#set par(justify: true, leading: 0.66em, first-line-indent: 2em)
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
#show figure: set block(breakable: false, above: 0.7em, below: 0.7em)

// ---------- 封面（无页眉页码）----------
#page(header: none, footer: none, margin: (x: 2.4cm, top: 5.0cm, bottom: 2.4cm))[
  #align(center)[
    #text(size: 36pt, font: "Noto Sans CJK SC")[哈尔滨工业大学] #v(1.18cm)
    #text(size: 26pt, font: "Noto Sans CJK SC", weight: "bold")[自然语言处理] #v(0cm)
    #text(size: 26pt, font: "Noto Sans CJK SC", weight: "bold")[实验报告] #v(2.14cm)
    #text(size: 21pt, font: "Noto Serif CJK SC", weight: "bold")[(2026 年秋季学期)] #v(4.0cm)
  ]
  #align(center)[
    #text(size: 12pt, font: "Noto Sans CJK SC")[
    #grid(
      columns: (3.25cm, 6.5cm),
      rows: (1.12cm, 1.12cm, 1.12cm, 1.12cm),
      stroke: 0.6pt,
      inset: 4pt,
      align: center,
      fill: (x, y) => if x == 0 { rgb("#CCEDFF") },
      text(size: 14pt, font: "Noto Sans CJK SC", weight: "bold")[姓名：], [邴海诺、高新来],
      text(size: 14pt, font: "Noto Sans CJK SC", weight: "bold")[学号：], [2024112353、2024112774],
      text(size: 14pt, font: "Noto Sans CJK SC", weight: "bold")[学院：], [计算学部],
      text(size: 14pt, font: "Noto Sans CJK SC", weight: "bold")[教师：], [杨沐昀],
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
    实验三：Qwen2-7B-Instruct 的中文医疗对话微调与交互质量评估
  ]
  #text(size: 11pt, font: "Noto Sans CJK SC")[
    ——LoRA 参数分析、DoRA 对比与格式遵循诊断
  ]
]
#v(0.4em)

*摘要*　本文在 DISC-Med-SFT 上对 Qwen2-7B-Instruct 做 LoRA 有监督微调，并在 Chinese-medical-dialogue（前 100/500 条，ROUGE-1/2/L F1）与 CMMLU computer_science（204 题）上对比基座与六个参数高效适配器，另做格式对齐续训与同预算回放对照。医疗上，仅 252 万参数（0.033%）的 L8-QV 把 ROUGE-L 从 0.1359 提到 0.1807（95% 区间 $[+0.0403,+0.0495]$），前 500 条从 0.1446 提到 0.1915，生成长度由 437 字符降到 257；扩容到 16 倍没有可靠增益。CMMLU 官方短提示下基座/L8-QV 严格准确率为 78.9%/77.5%，差异不显著；训练系统提示下的 61.3%/27.0% 主要是输出格式不匹配。仅用训练语料合成 2,000 条四选一格式样本并混入 1,000 条医疗回放后续训 L8-QV，官方严格准确率升到 83.3%，相对基座 +4.4 个百分点（McNemar $p=0.093$，边缘显著），显著高于 L8-QV（$p=0.002$）与同预算纯回放（$p=0.001$），医疗 ROUGE 同时提高。DoRA-8 相对基座把医疗 R-1/L 提高 0.0104/0.0443（区间不跨零），CMMLU 仅低 1.0 个百分点且不显著，是可行的替代方法；DoRA-16 医疗 R-L 也提高 0.0429，但 CMMLU 降至 66.2%。综合参数效率与通用能力，医疗生成推荐 L8-QV；官方选择题口径下推荐 L8-QV+FMT。交互上，微调把时延从约 8.4 秒降到 2--4 秒，角色代入与三点硬约束仍不完整。

*关键词*　大语言模型；LoRA；DoRA；参数高效微调；医疗对话；CMMLU；人机交互评价

= 任务与数据

== 模型与任务

基座为 Qwen2-7B-Instruct[3]（7.62B，28 层解码器）。任务是在中文医疗对话上做有监督微调：给定系统提示与患者提问，自回归生成医生回复。

训练数据为 DISC-Med-SFT[4]（464,898 段多轮对话、1,362,284 条医生回复）。按指导书把每条医生回复与其前一条用户提问配对，随机抽取 100,000 条并去重得 96,293 条训练对；验证集按对话级抽取 2,000 条，与医疗测试集前 500 条零重叠。主实验按固定种子取 12,000 条，六组主配置共享同一子集与顺序，1 epoch 共 375 步；4.2 节另构造 3,000 条续训数据。

== 评测数据

*Chinese-medical-dialogue* 为 Alpaca 格式医疗问答，取前 100 条为主评测集、前 500 条为扩展集；源文取 `input`（缺失时用 `instruction`），参考取 `output`。*CMMLU computer_science*[5]共 204 道四选一题，题干与选项拼成源文，正确字母为目标。评测使用课程两版脚本完全相同的三级解析器（整句字母、显式表述、行首字母），无法解析计错。远程 `gen_batch.py` 的短系统提示、256 新词元、重复惩罚 1.05 作为主口径；本地指导书第 3.17 节的专用提示、32 新词元作为复核口径。为区分答错与格式不遵循，另报告只重判未解析输出的内容匹配上界与代理 MRR；训练系统提示下的结果作为提示敏感性对照。

= 方法

== LoRA 低秩适配

LoRA[1]冻结预训练权重 $bold(W)_0 in RR^(d times k)$，在旁路注入低秩矩阵 $bold(B) in RR^(d times r)$ 与 $bold(A) in RR^(r times k)$：

$ bold(h) = bold(W)_0 bold(x) + (alpha \/ r) bold(B) bold(A) bold(x), $

其中 $r$ 为秩，$alpha$ 为缩放系数。训练只更新 $bold(A)$ 与 $bold(B)$，$bold(B)$ 零初始化，因此训练起点与基座等价。单层可训练参数为 $r(d+k)$。Qwen2 的 $q,o$ 投影为 $3584 times 3584$，$k,v$ 为 $3584 times 512$，MLP 的 gate/up 为 $3584 times 18944$、down 为 $18944 times 3584$。$r=8$ 仅 $q,v$ 时每层增加 90,112 个参数，28 层共 2,523,136 个（0.033%）；$r=16$ 全线性层每层 1,441,792 个，共 40,370,176 个（0.530%）；$r=32$ 为 80,740,352 个（1.060%）。

== 参数分析设计

固定 $alpha=2r$、dropout 0.05、有效批大小 32、余弦退火（预热 3%），比较四组 LoRA：L8-QV（$r=8$、$q,v$、$1e-4$，指导书默认）；L16-ALL-1e4（$r=16$、全线性层、$1e-4$，隔离秩与目标模块的影响）；L16-ALL-2e4（在上一组把学习率提到 $2e-4$）；L32-ALL（$r=32$、$2e-4$，检验容量是否在 $r=16$ 附近饱和）。

== DoRA：权重分解的低秩适配

DoRA[2]把冻结权重分解为幅度与方向，并只对方向做低秩更新：

$ bold(W)' = bold(m) (bold(W)_0 + bold(B) bold(A)) / (||bold(W)_0 + bold(B) bold(A)||_c), $

其中 $bold(m) in RR^d$ 为输出通道的幅度向量，$||dot||_c$ 为按列范数。除 LoRA 参数外，每个适配层增加一个长度等于输出维度的幅度向量。本文设两个对照：DoRA-8-QV 与 L8-QV 同秩、同目标、同学习率，总可训练参数 2,637,824（0.0346%）；DoRA-16-ALL 与 L16-ALL-2e4 完全一致，总参数 41,760,768（0.548%），仅比同 rank LoRA 多 3.4%。DoRA 的额外开销主要来自每步的权重范数计算；推理时更新可合并回原权重，不引入额外延迟。

== 交互评估设计

按人机交互评价[11]补充四类探针：在 6 个固定问题上测量端到端时延与吞吐；“不超过 30 字”“分三点、每点不超过 20 字”两条硬约束及儿科医生/药剂师角色提示检查指令遵循[9]；三轮头痛对话与急症题检查上下文和安全升级；从测试集抽 15 题，把基座、L8-QV、DoRA-16 的回答盲化后由单个 LLM 评审按相关性、准确性、流畅性打 1--5 分[10]。单评审结果仅作探索性证据。

= 实验设置

== 训练配置

全部训练与评测在百度 AI Studio 的免费算力上完成（单卡 Iluvatar BI-V150S32G，32 GB 显存）。课程 PaddleFormers 环境在实验期间不可用，改用 PyTorch + Transformers + PEFT，数据格式、训练目标与官方评测口径保持一致；官方 bfloat16/eager 改为 fp16 + SDPA，预实验未观察到乱码或重复退化。训练采用梯度检查点、截断长度 1,024、AdamW、权重衰减 0、梯度裁剪 1.0；所有配置有效批大小均为 32（微批 4 × 累积 8，DoRA-8 为 2 × 16）。表 1 汇总可训练参数，表 2 汇总各配置，表 3 给出训练开销。

#figure(
  kind: table,
  supplement: none,
  caption: [表 1　LoRA / DoRA 可训练参数（Qwen2-7B-Instruct，28 层）],
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, left, right, right, right, right, right),
    stroke: (x, y) => (
      top: if y == 0 { 0.9pt } else { 0pt },
      bottom: if y == 1 { 0.5pt } else if y == 7 { 0.9pt } else { 0pt },
    ),
    inset: 3.2pt,
    text(size: 8.2pt)[*配置*], text(size: 8.2pt)[*方法*], text(size: 8.2pt)[*秩*],
    text(size: 8.2pt)[*目标模块*], text(size: 8.2pt)[*学习率*],
    text(size: 8.2pt)[*总参数*], text(size: 8.2pt)[*占比*],
    [L8-QV], [LoRA], [8], [$q,v$], [$1e-4$], [2,523,136], [0.033%],
    [L16-ALL-1e4], [LoRA], [16], [全部], [$1e-4$], [40,370,176], [0.530%],
    [L16-ALL-2e4], [LoRA], [16], [全部], [$2e-4$], [40,370,176], [0.530%],
    [L32-ALL], [LoRA], [32], [全部], [$2e-4$], [80,740,352], [1.060%],
    [DoRA-8-QV], [DoRA], [8], [$q,v$], [$1e-4$], [2,637,824], [0.0346%],
    [DoRA-16-ALL], [DoRA], [16], [全部], [$2e-4$], [41,760,768], [0.548%],
  ),
)

#figure(
  kind: table,
  supplement: none,
  caption: [表 2　六组主配置的共享训练超参数],
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    stroke: (x, y) => (
      top: if y == 0 { 0.9pt } else { 0pt },
      bottom: if y == 1 { 0.5pt } else if y == 8 { 0.9pt } else { 0pt },
    ),
    inset: 3.2pt,
    text(size: 8.2pt)[*项目*], text(size: 8.2pt)[*取值*], text(size: 8.2pt)[*说明*],
    [基座 / 框架], [Qwen2-7B-Instruct / PyTorch + PEFT], [7.62B 参数],
    [训练 / 验证样本], [12,000 / 2,000], [96,293 条去重对中抽取],
    [截断长度], [1,024], [超长样本截断],
    [有效批大小], [32], [微批 4 × 累积 8（DoRA-8 为 2 × 16）],
    [训练轮数 / 步数], [1 / 375], [所有配置一致],
    [学习率调度], [余弦退火], [预热 3%，峰值见表 1],
    [精度], [fp16 + SDPA], [梯度检查点开启],
    [随机种子], [42], [单种子，未做多种子重复],
  ),
)

#figure(
  kind: table,
  supplement: none,
  caption: [表 3　训练开销（单卡，fp16，梯度检查点）],
  table(
    columns: (auto, auto, auto),
    align: (left, right, right),
    stroke: (x, y) => (
      top: if y == 0 { 0.9pt } else { 0pt },
      bottom: if y == 1 { 0.5pt } else if y == 7 { 0.9pt } else { 0pt },
    ),
    inset: 3.2pt,
    text(size: 8.2pt)[*配置*], text(size: 8.2pt)[*训练时长*], text(size: 8.2pt)[*峰值显存*],
    [L8-QV], [31.3 min], [21.55 GB],
    [L16-ALL-1e4], [40.9 min], [22.19 GB],
    [L16-ALL-2e4], [41.0 min], [22.19 GB],
    [L32-ALL], [41.1 min], [22.86 GB],
    [DoRA-8-QV], [42.9 min], [18.12 GB],
    [DoRA-16-ALL], [2 h 05 min], [22.21 GB],
  ),
)

== 评测协议与指标

所有模型使用同一聊天模板、贪心搜索、批大小 8 与左填充；适配器推理前合并回基座，使参数量差异只体现在训练阶段。医疗主实验使用训练系统提示，另以官方短提示做敏感性对照；CMMLU 主结果使用官方短提示，复核使用指导书专用提示，训练提示下的结果作为格式对照。

ROUGE[6]先经 jieba 分词[12]再计算 ROUGE-1/2/L F1，空输出计 0；BLEU-4[7]为语料级 4-gram 精度加简短惩罚，EM 为规范化完全匹配，F1 为词元多重集重叠。CMMLU 按三级规则解析，未解析计错，并报告内容匹配与代理 MRR。时延在 6 个固定问题上测量，医疗主表取 5 个医学问题的平均时延。统计上，医疗与 CMMLU 分别使用 500/2000 次配对 bootstrap，只对不跨零的差异作结论；单种子、12k 子集与 1 epoch 限制外推。

= 结果与分析

== 基座与 LoRA 基线

表 4 给出主结果。基座医疗 ROUGE-1/2/L 为 0.2150/0.0338/0.1359，平均生成 437 字符（约为参考的 2.4 倍），99/100 条在 256 词元处截断；L8-QV 仅训练 252 万参数，即把 ROUGE-L 提到 0.1807（+0.0450，95% 区间 $[+0.0403,+0.0495]$），长度压到 257 字符、截断降到 8 条；前 500 条 ROUGE-L 从 0.1446 升到 0.1915（图 1），说明 100 条子集低估了基座的冗长程度。EM 对所有模型均为 0，说明该任务为开放式生成，字面匹配不适用。换用官方短提示后，基座/L8-QV 的医疗 ROUGE-L 为 0.1809/0.1988，L8-QV 仍显著领先（+0.0180，$[+0.0032,+0.0321]$），但增益小于训练提示，部分优势来自对训练提示的适配。

#figure(
  image("figs/rouge_compare.png", width: 75%),
  caption: [各模型在医疗对话前 100 条上的 ROUGE-1/2/L F1],
)

#figure(
  kind: table,
  supplement: none,
  caption: [表 4　主结果（医疗为训练系统提示下前 100 条 F1；CMMLU 为官方短提示下 204 题严格准确率 / 内容匹配上界）],
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right, right, right, right),
    stroke: (x, y) => (
      top: if y == 0 { 0.9pt } else { 0pt },
      bottom: if y == 1 { 0.5pt } else if y == 8 { 0.9pt } else { 0pt },
    ),
    inset: 2.6pt,
    text(size: 7.4pt)[*模型*], text(size: 7.4pt)[*R-1*], text(size: 7.4pt)[*R-2*],
    text(size: 7.4pt)[*R-L*], text(size: 7.4pt)[*BLEU-4*], text(size: 7.4pt)[*F1*],
    text(size: 7.4pt)[*严格*], text(size: 7.4pt)[*内容*], text(size: 7.4pt)[*长度*], text(size: 7.4pt)[*时延*],
    [基座], [0.2150], [0.0338], [0.1359], [0.0071], [0.258], [78.9], [82.8], [437], [8.4],
    [L8-QV], [0.2284], [0.0476], [0.1807], [0.0166], [0.303], [77.5], [81.9], [257], [3.7],
    [L16-1e4], [0.2323], [0.0424], [0.1813], [0.0174], [0.299], [74.0], [78.9], [233], [2.3],
    [L16-2e4], [0.2277], [0.0400], [0.1786], [0.0159], [0.299], [65.7], [77.5], [254], [2.2],
    [L32-ALL], [0.2311], [0.0423], [0.1815], [0.0158], [0.299], [60.3], [75.0], [230], [2.1],
    [DoRA-8], [0.2254], [0.0486], [0.1802], [0.0175], [0.296], [77.9], [82.4], [236], [3.5],
    [DoRA-16], [0.2236], [0.0387], [0.1787], [0.0151], [0.297], [66.2], [77.5], [252], [2.2],
  ),
)

官方短提示下，基座/L8-QV 的 CMMLU 严格准确率为 78.9%/77.5%（差 1.5 个百分点，2000 次配对 bootstrap 区间 $[-3.4,+6.4]$ 跨零），内容匹配 82.8%/81.9%，MRR 0.840/0.819，通用知识基本保留。参数扩容的代价更明显：L16-1e4、L16-2e4、L32 的严格准确率依次为 74.0%、65.7%、60.3%，内容匹配降至 78.9%、77.5%、75.0%；同预算 DoRA-8 为 77.9%/82.4%，与基座接近，DoRA-16 降至 66.2%/77.5%。可见该预算下容量越大、学习率越高，领域适配越充分，但通用知识漂移越明显（图 2）。

*格式与知识的分离。* 本地指导书的专用提示把基座/L8-QV 提到 85.3%/82.4%（未解析 0），差距缩至 2.9 个百分点（$[0.0,+5.9]$，触及零）。若沿用训练系统提示，两者严格准确率只有 61.3%/27.0%，未解析 48/131 题；但内容匹配仍有 75.5%/69.6%，抽样可见模型直接复述正确选项内容而不写字母。因此训练提示下的大幅下降主要是格式不遵循而非知识遗忘；严格准确率随容量上升（27.0→48.0→51.0）同样只是格式遵循改善[8]。

#figure(
  image("figs/cmmlu_compare.png", width: 92%),
  caption: [CMMLU computer_science（204 题）严格准确率：左为六个参数配置在训练提示与官方短提示下的对照，右为基座与 L8-QV 在三种口径下的对照],
)

== 格式对齐续训

官方短提示下 L8-QV 的 16 条未解析题多数已包含正确内容（内容匹配 81.9% 对严格 77.5%），差距主要是“答对但不按选项格式作答”。为此，仅用同一训练语料构造 3,000 条续训集：2,000 条合成四选一题（正确选项为配对的医生回复，干扰项为其他训练对的回复，随机打乱，要求输出“字母. 选项文本”，系统提示与官方评测一致），另混入 1,000 条原始医疗对回放。以 L8-QV 为初始化续训 1 epoch（93 步，$5e-5$）得到 L8-QV+FMT；控制组 L8-QV+REPLAY 用同预算纯医疗回放、同初始化与同步数训练。未使用任何 CMMLU 题目或选项。

#figure(
  kind: table,
  supplement: none,
  caption: [表 5　格式对齐续训与对照（医疗为训练系统提示下前 100 条 F1；CMMLU 为官方短提示下 204 题）],
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right, right),
    stroke: (x, y) => (
      top: if y == 0 { 0.9pt } else { 0pt },
      bottom: if y == 1 { 0.5pt } else if y == 5 { 0.9pt } else { 0pt },
    ),
    inset: 3pt,
    text(size: 7.8pt)[*模型*], text(size: 7.8pt)[*R-1*], text(size: 7.8pt)[*R-2*],
    text(size: 7.8pt)[*R-L*], text(size: 7.8pt)[*严格*], text(size: 7.8pt)[*内容*],
    text(size: 7.8pt)[*MRR*], text(size: 7.8pt)[*未解析*],
    [基座], [0.2150], [0.0338], [0.1359], [78.9], [82.8], [0.840], [17],
    [L8-QV], [0.2284], [0.0476], [0.1807], [77.5], [81.9], [0.819], [16],
    [L8-QV+FMT], [0.2448], [0.0510], [0.1915], [83.3], [83.8], [0.838], [3],
    [L8-QV+REPLAY], [0.2285], [0.0479], [0.1806], [77.0], [81.9], [0.819], [17],
  ),
)

表 5 显示，FMT 把严格准确率从 77.5% 提到 83.3%（170/204），未解析从 16 降到 3；相对基座高 4.4 个百分点（$[0.0,+8.8]$，McNemar $p=0.093$），相对 L8-QV 高 5.9（$[+2.5,+9.3]$，$p=0.002$），相对纯回放高 6.4（$[+2.9,+10.3]$，$p=0.001$）。纯回放仅 77.0%，与 L8-QV 持平，说明增益来自格式样本而非额外训练。内容匹配 83.8%、MRR 0.838 与基座相当，字母分布均衡（A 54/B 50/C 48/D 49），说明模型学会按选项作答，而不是知识增长。医疗侧同步提高：前 100 条 R-1/R-L 为 0.2448/0.1915（相对 L8-QV +0.0164/+0.0109），前 500 条 R-1/R-L 为 0.2637/0.1999（+0.0115/+0.0084），未牺牲生成质量。

== 参数分析

容量扩展没有带来可靠的医疗增益。L16-ALL-1e4 的 ROUGE-1 为 0.2323，比 L8-QV 高 0.0040（$[-0.0079,+0.0157]$），ROUGE-L 高 0.0007（$[-0.0059,+0.0070]$），均不显著，ROUGE-2 反而从 0.0476 降到 0.0424；学习率翻倍后 ROUGE-1/L 分别下降 0.0051/0.0029，方向一致偏负；$r=32$ 相对 L16-2e4 的回升也不显著。参数从 252 万增至 4,037 万（16 倍）乃至 8,074 万（32 倍），医疗 ROUGE 绝对提升不足 0.5 个点，说明领域适配主要来自句长、语气与叙述结构的低秩调整，$q,v$ 在 $r=8$ 时已覆盖大部分信号。CMMLU 的方向相反：官方短提示下严格准确率随容量上升而下降（77.5→74.0→65.7→60.3），内容匹配同步下降；训练提示下的“容量增益”只是格式遵循改善。训练损失则随容量下降（最后 10 次均值 1.314→1.231→1.219→1.211），但测试指标没有同步提高，说明更低损失更多反映对 12,000 条子集的过拟合与通用知识漂移。

== DoRA 与基座的对比

同预算的 DoRA-8-QV（$r=8$、$q,v$、$1e-4$，2,637,824 个参数，训练 42.9 分钟）相对基座，医疗 ROUGE-1/2/L 从 0.2150/0.0338/0.1359 提高到 0.2254/0.0486/0.1802，R-1 与 R-L 的 95% 区间分别为 $[+0.0005,+0.0201]$ 与 $[+0.0402,+0.0487]$，均不跨零；前 500 条 R-L 为 0.1911（基座 0.1446）。官方短提示下 CMMLU 严格准确率 77.9%、内容匹配 82.4%、MRR 0.824，与基座的 78.9%/82.8%/0.840 相比仅低 1.0 个百分点且不显著。可见 DoRA 作为替代方法，能够在不显著损失通用知识的前提下提升医疗生成。

更大预算的 DoRA-16-ALL（$r=16$、全线性层、$2e-4$，训练 2 h 05 min）医疗 R-L 为 0.1787，相对基座提高 0.0429（$[+0.0380,+0.0480]$），R-1 提高 0.0086（$[-0.0017,+0.0189]$，区间跨零）；但 CMMLU 严格准确率降到 66.2%、内容匹配 77.5%，相对基座 78.9%/82.8% 的退化显著。可见 DoRA 的领域适配有效，但大容量配置以通用知识漂移为代价；综合效果与开销，DoRA-8 是更合适的替代方案。

== 交互质量评估

*响应时间与吞吐。* 基座在 5 个医学问题上的平均时延为 8.4 秒，L8-QV 为 3.7 秒，其余适配器为 2.1--2.3 秒；解码吞吐稳定在 28.7--30.5 词元/秒（图 3）。差异主要来自生成长度：基座几乎每题都生成满 256 词元，微调模型多在 45--95 词元内结束。交互等待从约 8 秒降到 2--4 秒，但单位时间计算效率未变。

#figure(
  image("figs/latency_compare.png", width: 68%),
  caption: [5 个医学问题的平均端到端时延与解码吞吐],
)

*指令遵循与角色扮演。* 六个模型都满足 30 字、急症与多轮规则，三轮头痛对话能引用上一轮信息；三点约束只有基座与 L8-QV 完全满足，四个更大预算的适配器至少一点超过 20 字。角色代入也只有基座完成，五个适配器均未代入，其中四个直接自称“AI 医疗助手”（表 6）。

#figure(
  kind: table,
  supplement: none,
  caption: [表 6　交互探针结果（√ 表示满足规则，× 表示未满足）],
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center, center),
    stroke: (x, y) => (
      top: if y == 0 { 0.9pt } else { 0pt },
      bottom: if y == 1 { 0.5pt } else if y == 7 { 0.9pt } else { 0pt },
    ),
    inset: 3pt,
    text(size: 7.8pt)[*模型*], text(size: 7.8pt)[*角色*], text(size: 7.8pt)[*共情*],
    text(size: 7.8pt)[*≤30 字*], text(size: 7.8pt)[*三点*],
    text(size: 7.8pt)[*急症*], text(size: 7.8pt)[*多轮上下文*],
    [基座], [√], [√], [√], [√], [√], [√],
    [L8-QV], [×], [√], [√], [√], [√], [√],
    [L16-1e4], [×], [√], [√], [×], [√], [√],
    [L16-2e4], [×], [√], [√], [×], [√], [√],
    [L32-ALL], [×], [√], [√], [×], [√], [√],
    [DoRA-16], [×], [√], [√], [×], [√], [√],
  ),
)

*格式消融。* 训练系统提示下追加“只输出字母”指令后，严格准确率变化均在 1 个百分点以内，未解析数没有实质下降（表 7）；未解析输出是直接复述正确选项内容而非乱答。可见 SFT 风格是强先验，追加用户指令难以覆盖系统提示；换成官方短提示即恢复到 77.5%，专用提示达到 82.4% 且未解析为 0。部署时应固定任务对应的系统提示并做格式约束或后处理，无需重新训练。

#figure(
  kind: table,
  supplement: none,
  caption: [表 7　CMMLU 格式消融（训练系统提示下追加“只输出字母”指令；变化为准确率百分点差）],
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    stroke: (x, y) => (
      top: if y == 0 { 0.9pt } else { 0pt },
      bottom: if y == 1 { 0.5pt } else if y == 7 { 0.9pt } else { 0pt },
    ),
    inset: 3pt,
    text(size: 7.8pt)[*模型*], text(size: 7.8pt)[*严格准确率*], text(size: 7.8pt)[*未解析*],
    text(size: 7.8pt)[*加指令准确率*], text(size: 7.8pt)[*加指令未解析*], text(size: 7.8pt)[*变化*],
    [基座], [61.3], [48], [60.8], [50], [-0.5],
    [L8-QV], [27.0], [131], [27.0], [131], [0.0],
    [L16-1e4], [48.0], [78], [48.0], [78], [0.0],
    [L16-2e4], [47.5], [75], [47.5], [74], [0.0],
    [L32-ALL], [51.0], [67], [51.5], [66], [+0.5],
    [DoRA-16], [49.0], [72], [48.0], [74], [-1.0],
  ),
)

*盲评与 A/B。* 15 道医疗题、3 个系统共 45 条回答盲化后由单个 LLM 评审打分（表 8）。这是代理评价，评审可能偏好信息量更大的长回答，不能替代多名真人评审。

#figure(
  kind: table,
  supplement: none,
  caption: [表 8　盲评三维度得分（1--5）与成对胜负（W/T/L，相对基座）],
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    stroke: (x, y) => (
      top: if y == 0 { 0.9pt } else { 0pt },
      bottom: if y == 1 { 0.5pt } else if y == 5 { 0.9pt } else { 0pt },
    ),
    inset: 3pt,
    text(size: 7.8pt)[*模型*], text(size: 7.8pt)[*相关性*], text(size: 7.8pt)[*准确性*],
    text(size: 7.8pt)[*流畅性*], text(size: 7.8pt)[*总分*], text(size: 7.8pt)[*vs 基座*],
    [基座], [5.00], [4.33], [3.07], [4.13], [—],
    [L8-QV], [3.60], [3.80], [4.07], [3.82], [0/6/9],
    [DoRA-16], [3.47], [3.53], [4.07], [3.69], [1/1/13],
  ),
)

结果与自动指标方向相反：基座在相关性/准确性上更高（5.00/4.33，L8-QV 3.60/3.80，DoRA-16 3.47/3.53），微调模型在流畅性上更高（4.07 对 3.07）；成对胜负 L8-QV 0/6/9、DoRA-16 1/1/13。逐条检查可见，基座的长清单覆盖更多要点（尽管多在 256 词元处截断），微调回答更短、更常直接建议就医。ROUGE 奖励与参考的措辞和长度对齐，评审偏好信息量，二者衡量不同；单评审的长度偏差说明医疗“好回答”仍需多维度、多评审定义。

== 定性分析与讨论

定性对比表明，微调首先改变回复形态。第 33 条的参考为口语化段落，基座输出分点 Markdown 并在 256 词元处截断，L8-QV 生成约 200 字连贯答复，涵盖运动、饮食与就医建议，ROUGE-L 从 0.154 升到 0.262；第 497 条同样从分点标题转为直接建议。但长度对齐并非总是有益：第 225 条的参考详细列举咽痛、发热、吞咽困难等要素，基座虽冗长却覆盖大部分，L8-QV 只保留三项，ROUGE-L 从 0.185 降到 0.123。医疗场景还需关注安全性，盲评样本中个别回答建议自行购买滴耳液或口服抗生素，部署需要额外规则或人工审核；ROUGE 与内容匹配都不能验证医学正确性。

综合来看，医疗生成推荐小容量、低学习率、$q,v$ 的 L8-QV：以 0.033% 参数获得显著 ROUGE 增益，短提示下仍有 +0.0180 的优势，时延降到 3.7 秒；扩容没有可靠额外收益。DoRA-8 相对基座同样有效，DoRA-16 通用知识退化明显。若以官方选择题口径为主，则推荐 L8-QV+FMT（严格 83.3%，医疗 ROUGE 也高于 L8-QV）。CMMLU 的多口径结果表明领域 SFT 的知识损失很小，真正的风险是系统提示与输出格式不匹配，部署与评测都应把格式和知识分开。局限包括单种子、12,000 条子集、1 epoch、医疗主集仅 100 条、CMMLU 仅一个学科、盲评仅一名 LLM 评审，且未做事实性核查与多轮安全红队。

= 结论

本文完成了 Qwen2-7B-Instruct 在 DISC-Med-SFT 上的 LoRA 微调、参数分析、DoRA 对照与交互评估。L8-QV 以 0.033% 参数显著提升医疗 ROUGE，并保持 CMMLU 稳定；格式对齐续训把官方严格准确率提到 83.3%，相对基座 +4.4 个百分点（$p=0.093$），医疗 ROUGE 同步提高。DoRA-8 是可行的替代方法，DoRA-16 通用知识退化明显。后续应补多种子、真人评审、事实性核查与安全红队，并在部署侧固定系统提示、加入格式约束与急症升级规则。

= 参考文献

#set par(first-line-indent: 0em)
#text(size: 8.2pt)[
  #set par(spacing: 0.3em, leading: 0.55em)
  [1] Hu E J, Shen Y, Wallis P, et al. LoRA: Low-rank adaptation of large language models[C]//Proceedings of ICLR 2022. Virtual: OpenReview, 2022.

  [2] Liu S Y, Wang C Y, Yin H Z, et al. DoRA: Weight-decomposed low-rank adaptation[C]//Proceedings of ICML 2024. Vienna: PMLR, 2024: 32100--32121.

  [3] Qwen Team. Qwen2 technical report[R]. arXiv preprint arXiv:2407.10671, 2024.

  [4] Bao Z, Chen W, Xiao S, et al. DISC-MedLLM: Bridging general large language models and real-world medical consultation[R]. arXiv preprint arXiv:2308.14346, 2023.

  [5] Li H, Zhang Y, Koto F, et al. CMMLU: Measuring massive multitask language understanding in Chinese[C]//Findings of ACL 2024. Bangkok: ACL, 2024: 11260--11285.

  [6] Lin C Y. ROUGE: A package for automatic evaluation of summaries[C]//Text Summarization Branches Out: Proceedings of the ACL-04 Workshop. Barcelona: ACL, 2004: 74--81.

  [7] Papineni K, Roukos S, Ward T, et al. BLEU: A method for automatic evaluation of machine translation[C]//Proceedings of ACL 2002. Philadelphia: ACL, 2002: 311--318.

  [8] Biderman D, Prashanth R, Ranjan K, et al. LoRA learns less and forgets less[J]. Transactions on Machine Learning Research, 2024.

  [9] Zhou J, Lu T, Mishra S, et al. Instruction-following evaluation for large language models[R]. arXiv preprint arXiv:2311.07911, 2023.

  [10] Zheng L, Chiang W L, Sheng Y, et al. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena[C]//Advances in Neural Information Processing Systems 36. New Orleans: NeurIPS, 2023.

  [11] Amershi S, Weld D, Vorvoreanu M, et al. Guidelines for human-AI interaction[C]//Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems. Glasgow: ACM, 2019: 1--13.

  [12] Sun J. jieba: Chinese text segmentation[CP/OL]. https://github.com/fxsjy/jieba, 2012.
]
#set par(first-line-indent: 2em)

= 附录 A　实验过程截图

以下截图为实验三各环节终端日志的回放记录。

#set par(first-line-indent: 0em)
#figure(
  image("figs/shots/01_prepare_data.png", width: 92%),
  caption: [数据准备：DISC-Med-SFT 训练对构造，以及医疗测试集与 CMMLU 处理脚本的运行。],
)
#figure(
  image("figs/shots/02_train_lora_r8_qv.png", width: 92%),
  caption: [LoRA 微调训练（$r=8$，目标 $q,v$，12,000 条，1 epoch）：可训练参数 2,523,136（占总量 0.0331%），375 步，训练耗时 1,850 秒，GPU 峰值显存 21.55 GB。],
)
#figure(
  image("figs/shots/03_train_dora_r8_qv.png", width: 92%),
  caption: [DoRA 同预算对照训练（$r=8$，目标 $q,v$）：与 LoRA 使用同一份数据与配置，仅将 `use_dora` 置为 true。],
)
#figure(
  image("figs/shots/04_merge_export.png", width: 92%),
  caption: [导出合并：把 LoRA adapter 合并回基座并导出独立 fp16 checkpoint 的四步流程。],
)
#figure(
  image("figs/shots/05_gen_test_inference.png", width: 92%),
  caption: [批量推理：6 个固定医疗问题的贪心解码（`max_new_tokens=256`），输出写入 `results/official`。],
)
#figure(
  image("figs/shots/06_eval_medical.png", width: 92%),
  caption: [医疗评测：主集 100 条计算 ROUGE/BLEU-4/EM/F1，扩展 500 条用于推理统计。],
)
#figure(
  image("figs/shots/07_eval_cmmlu.png", width: 92%),
  caption: [CMMLU 评测：computer_science 204 题，采用官方短提示与官方严格选项解析口径。],
)
#figure(
  image("figs/shots/08_cmmlu_format.png", width: 92%),
  caption: [CMMLU 格式诊断：官方短提示下的严格解析、仅重判未解析输出的内容匹配与代理 MRR，用于区分知识缺失与格式不遵循。],
)
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
