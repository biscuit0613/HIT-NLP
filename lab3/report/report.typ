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
      [姓名：], [#box(width: 5.4cm, stroke: (bottom: 0.6pt))],
      [学号：], [#box(width: 5.4cm, stroke: (bottom: 0.6pt))],
      [学院：], [#box(width: 5.4cm, stroke: (bottom: 0.6pt))],
      [教师：], [#box(width: 5.4cm, stroke: (bottom: 0.6pt))],
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

*摘要*　本文在 DISC-Med-SFT 中文医疗对话数据上对 Qwen2-7B-Instruct 做 LoRA 有监督微调，并按课程官方推理与解析口径，在 Chinese-medical-dialogue（前 100 条及前 500 条，ROUGE-1/2/L F1）与 CMMLU computer_science（204 题）上评测基座与六个参数高效适配器。数据处理将 46.5 万段多轮对话中的 136.2 万条医生回复与前一条用户提问配对，去重得到 96,293 条训练对，按对话级切分验证集，并确认与医疗测试集零重叠。医疗指标上，在训练系统提示下，仅训练 252 万参数（占基座 0.033%）的 L8-QV 把 ROUGE-1/2/L 从 0.2150/0.0338/0.1359 提升到 0.2284/0.0476/0.1807，ROUGE-L 的 95% 配对 bootstrap 区间为 $[+0.0403, +0.0495]$，平均生成长度从 437 字符降到 257 字符；前 500 条上 ROUGE-L 从 0.1446 升至 0.1915。把秩与目标模块扩展到 16 倍参数量没有带来可靠的 ROUGE 增益，提高学习率反而略差，说明该任务的领域适配主要表现为低秩的风格与长度调整。CMMLU 暴露出一个关键现象：按官方“只解析选项字母”的严格口径，基座准确率为 61.3%，而 L8-QV 仅 27.0%，未解析率从 23.5% 升至 64.2%；内容匹配诊断显示两者分别为 77.5% 与 70.1%。进一步的系统提示对照表明这是训练提示与 SFT 风格共同造成的格式缺失：换成官方短提示后，两者严格准确率升至 78.9% 与 77.4%，未解析降到 17 与 16 题，知识基本保留。本文据此把“格式遵循”与“知识保持”分开评估，并同时报告格式指令消融与提示敏感性。交互评估进一步测量了端到端时延、吞吐、指令遵循、角色扮演、多轮上下文与急症安全行为；时延从基座的约 8.4 秒降到微调后的 2--4 秒，吞吐稳定在约 29--30 词元/秒，说明时延收益主要来自更短的输出；单评审盲评则显示基座的长清单在相关性与准确性上得分更高，微调回答在流畅性上更好，提示自动指标与评审偏好并不一致。替代方法方面，同配置 DoRA-16 与 LoRA 的差异在统计噪声内；同预算 DoRA-8-QV 与 L8-QV 在 ROUGE、CMMLU 与 MRR 上同样无显著差异，说明 DoRA 的额外开销未换来收益。综合参数效率、医疗指标与通用能力，L8-QV 是本文条件下的推荐配置。

*关键词*　大语言模型；LoRA；DoRA；参数高效微调；医疗对话；CMMLU；人机交互评价

= 任务与数据

== 模型与任务

基座为 Qwen2-7B-Instruct（Qwen Team，2024）：28 层解码器，隐藏维度 3,584，前馈维度 18,944，28 个查询头与 4 个键值头，词表 152,064，总参数 7,615,616,512（7.62B）。任务是在中文医疗对话上做有监督微调：给定系统提示与患者提问，自回归生成医生回复。

训练数据为 DISC-Med-SFT（Bao 等，2023），共 464,898 段多轮对话、1,362,284 条医生回复。按指导书把每条医生回复与紧邻的前一条用户提问配成单轮样本，一段含 $k$ 条医生回复的对话产生 $k$ 条样本。为控制训练预算并保持多样性，从候选中随机抽取 100,000 条，按源文与目标文的规范化文本去重后得到 96,293 条训练对；验证集从其余对话中按对话级抽取 2,000 条，保证同一对话的不同轮次不会跨集合。将训练样本与医疗测试集前 500 条做去空白、去标点的精确比对，重叠为 0。实际训练按固定种子从 96,293 条中取 12,000 条，所有配置共享同一子集与顺序，1 个 epoch 共 375 步。

== 评测数据

*Chinese-medical-dialogue* 为 Alpaca 格式医疗问答，按指导书取前 100 条为主评测集，并在前 500 条上补充评测以降低小样本波动；源文取 `input`（缺失时用 `instruction`），参考取 `output`。*CMMLU computer_science*（Li 等，2024）共 204 道四选一题，选项分布 A 52、B 51、C 51、D 50；题干与四个选项拼成源文，正确字母为目标。主评测严格沿用官方脚本：不追加“只输出字母”等格式指令，从输出中依次匹配整句字母、显式表述与行首字母，无法解析记为错误。为区分“答错”与“未按格式作答”，另设两个诊断：一是内容匹配上界，对未解析输出按选项文本与输出的字符三元组重叠选择最可能的选项；二是格式消融，在题干后追加“请回答上述单项选择题，只输出正确选项的字母（A、B、C、D）”，比较两种口径下的准确率与未解析率。

= 方法

== LoRA 低秩适配

LoRA（Hu 等，2022）冻结预训练权重 $bold(W)_0 in RR^(d times k)$，在旁路注入低秩矩阵 $bold(B) in RR^(d times r)$ 与 $bold(A) in RR^(r times k)$：

$ bold(h) = bold(W)_0 bold(x) + (alpha \/ r) bold(B) bold(A) bold(x), $

其中 $r$ 为秩，$alpha$ 为缩放系数。训练只更新 $bold(A)$ 与 $bold(B)$，$bold(B)$ 零初始化，因此训练起点与基座等价。单层可训练参数为 $r(d+k)$。Qwen2 的 $q,o$ 投影为 $3584 times 3584$，$k,v$ 为 $3584 times 512$，MLP 的 gate/up 为 $3584 times 18944$、down 为 $18944 times 3584$。$r=8$ 仅 $q,v$ 时每层增加 90,112 个参数，28 层共 2,523,136 个（0.033%）；$r=16$ 全线性层每层 1,441,792 个，共 40,370,176 个（0.530%）；$r=32$ 为 80,740,352 个（1.060%）。

== 参数分析设计

固定 $alpha=2r$、dropout 0.05、有效批大小 32、学习率余弦退火（预热 3%），逐项考察秩、目标模块与学习率：

*（1）L8-QV。* $r=8$、仅 $q,v$、学习率 $1 times 10^(-4)$，参数量最小，对应指导书默认设置。

*（2）L16-ALL-1e4。* $r=16$、全部线性层、$1 times 10^(-4)$；参数量增至 16 倍，单独隔离“秩与目标模块”的影响。

*（3）L16-ALL-2e4。* 在（2）上把学习率提高到 $2 times 10^(-4)$；单 epoch、375 步的短预算下，较高学习率可能更充分拟合，也可能加剧漂移。

*（4）L32-ALL。* $r=32$、$2 times 10^(-4)$，检验容量是否在 $r=16$ 附近饱和。

== DoRA：权重分解的低秩适配

DoRA（Liu 等，2024）把冻结权重分解为幅度与方向，并只对方向做低秩更新：

$ bold(W)' = bold(m) (bold(W)_0 + bold(B) bold(A)) / (||bold(W)_0 + bold(B) bold(A)||_c), $

其中 $bold(m) in RR^d$ 为输出通道的幅度向量，$||dot||_c$ 为按列范数。除 LoRA 参数外，每个适配层增加一个长度等于输出维度的幅度向量。本文设两个对照：DoRA-8-QV 与 L8-QV 同秩、同目标、同学习率，总可训练参数 2,637,824（0.0346%）；DoRA-16-ALL 与 L16-ALL-2e4 完全一致，总参数 41,760,768（0.548%），仅比同 rank LoRA 多 3.4%。DoRA 的额外开销主要来自每步的权重范数计算；推理时更新可合并回原权重，不引入额外延迟。

== 交互评估设计

除批量指标外，本文按人机交互评价的要求（Amershi 等，2019）补充四类测量。第一，*响应时间与吞吐*：在相同 fp16、批大小 1、贪心解码下，用 6 个固定问题逐条测量端到端时延（含分词、前向与解码），记录新词元数与词元/秒；模型加载不计入。第二，*指令遵循与角色扮演*（Zhou 等，2023）：设计“不超过 30 字”“分三点、每点不超过 20 字”两条硬约束，以及儿科医生、药剂师两个角色提示，检查模型是否遵守长度、格式与人设。第三，*多轮与安全*：构造三轮头痛对话，检查第二轮是否引用上一轮信息；用“胸痛、左臂发麻、出冷汗”的急症问题检查是否建议立即就医而不是仅推荐药物。第四，*盲评与 A/B*：从医疗测试集随机抽取 15 题，将基座、L8-QV 与 DoRA-16 的回答随机排成 A/B/C，评分时只看到随机标签、模型身份在评分完成前不可见，评审按相关性、准确性、流畅性各打 1--5 分，并统计成对胜负。由于课程环境无法组织多名真人评审，本文以一名 LLM 评审（Claude，单评审、盲化输入）作为人工评价的代理，同时保留完整评分表供真人复核；该结果应视为探索性证据，而非严格的人类受试者研究（Zheng 等，2023）。

= 实验设置

== 训练配置

全部训练与评测在百度 AI Studio 的免费算力上完成（单卡 32 GB 显存）。课程提供的 PaddleFormers 环境在实验期间无法使用，因此改用 PyTorch、Hugging Face Transformers 与 PEFT 实现；数据格式、训练目标与官方评测口径保持一致。官方脚本使用 bfloat16 与 eager attention，而所用免费算力卡的 fp16 路径更稳定，故统一采用 fp16 与 PyTorch SDPA，并在预实验中逐条核对输出，未观察到乱码或重复退化。训练采用梯度检查点、截断长度 1,024、AdamW、权重衰减 0、梯度裁剪 1.0。除 DoRA-8-QV 为降低显存峰值采用微批 2、累积 16 外，其余配置均为微批 4、累积 8；所有配置有效批大小均为 32。表 1 汇总可训练参数，表 2 汇总各配置，表 3 给出训练开销。

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
  caption: [表 2　共享训练超参数],
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

所有模型使用同一聊天模板与解码参数：贪心搜索、重复惩罚 1.05、最多 256 个新词元、批大小 8、左填充；医疗与 CMMLU 均如此。主实验的系统提示为训练时的 “You are Qwen, created by Alibaba Cloud. You are a helpful assistant.”；另以官方脚本的短提示 “You are a helpful assistant.” 对基座与 L8-QV 做提示敏感性对照。所有适配器在推理前合并回基座，使参数量差异只体现在训练阶段。

ROUGE（Lin，2004）采用 `rouge_chinese`：先对假设与参考做 jieba 分词（Sun，2012）再计算 ROUGE-1/2/L 的 F1、精确率与召回率，空输出按 0 分计入。BLEU-4（Papineni 等，2002）为语料级 4-gram 精度加简短惩罚（add-1 平滑）；EM 为规范化后的完全匹配；F1 为 jieba 词元的多重集重叠。CMMLU 按官方三级规则解析，无法解析计错，并记录未解析数与内容匹配诊断。由于生成式接口不返回选项概率，本文另用代理 MRR 做排序诊断：若输出可解析出字母，该字母记为第 1 位；否则按输出与各选项的字符三元组重叠排序，取正确选项倒数位次的均值。时延在 6 个固定问题上测量；医疗主表报告 5 个医学问题的平均时延，避免“首都是哪里”这类极短回答拉低均值。统计上，医疗主集仅 100 条、CMMLU 204 题，ROUGE 句级方差大；本文用 500 次配对 bootstrap 给出区间，只对不跨零的差异作结论。单种子、12,000 条子集与 1 个 epoch 仍限制结论外推。

= 结果与分析

== 基座与 LoRA 基线

表 4 给出主结果。基座在医疗测试集上的 ROUGE-1/2/L 为 0.2150/0.0338/0.1359，平均生成 437 字符，约为参考的 2.4 倍；其输出几乎全部为 Markdown 分点列表，99/100 条在 256 词元处被截断。L8-QV 只训练 252 万参数，却把 ROUGE-1/2/L 提到 0.2284/0.0476/0.1807，ROUGE-L 提升 0.0450（95% 区间 $[0.0403, 0.0495]$），平均长度压到 257 字符，截断条数降到 8。BLEU-4 从 0.0071 升到 0.0166，词元 F1 从 0.258 升到 0.303；EM 对所有模型均为 0，说明医疗回答是开放式生成，字面完全匹配并不适用。

前 500 条上的趋势一致：基座 ROUGE-1/2/L 为 0.2389/0.0380/0.1446，L8-QV 为 0.2522/0.0531/0.1915；基座 494/500 条达到 256 词元上限，L8-QV 仅 14 条。500 条上增益更大，说明 100 条子集低估了基座的冗长程度；两套结果方向一致（图 1）。

#figure(
  image("figs/rouge_compare.png", width: 75%),
  caption: [各模型在医疗对话前 100 条上的 ROUGE-1/2/L F1],
)

#figure(
  kind: table,
  supplement: none,
  caption: [表 4　主结果（训练系统提示；医疗为前 100 条 F1，CMMLU 为 204 题）],
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
    [基座], [0.2150], [0.0338], [0.1359], [0.0071], [0.258], [61.3], [77.5], [437], [8.4],
    [L8-QV], [0.2284], [0.0476], [0.1807], [0.0166], [0.303], [27.0], [70.1], [257], [3.7],
    [L16-1e4], [0.2323], [0.0424], [0.1813], [0.0174], [0.299], [48.0], [75.0], [233], [2.3],
    [L16-2e4], [0.2277], [0.0400], [0.1786], [0.0159], [0.299], [47.5], [71.6], [254], [2.2],
    [L32-ALL], [0.2311], [0.0423], [0.1815], [0.0158], [0.299], [51.0], [72.6], [230], [2.1],
    [DoRA-8], [0.2254], [0.0486], [0.1802], [0.0175], [0.296], [25.5], [69.6], [236], [3.5],
    [DoRA-16], [0.2236], [0.0387], [0.1787], [0.0151], [0.297], [49.0], [71.6], [252], [2.2],
  ),
)

CMMLU 的结果需要特别解读。按官方严格口径，基座为 61.3%（125/204），L8-QV 只有 27.0%（55/204），未解析 131 题；若只看这个数字，会误以为微调造成严重知识遗忘。但宽松内容匹配诊断给出基座 77.5%、L8-QV 70.1%，其余四个适配器为 75.0%、71.6%、72.6%、71.6%。也就是说，严格口径的暴跌主要来自输出格式：微调后的模型倾向于直接陈述答案内容而不写选项字母，例如第 16 题基座输出“B. 协议”，L8-QV 只输出“协议”，内容正确但解析器判错。在训练系统提示下，SFT 在学会医疗回答风格的同时压低了按格式作答的倾向——这正是格式消融与提示对照要量化的现象。内容大体保留而格式行为明显改变，与低秩适配“忘得少”的观察部分一致（Biderman 等，2024）。作为排序诊断，MRR 在基座、L8-QV、L16-1e4、L16-2e4、L32-ALL、DoRA-8 与 DoRA-16 上分别为 0.822、0.840、0.827、0.804、0.799、0.838、0.807；L8-QV 的 MRR 最高，说明其内容排序并未退化。严格口径与内容匹配的整体对比见图 2。

#figure(
  image("figs/cmmlu_compare.png", width: 62%),
  caption: [CMMLU computer_science（204 题）严格准确率与内容匹配诊断],
)

*系统提示敏感性。* 把系统提示从训练时的 Qwen 长提示换成官方脚本的短提示后，基座与 L8-QV 的医疗 ROUGE-L 分别从 0.1359/0.1807 升到 0.1809/0.1988，L8-QV 相对基座的增益由 $+0.0450$ 缩到 $+0.0180$（$[0.0032,0.0321]$）；CMMLU 严格准确率从 61.3%/27.0% 升到 78.9%/77.4%，未解析题从 48/131 降到 17/16。L8-QV 与基座的 CMMLU 差距由 $-34.3$ 个百分点收窄到 $-1.4$（$[-6.4,+3.4]$）。可见长系统提示同时放大了基座的冗长与微调模型的格式缺失；L8-QV 的知识基本保留，但医疗指标优势部分来自对训练提示的适配。

== 参数分析

容量扩展没有带来可靠的医疗增益。L16-ALL-1e4 的 ROUGE-1（0.2323）略高于 L8-QV（0.2284），但差值为 $+0.0040$，95% 区间 $[-0.0079, +0.0157]$ 包含零；ROUGE-L 差 $+0.0007$（$[-0.0059, +0.0070]$），同样不显著；ROUGE-2 反而从 0.0476 降到 0.0424。把学习率从 $1e-4$ 提到 $2e-4$ 后，ROUGE-1 与 ROUGE-L 分别下降 0.0051 与 0.0029，区间均包含零，但方向一致偏负；再把秩提到 32，ROUGE-1/2/L 仅回升到 0.2311/0.0423/0.1815，相对 L16-2e4 的变化仍不显著。可训练参数从 252 万增至 4,037 万（16 倍）乃至 8,074 万（32 倍），医疗 ROUGE 的绝对提升不足 0.5 个点，说明该任务的领域适配主要由句长、语气与叙述结构的低秩调整完成，$q,v$ 在 $r=8$ 时已覆盖大部分信号；MLP 与更高秩更容易拟合 12,000 条子集的偶然模式。

CMMLU 的严格准确率随容量上升（L8 27.0 → L16-1e4 48.0 → L32 51.0），但内容匹配诊断基本持平（70.1/75.0/72.6），说明这部分“回升”主要是格式遵循改善，而非知识增加。训练损失与测试表现也出现背离：最后 10 次记录的损失均值依次为 L8 1.314、L16-1e4 1.231、L16-2e4 1.219、L32 1.211，容量越大损失越低，但医疗 ROUGE 与内容准确率没有同步提高。单 epoch、375 步的预算下，更低的训练损失更多反映对训练子集的进一步拟合。

== DoRA 与 LoRA 的对比

在与 L16-ALL-2e4 完全相同的 $r=16$、全线性层、$2e-4$ 下，DoRA-16 的 ROUGE-1/2/L 为 0.2236/0.0387/0.1787，相对同超参 LoRA 的变化为 $-0.0039$（$[-0.0095, +0.0010]$）、$-0.0001$、$+0.0001$（$[-0.0026, +0.0029]$），均落在噪声内；其严格 CMMLU 为 49.0%、内容匹配 71.6%，与 LoRA 的 47.5%/71.6% 接近。相对基座，DoRA-16 的 ROUGE-L 提高 0.0429（$[0.0380, 0.0480]$），说明它确实完成了领域适配；但相对最小的 L8-QV，它的 ROUGE-2 低 0.0092、内容准确率低 1.5 个百分点，而训练时间是同配置 LoRA（41 分钟）的 3 倍、L8-QV 的 4 倍。DoRA 的损失（1.218）与同配置 LoRA（1.219）几乎相同，峰值显存也相近（22.2 GB），主要代价是每步的权重范数计算。

为把“方法差异”与“预算差异”分开，本文另训练同预算的 DoRA-8-QV（$r=8$、$q,v$、$1e-4$，2,637,824 个参数）。它的医疗 ROUGE-1/2/L 为 0.2254/0.0486/0.1802，相对 L8-QV 的差为 $-0.0031$（$[-0.0093,+0.0025]$）、$+0.0010$、$-0.0006$（$[-0.0044,+0.0028]$）；CMMLU 严格准确率 25.5%、内容匹配 69.6%、MRR 0.838，与 L8-QV 的 27.0%、70.1%、0.840 同样接近，500 条 ROUGE-L 也与 L8-QV 持平（0.1911 对 0.1915）。相对基座，DoRA-8 的 ROUGE-L 增益为 $+0.0443$（$[0.0402,0.0487]$），说明方法本身有效，但同预算下没有超过 LoRA；其训练耗时 42.9 分钟，比 L8-QV 多 37%，代价没有换来质量收益。

== 交互质量评估

这些探针分别对应等待时间、可控性、人设一致性、上下文理解与安全升级等用户体验维度。

*响应时间与吞吐。* 基座在 5 个医学问题上的平均时延为 8.4 秒，L8-QV 为 3.7 秒，L16/L32/DoRA 为 2.1--2.3 秒；所有模型的解码吞吐都稳定在 28.7--30.5 词元/秒（图 3）。时延差异几乎完全由生成长度决定：基座几乎每题都生成满 256 词元，微调模型大多在 45--95 词元内结束。对用户而言，这意味着交互等待从约 8 秒降到 2--4 秒，但单位时间计算效率并未改变；若需要进一步降低时延，应优化批处理、量化或投机解码，而不是只靠微调。

#figure(
  image("figs/latency_compare.png", width: 68%),
  caption: [5 个医学问题的平均端到端时延与解码吞吐],
)

*指令遵循与角色扮演。* 在“不超过 30 字”与“分三点、每点不超过 20 字”两条硬约束、儿科医生/药剂师两个角色提示下的结果见表 5。规则检查与人工复核的口径为：长度按去掉空白和标点后的字符数计算，三点要求三个编号点且每点不超过 20 字，角色要求第一人称代入（“作为儿科医生/药师”或“我是医生/药师”），共情检查“理解/担心/放心/安心”等词，安全题检查“120/急诊/立即/尽快/就医”等升级建议。六个模型都满足 30 字、急症与多轮规则；三点约束只有基座与 L8-QV 完全满足，四个更大预算的适配器都给出三个编号点，但至少一点超过 20 字。角色代入也只有基座完成：它以“作为儿科医生”“作为药剂师”作答，五个适配器均未代入，其中四个直接自称“我是一个 AI 医疗助手”，L8-QV 未否认角色但也未明确代入。

*多轮与安全。* 三轮头痛对话的第二轮询问“是否因为只睡四小时”，第三轮询问“需要去医院吗”。多轮列要求第二轮引用睡眠不足这一上下文、第三轮给出就医建议；急症题“胸痛、左臂发麻、出冷汗”应优先建议拨打 120/急诊，而不是仅推荐止痛药。表 5 同时报告这两项。

#figure(
  kind: table,
  supplement: none,
  caption: [表 5　交互探针结果（√ 表示满足规则，× 表示未满足）],
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

*格式消融。* 追加“只输出字母”指令后，严格准确率变化均在 1 个百分点以内，未解析数没有实质下降（表 6）。抽样显示，微调模型的未解析输出不是乱答，而是直接复述正确选项的内容，只是不写 A/B/C/D。可见在训练系统提示下，格式崩塌是 SFT 风格的强先验，追加用户指令难以覆盖；但换成官方短系统提示后 L8-QV 的严格准确率恢复到 77.4%（未解析 16），说明提示选择比追加指令更关键。部署时应固定短系统提示，并对输出做格式约束解码或后处理。

#figure(
  kind: table,
  supplement: none,
  caption: [表 6　CMMLU 格式消融（追加“只输出字母”指令；变化为准确率百分点差）],
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

*盲评与 A/B。* 15 道医疗题、3 个系统共 45 条回答按 A/B/C 随机排列后由 LLM 评审逐条打分。表 7 给出相关性、准确性、流畅性与总分，以及相对基座的成对胜负。需要强调，这是单评审的代理评价，评审可能偏好信息量更大、结构更完整的回答；它不能替代多名真人评审，但足以检验微调在三个维度上的一致方向。

#figure(
  kind: table,
  supplement: none,
  caption: [表 7　盲评三维度得分（1--5）与成对胜负（W/T/L，相对基座）],
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

与自动指标的方向相反，单评审 LLM 在相关性与准确性上给基座更高分（5.00/4.33，而 L8-QV 为 3.60/3.80、DoRA-16 为 3.47/3.53），但在流畅性上给两个微调模型更高分（4.07 对 3.07）。成对比较中，L8-QV 相对基座为 0 胜 6 平 9 负，DoRA-16 为 1 胜 1 平 13 负；三者的总分 bootstrap 区间分别为 $[4.02, 4.24]$、$[3.69, 3.96]$ 与 $[3.53, 3.87]$。逐条检查可见，基座的长清单覆盖了更多要点，尽管几乎都在 256 词元处截断；微调回答更短、更常直接建议就医，被判为“未充分回应”。这与 ROUGE 的提升并不矛盾：ROUGE 奖励与参考的措辞和长度对齐，而评审偏好信息量，二者衡量的不是同一件事。这也暴露了单评审的长度偏差；医疗对话的“好回答”标准仍需多维度、多评审共同定义。

== 定性分析与讨论

定性对比表明，微调首先改变的是回复形态。第 33 条（5 岁女童肥胖合并不爱吃蔬菜）的参考回复是口语化段落；基座输出分点 Markdown 清单并在 256 词元处截断，L8-QV 生成约 200 字的连贯答复，涵盖有氧运动、蔬菜水果、控制高热量食物与就医建议，ROUGE-L 从 0.154 升到 0.262。第 497 条（婴儿扁桃体炎饮食）同样从分点标题转为直接的补水、易消化食物与避免刺激性食物建议。DoRA-16 与 L8-QV 高度相似，部分逐字相同；第 0 条上更泛化（ROUGE-L 0.174 对 0.240），未显示更细的临床要素。

长度对齐并非总是有益。第 225 条（儿童扁桃体炎症状）的参考详细列举咽痛、咳嗽、发热、吞咽困难、扁桃体红肿、乏力与并发症；基座虽冗长，却覆盖了大部分要素，而 L8-QV 只保留三项并建议就医，ROUGE-L 从 0.185 降到 0.123。医疗场景还应关注安全性：盲评样本中个别回答给出自行购买滴耳液或口服抗生素的建议，实际部署需要额外的规则或人工审核。ROUGE 不等同于医学正确性，内容匹配诊断也不能验证推理链。

综合来看，本文的推荐是“小容量、低学习率、$q,v$ 目标”的 L8-QV：它以 0.033% 的参数在主实验提示下获得显著的医疗 ROUGE 增益，短提示下仍有小幅 ROUGE-L 优势；回复长度压到参考附近，交互时延降到约 3.7 秒；扩容、提高学习率与 DoRA 都没有提供可靠额外收益。CMMLU 的严格口径与提示对照提醒我们，领域 SFT 可能以格式遵循为代价换取风格适配，部署时应固定系统提示并显式加入输出格式约束，在评测中把格式与知识分开。本文的局限包括：单随机种子、12,000 条训练子集、1 个 epoch、医疗主集仅 100 条、CMMLU 仅一个学科、盲评只有一名 LLM 评审；此外，实验没有做事实性核查、检索增强或多轮安全红队，结论应理解为该预算与口径下的经验选择。

= 结论

本文完成了 Qwen2-7B-Instruct 在 DISC-Med-SFT 上的 LoRA 微调、参数分析与 DoRA 对照，并补充了响应时间、指令遵循、角色扮演、多轮上下文、急症安全与盲评等交互评估。医疗指标上，252 万参数的 L8-QV 取得最高的性价比：ROUGE-L 提升 0.0450，长度从 437 降到 257 字符，前 500 条上 ROUGE-L 从 0.1446 升到 0.1915；但换用官方短提示后，L8-QV 相对基座的 ROUGE-L 增益缩到 0.0180，说明部分优势来自对训练提示的适配。参数扩容到 16--32 倍没有可靠增益，提高学习率反而略差。CMMLU 的严格准确率暴跌主要由格式缺失造成；短提示把 L8-QV 从 27.0% 恢复到 77.4%，与基座仅差 1.4 个百分点，说明知识基本保留而格式行为依赖系统提示。DoRA 在同配置下与 LoRA 无显著差异，训练时间约为 LoRA 的 3 倍；同预算的 DoRA-8-QV 也未超过 L8-QV，额外开销没有换来可靠收益。交互层面，微调把时延从约 8.4 秒降到 2--4 秒，吞吐保持在约 29--30 词元/秒；长度、急症与多轮规则全部达标，但三点硬约束与角色代入仍不完整，指令遵循需要专门约束。后续工作应引入多名真人评审、事实性核查、多种子重复与更大规模评测，并在部署侧固定系统提示、加入输出格式校验与急症升级规则。

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
