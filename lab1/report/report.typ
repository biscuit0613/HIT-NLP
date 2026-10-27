// ============================================================
// 实验一：基于分类模型的情感分析 —— 实验报告（typst）
// 编译：typst compile report.typ
// 数值来源：实验运行结果（逐项核对一致；完整留档可经 git 历史回溯，禁止修改）
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
    实验一：基于分类模型的情感分析
  ]
  #text(size: 11pt, font: "Noto Sans CJK SC")[
    ——经典机器学习与预训练模型微调
  ]
]
#v(0.4em)

*摘要*　本文利用中文商品评论数据集完成正面/负面情感二分类，比较了经典机器学习与预训练模型微调两条路线。经典路线以字符 1--3 gram 的 TF-IDF 为统一表示，比较 15 种分类器，并通过特征空间、特征选择与分类器参数的单因素对照构造改进管线，验证集 Macro-F1 由 0.8467 提升至 0.8495。微调路线以 UER 中文 RoBERTa-Medium（约 3656 万参数）为基础，比较输入长度、训练预算与参数高效微调等方案，最优单模型 Macro-F1 达 0.8787，较经典路线高约 3 个百分点；三种子概率平均集成为 0.8777。误分类分析显示错误样本中转折表达占比 37.9%、高于全样本基准 32.5%，进一步核查表明相当比例的错误源于评分标签与评论文本情感不一致；两模型分歧样本中微调正确 261 条、经典正确 133 条。

*关键词*　情感分析；TF-IDF；预训练微调；UER RoBERTa；LoRA；误分类分析

= 任务与数据

本实验使用课程提供的 MARC 中文商品评论数据完成二分类情感分析：用户评分 1--2 星映射为负面（标签 0），4--5 星映射为正面（标签 1），3 星样本剔除以降低中性歧义。训练集 20,000 条、验证集 4,000 条，两类别均严格均衡，两条技术路线共用同一划分，保证成绩可比。

对文本长度的核验表明，评论整体较短：字符长度中位数 37，按 UER 配套 WordPiece 编码后词元数中位数 38；以 128 词元截断仅影响 4.85% 的样本，256 词元截断影响 0.75%。这为后续输入长度设置提供了依据。需要说明的是，评分标签与文本情感并非始终一致（真实评论中存在高分差评等现象），这一点在误分类分析中会重新讨论。

= 方法

== 经典机器学习路线

该路线以 TF-IDF 将评论文本转换为数值特征，再调用 scikit-learn 分类器。文本表示采用字符 1--3 gram：以逻辑回归探针对比六种表示（字符 1--2/1--3/1--4 gram、jieba 词 1 gram/1--2 gram、词字拼接）后，char13 以其 0.8442 的验证 Macro-F1 居首（词级最佳 word12 为 0.8407），且对分词错误不敏感。TF-IDF 开启亚线性词频（sublinear_tf）、过滤低频词（min_df=2），并对向量做 L2 归一化。

主实验比较 15 种分类器：朴素贝叶斯 3 种（MultinomialNB、ComplementNB、BernoulliNB）、线性模型 4 种（逻辑回归、LinearSVC、Ridge、感知机）、距离与核方法 3 种（KNN、最近质心、RBF 核 SVC）、决策树与集成 4 种（决策树、随机森林 200 棵、AdaBoost 200 轮、梯度提升 200 轮）以及 MLP（先用 TruncatedSVD 降至 300 维，隐层 256，含早停）。结果见 4.1 节表 1。

在 15 方法基线之上，按单因素原则做四组改进对照：A 组调整特征空间（n-gram 范围、min_df、词字拼接）；B 组调整 TF-IDF 设置并引入卡方特征选择；C 组对照距离类方法在稀疏特征与 TruncatedSVD 稠密特征上的表现；D 组扫描各分类器关键参数（逻辑回归/LinearSVC/SVC 的 C、朴素贝叶斯 alpha、KNN 的 k）。各组胜者再组合成最终候选管线，在验证集上重算取最优。

== 预训练模型微调路线

微调基于 UER 中文 RoBERTa-Medium（8 层、隐维 512、约 3656 万参数）及其配套 WordPiece 分词器（词表 21,128）。对编码过程的观察表明：中文基本按字切分，未登录英文词退化为子词片段（keyword → key + \#\#board），表情符号映射为 [UNK]；模型输入由 [CLS]、正文词元、[SEP] 组成，批内按最长样本补齐 [PAD]，attention_mask 标记有效位置。

训练采用 AdamW，权重矩阵施加 0.01 的权重衰减、偏置与 LayerNorm 不衰减；学习率线性预热 10% 后线性衰减；批大小 32，梯度裁剪阈值 1.0。每轮结束评价验证集，按验证 Macro-F1 保存最优模型，训练结束后恢复保存的模型复评核对。

围绕该基线设计变体对照：base128 为截断 128 词元的基线；len256 将截断提高至 256 词元以覆盖长评论；len256ep4 在 len256 上提高学习率并延长训练；lora256 采用 LoRA（r=8，作用于注意力 Q/V 投影，并将随机初始化的 pooler 与分类头一并纳入训练与保存），可训练参数仅 1.07%，对照参数高效方案；另以相同配置更换随机种子（8888、2024）重复训练，用于评估成绩波动并构造多种子集成（三模型正面概率平均后阈值 0.5 决策）。

= 实验设置

所有实验在本地完成：Python 3.11 环境由 uv 管理，经典路线使用 scikit-learn 1.5.2；微调使用 PyTorch 2.5.1（CUDA，RTX 4060 8 GB）、Transformers 4.44.2 与 PEFT 0.12.0，预训练权重离线加载。评价统一报告 Accuracy、Macro-F1、宏平均 Precision/Recall 与混淆矩阵，阈值取 0.5。微调路线的完整流程（训练、每轮验证、最优模型保存与恢复、新评论预测）同时以教学包验收 Notebook 形式提供，其运行结果与本文引用的数值一致。

= 结果与分析

== 两条路线成绩对比

15 种经典分类器在 char13 TF-IDF 上的成绩如表 1 所示。Ridge 以 0.8467 居首，RBF 核 SVC 与其仅差 0.0017，但训练耗时相差三个数量级；深层决策树、KNN、最近质心表现明显偏低；MLP 在 SVD 降维特征上亦未超过线性模型，说明高维稀疏 TF-IDF 上线性决策面与本文任务高度匹配。

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, left, right, right, right),
  stroke: (x, y) => (
    top: if y == 0 { 0.9pt } else { 0pt },
    bottom: if y == 1 { 0.5pt } else if y == 15 { 0.9pt } else { 0pt },
  ),
  inset: 3.2pt,
  text(size: 8.2pt)[*方法*], text(size: 8.2pt)[*方法组*],
  text(size: 8.2pt)[*Accuracy*], text(size: 8.2pt)[*Macro-F1*],
  text(size: 8.2pt)[*训练耗时/s*],
  [Ridge], [线性], [0.8468], [0.8467], [0.2],
  [SVC(RBF)], [核方法], [0.8450], [0.8450], [202.5],
  [LinearSVC], [线性], [0.8442], [0.8442], [0.3],
  [逻辑回归], [线性], [0.8400], [0.8400], [0.5],
  [MLP(SVD300)], [神经网络], [0.8320], [0.8320], [25.2],
  [MultinomialNB], [贝叶斯], [0.8280], [0.8280], [0.0],
  [ComplementNB], [贝叶斯], [0.8280], [0.8280], [0.0],
  [随机森林], [集成], [0.8195], [0.8195], [5.8],
  [AdaBoost], [集成], [0.8195], [0.8195], [44.5],
  [感知机], [线性], [0.8147], [0.8145], [0.1],
  [梯度提升], [集成], [0.8125], [0.8125], [116.1],
  [BernoulliNB], [贝叶斯], [0.7987], [0.7970], [0.0],
  [最近质心], [距离], [0.7895], [0.7895], [0.0],
  [KNN(k=15)], [距离], [0.7808], [0.7805], [0.0],
  [决策树], [树], [0.7143], [0.7142], [18.2],
)
#figure(caption: [表 1　15 种经典分类器验证集成绩（统一字符 1--3 gram TF-IDF 表示）], kind: table, supplement: none)[]

改进对照中，A 组引入稀疏度更低的 min_df=1（0.8460）与卡方 Top-50,000 特征选择（0.8457）均带来稳定增益；D 组中 RBF SVC（C=3）达 0.8492，为单项最优；将 A、B 组的最优表示与选择组合后接线性分类器，得到最终改进管线 char13+min_df=1 + 卡方 Top-50,000 + 逻辑回归，验证 Macro-F1 为 0.8495，较 15 方法基线提升 0.0028，且训练仅需约 1 秒——以约 1 秒的训练成本超过 D 组最优的 RBF SVC（训练 341 秒），体现了特征工程对线性模型的价值。相对地，词字拼接、sublinear_tf 关闭、min_df=5 及 SVD 降维均未超过各自基线。

微调方面，六个变体与集成的成绩汇总于表 2 与图 1。微调的最好单模型（0.8787）较经典最优提升 0.0292；len256 较 base128 提升 0.0015，说明长评论截断的减少有正面作用但幅度有限；提高学习率并延长至 4 轮的 len256ep4 未获提升且在末轮出现验证下滑，提示该规模数据上训练预算已饱和；LoRA 变体以 1.07% 的可训练参数取得 0.8720，逼近全参数水平。

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  align: (left, right, right, right, right, right),
  stroke: (x, y) => (
    top: if y == 0 { 0.9pt } else { 0pt },
    bottom: if y == 1 { 0.5pt } else if y == 7 { 0.9pt } else { 0pt },
  ),
  inset: 3.2pt,
  text(size: 8.2pt)[*变体*], text(size: 8.2pt)[*截断*],
  text(size: 8.2pt)[*学习率*], text(size: 8.2pt)[*轮数*],
  text(size: 8.2pt)[*可训练参数*], text(size: 8.2pt)[*Macro-F1*],
  [base128], [128], [2e-5], [3], [100%], [0.8732],
  [len256], [256], [2e-5], [3], [100%], [0.8747],
  [len256ep4], [256], [3e-5], [4], [100%], [0.8722],
  [lora256], [256], [3e-4], [5], [1.07%], [0.8720],
  [len256,种子 8888], [256], [2e-5], [3], [100%], [0.8782],
  [len256,种子 2024], [256], [2e-5], [3], [100%], [0.8787],
  [集成(3 种子平均)], [256], [2e-5], [3], [100%], [0.8777],
)
#figure(caption: [表 2　预训练微调各变体与集成验证集成绩（最优轮）], kind: table, supplement: none)[]

#figure(
  image("figs/route_comparison.png", width: 100%),
  caption: [两条路线成绩对比（橙=经典路线，蓝=预训练微调；实色=Macro-F1，斜纹=Accuracy）],
)

== 误分类与分歧分析

以最优单模型（len256 种子 2024）的恢复模型为准，验证集共 485 条错误（12.1%），假负例与假正例基本均衡（246 对 239）。对错误样本表达的统计显示（图 2）：含否定词的样本占 75.5%、全样本为 73.1%，二者接近；含转折词的样本占 37.9%，明显高于全样本基准 32.5%，说明含"但、不过、却"等转折结构的评论更易被误判，模型对句内情感的转折把握不足。按文本长度分箱比较错误率未发现单调趋势。

#figure(
  image("figs/error_flag_rates.png", width: 52%),
  caption: [错误样本与全样本中否定/转折表达占比],
)

对错误实例的逐条核查表明，高置信度错误中相当比例并非语言理解失败，而是评分标签与文本情感不一致所致。假正例中，如"很有意思，贴切小孩子生活！孩子们下课抢着看！推荐！"（文本高度正面，用户评分却为 1--2 星）模型以 0.98 的正面概率预测为正面；假负例中，如"一次糟糕的购物体验，给差评系统还不通过……"（文本明显负面却被标为正面）模型以 0.99 的负面概率预测为负面，以及"书的装帧丑到令人绝望……四星是给内容，排版设计一星都不想给"（用户评 4 星但正文以批评为主）。这类样本构成当前准确率的主要瓶颈，也说明 0.88 左右的验证成绩已接近标签噪声下的期望上限。

该模型与经典最优管线（表 1 改进最优，0.8495）的分歧样本共 746 条，其中微调正确而经典错误 261 条、经典正确而微调错误 133 条、两者都错 352 条，微调模型优势稳定。新评论测试中，模型对明确正/负面、否定与转折表达（如"包装很好，但是商品质量差"判为负面 0.89；"物流有点慢，不过东西很好用"判为正面 0.97）均正确，对双重否定式表达"不是不好用，只是价格实在太贵了"给出 0.61 的正面概率——正确理解否定，但对转折后的抱怨权重不足，与前述统计一致。

== 技术讨论

三种子重复训练暴露了验证成绩的波动幅度：相同配置、不同随机种子下 Macro-F1 介于 0.8747 与 0.8787 之间（跨度 0.004），若报告仅取单次成绩判断"改进有效"并不可靠。相应的多种子概率平均集成为 0.8777，未超过最优单模型——成员模型同配置、同数据，错误高度相关，集成主要起到稳定与降方差作用而非抬升上限；这从侧面说明 0.88 附近的波动更多来自数据与优化噪声，而非模型选择差异。

LoRA 变体在调试中曾出现适配器重载后成绩骤降至 0.44 的问题。定位发现：UER RoBERTa 权重不含 pooler（其预训练无 NSP 任务），加载时 pooler 与分类头均为随机初始化；而早期实现仅将分类头纳入适配器保存，重载时 pooler 重新随机化，特征空间与训练时错位。将 pooler 一并纳入训练与保存后，合并导出的模型复评成绩与最优轮完全一致（0.8720）。该问题说明参数高效微调中"哪些参数参与训练、哪些随检查点保存"必须一一对应，恢复流程中的核对断言不可省略。

= 结论

本文在统一数据划分上完成了情感分析两条路线的系统比较。经典路线中，字符 1--3 gram TF-IDF 加线性模型是性价比最高的方案，特征选择等改进可小幅提升且几乎没有推理代价；预训练微调整体高出约 3 个百分点，其中输入长度 256 词元与三训练轮为最稳妥配置，LoRA 以 1.07% 参数逼近全参数成绩，适合资源受限场景。误差分析表明，在当前验证集上模型错误的主要来源是评分标签与文本情感不一致，转折与否定结构的影响次之；继续在该数据上追求更高的验证分数收益有限，改进方向更应放在数据清洗与更细粒度（如属性级）的情感建模。

= 参考文献

#set par(first-line-indent: 0em)
#text(size: 9pt)[
  [1] Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine learning in Python[J]. Journal of Machine Learning Research, 2011, 12: 2825--2830.

  [2] Devlin J, Chang M W, Lee K, et al. BERT: Pre-training of deep bidirectional transformers for language understanding[C]//Proceedings of NAACL-HLT 2019. Minneapolis: ACL, 2019: 4171--4186.

  [3] Liu Y, Ott M, Goyal N, et al. RoBERTa: A robustly optimized BERT pretraining approach[EB/OL]. arXiv:1907.11692, 2019.

  [4] Hu E J, Shen Y, Wallis P, et al. LoRA: Low-rank adaptation of large language models[C]//International Conference on Learning Representations (ICLR), 2022.

  [5] Zhao Z, Chen H, Zhang J, et al. UER: An open-source toolkit for pre-training models[C]//Proceedings of EMNLP-IJCNLP 2019: System Demonstrations. Hong Kong: ACL, 2019: 241--246.

  [6] Wolf T, Debut L, Sanh V, et al. Transformers: State-of-the-art natural language processing[C]//Proceedings of EMNLP 2020: System Demonstrations. Online: ACL, 2020: 38--45.
]
#set par(first-line-indent: 2em)

= 附录 A　实验工具清单

#set par(first-line-indent: 0em)
#table(
  columns: (auto, auto, auto),
  align: (left, left, left),
  stroke: (x, y) => (
    top: if y == 0 { 0.9pt } else { 0pt },
    bottom: if y == 1 { 0.5pt } else if y == 11 { 0.9pt } else { 0pt },
  ),
  inset: 3pt,
  text(size: 8.5pt)[*工具*], text(size: 8.5pt)[*版本*], text(size: 8.5pt)[*用途*],
  [Python], [3.11], [实验主语言，两条路线共用],
  [uv], [0.9.15], [虚拟环境与依赖管理（含 PyPI 镜像/代理）],
  [NumPy / Pandas], [1.26 / 2.2], [数值计算与数据表格处理],
  [scikit-learn], [1.5.2], [TF-IDF 向量化、特征选择、15 种经典分类器与指标],
  [PyTorch], [2.5.1+cu124], [预训练模型微调的深度学习框架（GPU）],
  [Transformers], [4.44.2], [UER RoBERTa 模型与 WordPiece 分词器的加载与训练接口],
  [PEFT], [0.12.0], [LoRA 参数高效微调对照实验],
  [jieba], [0.42.1], [词级表示对照中的中文分词],
  [Matplotlib], [3.9.2], [全部图表绘制],
  [JupyterLab], [4.2.5], [验收 Notebook（微调训练与预测全流程）的运行环境],
  [Typst], [0.15.1], [本报告的排版与编译],
)
#set par(first-line-indent: 2em)

// ---------- 末页：实验成绩与教师签名（对应模板末页）----------
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
