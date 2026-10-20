# 检索记录与限制

日期：2026-10-13。目标：2022-2026 年 AQA、具体动作诊断和训练反馈相关工作，用于组会四宫格，而非穷尽式系统综述。

## 入口

- [Awesome-AQA](https://github.com/ZhouKanglei/Awesome-AQA)
- [该仓库的 BibTeX](https://raw.githubusercontent.com/ZhouKanglei/Awesome-AQA/main/data/aqa.bib)
- 官方 CVF/ECVA、AAAI、IJCAI、NeurIPS、PMLR、ACM、出版社、作者网站及 arXiv/OpenReview。

本地原样快照包含 158 条条目，按列表的 year 字段筛得 125 条 2022-2026 年候选。这个数字是候选条目数，不是核实后的独立论文数；可能有重名、重复、年份和链接错误。独立补充检索发现 QAQA、VLM 实证评分、AgentCoach、LLM-FMS、BioCoach、BoT-Feedback 等，不要求其已经被列表收录。

## 检索词与跟进

- `training-free action quality assessment dynamic time warping`
- `zero-shot action quality assessment scoring ranking VLM`
- `skeleton action quality assessment anomaly aware DTW`
- `Universal Physical Education Training-Free Action Quality Assessment`
- `Can Vision Language Models Judge Action Quality`
- `Can Vision-Language Models Judge Olympic Diving`
- `LLM-FMS training fine tuning feedback`
- `AgentCoach CHI 2026`
- `3D pose biomechanics coaching feedback training-free`
- `BoT-Feedback`、`BioCoach From 3D Pose to Prose`
- 对 TPT、FineDiving/TSA、LOGO/GOAT、FineParser、RICA2、Skating-Mixer、MAGR、PECoP、CoFInAl、PAMFN、LAV、NAE、ExpertAF、TechCoach、CrossTrainer、ProfVLM、CPR-Coach、CoT-AFA、FitAQA、TaiChi-AQA 分别核对原始来源。
- 从 FitAQA 原文参考文献追加 3D Pose Based Feedback、QEVD/Stream-VLM 和 BioCoach；从作者主页与论文核对 Consistent Temporal Grounding 的方法和接受状态。

## 分类方法

1. 先确认真正输出：数值分数/排序、评价性叙述、具体错误、可执行指导或姿态纠正。
2. 再检查评分头、反馈模块、适配器、LoRA、领域检测器等是否需要训练；不能只看视觉/语言骨干是否冻结。
3. 区分具体配置：同一论文的 zero-shot/fine-tuned、直接评分/训练回归、评分/诊断分支分别标注。
4. 使用主来源确认正式 venue；初始 arXiv 年份与正式出版年份分开说明。
5. 标记局部训练自由和全文受限的案例，不让条件归类承担主代表的作用。

## 可核实范围

`classified_papers.csv` 的 `confidence` 是本轮分类证据的置信程度，不是对方法效果的评分。`high` 也不表示阅读了全部附录或运行了代码。

- QAQA：PMC 可读全文，核对无监督框架、预训练姿态依赖及评分公式；正式 Sensors 2025。
- 现成 VLM 评分/诊断：官方 CVPR Workshops 2026 页面及论文全文/对应 arXiv HTML。其评分/排名和具体错误任务分开归类。
- AgentCoach：作者提供论文及项目、CHI DOI；核对检查点、预训练模型和提示式反馈。
- LLM-FMS：PMC 可读全文及 PLOS 原刊；核对规则与姿态信息提示，不因为同时输出分数就只放上排。
- BioCoach：官方 CVPR 2026 页面、论文及作者代码，确认是已发表工作，而且训练了跨注意力模块。
- BoT-Feedback：仅核实 arXiv v1；主实验 zero-shot，正文另述最终反馈监督及微调补充实验，因此配置分开列。未将其称为正式会议论文。
- MoAKE：作者仓库和 arXiv 标注 ECCV 2026；明确有训练命令，未取得正式 proceedings 页面。状态为作者确认录用，而非仅凭社区列表宣称正式发表。
- Consistent Temporal Grounding：作者主页称 ECCV 2026 待刊，论文明确有注意力一致性训练；正式 proceedings 未核实。
- EFA / CoT-AFA：核实 arXiv 2025 首稿及 2026 修订；未找到足够证据把它标为正式期刊/会议。
- Hou 等 2025 太极两阶段 AQA：公开摘要可读，正文访问受限；质量阶段看似几何 DTW，姿态估计器本身训练，因此仅列条件归类。
- MED-ACDTW：搜索索引可检索投稿 PDF 段落，首页标注 ICLR 2026 在审；OpenReview forum/pdf 标准地址要求浏览器验证，缓存 PDF 地址重新直接访问为 404。未核实完整方法、最终决定及首次提交日期，保留候选和状态未知，不用其填充“正式论文”标签。

## 原始列表的元数据问题

- `xu2025humancentric` 和 `zhang2025poseguided` 使用同一个 IEEE 文档号 10946879，但题目/venue 不同。未擅自认定两个条目都正确，候选 CSV 标记冲突。
- TechCoach 正式发表是 AAAI 2026，原始预印本在 2024。TaiChi-AQA 是 2026 卷期、2025-12-29 在线发表。
- `Interpretable Feedback` 标签可能只指注意力/分数归因，不能直接用作“生成训练反馈”的证据。
- 列表原始作者顺序等元数据不全部校正；需要正式引用时采用出版社/正式论文的 BibTeX，而不是直接复制候选表。

## 未计入主表的历史或相邻工作

- [Classification of motor errors to provide real-time feedback for sports coaching in virtual reality: A case study in squats and Tai Chi pushes](https://www.sciencedirect.com/science/article/pii/S0097849318301304)：2018，超出主范围。
- AIFit (CVPR 2021)：可用于历史脉络，但不纳入 2022-2026 主分类统计。
- GAIA (NeurIPS 2024)：评价 AI 生成视频中的动作，不等同于真实练习者训练质量；不能用其替代右上角真实人体 AQA 的核心代表。
- 通用姿态重建、动作识别或视频时序定位论文：仅作为组件相关工作；没有质量/诊断/指导输出时，不为凑满四宫格强行加入。

## 复现

在装有 `bibtexparser==1.4.3` 的环境中运行：

```bash
python build_review.py --bib sources/awesome_aqa_20261013.bib --output /tmp/aqa_quadrants_rebuilt
```

候选表和分类表均通过结构化 BibTeX/CSV API 生成。构建脚本只处理本地整理的元数据，不访问 API 或重新执行实验。日期是本次检索截止，运行脚本不会使证据自动更新。
