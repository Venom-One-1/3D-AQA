"""Build a dated literature snapshot from manually checked primary sources."""

import argparse
import csv
import hashlib
import json
import logging
import shutil
from collections import Counter
from pathlib import Path

import bibtexparser


DATE = "2026-10-13"
logging.getLogger("bibtexparser").setLevel(logging.ERROR)
parser = argparse.ArgumentParser()
parser.add_argument("--bib", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
database = bibtexparser.loads(args.bib.read_text(encoding="utf-8"))
bibliography = {entry["ID"]: entry for entry in database.entries}
papers = []


def add(bib_id, name, year, venue, status, quadrant, task, output, training,
        evidence, source, *, title=None, variant="main", publication="",
        caveat="", representative="", confidence="high", location="摘要及方法",
        paper_id=None):
    entry = bibliography.get(bib_id, {})
    papers.append({
        "paper_id": paper_id or bib_id or name,
        "variant": variant,
        "short_name": name,
        "title": title or entry.get("title", name),
        "year": year,
        "venue": venue,
        "publication_status": status,
        "quadrant": quadrant,
        "task": task,
        "output_type": output,
        "task_specific_training": training,
        "classification_evidence": evidence,
        "evidence_location": location,
        "primary_url": source,
        "publication_url": publication or source,
        "scope_caveat": caveat,
        "representative_priority": representative,
        "confidence": confidence,
        "awesome_bib_id": bib_id,
        "checked_on": DATE,
    })


VLM_TITLE = "Can Vision Language Models Judge Action Quality? An Empirical Evaluation"
VLM_PUB = "https://openaccess.thecvf.com/content/CVPR2026W/SAUAFG/html/Monte_e_Freitas_Can_Vision_Language_Models_Judge_Action_Quality_An_Empirical_Evaluation_CVPRW_2026_paper.html"
VLM_SRC = "https://arxiv.org/html/2604.08294v1"
DIVING_TITLE = "Can Vision-Language Models Judge Olympic Diving? From Reasoning to Scores in Zero-Shot Action Quality Assessment"
DIVING_SRC = "https://arxiv.org/abs/2609.19354"
BOT_TITLE = "BoT-Feedback: Grounding Multimodal Reasoning in Biomechanical Evidence for Explainable Human Action Feedback"
BOT_SRC = "https://arxiv.org/html/2610.06972v1"

# Q1: score/ranking without AQA-specific fitting.
add("", "QAQA / Anomaly-Aware DTW", 2025, "Sensors 25(23):7160", "published", "Q1",
    "八段锦、易筋经的参考-学习者动作评分", "score", "no",
    "预训练姿态估计后计算几何特征、异常检测和自适应约束 DTW；质量分数由距离公式聚合，方法明确称为无监督、无需重训练。",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12693942/",
    title="Skeleton-Based Action Quality Assessment with Anomaly-Aware DTW Optimization for Intelligent Sports Education",
    publication="https://www.mdpi.com/1424-8220/25/23/7160",
    caveat="仍依赖预训练姿态模型、参考动作、参数设计及 DBSCAN 的逐序列拟合；不是完全无先验或无需任何数据处理。正式实验主要输出数值质量分数。",
    representative="primary", location="Sec. 3 总体框架；Sec. 3.5.3 评分公式 (12)-(14)", paper_id="fu2025qaqa")
add("", "Off-the-shelf VLM scoring", 2026, "CVPR Workshops / SAUAFG", "published_workshop", "Q1",
    "跳水和花滑的直接评分与成对质量比较", "score;pairwise_ranking", "no",
    "使用现成 Gemini、Qwen、InternVL 的提示式推理；评分任务直接预测执行分/GOE，成对任务判断哪次执行更好，不进行额外训练。",
    VLM_SRC, title=VLM_TITLE, publication=VLM_PUB, variant="score_and_ranking",
    caveat="代表一个正式发表的训练自由评测配置，不是证明高性能的新评分算法；同篇的具体错误诊断配置另列 Q4。",
    representative="secondary", location="Sec. 2.1.4-2.1.5；Sec. 3.5", paper_id="freitas2026vlm")
add("", "MED-ACDTW", "2026 manuscript", "OpenReview / ICLR 2026 submission", "submission_decision_unverified", "Q1",
    "体育教学动作与参考动作的 DTW 质量评分", "score", "no",
    "公开投稿 PDF 描述基于骨架几何特征与 DTW 的训练自由评分，不学习质量回归器。",
    "https://openreview.net/pdf?id=SB4LrFt5Zu",
    title="Universal Physical Education: Training-Free Action Quality Assessment Based on Dynamic Time Warping",
    publication="https://openreview.net/forum?id=SB4LrFt5Zu",
    caveat="检索索引中的投稿 PDF 标为 ICLR 2026 在审；直接论坛/PDF 被浏览器验证拦截，缓存地址重新访问为 404。未核实最终决定、首次提交日期及完整方法，暂列候选，不能写成 ICLR 2026 正式论文。",
    confidence="medium", location="搜索引擎索引的投稿 PDF 首页/方法摘要；直接访问受限", paper_id="openreview_med_acdtw")
add("hou2025interpretable", "Two-stage 3D + DTW (Hou et al.)", 2025, "The Visual Computer", "published", "Q1",
    "太极拳的静态/动态特征质量评分", "score;feature_analysis", "conditional_assessment_stage",
    "公开摘要将评估阶段描述为静态/动态特征的改进 DTW 对齐与分数汇总，未描述质量回归器；同时明确训练了新的 MiETransformer 姿态估计器。",
    "https://link.springer.com/article/10.1007/s00371-025-04101-6",
    caveat="只能暂按质量评估阶段放在 Q1；整套系统包含新训练的姿态网络。全文访问受限，需取得全文后确认是否还存在质量监督拟合。不要用作无条件 training-free 代表作。",
    confidence="conditional", location="出版社公开摘要")
add("", "Olympic Diving VLM (direct scores)", 2026, "arXiv:2609.19354", "preprint", "Q1",
    "AQA-7 跳水直接评分", "score", "no", "原始 VLM 直接输出零样本数值分数；与该文另行拟合的集成回归方案区别对待。",
    DIVING_SRC, title=DIVING_TITLE, variant="direct_vlm_baseline",
    caveat="这里只归类原始零样本基线，不能将该文完整提案归为训练自由。", location="摘要中的 standalone VLM 与 regression-based framework 对照", paper_id="velesaca2026diving")

# Q2: trained score/ranking models, including interpretable but non-coaching outputs.
add("xu2022finediving", "TSA (FineDiving)", 2022, "CVPR", "published", "Q2",
    "细粒度跳水程序与质量评分", "score", "yes",
    "使用程序/时序标注学习程序感知表征与 Temporal Segmentation Attention，训练质量分数回归。",
    "https://openaccess.thecvf.com/content/CVPR2022/html/Xu_FineDiving_A_Fine-Grained_Dataset_for_Procedure-Aware_Action_Quality_Assessment_CVPR_2022_paper.html",
    caveat="FineDiving 是数据集/论文名，TSA 才是该论文提出的方法。", representative="secondary")
add("bai2022action", "TPT", 2022, "ECCV", "published", "Q2",
    "时序动作解析与分数预测", "score", "yes",
    "学习 Temporal Parsing Transformer 和对比式分数回归；输出数值分数而非教练反馈。",
    "https://arxiv.org/abs/2207.09270", publication="https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640416.pdf",
    representative="primary")
add("zhang2023logo", "GOAT (LOGO)", 2023, "CVPR", "published", "Q2",
    "长视频多人花样游泳质量评分", "score", "yes",
    "作者代码给出 GOAT 的训练流程，将群体关系表征用于 USDL/CoRe 评分；不生成动作纠正语言。",
    "https://github.com/shiyi-zh0408/LOGO",
    caveat="LOGO 是长视频数据集，方法名称为 GrOup-aware ATtention (GOAT)。长视频 AQA 并非空白。",
    representative="secondary", location="作者仓库 Abstract、训练命令和数据统计")
add("xia2023skatingmixer", "Skating-Mixer", 2023, "AAAI", "published", "Q2",
    "长时花滑的音视频评分", "score", "yes",
    "训练 memory-recurrent MLP 音视频模型预测花滑评分；面向多分钟视频，不输出自然语言训练反馈。",
    "https://ojs.aaai.org/index.php/AAAI/article/download/25392/25164")
add("xu2024fineparser", "FineParser", 2024, "CVPR", "published", "Q2",
    "人体相关细粒度时空解析与质量评分", "score;spatiotemporal_attribution", "yes",
    "学习人体空间掩码及细粒度时序解析并服务质量分数回归；可解释的解析结果不是自然语言指导。",
    "https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_FineParser_A_Fine-grained_Spatio-temporal_Action_Parser_for_Human-centric_Action_Quality_Assessment_CVPR_2024_paper.pdf",
    representative="primary")
add("majeedi2024rica2", "RICA2", 2024, "ECCV", "published", "Q2",
    "评分规则引导的程序动作评分与置信度", "score;uncertainty", "yes",
    "规则和语言步骤描述进入可训练表征、信息瓶颈及分数回归模块；训练目标包含分数监督。",
    "https://arxiv.org/pdf/2408.02138", publication="https://abrarmajeedi.github.io/rica2_aqa/",
    caveat="使用评分 rubric/语言知识并不意味着不训练，也不等于输出教练式反馈。", location="Method；附录训练细节")
add("zhou2024magr", "MAGR", 2024, "ECCV", "published", "Q2",
    "持续动作质量评分", "score", "yes",
    "以流式任务进行质量模型训练，使用流形对齐和图正则抑制遗忘，最终仍为分数预测。",
    "https://www.hubertshum.com/pbl_eccv2024caqa.htm")
add("dadashzadeh2024pecop", "PECoP", 2024, "WACV", "published", "Q2",
    "参数高效预训练与下游运动/康复评分", "score", "yes",
    "3D-Adapters 进行额外自监督预训练，随后用于 MUSDL、CoRe、TSA 等学习型 AQA 方法。",
    "https://arxiv.org/abs/2311.07603", publication="https://github.com/Plrbear/PECoP",
    caveat="自监督、参数高效都仍然属于有任务相关训练；不能归为 training-free。")
add("zhou2024cofinal", "CoFInAl", 2024, "IJCAI", "published", "Q2",
    "长期动作的粗到细等级评分", "score;grade", "yes",
    "训练粗等级原型并利用细等级原型完成质量判别；instruction alignment 服务等级/分数而非训练指导生成。",
    "https://www.ijcai.org/proceedings/2024/0196")
add("zeng2024multimodal", "PAMFN", 2024, "IEEE TIP", "published", "Q2",
    "花滑/艺术体操多模态评分", "score", "yes",
    "训练 RGB、光流、音频分支及渐进自适应融合，提升分数回归。",
    "https://arxiv.org/abs/2402.09444", publication="https://doi.org/10.1109/TIP.2024.3362135")
add("xu2025languageguided", "LAV", 2025, "CVPR", "published", "Q2",
    "长时运动的语言引导音视频评分", "score", "yes",
    "语言引导音视频表征学习最终用于长时体育动作评分，而非输出教练指导段落。",
    "https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_Language-Guided_Audio-Visual_Learning_for_Long-Term_Sports_Assessment_CVPR_2025_paper.pdf")
add("wang2026taichi", "TaiChi-AQA / attention-gMLP-DAE", 2026, "IET Computer Vision 20(1):e70053", "published", "Q2",
    "24 式太极的分段动作质量评分", "score;score_distribution", "yes",
    "训练 attention/gMLP 与分数分布自编码器；数据提供分数及动作描述，但不是针对每次错误的教练反馈生成模型。",
    "https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cvi2.70053",
    caveat="2025-12-29 首次在线发表，2026 年正式卷期；1313 个短片来自 55 个长视频。动作步骤描述不等于专家纠错反馈。")
add("xu2026moake", "MoAKE", 2026, "ECCV (author-confirmed acceptance)", "accepted_author_confirmed", "Q2",
    "多动作统一评分与零/少样本跨动作迁移", "score", "yes",
    "训练动作知识专家和动态聚合网络；作者代码明确提供 380 epoch 训练及训练后 checkpoint 测试。",
    "https://github.com/XuHuangbiao/MoAKE", publication="https://arxiv.org/abs/2607.19826",
    caveat="作者仓库及 arXiv 注释标为 ECCV 2026，未取得正式 proceedings 页面；zero/few-shot 测试不表示评分模型从未训练。")
add("xu2026limssr", "LIMSSR", 2026, "ICML / PMLR 306", "published", "Q2",
    "训练时缺模态的多模态质量评分", "score", "yes",
    "在训练时不完整模态条件下学习序列到分数推理及融合模块；LLM 使用不改变任务专用训练的事实。",
    "https://proceedings.mlr.press/v306/xu26bo.html")
add("", "Olympic Diving VLM (ensemble regression)", 2026, "arXiv:2609.19354", "preprint", "Q2",
    "AQA-7 跳水的文本特征/子分数集成评分", "score", "yes",
    "对 VLM 推理文本做 TF-IDF、降维及集成回归，拟合最终竞赛分数；整体包含训练的质量预测器。",
    DIVING_SRC, title=DIVING_TITLE, variant="ensemble_regression",
    caveat="标题的 zero-shot 指 VLM 使用方式，不代表完整的集成评分流程无需训练。", location="摘要的 regression-based framework", paper_id="velesaca2026diving")

# Q3 includes narrative evaluation, diagnostic labels, and prescriptive feedback.
add("", "3D Pose Based Feedback", 2022, "ACCV", "published", "Q3",
    "体能/康复动作错误识别与纠正", "error_diagnosis;pose_correction", "yes",
    "用双分支 GCN 学习动作错误类别及纠正措施，并非用手写角度规则直接输出结果。",
    "https://openaccess.thecvf.com/content/ACCV2022/html/Zhao_3D_Pose_Based_Feedback_For_Physical_Exercises_ACCV_2022_paper.html",
    title="3D Pose Based Feedback for Physical Exercises",
    caveat="反馈包括错误类型与姿态纠正，不等同于开放式 LLM 教练段落。", paper_id="zhao2022posefeedback")
add("zhang2024narrative", "NAE", 2024, "CVPR", "published", "Q3",
    "评分感知的动作叙述性评价", "narrative_evaluation;score", "yes",
    "训练可学习提示、多模态交互和文本匹配/生成，输出专业动作评价。",
    "https://arxiv.org/abs/2404.14471",
    caveat="叙述性评价可以描述优缺点，但不能把所有输出都称为可执行训练建议。", representative="secondary")
add("", "ImagineNet (CPR-Coach)", 2024, "CVPR", "published", "Q3",
    "心肺复苏单错误/复合错误诊断", "error_diagnosis", "yes",
    "在单错误训练数据上训练 ImagineNet，识别测试中的复合错误；输出细粒度错误类别。",
    "https://openaccess.thecvf.com/content/CVPR2024/html/Wang_CPR-Coach_Recognizing_Composite_Error_Actions_based_on_Single-class_Training_CVPR_2024_paper.html",
    title="CPR-Coach: Recognizing Composite Error Actions based on Single-class Training",
    caveat="属于具体诊断类，不是自由文本训练反馈生成。", paper_id="wang2024cprcoach")
add("okamoto2024hierarchical", "NS-AQA", 2024, "CVPR Workshops / CVSports", "published_workshop", "Q3",
    "跳水评分、具体错误分析与可视证据报告", "error_diagnosis;report;score", "yes_task_parser",
    "质量分析及动作/阶段识别由规则微程序完成，但作者另外标注并训练跳台和水花等检测器；整套系统存在领域专用训练。",
    "https://arxiv.org/html/2403.13798v2",
    caveat="如果只讨论已经取得符号后的规则分析器，该组件可称训练自由；四宫格按整套部署系统归入 Q3。不能误写成训练阶段分割网络。",
    location="Introduction 训练掩码贡献；Sec. 3.1、3.2")
add("", "Stream-VLM (QEVD)", 2024, "NeurIPS Datasets and Benchmarks", "published", "Q3",
    "流式健身视频中的及时训练反馈", "corrective_feedback;feedback_timing", "yes",
    "训练健身视觉网络、跨注意力适配器和语言 LoRA，并学习何时输出反馈。",
    "https://arxiv.org/html/2407.08101v1",
    title="What to Say and When to Say it: Live Fitness Coaching as a Testbed for Situated Interaction",
    publication="https://proceedings.neurips.cc/paper_files/paper/2024/hash/8aa695b5d6bf9e958d4e56b22c9b859d-Abstract-Datasets_and_Benchmarks_Track.html",
    caveat="这里归类该文训练后的 Stream-VLM；同文的现成 VLM 零样本基线可归 Q4。", location="Sec. 4 训练方案；Appendix D", paper_id="panchal2024qevd")
add("ashutosh2025expertaf", "ExpertAF", 2025, "CVPR", "published", "Q3",
    "体育技能专家反馈及纠正演示", "corrective_feedback;corrected_pose;demonstration", "yes",
    "用视频/姿态及处理后的专家评论、学习者-专家配对训练反馈生成与纠正模块。",
    "https://openaccess.thecvf.com/content/CVPR2025/papers/Ashutosh_ExpertAF_Expert_Actionable_Feedback_from_Video_CVPR_2025_paper.pdf",
    publication="https://vision.cs.utexas.edu/projects/ExpertAF/", representative="primary")
add("ashutosh2025learninga", "CrossTrainer", 2025, "NeurIPS", "published", "Q3",
    "跨运动技能属性、熟练度与训练反馈", "corrective_feedback;proficiency", "yes",
    "发现共享技能属性后训练多模态语言模型，生成反馈和熟练度标签；跨运动测试仍使用训练后的模型。",
    "https://proceedings.nips.cc/paper_files/paper/2025/hash/eafca59fcbabb555fdb5373e2f34385e-Abstract-Conference.html",
    caveat="未见运动的迁移/zero-shot 不等同于没有任务专用训练。", representative="secondary")
add("li2026techcoach", "TechCoach", 2026, "AAAI 40(8):6699-6707", "published", "Q3",
    "技术点感知的动作训练反馈及质量分数", "corrective_feedback;score", "yes",
    "在技术点级评论监督下学习质量表征，统一输出整体评论和分数。",
    "https://ojs.aaai.org/index.php/AAAI/article/view/37601",
    caveat="首次 arXiv 在 2024 年，正式发表为 AAAI 2026，不能仅按首次预印本年份写成正式 2024 论文。", representative="primary")
add("", "ProfVLM", 2026, "Computer Vision and Image Understanding 268:104749", "published", "Q3",
    "多视角熟练度评估与专家式反馈", "corrective_feedback;proficiency", "yes",
    "冻结 TimeSformer 不代表冻结全模型；训练 AttentiveGatedProjector 并对语言模型进行 LoRA 反馈微调。",
    "https://doi.org/10.1016/j.cviu.2026.104749",
    title="ProfVLM: A lightweight video-language model for multi-view proficiency estimation", paper_id="bianchi2026profvlm")
add("qi2025explainable", "EFA / CoT-AFA", "2025; revised 2026", "arXiv:2512.15153v2", "preprint", "Q3",
    "健身/武术动作规范性、原因解释与解决办法", "explanation;corrective_feedback;quality_class", "yes",
    "训练双流 Explainable Fitness Assessor，融合步骤语义与视觉信息，联合规范性判断及解释生成。",
    "https://arxiv.org/abs/2512.15153",
    caveat="2025-12 首稿、2026-06 修订；截至检索日期未核实正式发表 venue，不沿用未经核实的期刊/会议标记。")
add("", "BioCoach", 2026, "CVPR", "published", "Q3",
    "流式健身的三维生物力学条件化训练指导", "corrective_feedback;feedback_timing", "yes",
    "选择动作相关自由度，构造个体化三维运动学/阶段上下文，并训练跨注意力反馈模块；骨干冻结但新增模块学习。",
    "https://openaccess.thecvf.com/content/CVPR2026/html/Ji_From_3D_Pose_to_Prose_Biomechanics-Grounded_Vision-Language_Coaching_CVPR_2026_paper.html",
    title="From 3D Pose to Prose: Biomechanics-Grounded Vision-Language Coaching",
    caveat="正式 CVPR 2026 论文，不能仅凭 arXiv 页面将其标为未发表；与结构化人体证据生成反馈的路线直接相关。",
    representative="secondary", paper_id="ji2026biocoach")
add("rai2026learning", "Consistent Temporal Grounding", 2026, "ECCV (author-confirmed acceptance)", "accepted_author_confirmed", "Q3",
    "运动反馈/验证任务的关键帧关注一致性", "corrective_feedback;technical_verification", "yes",
    "以生成和验证任务的视觉注意力一致性为额外训练目标，训练后在 ExpertAF 等运动指导任务评估。",
    "https://arxiv.org/html/2603.18453v1", publication="https://people.cs.pitt.edu/~kovashka/publications.htm",
    caveat="作者主页称 ECCV 2026 待刊，未获得正式 proceedings 页面；无需额外帧级标注不等于无需训练。")
add("", "BoT-Feedback (fine-tuned variant)", 2026, "arXiv:2610.06972v1", "preprint", "Q3",
    "生物力学证据引导的训练反馈", "corrective_feedback", "yes_optional_variant",
    "正文说明只监督最终反馈，并称补充材料包含微调实验；按这个可选配置归入 Q3，而非覆盖零样本主实验。",
    BOT_SRC, title=BOT_TITLE, variant="fine_tuned_variant",
    caveat="主结果显式为零样本；微调配置细节未逐项核验，正文训练措辞与主实验需分开解读。",
    confidence="medium", location="Step 4 Training Objective；Comparison with SOTA", paper_id="dong2026botfeedback")

# Q4: pretrained inference / deterministic evidence with specific diagnosis or feedback.
add("", "AgentCoach", 2026, "CHI", "published", "Q4",
    "参考示范的检查点评估与个性化反馈", "corrective_feedback;checkpoint_pass_fail", "no",
    "预训练 MediaPipe 和 Gemini 提取参考检查点、构造参数判定并生成历史相关反馈；使用提示示例，不做任务微调。",
    "https://jiakunyu.github.io/assets/pdf/AgentCoach.pdf",
    title="AgentCoach: LLM-Based Adaptive Coaching Feedback for Motor Skill Learning",
    publication="https://doi.org/10.1145/3772318.3791652",
    caveat="training-free 并非无需人工先验：仍有检查点整理、示例、验证与参考视频。", representative="primary", paper_id="agentcoach2026")
add("", "LLM-FMS", 2025, "PLOS ONE 20(3):e0313707", "published", "Q4",
    "Functional Movement Screen 姿态评分、解释与改进反馈", "score;explanation;corrective_feedback", "no",
    "姿态估计后的角度/位置指标与专家评分规则进入提示，LLM 输出评分及反馈；所述流程没有拟合质量回归器。",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11896072/",
    title="LLM-FMS: A fine-grained dataset for Functional Movement Screen action quality assessment",
    publication="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0313707",
    caveat="以方法描述判断其质量评估模块无需训练；依赖专家规则与评测标注。因为有具体解释/反馈，不只放 Q1。",
    representative="secondary", location="Methods 提示构造、评分与个性化反馈", paper_id="xing2025llmfms")
add("", "BoT-Feedback (zero-shot)", 2026, "arXiv:2610.06972v1", "preprint", "Q4",
    "对齐后的三维生物力学证据与训练反馈", "corrective_feedback", "no",
    "BDP 提取结构化人体描述、RL-DTW 对齐，分阶段识别动作/关注部位/比较证据/生成反馈；主结果明确为 zero-shot 配置。",
    BOT_SRC, title=BOT_TITLE, variant="zero_shot_main",
    caveat="2026-10-03 首稿，未核实正式发表；同文有可选微调设置，不能给整篇无条件 training-free 标签。RL-DTW 是 Robust Local DTW，不是强化学习。",
    representative="preprint_watch", location="Methodology；Comparison with SOTA 的 zero-shot 声明", paper_id="dong2026botfeedback")
add("zheng2026fitaqa", "FitAQA (off-the-shelf judgement)", 2026, "arXiv:2608.08736v1", "preprint", "Q4",
    "健身错误判断、感知和错误区间定位评测", "error_diagnosis;temporal_grounding", "no_benchmark_adaptation",
    "主评测直接调用现成模型，检查感知、是否存在指定形式错误及时间区间；不是学习新的反馈生成器。",
    "https://arxiv.org/html/2608.08736v1",
    caveat="按无 FitAQA 专项训练的通用 MLLM 判断配置归类；包含另外预训练的专用时序模型。属于诊断 benchmark，不是自由文本训练指导方法。",
    location="Experiments / Evaluated Models / Evaluation Protocol")
add("", "Off-the-shelf VLM error diagnosis", 2026, "CVPR Workshops / SAUAFG", "published_workshop", "Q4",
    "健身技术点验证与具体姿态错误识别", "error_diagnosis;technical_verification", "no",
    "同篇直接提示现成 VLM 判断健身技术要点合规与具体错误标签，未训练下游错误识别器。",
    VLM_SRC, title=VLM_TITLE, publication=VLM_PUB, variant="error_diagnosis",
    caveat="这是诊断/验证配置，非完整自然语言纠正反馈；评分/成对比较配置另列 Q1。", location="Sec. 2.1.1-2.1.3", paper_id="freitas2026vlm")


def write_csv(path, records, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


out = args.output
out.mkdir(parents=True, exist_ok=True)
source_dir = out / "sources"
source_dir.mkdir(exist_ok=True)
shutil.copyfile(args.bib, source_dir / "awesome_aqa_20261013.bib")
write_csv(out / "classified_papers.csv", papers, list(papers[0]))

inventory = []
for entry in database.entries:
    year = entry.get("year", "")
    if not year.isdigit() or not 2022 <= int(year) <= 2026:
        continue
    linked = [p for p in papers if p["awesome_bib_id"] == entry["ID"]]
    note = "原始元数据，仅用于候选发现；未完成原文分类核验。"
    if entry["ID"] in {"xu2025humancentric", "zhang2025poseguided"}:
        note += " 两个不同题目关联同一 IEEE 文档号 10946879，存在元数据冲突，不能直接视为两篇独立论文。"
    inventory.append({
        "bib_id": entry["ID"], "title_as_listed": entry.get("title", ""),
        "year_as_listed": year,
        "venue_as_listed": entry.get("booktitle", entry.get("journal", "")),
        "authors_as_listed": entry.get("author", ""),
        "doi_as_listed": entry.get("doi", ""),
        "paper_url_as_listed": entry.get("pdf", entry.get("url", "")),
        "review_status": "primary_checked_or_conditional" if linked else "discovery_only",
        "classified_quadrants": ";".join(sorted({p["quadrant"] for p in linked})),
        "classified_paper_ids": ";".join(sorted({p["paper_id"] for p in linked})),
        "note": "请以 classified_papers.csv 的更正、状态及限制为准。" if linked else note,
    })
write_csv(out / "candidate_inventory.csv", inventory, list(inventory[0]))

counts = Counter(p["quadrant"] for p in papers)
stats = {
    "checked_on": DATE, "scope_years": [2022, 2026],
    "awesome_entries_total": len(database.entries),
    "awesome_candidates_2022_2026": len(inventory),
    "awesome_candidates_by_listed_year": dict(sorted(Counter(r["year_as_listed"] for r in inventory).items())),
    "classified_configurations": len(papers),
    "distinct_classified_papers": len({p["paper_id"] for p in papers}),
    "classified_configurations_by_quadrant": dict(sorted(counts.items())),
    "conditional_or_medium_confidence_configurations": sum(p["confidence"] != "high" for p in papers),
    "awesome_snapshot_sha256": hashlib.sha256(args.bib.read_bytes()).hexdigest(),
    "not_exhaustive": True,
}
(out / "provenance.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

INTRO = """# 近五年 AQA 四宫格文献调研

检索截止：2026-10-13。年份范围：2022-2026（2026 年仅纳入检索日前可取得的材料）。

## 核心结论

四宫格适合组会概览，但分类对象应是具体方法/实验配置，而不是只凭论文标题。右上角不是空白：正式发表的 QAQA / Anomaly-Aware DTW 可作为几何匹配评分的代表，CVPR Workshops 2026 的预训练 VLM 评测可作为提示式直接评分/排名的补充。

推荐横轴写成 **Task-Specific Training / No Task-Specific Training**。右侧不是“模型从未训练”，而是无需为 AQA/反馈任务进一步训练质量预测器、诊断模块或反馈生成模块。通用预训练姿态网络和 LLM 可以使用；任务专用检测器、训练的回归头、LoRA、适配器、专用自监督训练都计入左侧。参考动作、手写规则、提示示例及人工设计仍需单独说明。

推荐纵轴上排写 **Scoring / Ranking Only**，下排写 **Diagnosis / Coaching Feedback (Optional Scores)**。下排包括叙述性评价、具体错误诊断和纠正指导，但三者不能混为同一种输出。仅展示注意力、人体掩码或分数分解，不自动等于有诊断或训练反馈。

## 建议放进图中的代表作

| 输出类型 | 有任务专用训练：左侧 | 无任务专用训练：右侧 |
| --- | --- | --- |
| 评分 / 排名：上排 | **Q2：TPT** (ECCV 2022)、**FineParser** (CVPR 2024)；补充 TSA/FineDiving、GOAT/LOGO | **Q1：QAQA / Anomaly-Aware DTW** (Sensors 2025)；补充 **Can VLMs Judge Action Quality?** 的评分/排名配置 (CVPR Workshops 2026) |
| 诊断 / 训练反馈：下排，可兼具评分 | **Q3：ExpertAF** (CVPR 2025)、**TechCoach** (AAAI 2026)；补充 NAE、CrossTrainer、BioCoach | **Q4：AgentCoach** (CHI 2026)、**LLM-FMS** (PLOS ONE 2025)；虚线/星号标出 **BoT-Feedback** 的零样本配置 (arXiv 2026) |

推荐优先正式论文不是按 citation 数排序；选择依据是分类明确、机制有代表性、与当前研究问题直接相关。Workshop 已正式发表，但应保留 Workshop 标记，不与主会论文混写。预印本不能用正式会议缩写替代状态。

选择这些代表的原因：Q1 用 QAQA 和现成 VLM 评分覆盖解析几何匹配、基础模型提示推理两种机制；Q2 用 TPT/FineParser 覆盖时序解析与人体相关时空解析；Q3 用 ExpertAF/TechCoach 覆盖可执行纠正、技术点感知反馈；Q4 用 AgentCoach/LLM-FMS 覆盖参考检查点系统、结构化姿态与规则提示。具体证据与发表链接见各象限表。BioCoach 值得加入 Q3 的扩展代表，BoT-Feedback 则作为注明预印本状态的 Q4 近邻。

## 检索范围与证据强度

以 [Awesome-AQA](https://github.com/ZhouKanglei/Awesome-AQA) 的 BibTeX 为候选入口，补充骨架/DTW、VLM 评分、运动诊断、LLM coaching 和 3D 生物力学反馈检索。下表的分类依据来自原论文、官方 proceedings/出版社页面或作者项目；不是直接照搬列表中的标签。

这是一轮面向四宫格与相关工作定位的定向调研，不是穷尽检索或系统综述。候选清单不表示全部已精读。高置信条目核对了任务、训练和输出；中等/条件条目明确保留未核实之处。

状态：published=正式论文；published_workshop=正式 workshop；accepted_author_confirmed=作者确认录用但未取得正式 proceedings；preprint=仅核实预印本；submission_decision_unverified=投稿材料可见但最终决定未核实。

"""
sections = [INTRO, f"本轮保存 {len(inventory)} 条 2022-2026 年列表候选，重点分类 {stats['distinct_classified_papers']} 篇文献的 {len(papers)} 个方法/配置。Q1/Q2/Q3/Q4 分别有 {counts['Q1']}/{counts['Q2']}/{counts['Q3']}/{counts['Q4']} 条配置；其中含条件归类和预印本，不应视为领域总体占比。\n"]
headings = {
    "Q1": "第一象限：右上，训练自由的评分/排名",
    "Q2": "第二象限：左上，训练式评分/排名",
    "Q3": "第三象限：左下，训练式评价/诊断/反馈",
    "Q4": "第四象限：右下，训练自由的评价/诊断/反馈",
}
for quadrant, heading in headings.items():
    sections += [f"\n## {heading}\n", "| 方法 / 配置 | 年份、发表状态 | 任务与输出 | 分类依据 | 限制 / 注意事项 |\n| --- | --- | --- | --- | --- |\n"]
    for p in papers:
        if p["quadrant"] != quadrant:
            continue
        cells = [
            f"[{p['short_name']}]({p['primary_url']})",
            f"{p['year']}；[{p['venue']}]({p['publication_url']})；`{p['publication_status']}`",
            f"{p['task']}；`{p['output_type']}`", p["classification_evidence"], p["scope_caveat"] or "-",
        ]
        sections.append("| " + " | ".join(cell.replace("|", "\\|").replace("\n", " ") for cell in cells) + " |\n")
sections.append("""
## 右上角如何讲

1. **几何匹配路线**：预训练姿态模型提取骨架，计算角度/骨段/动态特征，与参考轨迹进行 DTW，距离经解析公式转换为质量分数。QAQA 是本轮找到的分类明确、已正式发表的代表。其在传统健身/Qigong 上的应用也比只列跳水模型更贴近太极。
2. **基础模型直接判断路线**：把动作视频及评分要求交给冻结的 VLM，直接输出数值或成对比较。CVPR Workshops 2026 的实证研究正式评测了这类配置，但结果不支持直接声称其已解决细粒度 AQA。
3. **两者的边界**：相似度分数不必然等于教练质量标准，VLM 给出一个数也不保证排序可靠。这里是在分类现有路线，不是在认可其有效性或宣布右上角性能达到左侧水平。

MED-ACDTW 可作为投稿补充，但状态未知；Hou 等的太极论文可作为质量阶段的条件归类，不能拿来无条件证明整套系统没有训练。

## 当前图需要改正的地方

- FineDiving、LOGO 首先是数据集/论文名；图中方法标签最好写成 **TSA (FineDiving)** 和 **GOAT (LOGO)**。已有 TPT 的位置正确。
- NAE、ExpertAF、TechCoach 放左下总体正确；NAE 侧重评价性叙述，ExpertAF/TechCoach 更直接针对反馈，不能省略这个区别。TechCoach 的正式年份是 2026，不是仅按 arXiv 首稿标 2024。
- 原图的 “A case study in squats and Tai Chi pushes” 对应 [Hülsmann 等，Computers & Graphics 2018](https://www.sciencedirect.com/science/article/pii/S0097849318301304)，不在此次五年范围内；其学习错误分类器，可放历史背景的左下，不应用作近五年代表。AIFit (CVPR 2021) 同样可作历史背景，但不纳入 2022-2026 主表。
- CrossTrainer/MoAKE 的未见运动测试不等于 training-free。前者训练反馈模型，后者训练统一评分模型，分别在左下/左上。
- Frozen backbone、parameter-efficient training、自监督、无帧级标注、reference-free 都不等于无需任务专用训练。ProfVLM/BioCoach 是冻结骨干但训练新增模块的实例。
- 同一论文可有不同配置：VLM 实证评测的评分配置在右上、具体错误诊断配置在右下；跳水 zero-shot 论文的原始 VLM 在右上，但集成回归方案在左上；BoT-Feedback 零样本主实验在右下、可选微调在左下。图中注明配置，避免重复计数。
- 把右上留白可能误导听众认为没有相关工作。建议只列两种正式路线的代表，避免为了视觉对称把分类存疑的论文硬塞进去。

## 对本研究定位的影响

**四宫格给出研究路线，不构成 novelty 证明。** 当前系统可在右下标出主要反馈路线，并说明还提供右上类型的几何相对排名。这并不意味着一篇同时输出排名和反馈的论文需要被整体重复放置；可以用小标记表达双输出。

最应补入直接相关工作的论文有：

- [AgentCoach](https://jiakunyu.github.io/assets/pdf/AgentCoach.pdf)：正式 training-free 检查点评估与反馈的近邻。
- [LLM-FMS](https://pmc.ncbi.nlm.nih.gov/articles/PMC11896072/)：正式的姿态数值、专家规则和 LLM 解释/反馈路线，不能再将这种组合本身当作首次提出。
- [BioCoach](https://openaccess.thecvf.com/content/CVPR2026/html/Ji_From_3D_Pose_to_Prose_Biomechanics-Grounded_Vision-Language_Coaching_CVPR_2026_paper.html)：已正式发表的三维运动学/阶段信息引导反馈工作，区别在于任务专用训练及具体应用设计，不是“只有我们给 LLM 身体几何”。
- [BoT-Feedback](https://arxiv.org/abs/2610.06972)：2026-10-03 的预印本；有结构化三维人体描述、参考-学生时序对齐和零样本反馈，与当前动机非常接近，应正面比较。正文也提到微调设置，需按配置解读。
- [Consistent Temporal Grounding](https://arxiv.org/abs/2603.18453)：关注反馈与关键帧的一致性，但通过额外训练目标改进注意力。它说明“让反馈对应具体时刻”已是一个研究问题，不宜笼统说前人没有做。

更稳妥的差异陈述是：**针对多招式长时太极演练，在无需太极专用反馈训练的条件下，以动作要领约束的参考关键事件为中间语义单位，建立跨演练的对应证据，并支持可追溯诊断与相对排名。** 这是待进一步实验证明的组合贡献，不是已验证的唯一/首次方法。

需要避免“长视频 AQA 很少，所以长视频本身就是新问题”的表述：LOGO、Skating-Mixer、CoFInAl、LAV 已处理长时评分；QEVD 也提供长时训练视频及反馈。可以强调 **连续多招式、招式间语义变化、动作要领对应事件及任务专用反馈标注缺乏**，而非只强调秒数。

## 建议组会使用的表述

“我们按两条相互独立的轴梳理已有方法：是否需要任务专用训练，以及输出是只有分数/排名，还是能够给出具体诊断/训练反馈。左上是成熟的学习式评分；左下用专门数据训练评价或教练反馈；右上已有几何匹配和预训练 VLM 直接评分，但细粒度可靠性仍需验证；右下已有 training-free coaching，因此我们的贡献不能仅概括为不训练或使用 LLM，而需要落实到太极连续多招式中的语义事件对应与证据组织。”

## 附件与使用

- `classified_papers.csv`：重点文献的配置级分类、状态、分类依据、原文位置、证据链接和置信度。UTF-8 BOM，方便 Excel 打开。
- `candidate_inventory.csv`：列表中 125 条候选的原始元数据及核验标记。空分类表示尚未原文核验，不代表论文不属于任何象限。
- `search_log.md`：检索策略、证据限制、元数据冲突及快照来源。
- `provenance.json`：日期、候选/分类计数、源文件哈希。
- `sources/awesome_aqa_20261013.bib`：检索入口快照；保留原样，不把其中所有 venue/tags 当作已证实结论。
- `build_review.py`：报告和 CSV 的构建脚本；依赖 `bibtexparser==1.4.3`。对整理结果重新生成不会调用付费 API。

后续改图只需要挑选本报告推荐的少量标签；本轮未修改已有四宫格图片或任何实验结果。
""")
(out / "review_zh.md").write_text("".join(sections), encoding="utf-8")
shutil.copyfile(Path(__file__), out / "build_review.py")
print(json.dumps(stats, ensure_ascii=False, indent=2))
