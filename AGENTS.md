# Repository Guidelines

## 项目范围与阅读入口

本仓库用于太极拳 3D 动作质量评估，包含片段姿态对齐与评分、完整 24 式视频分割、结束定势指标与规则反馈、流畅性和骨盆轨迹诊断、排名融合，以及人工复核和时序查询标注工具。大多数流程消费已生成的 PHALP/4D-Humans tracking，不负责从视频运行完整人体重建。

- 先阅读 `README.md` 中对应流程，再核对入口脚本的 `build_parser()`、`main()` 或 `--help`；历史文档中的路径和实验数值不保证适用于当前工作区。
- `LateFusion_Technical_Document.md` 解释姿态、停顿、骨盆指标及 Late Fusion；`TechPoint.md`、`TechPoint_修订版.md`、`move.md` 保存动作要领与招式说明。修改规则时应确认入口实际使用哪一版。
- `temporal_query_annotator/USAGE.zh-CN.md` 说明标注工具；`prompt/` 保存实验提示词与历史计划。`task.md` 是早期 geodesic 模块设计，不代表当前所有入口的输出行为。

## 代码地图

可复用算法主要放在 `aqa3d/`；根目录脚本负责参数、数据集路径、实验编排和导出。现有脚本之间也有导入关系，修改公共函数前需搜索调用方和测试。

| 功能 | 核心模块 | 主要入口 |
| --- | --- | --- |
| 原始 YOLO 2D 对齐、学生关键帧与 3D 距离 | `alignment.py`、`geodesic.py`、`tracking.py`、`pipeline.py` | `run_3d_aqa.py`、`run_batch.py` |
| 固定教师锚点排名 | `alignment.py`、`smpl_dtw.py` | `run_teacher_keyframe_batch.py`、`run_teacher_keyframe_smpl_dtw_batch.py`、`run_teacher_1fps_smpl_dtw_batch.py` |
| 完整视频 TAS/SMPL-DTW 分割 | `smpl_dtw.py`、`smpl_pose_segmentation.py` | `run_tas_smpl_dtw_mapping.py`、`run_student_tas_smpl_dtw.py`、`run_smpl_pose_dtw_segmentation.py`、`run_smpl_pose_dtw_teach_batch.py` |
| 稀疏 KeyPose 与结束定势迁移 | `keypose_alignment.py`、`reference_manifest.py`、`endpoint_transfer.py` | `run_keypose_reference_alignment.py`、`build_endpoint_keypose_reference.py`、`run_student_endpoint_keyposes.py` |
| 关节几何、定势指标与反馈 | `angle_metrics.py`、`endpoint_metrics.py`、`endpoint_feedback.py` | `export_teacher_keyframe_angle_metrics.py`、`run_endpoint_metrics_experiment.py`、`run_endpoint_feedback.py`、`build_full_technique_feedback_rules.py` |
| 流畅性、速度和骨盆轨迹 | `motion_quality.py`、`velocity_quality.py`、`pelvis_quality.py` | `run_motion_quality_analysis.py`、`run_velocity_quality_analysis.py`、`run_pelvis_quality_analysis.py` |
| 教师标定分数与排名融合 | `pose_score.py`、`center_fusion.py`、`rank_fusion.py` | `run_pose_score_experiment.py`、`run_uncapped_pause_penalty_experiment.py`、`run_center_fusion_experiment.py`、`run_late_fusion_experiment.py` |
| 规则反馈/F-group 人工复核 | `feedback_review.py`、`f_group_review.py` | `run_feedback_review_app.py`、`run_f_group_experiment.py`、`run_f_group_api.py`、`run_f_group_review_app.py` |

表中模块路径相对于 `aqa3d/`。其他目录与工具：

- `temporal_query_annotator/` 是独立的本地标注工具：`domain.py` 做 DSL/标注校验，`project.py` 加载任务并持久化，`llm.py` 生成候选，`server.py` 提供 HTTP 服务，`static/` 保存原生 HTML/CSS/JS。
- `f_group_review_static/` 是 F-group 复核界面的静态资源；规则反馈界面的页面代码位于 `run_feedback_review_app.py`。
- `video_utils/`、`trim_full_teach_videos.py`、`split_video_by_boundaries.py` 负责抽帧与视频准备；`export_tas_*.py`、`visualize_*.py`、`plot_*.py` 负责标注导出、边界检查与实验图表。
- `export_keypose_smpl.py` 是依赖外部 HMR2/4D-Humans 的图像推理入口；`inspect_tracking_pkl.py`、`diagnose_smpl_dtw_alignment.py` 用于诊断。
- `tests/` 是主测试集；`temporal_query_annotator/tests/` 需单独发现。`report_assets/` 包含报告插图生成脚本。
- `*_results/`、`f_group_experiment/` 中已有部分报告和模型原始回答，是实验记录；不要把它们当成自动可重建的源码或随手覆盖。

## 环境与数据前提

项目直接从仓库根目录运行 Python 脚本，没有 `pyproject.toml`/`setup.py` 或前端构建步骤。文档使用 `4d-humans` Conda 环境；先确认当前解释器和环境是否存在，不要假定系统自带 `python` 或指定 GPU。

```bash
conda activate 4d-humans
python -m pip install -r requirements.txt
```

`requirements.txt` 仅列出 `numpy`、`scipy`、`opencv-python`、`joblib`、`pandas`、`ultralytics`，不是各流程完整的环境清单：

- `aqa3d/__init__.py` 会导入 `angle_metrics.py`，后者直接依赖 PyTorch，因此许多数值测试也需要 `torch`。
- SMPL-24 重建需要 `smplx`、PyTorch 和 SMPL 模型，默认模型目录是 `~/.cache/4DHumans/data/smpl`。图表、图像及 F-group API 相关代码还使用 `matplotlib`、Pillow、`requests`。
- 视频裁剪/切分需要系统可执行程序 `ffmpeg` 和 `ffprobe`。KeyPose 图像推理还需要外部 4D-Humans/HMR2 代码和权重。
- 多个入口默认读取 `/home/sqw/VisualSearch/aqa/` 下的视频、tracking、人工边界或既有实验输出；YOLO 默认权重在相邻 `aqa/model_weights/yolo11m-pose.pt`。这些是本机历史配置，不能认为新检出就具备。优先用已有 CLI 参数传入实际路径。
- 当前检出不含 `validate_dtw/validate_implementations.py`；README 中的新旧 DTW 对比命令需要另备外部文件，不能作为仓库内可直接运行的检查。
- 当前检出也不含标注工具文档提到的默认 `projects/bv1we411w7jb_first3.json` 和 DSL schema JSON。启动时需通过 `--project` 提供真实配置及视频、JPG 帧目录和规则文件；可参照 `domain.py`、`project.py` 与合成测试了解结构。不要仅根据文档假定这些文件存在。

## 常用命令

以下命令从仓库根目录执行。尖括号表示必须替换的输入；批处理入口还依赖各自的默认数据或显式路径参数。运行实验前核对输入和输出目录。

```bash
# 单视频对：YOLO 对齐后导出 SMPL 距离
python run_3d_aqa.py --student-video <student.mp4> --teacher-video <teacher.mp4> \
  --student-tracking <student.pkl> --teacher-tracking <teacher.pkl> \
  --yolo-weights <yolo11m-pose.pt> --output-dir results/<case> --device auto

# 固定教师锚点；第一项使用 YOLO 2D DTW，后两项使用 SMPL DTW
python run_teacher_keyframe_batch.py --output-root teacher_keyframe_results
python run_teacher_keyframe_smpl_dtw_batch.py --output-root teacher_keyframe_smpl_dtw_results
python run_teacher_1fps_smpl_dtw_batch.py --output-root teacher_1fps_smpl_dtw_results

# 完整视频分割，显式指定参考及人工边界
python run_smpl_pose_dtw_segmentation.py --input-video <input.mp4> \
  --input-tracking <input.pkl> --reference-video <reference.mp4> \
  --reference-tracking <reference.pkl> --gold-boundaries <boundaries.txt> \
  --gold-url-id <reference_5FPS_id> --output-root <new-output-dir>

# 以已确认 manifest 迁移学生结束定势
python run_student_endpoint_keyposes.py --reference-manifest <reference_manifest.json> \
  --input-video <student.mp4> --input-tracking <student.pkl> --output-root <new-output-dir>

# 在已有指标上生成规则反馈
python run_endpoint_feedback.py --input-root <endpoint-metric-dir> --output-root <new-feedback-dir>

# 本地人工复核服务，分别默认使用 8501、8502、8765 端口
python run_feedback_review_app.py
python run_f_group_review_app.py
python -m temporal_query_annotator.server --project <project.json>
```

融合实验的输入链为姿态分/已复核停顿与骨盆诊断，再到 center fusion 和 late fusion。参照 `README.md` 与技术文档确认中间 CSV 和路径；不要把 `run_late_fusion_experiment.py` 当作从原视频开始的一键入口。

## 必须保留的数据与算法约定

- **姿态距离与几何指标不同**：geodesic scoring 比较 23 个局部 `body_pose` 旋转，排除 `global_orient`；旋转矩阵通常为 `(T, 23, 3, 3)`，误差图为 `(T, 23)`。SO(3) 距离在 `acos` 前将余弦截断至 `[-1, 1]`，明确弧度/角度单位。不要替换成 axis-angle 的欧氏距离。
- **原生 SMPL-24**：关节几何使用 `SMPL_24_JOINTS` 及 SMPLLayer 重建的原生 24 关节。PHALP 保存的 `3d_joints` 有不同顺序，不能直接套用原生 SMPL 索引。
- **索引与时间**：内部原视频帧和采样索引通常为 0-based，PHALP 帧号为 1-based（`source_index + 1`）。部分 TAS CSV 导出 1-based 闭区间；按实际列名和导出函数核对。5 FPS 网格时间和实际原视频帧时间需要分别保存，不能混用。
- **分段与边界**：结束边界必须严格递增；离散区间为 `[previous_boundary + 1, current_boundary]`，第一段从内部索引 0 开始。全局 SMPL-DTW 路径采用 `(target_index, reference_index)`；一个参考边界的多个路径候选按 local geodesic cost 选帧。重复或逆序边界应保留诊断并报错，不静默修正。
- **Tracking**：默认选择有效记录最多的 track；stitched loader 在当前 ID 消失后按姿态连续性衔接。保持 `source_track_ids` 等溯源信息。缺失精确帧默认失败，近邻替代须显式启用并限定距离；结束定势迁移还需记录请求帧、实际使用帧及复核标记。
- **流程差异**：原始 `run_batch.py` 以学生关键帧为锚点，教师关键帧批处理使用固定教师锚点；运动峰值和均匀 1 FPS 不是同一策略。原始 `qishi` 片段流程截取学生末 `15 * int(fps)` 帧，不能把此规则自动推广到完整视频分割。
- **参考与反馈**：结束事件和动作要领来自 reference manifest，保留其快照、哈希与事件 ID。窗口指标、中心帧值及教师区间各有意义，不重新挑选“最佳帧”来消除偏差。超出教师区间或没有复核标记均不能直接判定动作合格与否。
- **评估边界**：骨盆/root 信号是身体重心的代理，不是真实人体质心；SMPL 的手腕/头部位置不能证明掌心朝向、精确视线或承重。保留规则的可观测性和不支持事项。
- **分数与排名**：基础管线导出距离；`pose_score.py` 才实现教师标定分数。Late Fusion 输出当前学生集合内的相对排名。保留指标方向、并列排名规则及 leave-one-move-out 划分，不把训练集调参结果描述成独立验证性能。

## 开发与测试

使用 4 空格缩进、`snake_case`、公共函数类型注解和 `pathlib.Path`。数值逻辑优先使用 NumPy，检查输入形状、有限值和索引范围；成本矩阵关注分块计算和内存占用。沿用现有 dataclass、显式参数与结果导出结构，注释说明不明显的数学或帧号转换。

主测试使用标准库 `unittest`，没有必要引入其他测试框架：

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s temporal_query_annotator/tests -v

# 聚焦某个模块；tests/ 没有 __init__.py，优先使用 discover
python -m unittest discover -s tests -p 'test_smpl_dtw.py' -v
```

算法变更应使用小型合成旋转、轨迹、临时文件和 mock 覆盖关键行为，避免依赖私有视频、真实 GPU 推理或付费 API。数值断言同时检查形状与容差，使用 `numpy.testing.assert_allclose`。涉及帧号、边界、缺帧替代、track 切换、复核持久化或排名划分时，运行相应回归测试；跨模块改动再运行完整主测试集。修改标注工具时单独运行其测试集，根目录 `tests/` 的 discover 不包含它。

纯文档改动检查路径、命令参数和 `git diff --check` 即可。缺依赖或数据时明确记录未能运行的检查及原因，不把导入失败算成通过，也不为文档更新安装整套重建环境或启动批量实验。

当前已知测试问题：`temporal_query_annotator/tests/test_domain.py` 的 `test_prompt_contains_task_and_metric_catalog` 断言 `build_user_prompt()` 返回值包含“不直接输出绝对时间戳”，当前实现未包含该文字；本次文档更新时该独立测试集为 6 通过、1 失败。后续修改提示词或测试契约时需核对，不能以此推定整套测试已通过。

## 实验文件、API 与交付

- 保留输入身份、参数、哈希、tracking 来源、参考快照和人工复核状态。需要新实验时使用新输出目录；不要移除既有的防覆盖、原子保存、revision 检查或文件锁。F-group 复核服务运行时不能删除其 lock 文件。
- F-group 流程区分冻结输入、首次模型原始回答、自动校验和人工判断；格式错误的回答也属于实验结果，不能静默重生成。`run_f_group_api.py --dry-run` 不发请求，但会创建输出目录和冻结实验文件。
- F-group API 使用 `DASHSCOPE_BASE_URL`/`DASHSCOPE_API_KEY`；标注工具使用 `TQA_LLM_API_URL`、`TQA_LLM_API_KEY`、`TQA_LLM_MODEL`，也支持手动导入候选。保持密钥在服务端环境变量中，不写入报告、浏览器资源或日志。常规验证应 mock API。
- `.gitignore` 全局忽略 `*.json`、`*.csv`、`*.npz`、tracking、权重和多数图片；新配置/fixture 可能因此不出现在 `git status`。需要随源码提交的文件应检查 `git check-ignore -v <path>` 并有针对性地处理，避免取消整个数据忽略规则。
- 不提交私有视频、模型权重、完整 tracking 或批量实验产物。已有受版本控制的报告和回答不代表其他大文件也应加入。
- 提交信息简短描述实际变更。PR/交付说明交代行为变化、验证结果、所需外部数据/模型以及未验证项；可视化变更按需附代表性输出。完成前检查 `git diff --check` 和变更范围。
