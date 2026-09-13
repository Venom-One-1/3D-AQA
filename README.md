# 3D-AQA

本项目保留原 AQA 项目的 YOLO Pose + DTW 时序对齐和学生关键帧提取，改用
4D-Humans/PHALP 输出的 SMPL `body_pose` 计算动作相似度。评分只使用 23 个
局部关节旋转，明确排除 `global_orient`。

对每个 DTW 匹配的关键帧对，计算：

`d = acos(clamp((trace(R_student @ R_teacher.T) - 1) / 2, -1, 1))`

第一版使用均匀帧权重与均匀关节权重，输出动作的平均距离（弧度和角度），暂不
将距离映射成百分制分数。

## 环境

在 `4d-humans` 环境中安装项目依赖：

```bash
python -m pip install -r requirements.txt
```

## 单视频对

```bash
cd /home/sqw/Projects/3D-AQA
python run_3d_aqa.py \
  --student-video /home/sqw/VisualSearch/aqa/ActionSegments/student/1_1_qishi.mp4 \
  --teacher-video /home/sqw/VisualSearch/aqa/ActionSegments/teach/QxVvRcRn2TA_1_qishi.mp4 \
  --student-tracking /home/sqw/VisualSearch/aqa/Tracking/student/1_1_qishi/results/demo_1_1_qishi.pkl \
  --teacher-tracking /home/sqw/VisualSearch/aqa/Tracking/teach/QxVvRcRn2TA_1_qishi/results/demo_QxVvRcRn2TA_1_qishi.pkl \
  --output-dir results/1_1_qishi \
  --device cuda:2 --yolo-batch-size 8
```

第一式 `qishi` 会按旧项目规则只取学生视频的末 `15 * int(fps)` 帧。PHALP
帧号从 1 开始，AQA/OpenCV 帧号从 0 开始；程序会显式完成这一转换并记录在结果中。

## 批量运行

默认扫描全部 15 个学生动作片段以及 3 个教师片段：

```bash
cd /home/sqw/Projects/3D-AQA
python run_batch.py --output-root results --device cuda:2 --yolo-batch-size 8
```

也可以只运行一个片段：

```bash
python run_batch.py --student 1_1_qishi --output-root results --device cuda:2
```

每个学生片段会保存到独立目录：

- `summary.json`：平均 geodesic distance、数据来源及轨迹 ID；
- `matched_keyframes.csv`：关键帧和 DTW 匹配帧的 0/1 基索引及每帧平均误差；
- `geodesic_errors.npz`：形状为 `(关键帧数, 23)` 的逐帧逐关节误差图（弧度和角度）。

## 完整教学视频的 TAS 边界映射

下面的命令以 `QxVvRcRn2TA` 为参考，将裁剪后的完整教学视频统一采样到 5 FPS，
使用 23 个 SMPL 局部关节旋转的平均 geodesic distance 运行全局 DTW，并迁移
24 式的结束边界：

```bash
cd /home/sqw/Projects/3D-AQA
conda run -n 4d-humans python run_tas_smpl_dtw_mapping.py
```

当一个参考边界帧在 DTW 路径上对应多个目标帧时，选择 local geodesic distance
最小的候选帧。程序要求 24 个映射终点严格递增，不会静默修正重复或逆序边界。
PHALP 在长视频中发生连续 track ID 切换时，会保持当前 ID 直到其消失，再按相邻
SMPL 姿态连续性连接后继 ID，并在 `summary.json` 中记录实际使用的 ID。

默认结果保存在 `tas_smpl_dtw_results/`：

- `all_mapped_segments_5fps.csv`：所有目标视频的 1-based、闭区间 5 FPS 分段；
- `<video_id>/segments_5fps.csv`：单个视频的 24 式映射结果；
- `<video_id>/boundary_mapping.csv`：边界候选数、源帧号和选中的 local distance；
- `<video_id>/dtw_path.csv` 与 `dtw_path.npz`：完整 DTW 路径；
- `<video_id>/dtw_diagnostics.png`：cost matrix、完整路径及 24 个映射边界；
- `mapping_summary.json`：样本数、DTW 距离、单调性、末尾覆盖和运行时间摘要。

## 完整学生视频的 TAS 边界映射

下面的命令使用 `QxVvRcRn2TA` 的 Ground Truth 边界，通过 5 FPS SMPL
Geodesic DTW 分割所有已经完成 tracking 的学生视频：

```bash
cd /home/sqw/Projects/3D-AQA
conda run -n 4d-humans python run_student_tas_smpl_dtw.py
```

只处理指定学生时，可以重复传入 `--student-video-id`：

```bash
conda run -n 4d-humans python run_student_tas_smpl_dtw.py \
  --student-video-id 00 --student-video-id 01
```

默认结果保存在 `student_segmentation_results/<video_id>/`。`segments.csv` 同时
记录原视频的 1-based 闭区间 `start_frame/end_frame`、时间范围以及 5 FPS 序列的
`start_frame_5fps/end_frame_5fps`。完整 DTW 路径、边界映射、诊断图和运行摘要也会
保存在同一目录。参考视频、参考 tracking 和边界 CSV 均可通过命令行参数切换。

可视化已经完成分割的学生视频：

```bash
conda run -n 4d-humans python visualize_student_tas_boundary_frames.py
```

程序为每个学生生成一张包含全部 24 式的两列对比图：左侧是参考视频的 Ground
Truth 结束边界帧，右侧是学生视频的 DTW 预测结束边界帧。图片保存在
`student_segmentation_results/<video_id>/boundary_frames.jpg`。使用
`--student-video-id 00` 可以只生成指定学生的图片。

## 通用 24 式 SMPL Pose DTW 分割

`run_smpl_pose_dtw_segmentation.py` 默认以完整教学视频 `BV1WE411W7JB` 为参考，
读取 `BV1WE411W7JB_5FPS_Boundary_ft.txt` 中人工标注的 24 个 5 FPS 金标准结束边界。
输入视频需预先裁掉动作前后的背景，并已单独完成 PHALP tracking：

```bash
cd /home/sqw/Projects/3D-AQA
conda run -n 4d-humans python run_smpl_pose_dtw_segmentation.py \
  --input-video /path/to/input.mp4 \
  --input-tracking /path/to/demo_input.pkl
```

脚本从 MP4 读取真实 FPS 和帧数，将参考与输入序列统一采样到 5 FPS，使用 23 个
SMPL 局部 `body_pose` 关节的平均 Geodesic Distance 运行完整序列全局 DTW。
一个参考边界对应多个路径候选时，选择 local distance 最小的输入帧，并要求全部
24 个结果严格递增。默认输出到
`smpl_pose_dtw_segmentation_results/<video_id>/`：

- `boundaries.csv`：精确边界时间、5 FPS 索引、原视频帧号及 PHALP 帧号；
- `segments.csv`：由相邻边界构造的 24 个互不重叠区间；
- `dtw_path.csv` 和 `dtw_path.npz`：完整 DTW 路径及逐点 local distance；
- `dtw_diagnostics.png`：cost matrix、路径及 24 个边界；
- `segmentation.json` 和 `summary.json`：结构化结果、输入元数据与质量诊断。

添加 `--save-local-costs` 可额外保存完整 local cost matrix。参考视频、参考 tracking
和金标准文件也保留了可选参数，便于后续替换参考教学视频。

将输入视频自身的人工 Ground Truth 与 SMPL Pose DTW 预测结束边界绘制为一张
24 行双列图：

```bash
conda run -n 4d-humans python visualize_smpl_pose_dtw_boundaries.py
```

第一列是人工结束边界画面，第二列是预测结束边界画面；每行同时标注两个时间、
帧号、时间误差和边界 local Geodesic Distance。默认图片保存为预测结果目录下的
`boundary_frames.jpg`，视频、人工边界、预测 CSV 和输出路径均可通过参数切换。

## 结束边界 KeyPose 参考包

第一版定性分析将每一式的 Gold 结束边界事件作为该式唯一 KeyPose。下面的命令
根据新版边界文件、参考视频、stitched PHALP tracking 和 `TechPoint.md` 生成统一
reference manifest：

```bash
conda run -n 4d-humans python build_endpoint_keypose_reference.py
```

默认输出到 `reference_data/BV1WE411W7JB/`：

- `reference_manifest.json`：参考视频身份、路径、采样率和统一帧索引规范；
- `endpoint_keyposes.csv/json`：24 个结束事件及对应的最后一条动作要领；
- `endpoint_keypose_overview.jpg`：24 个 Gold 边界帧的一图总览。

内部统一使用 0-based 原视频帧和 0-based 5 FPS sample index，PHALP 帧号为
1-based。离散招式区间采用闭区间
`[previous_boundary + 1, current_boundary]`，第 1 式从 sample/source index 0
开始。参考 tracking 必须在全部 24 个精确边界帧上存在 SMPL pose，否则程序默认
报错；仅诊断缺失数据时可以显式添加 `--allow-missing-tracking-keyposes`。

## 全视频帧动作流畅性诊断

下面的命令使用原始视频帧率的 SMPL 局部旋转分析动作停顿、节奏和幅度。DTW
只负责建立教师与学生的动作阶段对应；停顿时长和动作幅度仍在原始时间轴上计算。

```bash
cd /home/sqw/Projects/3D-AQA

# 分析已有24式分割结果的完整学生视频
conda run -n 4d-humans python run_motion_quality_analysis.py --dataset full

# 分析学生1、2、3、4、10的三个独立动作片段
conda run -n 4d-humans python run_motion_quality_analysis.py --dataset clips

# 同时分析两套数据，并保存逐式诊断图
conda run -n 4d-humans python run_motion_quality_analysis.py \
  --dataset both --save-diagnostics

# 不重算逐帧指标，仅从已有汇总表重建两套幅度排名实验
conda run -n 4d-humans python run_motion_quality_analysis.py --rankings-only
```

可以使用 `--student-video-id 00`、`--clip 4_1_qishi` 或
`--move qishi` 缩小处理范围。默认结果保存在 `motion_quality_results/`：

- `full/<video_id>/`：完整学生视频的24式指标；
- `clips/<clip_id>/`：独立动作片段指标；
- `motion_quality_summary.csv`：每式、每身体区域的停顿、节奏、时长和幅度指标；
- `pause_events.csv`：每个连续停顿事件的原视频帧范围和教师阶段；
- `tempo_bins.csv`：教师进度均分为10段后的局部持续时间比例；
- `motion_signals.npz`：逐帧速度、有效帧、停顿掩码、DTW进度与停滞掩码；
- `metric_rankings.csv`：各单项指标的学生排名；
- `human_rank_correlations.csv`：动作片段单项指标与人工排名的 Spearman 相关性。
- `*_amplitude_larger_is_better.csv`：按幅度比越大越好的当前实验结果；
- `*_amplitude_teacher_closeness.csv`：按幅度比越接近1越好的对照实验结果。
- `*_amplitude_absolute_difference.csv`：按 `|amplitude_ratio - 1|` 从小到大排名。

第一版只输出独立诊断指标，不将其合成为总分。通用文件名
`metric_rankings.csv` 和 `human_rank_correlations.csv` 使用
`|amplitude_ratio - 1|` 越小越好的假设；`duration_ratio` 仍按与1的对数偏差排名，
其余误差、停顿和节奏指标均为越小越好。相同指标值使用平均秩处理。

## 关节角速度与加速度流畅性分析

该实验使用5个具备人工24式边界的教学视频，建立
`招式 × 身体区域 × Qx参考进度` 的教师运动学模板。角速度和角速度变化率在原始
视频帧率上计算；5 FPS SMPL pose-DTW只用于提供动作阶段对应，不压缩学生的真实
停顿和持续时间。

```bash
cd /home/sqw/Projects/3D-AQA

# 构建五教师模板、留一验证和24式模板诊断图
conda run -n 4d-humans python run_velocity_quality_analysis.py \
  build-teacher-model

# 分析学生1、2、3、4、10的三式片段
conda run -n 4d-humans python run_velocity_quality_analysis.py \
  analyze --dataset clips --save-diagnostics

# 分析已有完整学生视频；可重复指定视频ID
conda run -n 4d-humans python run_velocity_quality_analysis.py \
  analyze --dataset full --student-video-id 00 --student-video-id 04 \
  --save-diagnostics
```

默认结果保存在 `velocity_quality_results/`：

- `teacher_model/teacher_motion_model.npz`：101点教师速度、加速度和活动概率模板；
- `teacher_model/teacher_leave_one_out.csv`：五教师留一稳定性验证；
- `clips/<clip_id>/` 和 `full/<video_id>/`：逐样本指标、事件、信号和诊断图；
- `all_kinematic_quality_summary.csv`：全部招式和身体区域的批量汇总；
- `metric_rankings.csv` 和 `human_rank_correlations.csv`：单指标探索性排名与相关性。

`angular_speed_change` 是角速度大小的中心差分，单位为 `degree/s²`，不是完整三维
角加速度向量。速度均值和方差只作描述；流畅性解释应结合速度轮廓偏差、加速度
异常、运动碎片、停顿、动作幅度和持续时间，不根据单一统计量生成总分。

## 教师标定姿态分与异常停顿扣分实验

该实验用5位教师之间的20个有向配对，为每一式分别标定关键帧 Geodesic Distance
和完整 DTW 路径平均 Geodesic Distance。姿态基础分默认按关键帧40%、DTW路径60%
融合；随后只统计教师活动阶段内的学生异常停顿，每次扣5分，最多扣15分。

```bash
cd /home/sqw/Projects/3D-AQA
conda run -n 4d-humans python run_pose_score_experiment.py
```

默认分析学生1、2、3、4、10的 `qishi`、`yemafenzong`、`baiheliangchi`，结果保存在
`pose_score_experiment_results/`：

- `teacher_pairwise_distances.csv`：教师有向配对的两类距离；
- `teacher_calibration.csv`：每式距离中位数、MAD和稳健尺度；
- `student_pose_scores.csv`：两类距离、标准化值、姿态分、停顿扣分和最终分；
- `unexpected_pause_events.csv`：异常停顿的帧范围、时长和教师活动占比；
- `score_rankings.csv` 与 `human_rank_correlations.csv`：单项及综合分的排名验证；
- `penalty_sensitivity.csv` 与 `penalty_sensitivity_correlations.csv`：每次扣0、2、5、8分的敏感性实验。

当前综合分仍是探索性基线，不应直接解释为最终动作质量分。固定停顿扣分需结合更多
人工标注样本验证，并重点检查正常定势是否被误判。

取消停顿扣分上限，并使用人工复核事件测试每次扣2、4、5、6、8分：

```bash
conda run -n 4d-humans python run_uncapped_pause_penalty_experiment.py
```

该脚本读取已有姿态基础分，不会重新运行 tracking 或 DTW。默认输出到
`pose_score_experiment_results/uncapped_pause_penalty/`，并保留自动事件数和人工
复核后的有效事件数。

## 骨盆轨迹与重心转换实验

第一版实验使用10个教学视频，为起势、野马分鬃和白鹤亮翅建立骨盆轨迹模板，并
分析学生1、2、3、4、10。核心信号是骨盆相对双踝中点的高度、身体左右轴位移和
前后轴位移；身体左右轴由70%髋连线和30%肩连线的地面投影构成。PHALP camera
平移仅作为辅助 root 轨迹，不视为真实人体质心。

```bash
cd /home/sqw/Projects/3D-AQA
conda run -n 4d-humans python run_pelvis_quality_analysis.py
```

默认结果保存在 `pelvis_quality_results/`。其中
`teacher_trajectory_template.csv` 是10教师的中位数和P10-P90模板，
`all_pelvis_quality_summary.csv` 是学生指标汇总，`clips/<clip_id>/` 保存逐帧
轨迹、NPZ信号、配置摘要和诊断图。该实验不生成综合分。

## 重心指标融合实验

下一版实验将教师校准的重心异常量与姿态基础分、人工复核后的异常停顿次数融合，
并扫描每次停顿扣分和重心扣分系数：

```bash
cd /home/sqw/Projects/3D-AQA
conda run -n 4d-humans python run_center_fusion_experiment.py
```

默认结果保存在 `center_fusion_experiment_results/`。教师阈值采用10个教学视频的
leave-one-teacher-out误差进行稳健校准；`ranking_results.csv` 保存全部参数组合
的学生排名，`configuration_summary.csv` 汇总与人工排名的一致性，
`student_center_components.csv` 则保留每项重心指标的误差、稳健z值和扣分来源。
该实验仅用于五名学生上的敏感性分析，最优参数不能视为已经验证的正式评分规则。

## 可学习的 Late Rank Fusion 实验

使用姿态分、人工复核后的异常停顿次数和综合重心异常分别建立相对排名，并比较
单指标、等权 Borda、可学习排名融合及连续指标融合：

```bash
cd /home/sqw/Projects/3D-AQA
conda run -n 4d-humans python run_late_fusion_experiment.py
```

默认结果保存在 `late_fusion_results/`。`evaluation_summary.csv` 同时报告每式
Spearman、平均排名误差和完整排序，`oof_rankings.csv` 保存每名学生的预测名次，
`fold_weights.csv` 保存 leave-one-move-out 的逐折权重及均值、标准差，
`regularization_sensitivity.csv` 保存正则系数敏感性分析。现有规则扣分方法仅作为
in-sample参考，不与留一招式结果混作泛化性能。该模型只提供当前学生集合内的
相对排名，不产生绝对动作质量分数。

## 测试

```bash
cd /home/sqw/Projects/3D-AQA
python -m unittest discover -s tests -v
```

## 批量分割教学视频

以 `BV1WE411W7JB` 的 5 FPS 金标准边界为参考，批量分割
`Tracking/teach_trimmed` 下所有其他教学视频，并为每个视频生成
`boundary_frames.jpg`：

```bash
conda activate 4d-humans
cd /home/sqw/Projects/3D-AQA
python run_smpl_pose_dtw_teach_batch.py
```

具备完整24式人工标注的视频显示“参考 GT + 目标 GT + 目标预测”三列，其他视频显示
“参考 GT + 目标预测”两列。结果保存到
`smpl_pose_dtw_segmentation_results/<video_id>/`，批量状态保存在
`smpl_pose_dtw_segmentation_results/teach_batch_summary.json`。若 tracking ID 切换处存在
短缺口，可显式增加最近有效姿态容差，例如
`--max-tracking-gap-seconds 0.65`；默认仍为严格匹配。

## 按人工边界导出24式子视频

使用5 FPS边界文件中的前23个结束边界，将完整视频按原始帧精确切分为24个
互不重叠的子视频：前23式包含各自的结束边界帧，第24式从第23式边界的下一帧
延伸到原视频最后一帧。

```bash
conda activate 4d-humans
cd /home/sqw/Projects/3D-AQA
python split_video_by_boundaries.py
```

默认输入为 `BV1WE411W7JB.mp4` 及其金标准边界，输出到
`/home/sqw/VisualSearch/aqa/ActionSegments/teach/BV1WE411W7JB/`。切换视频时使用
`--input-video`、`--boundary-file`、`--video-id` 和 `--output-root`；已有输出需要显式
指定 `--overwrite`。`segments.csv` 和 `segments.json` 保存可复核的原始帧范围。

## KeyPose 对齐到完整教学视频

先对 `FrameData/KeyPose` 下24式的人工 KeyPose 运行 HMR2 图像推理。脚本递归读取
图片、选择检测面积最大的教师，并保存23个局部 SMPL 关节旋转及重建 overlay；
逐图结果支持断点续跑：

```bash
conda activate 4d-humans
cd /home/sqw/Projects/3D-AQA
CUDA_VISIBLE_DEVICES=0 python export_keypose_smpl.py
```

随后在 `i8kMrJmAfjU` 的每式 Ground Truth 边界内，以原始30 FPS tracking
独立运行 SMPL Geodesic DTW：

```bash
python run_keypose_reference_alignment.py
```

默认结果位于
`/home/sqw/VisualSearch/aqa/keypose_alignment_results/i8kMrJmAfjU/`。
`keypose_matches.csv` 同时保存裁剪视频和原视频的帧号、秒数及格式化时间戳；
`moves/<move_id>_<move_name>/` 保存完整 DTW 路径、cost matrix、逐式 CSV、
路径诊断图和三列人工核验图。KeyPose 只按每式文件名顺序编号，不自动绑定尚未
人工确认的动作要领 `pose_id`。

复用同一份 KeyPose SMPL 参数，对齐其他具有24式人工边界和 tracking 的教学视频：

```bash
python run_keypose_reference_alignment.py \
  --video-id BV1WE411W7JB \
  --annotation-path /home/sqw/Projects/annotation-tool/annotations/instruction_2026-07-28_13.35.36.txt
```

未指定 `--output-root` 时，结果自动保存到
`keypose_alignment_results/<video-id>/`。通过 `--keypose-smpl-archive` 和
`--keypose-overlay-root` 可以切换 KeyPose 重建来源，不会重复运行 HMR2。

## 查看 Tracking `.pkl`

`.pkl` 是 `joblib` 压缩的逐帧 PHALP 结果。下面的脚本会显示文件摘要和指定帧中
每个追踪人物的 SMPL 字段、数组形状及取值范围：

```bash
python inspect_tracking_pkl.py \
  /home/sqw/VisualSearch/aqa/Tracking/teach/QxVvRcRn2TA_1_qishi/results/demo_QxVvRcRn2TA_1_qishi.pkl \
  --frame 1
```

## 验证新旧 DTW

验证脚本会直接调用旧 AQA 项目的 `dynamic_time_warpping`，并与本项目
`dtw_alignment` 对比。它会验证固定种子的合成数据、全平局数据，以及真实的
`1_1_qishi` 视频特征；两种实现的匹配路径会分别保存为 CSV。

旧实现的 `dtaidistance` 安装在 `aqa` 环境中，因此请使用该环境运行：

```bash
conda activate aqa
cd /home/sqw/Projects/3D-AQA
python validate_dtw/validate_implementations.py --device cuda:2
```

结果会写入 `validate_dtw/results/validation_summary.json`，每个数据集目录还包含
`legacy_matching.csv`、`new_matching.csv`、`comparison.json` 与可复现实验输入的
`input_features.npz`。

# DTW 算法分割学生视频
```bash
conda activate 4d-humans

python run_student_tas_smpl_dtw.py \
  --student-video-id 14 \
  --student-video-id 15 \
  --student-video-id 16 \
  --student-video-id 17 \
  --student-video-id 18 \
  --student-video-id 19 \
  --student-video-id 20 \
  --student-video-id 21 \
  --student-video-id 22 \
  --student-video-id 23 \
```
