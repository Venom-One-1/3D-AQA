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

## 测试

```bash
cd /home/sqw/Projects/3D-AQA
python -m unittest discover -s tests -v
```

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
