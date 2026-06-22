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

## 测试

```bash
cd /home/sqw/Projects/3D-AQA
python -m unittest discover -s tests -v
```
