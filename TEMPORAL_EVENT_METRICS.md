# 多事件几何测量

入口 `run_temporal_event_metrics.py` 消费已冻结的前三式事件迁移结果。它不运行
视频人体重建、DTW 或 LLM，只从已有 SMPL 参数回归原生 SMPL-24 关节。

```bash
conda activate 4d-humans
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python run_temporal_event_metrics.py \
  --transfer-root temporal_keypose_transfer_results/first3_v2 \
  --endpoint-root endpoint_metric_results/first3_five_students \
  --output-root temporal_event_metric_results/first3_five_students_v1
```

输出目录必须不存在。输入为修正白鹤亮翅抱手代表帧到 frame0 1328 的 `first3_v2`，
固定十位教师、学生 01/02/03/04/10 与十个事件。参考锚点单独测量，不参与教师统计。

- `metric_definitions.json`：完整查询、分量、归一化、解剖方向、事件指标集合与局限。
- `measurements.json/csv`：代表帧短窗和过程首尾变化的数值、有效性、稳定性提醒。
- `teacher_reference.json`：每项十个教师原始值（含无效 null）、两种参考区间。
- `student_comparison.json/csv`：主/辅区间及中心值的一致性；仅为几何证据，不是错误标签。
- `comparison_report.md`、`events/`、`cases/`：图片、完整对照和集中导航。
- 视频独立目录：`native_joints.npz` 原生关节、源帧号、tracking 掩码与 ID；
  `frame_metrics.npz` 逐帧查询信号；`measurements.json` 保存事件、窗口来源与完整上下文。
- `endpoint_compatibility.json`：390 项旧 F 组结束定势数值的逐项校验。
- `input_provenance.json`、`snapshots/`：输入身份、冻结文件哈希、参考事件与源码快照。

代表帧窗口为前后各 0.2 秒，截断于当前招式。三个结束事件沿用已人工检查的
F 组关节缓存，完整保留 26 个指标槽，并要求与旧数值误差不超过 1e-6。
内部事件则使用本轮从已有 tracking 回归的关节。逐帧归一化，不补缺帧，不因偏差挑选新帧。

两个过程事件另算末窗减首窗：单侧窗口位于事件内部，不超过 0.2 秒且互不重叠。
同时保留原始末帧减首帧的变化，避免把过程变化当成代表帧姿势值。
过程量反映选定阶段的几何变化，不证明速度、节奏或手脚协调质量。

教师区间默认使用未缩放 MAD 的 median ± 2×MAD，另列 P10–P90。至少七位教师有效，
保留全部十位原始记录；短窗变化提醒不触发删教师。窗口至少三帧有效且有效比例不低于
0.7。角度中心/中位数差大于 5 度或范围大于 10 度、比例差大于 0.05 或范围大于 0.1
会提示谨慎解读。这些阈值是预定义的、未标定的诊断参数，不是专家错误标准。

`difference_evidence` 只表示两种区间一致、原始与稳健汇总判断一致且没有自动提醒。
`review_measurement` 表示需要谨慎解读，不要求人工逐行复核全部条目。
`no_interval_difference` 不表示动作合格。报告保留所有指标，不按显著性缩小指标覆盖。

```bash
python -m unittest discover -s tests -p 'test_temporal_event_metrics.py' -v
```
