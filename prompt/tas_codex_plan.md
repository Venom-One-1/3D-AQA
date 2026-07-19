# 太极拳 24 式伪标注方案

## Summary
- 方案: **用一个标注了24式边界的参考教学视频，结合SMPL Geodesic Distance DTW 对齐**，为其他完整教学视频生成边界标注(先选一部分教学视频)。
- 当前可复用资源：`/home/sqw/Projects/3D-AQA` 里的 Geodesic Distance、DTW 代码，`/home/sqw/Projects/annotation-tool/annotations` 路径下的标注，instruction_2026-07-08_22.42.55.txt 中有最新的标注结果。

## Method
- 输入数据：
  - 参考视频: URLID, QxVvRcRn2TA 含有 24 个招式的边界。注意: 一般来说，视频的开头或结尾可能会出现与 24 式无关的背景信息，伪标注前要进行预处理，筛除背景信息。背景只在开头或结尾出现，所以第 1 式的开始到第 24 式的结束就是完整的有效片段。
  - 其他教学视频在：`/home/sqw/VisualSearch/aqa/teach` 路径下。
  - 帧采样：统一采样为 5 FPS 做初版伪标注；后续若边界过粗，可以酌情提高。
- 分割算法：
  - 用参考视频的 24 式边界表，作为标准时间轴。
  - 对选择的完整教学视频和教师完整模板做 DTW 对齐。
  - 将参考视频的 24 个边界通过 DTW path 映射到待标注视频的时间轴，得到待标注视频的 24 段边界。
  - 加入约束：动作顺序固定、边界单调递增。
- 输出格式：
  - 输出为 CSV：`video_id, move_id, move_name, start_frame, end_frame, start_time, end_time`
  - `start_frame/end_frame` 保存 5 FPS 采样序列中的 1-based 闭区间帧号。
  - 边界帧不共享：如果第 n 式结束于 `t` 秒，则第 n+1 式从 `t * 5 + 1` 开始。

## Assumptions
- 每个教学视频都完整包含 24 式，动作顺序不变。
- 输出先用于“伪标注/初始标注”，需要少量人工抽查校准，不能直接视为高质量人工标签。
- 24 式动作表采用标准顺序，后续文件名沿用现有 MoveID_pinyin 格式，例如 `1_qishi, 2_yemafenzong, 3_baiheliangchi, 4_louxiaobu, ..., 9_danbian, 10_yunshou, 11_danbian, ...`。
