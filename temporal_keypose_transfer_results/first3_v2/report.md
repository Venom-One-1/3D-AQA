# 前三式关键事件 DTW 迁移

参考：BV1WE411W7JB。复用既有完整视频 5 FPS、23 个局部 SMPL 关节的 Geodesic DTW 路径，逐条核对路径的身份、索引、时间和实际 tracking 姿态距离。未重新运行 DTW、人体重建或 LLM。

共 10 个参考事件，15 个目标视频，150/150 个事件获得映射；2 个事件带自动诊断警告，其中 2 个已人工确认事件对应。

## 解读边界

内部代表帧取该参考 5 FPS 帧在 DTW 路径上、当前招式内候选的较早中位帧，不按学生姿态误差挑最好帧。结束事件复用已有最小局部距离的招式边界。参考原始帧、量化后的参考帧和目标实际帧分别保存。区间起止取候选包络，并限定在当前式内；越界候选全部留在 candidates.csv，且标记复核。

红框仅表示尚未确认的匹配或 tracking 诊断，不是动作错误；绿框表示该诊断条目的事件对应已人工确认。无框不代表匹配已人工确认。内部事件没有人工时间戳金标准，因此不能把映射数量、单调性或候选跨度称为定位准确率。事件允许重叠；重复代表帧不静默改帧。白鹤亮翅抱手已按人工更正改用区间起点作代表帧；原查询中点仅留作历史溯源，不再参与当前迁移。

候选跨度警告阈值为 0.80s，仅用于优先检查，未经标定。

## 人工更正参考帧

- 白鹤亮翅／转体抱手阶段：frame0 1338 → 1328（44.267s）。事件区间和招式边界不变；目标代表帧按更正后的参考帧重新迁移。

## 集中对照

### 1. 起势

![学生对照](figures/01_students.jpg)

[教师对照：第1组](figures/01_teachers_1.jpg)

[教师对照：第2组](figures/01_teachers_2.jpg)

### 2. 左右野马分鬃

![学生对照](figures/02_students.jpg)

[教师对照：第1组](figures/02_teachers_1.jpg)

[教师对照：第2组](figures/02_teachers_2.jpg)

### 3. 白鹤亮翅

![学生对照](figures/03_students.jpg)

[教师对照：第1组](figures/03_teachers_1.jpg)

[教师对照：第2组](figures/03_teachers_2.jpg)

## 逐视频区间与代表帧

| 类型 | 视频 | 映射 | 尚待确认的诊断 | 对照 |
|---|---|---:|---:|---|
| teacher | QxVvRcRn2TA | 10/10 | 0 | [查看](teacher/QxVvRcRn2TA/keypose_overview.jpg) |
| teacher | BV1iE411c7Ni_p03 | 10/10 | 0 | [查看](teacher/BV1iE411c7Ni_p03/keypose_overview.jpg) |
| teacher | BV1tk4y1r7Yr_p27 | 10/10 | 0 | [查看](teacher/BV1tk4y1r7Yr_p27/keypose_overview.jpg) |
| teacher | an5qNCspzUw | 10/10 | 0 | [查看](teacher/an5qNCspzUw/keypose_overview.jpg) |
| teacher | i8kMrJmAfjU | 10/10 | 0 | [查看](teacher/i8kMrJmAfjU/keypose_overview.jpg) |
| teacher | BV1vShdzLEsn_p1 | 10/10 | 0 | [查看](teacher/BV1vShdzLEsn_p1/keypose_overview.jpg) |
| teacher | 420p2gYFTa | 10/10 | 0 | [查看](teacher/420p2gYFTa/keypose_overview.jpg) |
| teacher | BV1MFZKYnEdM_pNA | 10/10 | 0 | [查看](teacher/BV1MFZKYnEdM_pNA/keypose_overview.jpg) |
| teacher | BV1svYmz6EFB_pNA | 10/10 | 0 | [查看](teacher/BV1svYmz6EFB_pNA/keypose_overview.jpg) |
| teacher | Bg3kJjFReAQ | 10/10 | 0 | [查看](teacher/Bg3kJjFReAQ/keypose_overview.jpg) |
| student | 01 | 10/10 | 0 | [查看](student/01/keypose_overview.jpg) |
| student | 02 | 10/10 | 0 | [查看](student/02/keypose_overview.jpg) |
| student | 03 | 10/10 | 0 | [查看](student/03/keypose_overview.jpg) |
| student | 04 | 10/10 | 0 | [查看](student/04/keypose_overview.jpg) |
| student | 10 | 10/10 | 0 | [查看](student/10/keypose_overview.jpg) |

## 诊断与人工确认

| 类型 | 视频 | 招式／事件 | 代表帧时间 | 警告 | 人工复核 |
|---|---|---|---:|---|---|
| teacher | BV1svYmz6EFB_pNA | 起势／双臂前举完成阶段 | 8.000s | interval_collapsed_to_single_sample | correct：人工确认区间压缩到同一帧可以接受，该帧确实对应起势双手前举事件。保留退化区间诊断，不将其等同于错误匹配。 |
| student | 10 | 左右野马分鬃／第3次弓步分掌（左势） | 37.571s | end_candidates_outside_form; end_wide_candidate_span; representative_candidates_outside_form; representative_wide_candidate_span | correct：人工确认当前迁移代表帧满足复核要求；约0.801秒候选跨度及跨式候选可接受。此确认只涉及事件对应，不表示动作质量合格。 |

警告释义：wide_candidate_span 为某一参考帧匹配到较宽的目标时间跨度；candidates_outside_form 为部分路径候选落在当前式以外；missing_tracking / nearest_substitution 为原始精确帧缺失或旧路径用了近邻姿态；near_track_switch 为 ID 切换附近；duplicate_event_representative 为相邻事件选到同帧；interval_collapsed_to_single_sample 为区间被压到一个 5 FPS 帧。


## 数据文件

- reference_event_manifest.json：冻结参考事件、动作要领、原始区间及来源哈希。
- all_events.csv：逐视频逐事件的时间、索引、候选跨度、选帧策略及 tracking 状态。
- candidates.csv：起点、代表帧、终点的全部 DTW 候选及是否越界／被选中。
- review_queue.csv：自动诊断警告的事件及已有人工复核，不重复要求已确认条目复核。
- manual_review_records_snapshot.json：人工参考帧更正和事件对应确认的冻结记录。
- 各视频 keypose_events.json/csv、dtw_path.npz、summary.json 和 keypose_overview.jpg。

本轮只验证跨视频事件对应，不生成质量分、教师指标范围或训练反馈。下一阶段再在确认可用的事件上测量姿态。
