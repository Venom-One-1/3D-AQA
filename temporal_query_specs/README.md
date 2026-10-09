# 动作要领驱动的时序查询

当前查询由 Codex 根据 `TechPoint.md` 编写，执行器消费已完成 tracking 的
原生 SMPL-24 关节序列。当前开发参考为 `BV1WE411W7JB`，覆盖前三式。
已知招式边界是合法输入：限定搜索范围，并作为每式末次完成事件的定势锚点。

## 运行

在项目根目录的 `4d-humans` 环境中，生成第 2、3 式的可编辑 JSON：

```bash
python build_temporal_query_plans.py
```

默认写入 `temporal_query_specs/first3_v2/`，拒绝覆盖已有查询。
查询源代码是根目录的 `build_temporal_query_plans.py`；可通过
`--manifest`、`--technique`、`--output-dir` 指定输入和新的输出目录。
末条动作要领必须与 manifest 一致。

运行单式查询：

```bash
python run_temporal_keypose_localization.py \
  --plan temporal_query_specs/first3_v2/02_yemafenzong_temporal_keyposes_v1.json \
  --output-dir temporal_keypose_results/first3_v2/BV1WE411W7JB/02_yemafenzong

python run_temporal_keypose_localization.py \
  --plan temporal_query_specs/first3_v2/03_baiheliangchi_temporal_keyposes_v1.json \
  --output-dir temporal_keypose_results/first3_v2/BV1WE411W7JB/03_baiheliangchi
```

历史起势查询仍在 `temporal_query_specs/qishi_v1.json`。重复实验请选择新的
`--output-dir`，程序拒绝覆盖已有结果。CPU 线程过多时，可在命令前加入
`env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`。

## 查询内容

| 招式 | 事件 | 主要测量 |
|---|---|---|
| 起势 | 前举完成、屈膝按掌 | 双腕高度与前向位置、膝角 |
| 左右野马分鬃 | 三次抱球收脚、左/右/左三次弓步分掌 | 双踝距离、单侧膝角、双腕高度次序、单侧腕位置 |
| 白鹤亮翅 | 抱手、上下分掌至结束定势 | 双腕距离、左腕高于右腕的关系、右腕减左腕的高度差变化 |

野马分鬃第三次抱球为完整招式左右交替顺序的展开，不假定原文最后一句独立
描述了全部过程。查询根据完整招式组织事件，不强制逐句一一对应。

局部高低由搜索区域内的 P5/P95 归一化信号表示；`positive/negative` 则直接
检验有符号几何量的正负，避免把演示者左右侧混为一谈。
`stable` 表示相对变化较慢，`stable_window` 不要求真正静止。
顺序依赖和宽进度窗口是显式先验，具体时间由条件匹配产生。
每个事件保留全部条件得分、掩码、连续区间和代表帧选择依据。

## 结果约定

已有定位结果可单独补绘辅助 KeyPose 对照，不重新执行查询：

```bash
python redraw_temporal_keypose_overviews.py
```

默认更新 `first3_v2/BV1WE411W7JB` 中第 2、3 式的 `keypose_overview.jpg`，
四列为辅助 KeyPose、区间起点、代表帧和区间终点。首次更新前的图保存在
`keypose_overview_without_auxiliary.jpg`；对应图片、展示阶段及输入哈希保存在
`auxiliary_keypose_mapping.json`。白鹤亮翅抱手的辅助图对应区间起点。
支持 `--move 2` 或 `--move 3`、`--results-root` 和 `--keypose-root`。

- `report.md`：事件时间、定位依据及两张对照图。
- `keypose_events.json/csv`：输出区间、原始信号匹配区间、代表帧来源。
- `keypose_overview.jpg`：每行一个事件，显示起点、中点、代表帧和终点；
  有明确辅助 KeyPose 的旧查询保留原对照列。
- `temporal_diagnostics.png`：逐帧信号和每个事件的查询得分；深色为信号匹配，
  浅色为最终输出区间，虚线为代表帧。
- `motion_signals.npz`：原始/平滑信号、导数、条件得分及 tracking 有效掩码。
- `sensitivity_runs.csv`：3 个平滑窗口与 3 个阈值倍率，共 9 组扰动。
- `summary.json`：输入及程序哈希、tracking 质量与稳定性统计。

原视频使用 0-based 索引；PHALP 使用 1-based；时间是裁剪后完整视频时间。
5 FPS 帧号表示最近网格，独立保存其网格时间与对应原视频帧，不能与精确查询帧混用。
跨招式的离散区间沿用 `[previous_boundary + 1, current_boundary]`。

`representative_source=known_move_end` 表示代表帧直接取已知结束边界；
`signal_query` 表示代表帧由信号选择。已知边界缺失有效 tracking 时返回
`missing_valid_move_end`，不自动改用临近帧。区间允许部分重叠，代表帧受先后依赖约束。

当前是同一参考视频上的开发实验。扰动定位成功率表示对参数的稳定性，不是
定位准确率；由边界决定的终点波动为零，也不代表独立定位能力。
候选区间不是统计置信区间，查询得分不是质量分或概率。
SMPL 腕、踝及身体坐标只描述几何关系，不能证明掌心朝向、真实承重或肌肉放松。

## 向教师和学生迁移事件

```bash
python run_temporal_keypose_transfer.py
```

默认读取前三式的已复核参考事件和 `endpoint_metric_results/first3_five_students`
中的完整 DTW 路径，覆盖十位教师及学生 01、02、03、04、10。
输出到新的 `temporal_keypose_transfer_results/first3_v2/`，拒绝覆盖已有结果。
`first3_v1` 保留为更正前的历史实验，不覆盖其数据和图。
可以用 `--students 01 04`、`--teachers <ID>`、`--moves 2 3` 筛选，
用不带值的 `--teachers` / `--students` 排除相应类别。
重复运行需显式传入新的 `--output-root`。

入口核对参考 manifest 的哈希、参考事件生成时的输入文件、原视频 FPS、
完整路径及其索引/时间，并重算路径上所有局部 Geodesic Distance 确认 tracking
相容。任一不匹配都会报错，不改用其他路径或静默重新运行 DTW。

内部代表帧先以实际原视频帧定位到最近的参考 5 FPS 帧（等距离取较早帧），
再选当前式内 DTW 候选的较早中位帧，避免按学生误差挑“最好”的姿势。
参考原始帧及量化偏移单独保留。区间起止取该参考起止帧的候选包络；
结束事件仍复用既有最小局部距离的招式结束边界。
区间允许重叠，跨式候选、重复代表帧和退化区间均保留诊断，不静默修正。
该程序只消费已有路径，不运行人体重建或 LLM。

- `report.md` 和 `figures/`：分招式的五学生／十教师对照。
- 各视频 `keypose_overview.jpg`：参考代表帧、目标区间起点、代表帧、终点。
- `reference_event_manifest.json`：冻结参考事件、动作要领、辅助对应和来源哈希。
- `all_events.csv`、各视频 `keypose_events.json/csv`：逐事件的时间、原视频帧号、
  候选范围、选择策略及 tracking 状态。
- `candidates.csv`：三个事件位置的全部候选，包括落在当前式以外的候选。
- `review_queue.csv`：带自动警告的事件，预留复核枚举和备注，未标注为 `pending`。

`--review-candidate-span-seconds` 默认 0.8s，只是人工检查的优先级阈值，
不是统计置信区间或动作质量判据。旧 DTW 使用过的 tracking 近邻替代会被
明确记录；目标原始帧缺失不会因此变成有效几何帧。
映射成功数量不等于定位准确率，内部事件仍无人工时间戳金标准。

### 人工更正与复核记录

默认加载 `reference_data/BV1WE411W7JB/temporal_event_review.json`，也可用
`--review-records` 提供明确的人工记录。白鹤亮翅 `3.gather_hands` 的参考代表帧
按人工确认从 frame0 1338 改为 1328（44.267s），然后重新通过 DTW 路径迁移
目标代表帧；事件区间和招式结束边界不变。原查询选择和敏感性信息保存在
`original_query_representative`，不会当作人工更正帧的自动查询得分。

教师 `BV1svYmz6EFB_pNA` 的起势前举及学生 `10` 的末次左野马分鬃已人工确认，
使用 `correct` 记录；自动诊断仍保留，图中改用绿框。该确认只涉及事件对应，
不是动作质量判断。复核记录绑定参考代表帧和目标起点／代表／终点帧，
若这些帧后续改变，不自动沿用旧复核结论。

`reference_event_manifest.json` 与 `reference/<move>/keypose_events.json` 是
更正后的参考事件；`source_keypose_events.json` 和 `source_summary.json` 是
更正前查询的原始快照。复核记录另冻结为 `manual_review_records_snapshot.json`。
