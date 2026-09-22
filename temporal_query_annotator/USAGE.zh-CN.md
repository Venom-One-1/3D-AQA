# 时序查询 Gold Standard 标注工具

该工具用于构建“动作要领 -> 运动学时序查询”的 Gold Standard。首个项目固定使用裁剪后的完整参考视频 `BV1WE411W7JB.mp4`，时间定位采用 5 FPS 网格，覆盖起势、左右野马分鬃和白鹤亮翅。

## 启动

在 `/home/sqw/Projects/3D-AQA` 中运行：

```bash
python -m temporal_query_annotator.server
```

浏览器访问：

```text
http://127.0.0.1:8765
```

默认标注实时保存到：

```text
temporal_query_annotator/data/bv1we411w7jb_first3.annotations.json
```

界面中的“导出 JSON”会额外导出包含项目任务文本和最终标注的完整 Gold Standard 文件。

## 标注内容

每个动作阶段保存以下信息：

- `observability`：SMPL-24 是否能够观测该动作要求。
- `temporal_scope`：瞬时事件、稳定区间、过渡区间或整式属性。
- `gold`：5 FPS 网格上的起始帧、结束帧和可选代表帧。
- `llm_candidate`：LLM 原始候选，用于统计接受率和编辑量。
- `query`：人工审核后的 DSL v0.1 查询。
- `review_status`：未标注、草稿或已审核。
- `audit`：候选生成、首次保存和最后保存时间，以及候选是否未经修改直接采用。

帧号 `N` 对应视频时间 `N / 5` 秒。例如帧 `80` 对应 `16.0s`。

## LLM 候选生成

服务支持兼容 Chat Completions 请求格式的模型接口。密钥只保存在服务端环境变量中，不会发送给浏览器：

```bash
export TQA_LLM_API_URL="https://your-provider.example/v1/chat/completions"
export TQA_LLM_API_KEY="your-api-key"
export TQA_LLM_MODEL="your-model-name"
python -m temporal_query_annotator.server
```

也可以指定其他密钥变量名：

```bash
python -m temporal_query_annotator.server \
  --llm-api-url "https://your-provider.example/v1/chat/completions" \
  --llm-model "your-model-name" \
  --llm-api-key-env "DASHSCOPE_API_KEY"
```

未配置接口时，工具进入手动导入模式。“查看提示词”可以复制完整提示词，在模型网页中运行后，将返回的 JSON 粘贴回工具。两种模式使用相同的候选审核流程。

默认采样参数为 `temperature=0.1`、`top_p=0.2`，用于降低同一动作要领重复生成时的随机性。

## DSL v0.1

Schema 位于 `schemas/temporal_query_dsl_v0.1.json`。DSL 不直接保存帧号，而是描述在招式内部执行的搜索逻辑：

```json
{
  "schema_version": "0.1",
  "scope": "stable_window",
  "search_region": {
    "unit": "move_progress",
    "start": 0.6,
    "end": 1.0
  },
  "conditions": [
    {
      "metric_id": "right_elbow_angle",
      "signal": "value",
      "aggregation": "mean",
      "operator": "within_teacher_range",
      "role": "required",
      "min_duration_seconds": 0.4
    }
  ],
  "temporal_relations": [],
  "selection": {
    "target": "window",
    "strategy": "best_score",
    "representative_frame": "min_motion"
  },
  "rationale": "在招式后半段寻找右肘角度合理且姿态稳定的完成阶段。"
}
```

## 扩展到其他视频

复制 `projects/bv1we411w7jb_first3.json`，修改以下字段：

- `video_id`
- `video_path`
- `frames_dir`
- `annotations_path`
- `sample_fps`
- `move_ids`
- `move_boundaries_seconds`

然后运行：

```bash
python -m temporal_query_annotator.server --project <project.json> --port 8766
```

## 测试

```bash
python -m unittest discover -s temporal_query_annotator/tests -v
```
