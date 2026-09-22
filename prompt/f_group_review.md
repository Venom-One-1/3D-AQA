# F 组与规则 baseline 对照复核

## 启动

```bash
conda activate 4d-humans
cd /home/sqw/Projects/3D-AQA
python run_f_group_review_app.py --port 8502
```

浏览器打开 `http://127.0.0.1:8502`。如果通过 VS Code Remote SSH 使用服务器，在“端口”面板转发 8502 后打开本地地址。默认只监听服务器本机，不向公网开放；端口占用时通过 `--port` 换一个端口。

## 数据与范围

- 直接读取当前 `f_group_experiment/first3_five_students_api/responses/` 下已人工修正的 15 份 JSON，不调用 API，不复用旧校验表中的解析失败结果。
- 图片使用 `endpoint_metric_results/first3_five_students/` 内 BV1WE411W7JB 的结束定势参考图和学生结束定势帧，保持完整宽高比，点击可看大图。
- 规则反馈来自本轮 `evaluation/baseline_snapshot.json`，总结复用既有 `coach_summary` 函数。
- 每案例展示动作要领、两侧总结、完整 findings 及规则候选建议／需复核原因。证据指标可展开查看学生窗口中位数、中心值及教师两套参考区间。
- 总计 167 个可标注条目：两种方法的 30 条总结、44 条 Qwen findings、93 条规则候选建议或需复核项。其余 37 条规则指标保留为折叠查看，不默认纳入复核。
- “正确／误判／无法判断”分别保存为 `correct / false_positive / uncertain`；“未标注”为 `pending`。复核总结时评估该总结，复核需复核项时评估其暂缓决定，不能把后者等同于已经确认的动作错误。

## 保存与导出

标注和备注先保留在页面，点击保存、切换案例或导出时写入服务端。关闭页面前若有未保存内容会提示。写入失败会显示错误，不会将失败的保存计为成功。

默认进度单独保存在：

`f_group_experiment/first3_five_students_api/ui_review/review_progress.json`

点击“导出 CSV”会先保存当前编辑，再下载 `f_group_review.csv`。导出含所有 167 项及其标注状态，字段包括 `case_id`、`source`（F / rule_baseline）、`review_scope`（coach_summary / finding / metric）、`item_id`、原始内容、`manual_verdict`、`notes`、`reviewer`、`updated_at`。

此 CSV 是独立的配对复核表：沿用英文枚举，但不覆盖、不自动导入旧 `manual_comparison.csv` 或其他复核工具的进度；不同复核单位不能简单按行合并。

为避免误关联标签，启动时会记录本轮 JSON、baseline 和参考 manifest 的哈希。开始标注后如果再修改源文件，程序拒绝覆盖已有标注。保留旧状态并指定新的 `--state-path /path/to/new_review.json` 开始新一轮。多个浏览器同时修改时使用版本检查，过时页面需刷新后再保存。

## 文件

- `run_f_group_review_app.py`：本地 HTTP 服务入口。
- `aqa3d/f_group_review.py`：当前数据加载、标注、CSV 导出。
- `f_group_review_static/`：本地 HTML、CSS、JS、Lucide 图标及许可证；无需 CDN。
- `tests/test_f_group_review.py`：原子保存、恢复、校验、冲突保护与导出测试。
