# 新方法与 F 组反馈复核

```bash
cd /home/sqw/Projects/3D-AQA
conda activate 4d-humans
python run_plan_feedback_review_app.py --port 8505
```

服务器本机地址：http://127.0.0.1:8505 。远程使用时将服务器 8505 转发到本地空闲端口（例如 18505），在本地浏览器访问 http://127.0.0.1:18505 。不要把其他程序的随机预览端口与该地址混用。

- 15 个案例：01、02、03、04、10 的前三式。
- 每条 finding 和每份 coach_summary 分别标注，共 105 项；两份自动校验未通过的回答仍完整保留，人工初始结论均为 pending。
- 左列新方法，右列原 F 组。上方展示动作要领、BV1WE411W7JB 参考锚点和学生结束定势；指标与参考范围可展开。
- 沿用 correct / false_positive / uncertain。标注评价当前建议或总结，而不是整套动作是否合格。
- 保存进度到 `plan_feedback_results/first3_five_students_v1/ui_comparison/review_progress.json`。与之前 F 组状态、实验输出和 `manual_feedback_review.csv` 相互独立，不自动继承历史标注。
- CSV 文件名 `plan_feedback_f_review.csv`，保留原复核程序的字段；source 为 plan_feedback / F，review_scope 为 finding / coach_summary。
- 导出不依赖服务器在线，包含未保存草稿，额外字段 save_status=saved/local_draft。CSV 下载不代表草稿已保存到服务器。
- 浏览器草稿随输入变化保存到 localStorage；冲突时明确选择服务器标注或本地草稿。清除浏览器数据会清除这些本地草稿，已保存服务器记录不受影响。
- 多个浏览器保存采用版本检查；重复启动会受到文件锁保护。不要删除正在使用的 `.lock` 文件；确需另开独立复核，可指定不同 `--state-path` 和 `--port`。

程序只读取实验，不调用 Qwen、不重跑 DTW、不修改模型原文。历史 F 组模型标签与新方法不同，这是开发性对照，不是严格受控消融。
