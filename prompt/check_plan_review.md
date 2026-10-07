# 动作要领与检查计划人工复核

## 启动

在 `4d-humans` 环境中执行：

```bash
cd /home/sqw/Projects/3D-AQA
python run_check_plan_review_app.py --port 8504
```

默认只监听服务器的 `127.0.0.1:8504`。通过 VS Code Remote SSH 转发该端口后访问本地对应端口。
如果 Windows 的现有 SSH 配置包含冲突的 RemoteForward，可另开 PowerShell：

```powershell
ssh -F NUL -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 127.0.0.1:18504:127.0.0.1:8504 sqw@202.112.113.90
```

保持该窗口运行，在普通浏览器地址栏输入 `http://127.0.0.1:18504/`。
不要重复启动使用同一标注文件的服务，也不要删除运行中的 `.lock` 文件。

## 范围与标注含义

- 输入为 `check_plan_experiment/first3_v1_repaired_v1` 的修复版计划：前三式、每式三次生成、共 117 个要领条目。默认第 1 次，其他轮次独立标注。
- 同时显示 BV1WE411W7JB 参考结束定势图、完整原文、当前条目、支持等级、中文测量定义、模型选取理由、预期关系、局限与原始结构化内容。
- 本次复核对象是“检查计划是否合理对应动作要领”，不是图中练习者的动作是否正确，也不是模型格式是否合规。
- `correct`（对应）：选点、度量、方向、预期关系和能力边界在该条要领下合理；合理地声明无法测量也可以标记为对应。
- `false_positive`（不对应）：方案存在不合理映射、遗漏、符号错误、越界解释等；请在备注中说明具体问题。
- `uncertain`（无法判断）：暂不能判断方案关联是否合理。未处理为 `pending`。
- 一条要领包含多个查询时作整体判断，备注记录具体查询的问题；不自动给未选条目打标。
- 参考图只供人工查看，没有重新发送给模型；既有修复记录与审查提示默认折叠，避免预先替代人工结论。

## 保存与导出

默认状态文件：`check_plan_review_results/first3_v1/review_progress.json`，可用 `--state-path` 指定独立路径。
保存采用原子替换、版本冲突检查和单进程文件锁；原始/修复版计划、图片及实验结果保持只读。
下次启动自动恢复已保存进度。浏览器会保留本地未保存草稿；换浏览器、换端口或清除浏览器数据会使这些草稿不可见，请及时保存到服务器。

“导出 CSV”下载当前全部条目，包括本地草稿，即使服务器暂时不可达也可导出。
`save_status=local_draft` 表示该行尚未保存到服务器；导出本身不等于保存。
服务器 `/api/export` 仅导出已保存状态。CSV 保留旧 `plan_review.csv` 的核心列，并增加原始查询定义、预期关系、轮次、复核人等字段；原 `plan_review.csv` 不会被覆盖。

若同一条目被其他浏览器修改，刷新后会明确列出冲突状态；可以保留本地草稿或采用服务器标注，不会静默覆盖。
原计划或参考资料变更时，使用新的状态文件，不能把旧结论自动套在新方案上。

## 验证

```bash
python -m unittest discover -s tests -p test_check_plan_review.py -v
```
