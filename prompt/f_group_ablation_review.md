# 原版 F 组与消融组人工复核

复用 `run_f_group_review_app.py`，新增 `--ablation-root` 模式。不提供此参数时仍为原版 F 组与 Rule Baseline 的复核，不改变旧标注。

## 本轮数据

- 学生 01、02、10，各前三式，共 9 个配对案例。
- 原版：`f_group_experiment/first3_five_students_api/responses`。
- 消融组：`f_group_experiment/no_metric_comparison_01_02_10_api/responses`，读取人工修正格式后的当前文件。
- 共 67 项：两组 18 条 coach_summary、原版 26 条 findings、消融组 23 条 findings。
- 两组全部重新标注，初始为 pending；不导入旧复核 CSV 或旧 review_progress.json。
- 默认状态文件：`f_group_experiment/no_metric_comparison_01_02_10_api/ui_comparison/review_progress.json`。
- 导出名 `f_group_ablation_review.csv`，source 枚举为 `F` 和 `F_ablation`；结论为 pending / correct / false_positive / uncertain。

标注针对各组原始总结和各条 finding，不强行把同序号 finding 当成同一个问题。CSV 中 record_id 同时包含 case_id、source、review_scope 和 item_id。

## 启动

```bash
cd /home/sqw/Projects/3D-AQA
conda activate 4d-humans
python run_f_group_review_app.py \
  --ablation-root f_group_experiment/no_metric_comparison_01_02_10_api \
  --port 8503
```

若同一状态文件已有服务运行，请使用已有服务，不要删除锁文件或重复启动。旧 8502 服务可继续运行。

本机浏览器访问服务器 127.0.0.1:8503。远程 Windows 用户在本机 PowerShell 建立转发：

```powershell
ssh -F NUL -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 127.0.0.1:18503:127.0.0.1:8503 sqw@202.112.113.90
```

保持窗口打开，在独立 Edge/Chrome 地址栏手动输入 `http://127.0.0.1:18503/`，避免内嵌预览重新映射端口。`-F NUL` 避免原 SSH 配置中的 RemoteForward 10808 冲突。

## 保存与实验边界

切换案例或导出前自动提交当前修改，也可点击保存。连接失败时不要刷新含未保存修改的页面；恢复 SSH 隧道后重试保存。导出包含全部记录，包括 pending。

程序校验两组案例身份、相同输入及回答结构。复核开始后若修改源回答，保存会被拒绝；应保留旧状态并用新的 `--state-path` 重新启动，不要覆盖已有复核。

页面展示教师锚点 BV1WE411W7JB 和学生结束定势、动作要领、双方 coach_summary、findings、依据与训练指导，以及可展开的指标值和两套参考范围。

原版模型标识为 qwen3.8-max，消融组为 qwen3.8-max-0902，不能保证底层模型相同。此次属于探索性对照；格式合规不代表动作判断正确。当前 UI 不调用模型 API。
