# 检查计划自动预审（试运行）

范围：前三式、三次生成，共 9 份修复版计划 / 117 个条目。只评价“动作要领到检查计划”的关系，不评价学生动作质量。

## 本轮结果

| 项目 | 数量 |
|---|---:|
| GPT-6 Astra 独立预审 | 9 次 |
| 数值探针通过 | 366 / 366 |
| 预审通过（未校准） | 84 |
| 语义疑问，需要人工裁决 | 33 |
| 通过项分层随机抽检 | 8 |
| 合并人工队列 | 41 / 117 |

33 条中包含模型判为 fail 的 31 条和 uncertain 的 2 条。模型 fail 并不等于专家确认错误，因此不直接改写计划。41 条人工队列按招式和要领可合并为 25 组，在界面中可以对照其他生成轮次。

相对全部逐条复核，当前队列条目减少约 65%，并非已经证明节省相同工时。84 条通过也不是正确率：还没有独立校准集，抽检量较小。应先检查抽检项中有无漏报，再决定是否将预审作为长期筛选机制。

## 判定来源

- 程序：严格计划格式、查询引用、变换不变性、镜像与左右交换、已知方向位移与符号。数值探针只验证测量接口，不自动理解中文预期关系。
- 模型：关联、方向、覆盖、推断边界四维评价，逐条附原文引用。JSON 格式、条目完备性及引用原文由程序复验。引用存在不等于理由正确。
- 路由：确定的程序错误退回；模型问题进入人工队列；缺少合法评审不放行；全通过仍标为“未校准”。
- 随机抽检：从通过项中按招式抽样，目标比例 10%，固定 seed=20261002，各招式取整且至少一项，实际 8/84。
- 既有人工标签：仅在独立判断完成后作描述性对照。18 个已标条目中有 4 条人机结论分歧，均加入人工队列。旧左右解释存在争议，不作为准确率金标准，不被自动覆盖。

本轮最常见的疑问是从静态几何越界推断承重、肌肉放松、真实竖直或动作过程。起势肘膝对齐的 right 使用在三个轮次均被认为方向上可成立，但部分计划仍存在 direct/proxy 分类或遗漏侧向分量的问题，应区别审查。

## 实验文件

- `manifest.json`：冻结输入、提示词与哈希、盲化条件。
- `inputs/J001.json` 等：只包含要领、接口及待审计划，不提供生成模型身份、人工标注、学生值或修复提示。
- `judge_prompt.md`：本轮冻结评审规则。
- `responses/J001.json` 等：最终原始评审。`J004.before_completion.json` 是保留的中间版本，最终版本只纠正一处错别字，未修改判定。
- `execution_log.json`：真实子任务 ID。通道为 Codex 独立子任务，请求模型 `gpt-6-astra`、high reasoning、无会话历史；不是 HTTP API 批量调用。
- `analysis_v1/report.md`、`summary.json`：统计摘要。
- `analysis_v1/clause_triage.csv`、`human_queue.csv`：全量与人工队列。
- `analysis_v1/human_queue_groups.json`：按同一要领合并。
- `analysis_v1/geometric_probes.json`：数值测试明细。
- `analysis_v1/review_overlay.json`：复核界面可选叠加层。

## 复核界面

旧服务及原始标注保持不变。新服务使用独立进度文件，复制已有 18 条人工标注作为起点，不会反向同步到旧文件。后续请只在新页面复核，避免两份进度分叉。

在项目目录运行（8505 已启动时不用重复运行）：

```bash
conda activate 4d-humans
cd /home/sqw/Projects/3D-AQA
python run_check_plan_review_app.py \
  --judge-report check_plan_judge_results/first3_v1_astra/analysis_v1/review_overlay.json \
  --state-path check_plan_review_results/first3_v1_astra/review_progress.json \
  --port 8505
```

Windows PowerShell 建立单独转发，保持终端打开：

```powershell
ssh -F NUL -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 127.0.0.1:18505:127.0.0.1:8505 sqw@202.112.113.90
```

在外部浏览器地址栏访问 `http://127.0.0.1:18505/`。服务器本机地址为 `http://127.0.0.1:8505/`。

默认只显示人工裁决与抽检；可切换全部、模型疑问、随机抽检及通过项。人工字段保持 `correct / false_positive / uncertain / pending`，模型字段单独导出。保持原有本地草稿、断线导出和并发保存保护。

## 重复运行

新实验先准备新的输出目录；不要覆写冻结结果：

```bash
python run_check_plan_judge.py prepare --root check_plan_judge_results/new_run
```

将其中的 `judge_prompt.md` 和一份 `inputs/Jxxx.json` 交给独立、无上下文的评审子任务，只允许其写对应答案文件。不向评审提供 manifest、人工标注或其他评审的结果。所有子任务结束后，再导入：

```bash
python run_check_plan_judge.py import --root check_plan_judge_results/new_run --answers /path/to/completed_answers
python run_check_plan_judge.py analyze --root check_plan_judge_results/new_run --output check_plan_judge_results/new_run/analysis_v1
```

入口不会自行访问付费 API。模型推理与本地导入分离；已导入但内容变化的响应会拒绝覆盖。执行日志、模型版本与抽检随机种子应随每轮保存。

本轮未修改生成计划、动作要领、测量结果或已有人工 CSV。不能由一次 judge 结果宣称模型已替代专家，也不能将 84/117 当作模型准确率。
