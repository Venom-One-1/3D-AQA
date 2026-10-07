# 计划驱动的选择性训练反馈实验

冻结三式首个有效轮次，复用 15 个结束定势和十位教师数据。只复核最终提示，不重新复核全部检查计划。

新模型：qwen3.8-max-0902；历史 F：qwen3.8-max。输入指标、提示词和筛选策略也不同，因此本轮是开发性对照，不是证明 LLM 优越性或因果效果的受控消融。

格式/引用校验通过：13/15；选中建议 26 条；其中多项证据 19 条、谨慎措辞 15 条。条数不等于正确率。

原始回答直接为 JSON：7；移除外围文字后提取 JSON：3；精确转义已核对的字符串引号：5。提取/转义未改写诊断内容、数值或原始回答；修复规则绑定原始文件哈希。日志在 normalization/，解析结果在 parsed_responses/。

## 检查覆盖

| 招式 | 可用于诊断的要领 | 空缺/不支持 |
|---|---:|---:|
| 1 起势 | 3 | 3 |
| 2 左右野马分鬃 | 3 | 10 |
| 3 白鹤亮翅 | 4 | 16 |

人工否定条目整条排除；不按学生结果换轮次。保留 Head/Wrist 代理及局限，但不把旧计划的 expected_relation 当作质量标准。未发现问题也可能是检查覆盖不足，不能解释为动作合格。

## 个案对照

| 学生 | 招式 | 新建议 | 多证据建议 | 历史 F | Rule 候选 | 对照报告 |
|---|---|---:|---:|---:|---:|---|
| 01 | 1 | 3 | 3 | 3 | 3 | [查看](reports/PF_01_01.md) |
| 01 | 2 | 2 | 1 | 3 | 1 | [查看](reports/PF_01_02.md) |
| 01 | 3 | 1 | 0 | 3 | 1 | [查看](reports/PF_01_03.md) |
| 02 | 1 | 2 | 1 | 2 | 0 | [查看](reports/PF_02_01.md) |
| 02 | 2 | 2 | 1 | 3 | 1 | [查看](reports/PF_02_02.md) |
| 02 | 3 | None | None | 3 | 2 | [查看](reports/PF_02_03.md) |
| 03 | 1 | 2 | 2 | 3 | 1 | [查看](reports/PF_03_01.md) |
| 03 | 2 | 1 | 1 | 3 | 1 | [查看](reports/PF_03_02.md) |
| 03 | 3 | 2 | 2 | 3 | 2 | [查看](reports/PF_03_03.md) |
| 04 | 1 | 2 | 2 | 3 | 3 | [查看](reports/PF_04_01.md) |
| 04 | 2 | 3 | 1 | 3 | 1 | [查看](reports/PF_04_02.md) |
| 04 | 3 | None | None | 3 | 2 | [查看](reports/PF_04_03.md) |
| 10 | 1 | 3 | 3 | 3 | 1 | [查看](reports/PF_10_01.md) |
| 10 | 2 | 1 | 1 | 3 | 0 | [查看](reports/PF_10_02.md) |
| 10 | 3 | 2 | 1 | 3 | 1 | [查看](reports/PF_10_03.md) |

## 下一步复核

先检查新方法选中的建议与总结：correct 表示建议有证据且合理，false_positive 表示误判或越界，uncertain 表示无法判断。填写 manual_feedback_review.csv 的 manual_verdict 与 notes，保留英文枚举。没有建议的案例也要检查总结是否恰当；未标注条目保留 pending。

该 CSV 不把此前 check-plan 对应性标注当作反馈正确性标签。最终建议的 precision、每例有效建议数、重复/矛盾或越界情况应在复核后统计；缺少穷尽式错误标注，暂不能估计召回率。

kind=flagged_summary / flagged_finding 的记录来自未通过自动校验的回答，单独保留供复核，不计入上述合规建议数量。自动校验失败也不等于人工判定误判。

本轮自动校验范围：JSON 结构、要领与查询关联、数字引用、P10-P90 方向、稳定性警告下的措辞等级。自由文本含义、训练价值和总结是否新增判断仍需复核。

## 复现

```bash
python run_plan_feedback_experiment.py prepare --output-root <新目录>
python run_plan_feedback_experiment.py run --output-root <目录> --api-key-file <权限600的密钥文件>
python run_plan_feedback_experiment.py analyze --output-root <目录>
```
run 默认不重试失败调用、不覆盖原始回答；显式 --retry-failed 才会重试。原始 API 回答、请求、耗时与用量在 responses/、api_calls/。输入和审查来源哈希冻结在 experiment_manifest.json。
