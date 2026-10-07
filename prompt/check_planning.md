# 动作要领到检查计划：第一轮开发实验

目的：验证 LLM 能否从动作要领选择可计算的几何证据，而不是根据预先指定的指标和区间做判断。
前三式属于开发集，不能用这一轮宣称跨招式或跨运动泛化。

## 冻结设计

- `qwen3.8-max-0902`；temperature=0.2，top_p=0.8，thinking_budget=4000，max_tokens=16384；关闭搜索。
- 起势、左右野马分鬃、白鹤亮翅各三次独立调用，共九次；无多轮历史，不发送重复编号。
- 只发送已人工确认的末条动作要领、原文短句编号、通用 SMPL 测量契约。
- 不发送学生数据、教师区间、旧规则、19 个默认示例查询、F 组回答或人工复核标签。
- 中文标点只用于保留原文和防止漏项，不等于专家完成了动作要领的语义分解。
- 每句输出直接测量、代理测量或不可测量，注明证据、相关分量及局限；同一查询可以供多句复用。
- 生成时没有学生数值，因此不能生成学生诊断或训练指导。
- 先保留原始回答，再做严格校验；不自动修复 JSON 或补写遗漏项。
- 每式按重复顺序选择首个通过结构校验且模型快照匹配的计划，绝不按学生结果挑选。
- 所有合法计划都在冻结的五名学生、十名教师关节缓存上执行，便于比较重复稳定性。
- 不重跑 tracking、DTW 或修改边界。后续测量数值不回传给本轮计划生成模型。

## 使用

```bash
conda activate 4d-humans
cd /home/sqw/Projects/3D-AQA
python run_check_plan_experiment.py --prepare-only
python run_check_plan_experiment.py --api-key-file /path/to/private-key-file
python run_check_plan_experiment.py --analyze-only
python run_check_plan_probes.py --analysis-root check_plan_experiment/first3_v1/analysis_<id>
```

密钥可从 `DASHSCOPE_API_KEY` 读取。文件方式要求本人所有、权限 600，不保存文件路径或密钥到实验配置。
请求会计费；首个传输失败会停止批量调用。默认不自动重试，明确使用 `--retry-failed` 才重试失败请求。
已完成请求自动跳过，即使回答格式有误也不重新生成。修改源码、输入或设置须换新的输出目录。

默认输出：`check_plan_experiment/first3_v1/`。每次分析新建 `analysis_<timestamp>_<id>/`，不覆盖人工记录。

| 文件 | 内容 |
|---|---|
| `experiment_manifest.json` | 模型、输入、代码哈希和九个独立请求 |
| `inputs/M01.json` 等 | 实际发送的动作要领和工具契约 |
| `planner_prompt.md` | 冻结提示词 |
| `responses/M01_R01.json` 等 | 原始最终回答，文件扩展名不保证内容是合法 JSON |
| `api_calls/` | 请求正文、原始流、响应模型名、用量和传输状态，不保存鉴权头 |
| `analysis_*/validation.json` | 结构、引用和工具合法性检查 |
| `analysis_*/plan_report.md` | 每句要领与生成的测量、理由、局限对照 |
| `analysis_*/plan_review.csv` | 待复核的语义对应关系；初始标签 pending |
| `analysis_*/stability.csv` | 查询集合 Jaccard、支持类型一致率、逐句证据一致率 |
| `analysis_*/M01_R01/measurements/` | 冻结教师与学生缓存执行结果，保留原始教师值 |
| `analysis_*/summary.json` | 首个合法计划选择及调用/执行统计 |
| `analysis_*/geometry_probes.json` | 独立离线核验：刚体变换/尺度不变性、无关点扰动、手腕向上分量、已知角度 |

## 解释边界

“格式合法、查询可执行、每句已引用”不等于动作要领理解正确。稳定地产生相同计划也可能稳定地犯错。
尤其要复核：左右方向、定势与过程的区别、Head/Wrist 代理点、身体坐标与重力坐标、承重与膝角的区别。
实验不计算诊断准确率，不输出新的训练反馈，不用查询数量判断计划优劣。

接口本身的已知角度、局部扰动、整体旋转/平移/缩放等测试属于几何单元测试；它们不等于真实太极拳错误标签。
完成这一轮语义审查后，再冻结提示词，在未使用的招式上验证泛化，最后接入测量后诊断，才能分别评价规划和诊断环节。
