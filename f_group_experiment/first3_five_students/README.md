# F 组：结束定势数值证据实验

已准备 15 份真实定势输入，覆盖 5 位学生的前三式、共 130 项指标。目前等待 Qwen 回答，尚不能报告模型准确率。

## 在百炼网页运行

1. 每案例新建独立会话，选择相同 Qwen3.8-max 模型及参数，关闭搜索。
2. 粘贴 [统一提示词](F_prompt.md)，只上传 inputs 内对应的一个 JSON。不上传报告、baseline、评价文件、图片或视频。
3. 保持所有案例参数一致，在 run_log.csv 记录实际模型名称、参数、时间及会话信息。使用界面支持的低随机性设置，不在看到个别结果后单独调参。
4. 把最终 JSON 回答按编号保存至 responses，保留第一次完整回答。若格式失败，记录后另建目录重试，不静默替换。
5. 全部完成后运行校验命令，程序保留原始回答。

| 编号 | 学生 | 招式 | 指标数 | 上传文件 | 回答保存位置 |
|---|---|---|---|---|---|
| F001 | 01 | qishi | 9 | [F001.json](inputs/F001.json) | responses/F001.json |
| F002 | 01 | yemafenzong | 8 | [F002.json](inputs/F002.json) | responses/F002.json |
| F003 | 01 | baiheliangchi | 9 | [F003.json](inputs/F003.json) | responses/F003.json |
| F004 | 02 | qishi | 9 | [F004.json](inputs/F004.json) | responses/F004.json |
| F005 | 02 | yemafenzong | 8 | [F005.json](inputs/F005.json) | responses/F005.json |
| F006 | 02 | baiheliangchi | 9 | [F006.json](inputs/F006.json) | responses/F006.json |
| F007 | 03 | qishi | 9 | [F007.json](inputs/F007.json) | responses/F007.json |
| F008 | 03 | yemafenzong | 8 | [F008.json](inputs/F008.json) | responses/F008.json |
| F009 | 03 | baiheliangchi | 9 | [F009.json](inputs/F009.json) | responses/F009.json |
| F010 | 04 | qishi | 9 | [F010.json](inputs/F010.json) | responses/F010.json |
| F011 | 04 | yemafenzong | 8 | [F011.json](inputs/F011.json) | responses/F011.json |
| F012 | 04 | baiheliangchi | 9 | [F012.json](inputs/F012.json) | responses/F012.json |
| F013 | 10 | qishi | 9 | [F013.json](inputs/F013.json) | responses/F013.json |
| F014 | 10 | yemafenzong | 8 | [F014.json](inputs/F014.json) | responses/F014.json |
| F015 | 10 | baiheliangchi | 9 | [F015.json](inputs/F015.json) | responses/F015.json |

## 校验与人工比较

```bash
conda activate 4d-humans
cd /home/sqw/Projects/3D-AQA
python run_f_group_experiment.py validate --experiment-root /home/sqw/Projects/3D-AQA/f_group_experiment/first3_five_students
```

自动核验案例、指标覆盖、数值引用、区间比较及证据 ID；动作诊断与指导正确性仍需人工判断。
输出 validation/ 内的问题清单、数值检查及模型/baseline 人工对照表。复查时用 --output-root 指定新目录，保留已有人工记录。
manual_comparison.csv 中 diagnosis_verdict / guidance_verdict 填 correct / false_positive / uncertain；multi_metric_supported 填 yes / no / not_applicable。
case_review.csv 中独立列出真实问题及双方漏报，记录多指标解释是否成立。不能仅按引用了多个指标就认定有效综合。
规则 baseline 的暂缓判断与模型的诊断错误分别记录，不强行一对一匹配不同数量的建议。
主要观察有依据建议比例、无依据判断、真实问题覆盖和人工确认的多指标综合。15 例仅用于预实验。
已有复核 UI 的标签评价规则建议或暂缓决定，不能直接作为模型诊断的完整真值，尤其不能用于计算漏报。

## 输入约定

输入包含完整动作要领、对应的所有指标、定义/单位/坐标系/归一化方式、学生中位数和中心值、教师双区间及十个教师原始值和窗口信息。
模型输入不含学生真实编号、人工排名、baseline 判断或建议、图片路径。case_index.csv 仅供研究者追溯。
沿用现有数值及区间；不附带规则系统的稳定性阈值和判定结果。
相同 JSON 加视频可用于后续配对实验；F 组单独不能证明模型是否使用视觉。
