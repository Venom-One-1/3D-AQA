# 检查计划离线修复报告

首次结果、原始回答、API 记录及冻结缓存均保持不变。本轮没有新增 API 调用。修复通过率不是模型首次成功率，也不是动作诊断准确率。

| 回答 | 首次严格校验 | 修复后 | 查询数 | 修改记录数 | 缓存执行 |
|---|---|---|---:|---:|---|
| [M01_R01](M01_R01/plan.json) | True | True | 10 | [0](M01_R01/repair_log.json) | ok |
| [M01_R02](M01_R02/plan.json) | False | True | 11 | [12](M01_R02/repair_log.json) | ok |
| [M01_R03](M01_R03/plan.json) | True | True | 10 | [0](M01_R03/repair_log.json) | ok |
| [M02_R01](M02_R01/plan.json) | False | True | 10 | [6](M02_R01/repair_log.json) | ok |
| [M02_R02](M02_R02/plan.json) | False | True | 9 | [1](M02_R02/repair_log.json) | ok |
| [M02_R03](M02_R03/plan.json) | False | True | 11 | [3](M02_R03/repair_log.json) | ok |
| [M03_R01](M03_R01/plan.json) | False | True | 13 | [7](M03_R01/repair_log.json) | ok |
| [M03_R02](M03_R02/plan.json) | False | True | 11 | [10](M03_R02/repair_log.json) | ok |
| [M03_R03](M03_R03/plan.json) | False | True | 17 | [1](M03_R03/repair_log.json) | ok |

首次通过 2/9；修复后通过 9/9。
改动 7 份；其中 4 份涉及助手核对后锁定的转义、引用或坐标修正，不能宣称为模型自行修好。
执行 9/9 份计划成功；覆盖 15 个不同学生定势案例。
数值核验 256/256 通过，0 项不可测。

## 修复边界

- 仅提取唯一完整 JSON；不拼接多个回答、不补写缺失字段。
- 未转义字符串由原文哈希和逐条文字锁定，保持解码后的中文内容。
- 合并相同测量时同步引用；同一要领重复引用则合并分量并保留解释。闲置查询删除，不删要领。
- M02_R01 的 left>0 转为 right<0，查询本身不变；不是仅把 left 改名。
- M03_R02 的引用通过关节顺序、坐标和尺度核对后修正；不进行模糊匹配。
- M03_R01 右侧却写 right<0 的矛盾改为 right>0，属于显式坐标修正，不算纯格式修复。

## 尚未证明的部分

几何可计算不等于要领理解正确。脊柱共线与上体正直、膝角与承重、手腕高度与肘部下坠等代理关系仍需语义复核。模型自行给出的角度范围未获得专家验证，不作为学生质量阈值。本轮不生成训练反馈或质量排名。

## 查看结果

[逐句计划对照](plan_report.md) · [语义复核表](plan_review.csv) · [所有修改前后对照](repair_actions.csv) · [汇总 JSON](summary.json)

按重复顺序选择首份可执行计划，未根据学生结果挑选：

- 第 1 式：[M01_R01](M01_R01/plan.json)；[学生 01 测量证据](M01_R01/measurements/cases/01_01.json)
- 第 2 式：[M02_R01](M02_R01/plan.json)；[学生 01 测量证据](M02_R01/measurements/cases/01_02.json)
- 第 3 式：[M03_R01](M03_R01/plan.json)；[学生 01 测量证据](M03_R01/measurements/cases/01_03.json)
