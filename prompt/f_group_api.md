# F 组 API 批处理

沿用冻结的 15 个定势案例及统一提示词。每个请求独立，只发送提示词和一个案例 JSON；不发送视频、baseline 或人工标签。默认先运行首例，首例请求成功后以 2 并发继续。不是上传 Batch File 的离线作业。

## 运行

```bash
conda activate 4d-humans
cd /home/sqw/Projects/3D-AQA
python run_f_group_api.py \
  --base-url https://ws-05bnmtv66yx74w8r.cn-beijing.maas.aliyuncs.com/compatible-mode/v1 \
  --output-root /home/sqw/Projects/3D-AQA/f_group_experiment/first3_five_students_api \
  --workers 2
```

密钥从环境变量 `DASHSCOPE_API_KEY` 读取，未设置时由终端隐藏输入。不要把密钥写入命令行参数、脚本或 Git。`DASHSCOPE_BASE_URL` 也可替代 `--base-url`。

默认模型 `qwen3.8-max`，temperature=0.2、top_p=0.8、enable_thinking=true、thinking_budget=4000、max_tokens=16384、enable_search=false。采用流式 Chat Completions 接口，保存服务端实际返回的模型名、Token 用量和结束原因。别名不保证模型权重长期不变；严格跨日期复现可在新目录中通过 `--model` 指定官方快照，但不可混用本轮结果。

参数依据：[百炼 Chat API](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions)、[深度思考参数](https://help.aliyun.com/zh/model-studio/deep-thinking)、[模型与快照](https://help.aliyun.com/zh/model-studio/qwen3-8-max)。API 调用按账户计费，不假设网页免费额度同样适用。

## 断点续跑

- 相同命令自动跳过已完成回答，并校验文件哈希；不覆盖或重新生成格式错误的完整回答。
- `--case-id F001` 可单独运行一个案例，允许重复该参数。
- `--dry-run` 只冻结输入和配置，不调用 API，也不需要密钥。
- HTTP 429/500/502/503/504 默认最多重试 2 次，保留每次请求记录。
- 鉴权错误不自动重试；连接中断、截断或拒答也不自动重试，因为可能已计费。确认后用 `--retry-failed` 显式重试，旧尝试仍保留。
- 变更模型或采样设置请指定新的 `--output-root`。运行器源码哈希也已冻结，修改脚本后须使用新目录。
- 同一结果目录仅允许一个批处理进程。程序不会自动删除结果或人工复核记录。

## 输出

默认目录 `/home/sqw/Projects/3D-AQA/f_group_experiment/first3_five_students_api/`：

| 文件 | 内容 |
|---|---|
| `api_config.json` | 请求参数、运行器和输入哈希；不含密钥 |
| `case_index.csv` | F001–F015 与学生、招式的映射 |
| `inputs/`、`F_prompt.md` | 与网页版完全相同的冻结输入和提示词 |
| `responses/F001.json` 等 | 模型最终文本原样保存，不修正模型数值、JSON 或判断 |
| `api_calls/<case_id>/<attempt>/request.json` | 请求正文，无 Authorization 头 |
| `api_calls/<case_id>/<attempt>/raw.sse` | 原始流式事件，包含模型返回的思考文本，与最终答案分开 |
| `api_calls/<case_id>/<attempt>/answer.txt` | 汇总的最终答案文本 |
| `api_calls/<case_id>/<attempt>/metadata.json` | HTTP 状态、耗时、模型名、请求 ID、Token 用量、结束原因 |
| `api_calls/<case_id>/status.json` | 当前状态及所有尝试索引 |
| `api_summary.json` | 已完成、失败数量和最新校验目录 |
| `validation_<id>/validation_summary.json` | 数值与格式错误清单；诊断准确率始终留待人工评价 |
| `validation_<id>/manual_comparison.csv` | F 组与规则 baseline 的逐问题人工对照 |
| `validation_<id>/case_review.csv` | 真问题、双方漏报、多指标综合是否成立的案例级复核 |

“completed”只表示收到了完整最终文本，不代表 JSON 正确、动作分析正确或优于 baseline。校验允许保留部分结果，每次生成新目录，不覆盖已有人工记录。教师区间和原始规则 baseline 均保持不变。
