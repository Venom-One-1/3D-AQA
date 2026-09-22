from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .domain import (
    OBSERVABILITY_VALUES,
    TEMPORAL_SCOPES,
    ValidationError,
    extract_json_object,
    validate_query,
)
from .project import compact_metric_catalog, serialize_pretty


SYSTEM_PROMPT = """你是运动学时序查询编译器。你的任务不是猜测视频帧号，而是把太极拳动作要领转换成受约束、可执行的 JSON 查询。

严格要求：
1. 只能使用给定 metric_catalog 中存在的 metric_id，不得发明指标。
2. 根据 SMPL-24 是否能可靠观测该要领，选择 observable、partially_observable、not_observable 或 uncertain。
3. 查询只描述如何在当前招式内部搜索，不直接输出绝对时间戳或帧号。
4. 静态完成姿势用 stable_window，短暂接触或极值用 instant_event，连续变化过程用 transition_interval，整式统计用 whole_move。
5. 掌心、视线、呼吸、肌肉发力和真实足底压力不能由 SMPL-24 可靠测量；相应内容应标为部分可观测或不可观测。
6. 优先使用少量、判别力强的 required 条件，其余设为 supporting。
7. 只输出一个 JSON 对象，不使用 Markdown 代码块，不添加 JSON 之外的文字。
"""


@dataclass(frozen=True)
class LLMConfig:
    api_url: str | None
    api_key: str | None
    model: str | None
    temperature: float = 0.1
    top_p: float = 0.2
    timeout_seconds: float = 90.0

    @property
    def enabled(self) -> bool:
        return bool(self.api_url and self.api_key and self.model)

    def public_status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model": self.model,
            "mode": "openai_compatible" if self.enabled else "manual_import",
        }


def config_from_environment(
    api_url: str | None = None,
    model: str | None = None,
    api_key_env: str = "TQA_LLM_API_KEY",
) -> LLMConfig:
    return LLMConfig(
        api_url=api_url or os.environ.get("TQA_LLM_API_URL"),
        api_key=os.environ.get(api_key_env),
        model=model or os.environ.get("TQA_LLM_MODEL"),
    )


def build_user_prompt(
    task: dict[str, Any], metrics: dict[str, dict[str, Any]]
) -> str:
    preferred_ids = {
        str(check.get("metric_id"))
        for check in task.get("existing_checks", [])
        if check.get("metric_id")
    }
    catalog = compact_metric_catalog(metrics, preferred_ids)
    output_contract = {
        "observability": "observable | partially_observable | not_observable | uncertain",
        "observability_reason": "简短中文说明",
        "temporal_scope": "instant_event | stable_window | transition_interval | whole_move",
        "query": {
            "schema_version": "0.1",
            "scope": "与 temporal_scope 一致",
            "search_region": {
                "unit": "move_progress",
                "start": "0到1",
                "end": "0到1",
            },
            "conditions": [
                {
                    "metric_id": "来自 metric_catalog",
                    "signal": "value | velocity | acceleration",
                    "aggregation": "instant | mean | median | min | max | range | std | slope",
                    "operator": "within_teacher_range | below_teacher_range | above_teacher_range | increasing | decreasing | stable | local_minimum | local_maximum",
                    "role": "required | supporting",
                    "min_duration_seconds": "非负数字或 null",
                }
            ],
            "temporal_relations": [
                {
                    "first_condition": 0,
                    "relation": "before | after | overlaps | during",
                    "second_condition": 1,
                }
            ],
            "selection": {
                "target": "frame | window | whole_move",
                "strategy": "best_score | first_match | last_match | longest_match",
                "representative_frame": "center | start | end | min_motion | max_score | none",
            },
            "rationale": "为什么这些运动学条件能定位该动作要领",
        },
    }
    context = {
        "move": {
            "move_id": task["move_id"],
            "name": task["move_display_name"],
        },
        "technique_id": task["technique_id"],
        "stage_name": task["stage_name"],
        "technique": task["technique"],
        "existing_static_checks_for_reference": task.get("existing_checks", []),
        "known_unsupported_observations": task.get("unsupported_observations", []),
    }
    return (
        "请根据下面的动作要领生成一个候选时序查询。\n\n"
        f"context:\n{serialize_pretty(context)}\n\n"
        f"metric_catalog:\n{serialize_pretty(catalog)}\n\n"
        f"output_contract:\n{serialize_pretty(output_contract)}"
    )


class OpenAICompatibleProvider:
    def __init__(self, config: LLMConfig):
        self.config = config

    def generate(self, prompt: str) -> dict[str, Any]:
        if not self.config.enabled:
            raise RuntimeError("LLM provider is not configured")
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        request = urllib.request.Request(
            str(self.config.api_url),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed ({error.code}): {detail}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(f"LLM request failed: {error}") from error

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("LLM response does not use the expected chat-completions shape") from error
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        return extract_json_object(str(content))


def validate_candidate(candidate: dict[str, Any], metric_ids: set[str]) -> None:
    observability = candidate.get("observability")
    if observability not in OBSERVABILITY_VALUES:
        raise ValidationError(f"Invalid candidate observability: {observability!r}")
    scope = candidate.get("temporal_scope")
    if scope not in TEMPORAL_SCOPES:
        raise ValidationError(f"Invalid candidate temporal_scope: {scope!r}")
    query = candidate.get("query")
    if observability != "not_observable":
        validate_query(query, metric_ids)
        if query.get("scope") != scope:
            raise ValidationError("Candidate temporal_scope and query.scope differ")
    elif query is not None:
        validate_query(query, metric_ids)
