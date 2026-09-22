from __future__ import annotations

import copy
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TEMPORAL_SCOPES = {
    "instant_event",
    "stable_window",
    "transition_interval",
    "whole_move",
}
OBSERVABILITY_VALUES = {
    "observable",
    "partially_observable",
    "not_observable",
    "uncertain",
}
REVIEW_STATUSES = {"unannotated", "draft", "reviewed"}
SIGNALS = {"value", "velocity", "acceleration"}
AGGREGATIONS = {
    "instant",
    "mean",
    "median",
    "min",
    "max",
    "range",
    "std",
    "slope",
}
OPERATORS = {
    "within_teacher_range",
    "below_teacher_range",
    "above_teacher_range",
    "increasing",
    "decreasing",
    "stable",
    "local_minimum",
    "local_maximum",
}
CONDITION_ROLES = {"required", "supporting"}
SELECTION_TARGETS = {"frame", "window", "whole_move"}
SELECTION_STRATEGIES = {
    "best_score",
    "first_match",
    "last_match",
    "longest_match",
}
REPRESENTATIVE_STRATEGIES = {
    "center",
    "start",
    "end",
    "min_motion",
    "max_score",
    "none",
}
RELATIONS = {"before", "after", "overlaps", "during"}


class ValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ProjectPaths:
    config: Path
    video: Path
    frames: Path
    rules: Path
    annotations: Path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValidationError(f"Expected a JSON object in {path}")
    return value


def resolve_project_paths(config_path: Path, config: dict[str, Any]) -> ProjectPaths:
    base = config_path.resolve().parent

    def resolve(value: str) -> Path:
        candidate = Path(value).expanduser()
        return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()

    return ProjectPaths(
        config=config_path.resolve(),
        video=resolve(str(config["video_path"])),
        frames=resolve(str(config["frames_dir"])),
        rules=resolve(str(config["rules_path"])),
        annotations=resolve(str(config["annotations_path"])),
    )


def frame_to_seconds(frame_index: int, sample_fps: float) -> float:
    if sample_fps <= 0:
        raise ValidationError("sample_fps must be positive")
    if frame_index < 0:
        raise ValidationError("frame index must be non-negative")
    return frame_index / sample_fps


def seconds_to_frame(seconds: float, sample_fps: float) -> int:
    if sample_fps <= 0:
        raise ValidationError("sample_fps must be positive")
    if seconds < 0:
        raise ValidationError("seconds must be non-negative")
    return int(math.floor(seconds * sample_fps + 0.5))


def _check_enum(value: Any, allowed: set[str], field: str) -> None:
    if value not in allowed:
        raise ValidationError(f"{field} must be one of {sorted(allowed)}, got {value!r}")


def validate_query(query: dict[str, Any], metric_ids: set[str]) -> None:
    if not isinstance(query, dict):
        raise ValidationError("query must be an object")
    if query.get("schema_version") != "0.1":
        raise ValidationError("query.schema_version must be '0.1'")
    _check_enum(query.get("scope"), TEMPORAL_SCOPES, "query.scope")

    region = query.get("search_region")
    if not isinstance(region, dict) or region.get("unit") != "move_progress":
        raise ValidationError("search_region.unit must be 'move_progress'")
    start = region.get("start")
    end = region.get("end")
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        raise ValidationError("search_region start/end must be numbers")
    if not 0.0 <= float(start) <= float(end) <= 1.0:
        raise ValidationError("search_region must satisfy 0 <= start <= end <= 1")

    conditions = query.get("conditions")
    if not isinstance(conditions, list):
        raise ValidationError("query.conditions must be a list")
    for index, condition in enumerate(conditions):
        if not isinstance(condition, dict):
            raise ValidationError(f"condition {index} must be an object")
        metric_id = condition.get("metric_id")
        if metric_id not in metric_ids:
            raise ValidationError(f"condition {index} uses unknown metric_id {metric_id!r}")
        _check_enum(condition.get("signal"), SIGNALS, f"condition {index}.signal")
        _check_enum(
            condition.get("aggregation"),
            AGGREGATIONS,
            f"condition {index}.aggregation",
        )
        _check_enum(condition.get("operator"), OPERATORS, f"condition {index}.operator")
        _check_enum(condition.get("role"), CONDITION_ROLES, f"condition {index}.role")
        duration = condition.get("min_duration_seconds")
        if duration is not None and (
            not isinstance(duration, (int, float)) or float(duration) < 0
        ):
            raise ValidationError(
                f"condition {index}.min_duration_seconds must be null or non-negative"
            )

    relations = query.get("temporal_relations", [])
    if not isinstance(relations, list):
        raise ValidationError("query.temporal_relations must be a list")
    for index, relation in enumerate(relations):
        if not isinstance(relation, dict):
            raise ValidationError(f"temporal relation {index} must be an object")
        _check_enum(relation.get("relation"), RELATIONS, f"relation {index}.relation")
        first = relation.get("first_condition")
        second = relation.get("second_condition")
        if not isinstance(first, int) or not isinstance(second, int):
            raise ValidationError(f"relation {index} condition indexes must be integers")
        if not 0 <= first < len(conditions) or not 0 <= second < len(conditions):
            raise ValidationError(f"relation {index} references a missing condition")

    selection = query.get("selection")
    if not isinstance(selection, dict):
        raise ValidationError("query.selection must be an object")
    _check_enum(selection.get("target"), SELECTION_TARGETS, "selection.target")
    _check_enum(selection.get("strategy"), SELECTION_STRATEGIES, "selection.strategy")
    _check_enum(
        selection.get("representative_frame"),
        REPRESENTATIVE_STRATEGIES,
        "selection.representative_frame",
    )


def blank_query(scope: str = "stable_window") -> dict[str, Any]:
    _check_enum(scope, TEMPORAL_SCOPES, "scope")
    target = "whole_move" if scope == "whole_move" else (
        "frame" if scope == "instant_event" else "window"
    )
    representative = "none" if target == "whole_move" else "center"
    return {
        "schema_version": "0.1",
        "scope": scope,
        "search_region": {"unit": "move_progress", "start": 0.0, "end": 1.0},
        "conditions": [],
        "temporal_relations": [],
        "selection": {
            "target": target,
            "strategy": "best_score",
            "representative_frame": representative,
        },
        "rationale": "",
    }


def default_annotation(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "technique_id": task["technique_id"],
        "observability": "uncertain",
        "observability_reason": "",
        "temporal_scope": "stable_window",
        "gold": {
            "start_frame_5fps": None,
            "end_frame_5fps": None,
            "representative_frame_5fps": None,
        },
        "llm_candidate": None,
        "query": blank_query(),
        "review_status": "unannotated",
        "annotator_notes": "",
        "audit": {
            "llm_generated_at": None,
            "first_saved_at": None,
            "last_saved_at": None,
            "llm_candidate_accepted_without_edit": None,
        },
    }


def validate_annotation(
    annotation: dict[str, Any],
    task: dict[str, Any],
    metric_ids: set[str],
) -> None:
    if annotation.get("technique_id") != task["technique_id"]:
        raise ValidationError("technique_id does not match the current task")
    _check_enum(annotation.get("observability"), OBSERVABILITY_VALUES, "observability")
    _check_enum(annotation.get("temporal_scope"), TEMPORAL_SCOPES, "temporal_scope")
    _check_enum(annotation.get("review_status"), REVIEW_STATUSES, "review_status")

    gold = annotation.get("gold")
    if not isinstance(gold, dict):
        raise ValidationError("gold must be an object")
    frame_keys = (
        "start_frame_5fps",
        "end_frame_5fps",
        "representative_frame_5fps",
    )
    for key in frame_keys:
        value = gold.get(key)
        if value is not None and (not isinstance(value, int) or value < 0):
            raise ValidationError(f"gold.{key} must be null or a non-negative integer")

    move_start = int(task["move_start_frame_5fps"])
    move_end = int(task["move_end_frame_5fps"])
    for key in frame_keys:
        value = gold.get(key)
        if value is not None and not move_start <= value <= move_end:
            raise ValidationError(
                f"gold.{key}={value} falls outside move range [{move_start}, {move_end}]"
            )
    start = gold.get("start_frame_5fps")
    end = gold.get("end_frame_5fps")
    representative = gold.get("representative_frame_5fps")
    if start is not None and end is not None and start > end:
        raise ValidationError("gold start frame must not exceed end frame")
    if representative is not None and start is not None and representative < start:
        raise ValidationError("representative frame must not precede gold start")
    if representative is not None and end is not None and representative > end:
        raise ValidationError("representative frame must not follow gold end")

    query = annotation.get("query")
    if annotation["observability"] != "not_observable":
        validate_query(query, metric_ids)


def deep_copy_json(value: Any) -> Any:
    return copy.deepcopy(value)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        if start < 0:
            raise ValidationError("LLM response does not contain a JSON object")
        depth = 0
        in_string = False
        escaped = False
        end = None
        for index, char in enumerate(stripped[start:], start=start):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            raise ValidationError("LLM response contains incomplete JSON")
        try:
            value = json.loads(stripped[start:end])
        except json.JSONDecodeError as error:
            raise ValidationError(f"Invalid JSON in LLM response: {error}") from error
    if not isinstance(value, dict):
        raise ValidationError("LLM response must be a JSON object")
    return value
