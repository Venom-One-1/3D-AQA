"""Execute constrained temporal key-pose queries on SMPL-24 joint sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .pose_measurements import MeasurementQuery, measure_pose


EPS = 1e-10
SUPPORTED_OPERATORS = {
    "increasing", "decreasing", "stable", "local_minimum", "local_maximum",
    "positive", "negative",
}
SUPPORTED_SCOPES = {"instant_event", "stable_window", "transition_interval"}
SUPPORTED_REDUCTIONS = {"mean", "median", "min", "max"}


@dataclass(frozen=True)
class MetricSignals:
    raw: np.ndarray
    smoothed: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


def _exact_fields(value: dict[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} must contain exactly {sorted(expected)}")


def validate_temporal_plan(plan: dict[str, Any]) -> None:
    """Validate the executable plan without accepting arbitrary expressions."""
    _exact_fields(
        plan,
        {"schema_version", "plan_id", "video_id", "move", "execution",
         "measurements", "metrics", "events", "notes"},
        "plan",
    )
    if plan["schema_version"] != "1.0" or not isinstance(plan["plan_id"], str):
        raise ValueError("Unsupported temporal plan version or invalid plan_id")
    _exact_fields(
        plan["move"],
        {"move_id", "move_name", "start_seconds", "end_seconds", "technique"},
        "move",
    )
    move = plan["move"]
    if (type(move["move_id"]) is not int or move["move_id"] < 1
            or not 0 <= move["start_seconds"] < move["end_seconds"]):
        raise ValueError("Invalid move interval")
    _exact_fields(
        plan["execution"],
        {"smoothing_seconds", "derivative_window_seconds", "bridge_gap_seconds",
         "minimum_valid_ratio", "sensitivity_smoothing_seconds",
         "sensitivity_threshold_multipliers"},
        "execution",
    )
    execution = plan["execution"]
    for key in ("smoothing_seconds", "derivative_window_seconds", "bridge_gap_seconds"):
        if not isinstance(execution[key], (int, float)) or execution[key] <= 0:
            raise ValueError(f"execution.{key} must be positive")
    if not 0 < execution["minimum_valid_ratio"] <= 1:
        raise ValueError("execution.minimum_valid_ratio must be in (0,1]")
    for key in ("sensitivity_smoothing_seconds", "sensitivity_threshold_multipliers"):
        values = execution[key]
        if not isinstance(values, list) or not values or any(
            not isinstance(x, (int, float)) or x <= 0 for x in values
        ):
            raise ValueError(f"execution.{key} must be a nonempty positive list")

    queries: dict[str, MeasurementQuery] = {}
    if not isinstance(plan["measurements"], list) or not plan["measurements"]:
        raise ValueError("measurements must be a nonempty list")
    for value in plan["measurements"]:
        query = MeasurementQuery.from_dict(value)
        if query.query_id in queries:
            raise ValueError(f"Duplicate measurement query: {query.query_id}")
        queries[query.query_id] = query

    metrics: set[str] = set()
    if not isinstance(plan["metrics"], list) or not plan["metrics"]:
        raise ValueError("metrics must be a nonempty list")
    for metric in plan["metrics"]:
        _exact_fields(metric, {"metric_id", "sources", "reduction", "description"}, "metric")
        metric_id = metric["metric_id"]
        if not isinstance(metric_id, str) or not metric_id or metric_id in metrics:
            raise ValueError(f"Invalid or duplicate metric_id: {metric_id!r}")
        metrics.add(metric_id)
        if metric["reduction"] not in SUPPORTED_REDUCTIONS:
            raise ValueError(f"Unsupported reduction: {metric['reduction']}")
        if not isinstance(metric["sources"], list) or not metric["sources"]:
            raise ValueError(f"Metric {metric_id} has no sources")
        for source in metric["sources"]:
            _exact_fields(source, {"query_id", "component"}, "metric source")
            query = queries.get(source["query_id"])
            if query is None or source["component"] not in query.definition()["components"]:
                raise ValueError(f"Metric {metric_id} uses an unknown query component")

    event_ids: set[str] = set()
    if not isinstance(plan["events"], list) or not plan["events"]:
        raise ValueError("events must be a nonempty list")
    for event in plan["events"]:
        _exact_fields(
            event,
            {"event_id", "label", "technique", "query", "after_event",
             "terminal_extension", "auxiliary_reference_image"},
            "event",
        )
        event_id = event["event_id"]
        if not isinstance(event_id, str) or not event_id or event_id in event_ids:
            raise ValueError(f"Invalid or duplicate event_id: {event_id!r}")
        query = event["query"]
        _exact_fields(
            query,
            {"schema_version", "scope", "search_region", "conditions",
             "temporal_relations", "selection", "rationale"},
            "event query",
        )
        if query["schema_version"] != "0.1" or query["scope"] not in SUPPORTED_SCOPES:
            raise ValueError(f"Unsupported query schema or scope for {event_id}")
        _exact_fields(query["search_region"], {"unit", "start", "end"}, "search_region")
        region = query["search_region"]
        if (region["unit"] != "move_progress"
                or not 0 <= region["start"] <= region["end"] <= 1):
            raise ValueError(f"Invalid search region for {event_id}")
        if not isinstance(query["conditions"], list) or not query["conditions"]:
            raise ValueError(f"Event {event_id} has no conditions")
        condition_ids: set[str] = set()
        required_count = 0
        for condition in query["conditions"]:
            _exact_fields(
                condition,
                {"condition_id", "metric_id", "signal", "aggregation", "operator",
                 "role", "min_duration_seconds", "score_threshold", "weight"},
                "condition",
            )
            cid = condition["condition_id"]
            if not isinstance(cid, str) or not cid or cid in condition_ids:
                raise ValueError(f"Invalid condition_id in {event_id}")
            condition_ids.add(cid)
            if condition["metric_id"] not in metrics:
                raise ValueError(f"Unknown metric in {event_id}: {condition['metric_id']}")
            if condition["signal"] not in {"value", "velocity", "acceleration"}:
                raise ValueError(f"Unsupported signal in {event_id}")
            if condition["aggregation"] != "instant":
                raise ValueError("Executable v1 currently requires aggregation='instant'")
            if condition["operator"] not in SUPPORTED_OPERATORS:
                raise ValueError(f"Unsupported operator in {event_id}")
            if condition["role"] not in {"required", "supporting"}:
                raise ValueError(f"Invalid condition role in {event_id}")
            required_count += condition["role"] == "required"
            if (not isinstance(condition["min_duration_seconds"], (int, float))
                    or condition["min_duration_seconds"] < 0
                    or not isinstance(condition["score_threshold"], (int, float))
                    or not 0 <= condition["score_threshold"] <= 1
                    or not isinstance(condition["weight"], (int, float))
                    or condition["weight"] <= 0):
                raise ValueError(f"Invalid condition parameters in {event_id}")
        if not required_count:
            raise ValueError(f"Event {event_id} needs at least one required condition")
        if query["temporal_relations"]:
            raise ValueError("Executable v1 does not yet accept intra-event temporal_relations")
        _exact_fields(query["selection"], {"target", "strategy", "representative_frame"}, "selection")
        selection = query["selection"]
        if selection["target"] not in {"frame", "window"}:
            raise ValueError(f"Unsupported selection target in {event_id}")
        if selection["strategy"] not in {"best_score", "first_match", "last_match", "longest_match"}:
            raise ValueError(f"Unsupported selection strategy in {event_id}")
        if selection["representative_frame"] not in {"center", "start", "end", "max_score"}:
            raise ValueError(f"Unsupported representative strategy in {event_id}")
        dependency = event["after_event"]
        if dependency is not None:
            _exact_fields(dependency, {"event_id", "anchor", "offset_seconds"}, "after_event")
            if dependency["event_id"] not in event_ids:
                raise ValueError(f"Event {event_id} depends on an event that is not earlier in the plan")
            if dependency["anchor"] not in {"start", "representative", "end"}:
                raise ValueError(f"Invalid dependency anchor in {event_id}")
            if dependency["offset_seconds"] < 0:
                raise ValueError(f"Negative dependency offset in {event_id}")
        if event["terminal_extension"] not in {"none", "move_end"}:
            raise ValueError(f"Invalid terminal_extension in {event_id}")
        if event["terminal_extension"] == "move_end" and region["end"] != 1:
            raise ValueError("move_end extension requires search_region.end=1")
        event_ids.add(event_id)


def rolling_median(values: np.ndarray, times: np.ndarray, window_seconds: float) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    radius = window_seconds / 2.0
    for index, center in enumerate(times):
        selected = (np.abs(times - center) <= radius + EPS) & np.isfinite(values)
        if selected.any():
            result[index] = float(np.median(values[selected]))
    return result


def rolling_slope(values: np.ndarray, times: np.ndarray, window_seconds: float) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    radius = window_seconds / 2.0
    for index, center in enumerate(times):
        selected = (np.abs(times - center) <= radius + EPS) & np.isfinite(values)
        if selected.sum() < 3:
            continue
        local_t = times[selected]
        local_v = values[selected]
        centered_t = local_t - local_t.mean()
        denominator = float(np.sum(centered_t * centered_t))
        if denominator > EPS:
            result[index] = float(np.sum(centered_t * (local_v - local_v.mean())) / denominator)
    return result


def compute_raw_metrics(
    plan: dict[str, Any],
    joints: np.ndarray,
    frame_ids: np.ndarray,
    tracking_status: np.ndarray,
) -> dict[str, np.ndarray]:
    validate_temporal_plan(plan)
    measured: dict[str, Any] = {}
    for value in plan["measurements"]:
        query = MeasurementQuery.from_dict(value)
        measured[query.query_id] = measure_pose(
            joints, query, frame_ids=frame_ids, tracking_status=tracking_status,
        )
    metrics: dict[str, np.ndarray] = {}
    for metric in plan["metrics"]:
        sources = []
        for source in metric["sources"]:
            result = measured[source["query_id"]]
            component = result.components.index(source["component"])
            sources.append(result.values[:, component])
        values = np.stack(sources, axis=1)
        valid = np.isfinite(values).all(axis=1)
        reduced = np.full(len(values), np.nan)
        reducer = getattr(np, metric["reduction"])
        reduced[valid] = reducer(values[valid], axis=1)
        metrics[metric["metric_id"]] = reduced
    return metrics


def prepare_metric_signals(
    raw_metrics: dict[str, np.ndarray],
    times: np.ndarray,
    *,
    smoothing_seconds: float,
    derivative_window_seconds: float,
) -> dict[str, MetricSignals]:
    prepared = {}
    for metric_id, raw in raw_metrics.items():
        smoothed = rolling_median(raw, times, smoothing_seconds)
        smoothed[~np.isfinite(raw)] = np.nan
        velocity = rolling_slope(smoothed, times, derivative_window_seconds)
        velocity[~np.isfinite(raw)] = np.nan
        acceleration = rolling_slope(velocity, times, derivative_window_seconds)
        acceleration[~np.isfinite(raw)] = np.nan
        prepared[metric_id] = MetricSignals(raw.copy(), smoothed, velocity, acceleration)
    return prepared


def _robust_high(
    values: np.ndarray, search: np.ndarray, *, constant_score: float = 0.0,
) -> np.ndarray:
    selected = values[search & np.isfinite(values)]
    result = np.full(len(values), np.nan)
    if not len(selected):
        return result
    low, high = np.percentile(selected, [5, 95])
    valid = np.isfinite(values)
    if high - low <= EPS:
        result[valid] = constant_score
    else:
        result[valid] = np.clip((values[valid] - low) / (high - low), 0, 1)
    return result


def _operator_score(values: np.ndarray, operator: str, search: np.ndarray) -> np.ndarray:
    if operator in {"positive", "negative"}:
        valid = np.isfinite(values)
        result = np.full(len(values), np.nan)
        result[valid] = (values[valid] > EPS if operator == "positive"
                         else values[valid] < -EPS).astype(float)
        return result
    if operator == "local_maximum":
        return _robust_high(values, search)
    if operator == "local_minimum":
        return _robust_high(-values, search)
    if operator == "increasing":
        return _robust_high(np.maximum(values, 0), search)
    if operator == "decreasing":
        return _robust_high(np.maximum(-values, 0), search)
    if operator == "stable":
        return _robust_high(-np.abs(values), search, constant_score=1.0)
    raise ValueError(f"Unsupported operator: {operator}")


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    runs: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for current_value in indices[1:]:
        current = int(current_value)
        if current != previous + 1:
            runs.append((start, previous))
            start = current
        previous = current
    runs.append((start, previous))
    return runs


def _remove_short(mask: np.ndarray, minimum_frames: int) -> np.ndarray:
    result = mask.copy()
    for start, end in _runs(result):
        if end - start + 1 < minimum_frames:
            result[start:end + 1] = False
    return result


def _bridge(mask: np.ndarray, maximum_gap_frames: int) -> np.ndarray:
    result = mask.copy()
    runs = _runs(result)
    for (_, previous_end), (next_start, _) in zip(runs, runs[1:]):
        if next_start - previous_end - 1 <= maximum_gap_frames:
            result[previous_end + 1:next_start] = True
    return result


def _representative(start: int, end: int, score: np.ndarray, strategy: str) -> int:
    if strategy == "start":
        return start
    if strategy == "end":
        return end
    if strategy == "center":
        return (start + end) // 2
    local = score[start:end + 1]
    finite = np.flatnonzero(np.isfinite(local))
    return start + int(finite[np.argmax(local[finite])]) if len(finite) else (start + end) // 2


def locate_events(
    plan: dict[str, Any],
    metrics: dict[str, MetricSignals],
    frame_ids: np.ndarray,
    times: np.ndarray,
    fps: float,
    *,
    threshold_multiplier: float = 1.0,
) -> dict[str, dict[str, Any]]:
    """Locate event windows in plan order and retain every diagnostic score."""
    validate_temporal_plan(plan)
    if len(frame_ids) != len(times) or not len(times) or fps <= 0:
        raise ValueError("Invalid temporal axis")
    execution = plan["execution"]
    move_start = float(plan["move"]["start_seconds"])
    move_end = float(plan["move"]["end_seconds"])
    duration = move_end - move_start
    outputs: dict[str, dict[str, Any]] = {}
    for event in plan["events"]:
        event_id = event["event_id"]
        region = event["query"]["search_region"]
        start_time = move_start + region["start"] * duration
        end_time = move_start + region["end"] * duration
        dependency = event["after_event"]
        if dependency is not None:
            prior = outputs[dependency["event_id"]]["result"]
            key = f"{dependency['anchor']}_time_seconds"
            if prior["status"] != "located" or prior[key] is None:
                outputs[event_id] = _empty_detection(event, len(times), "unresolved_dependency")
                continue
            start_time = max(start_time, prior[key] + dependency["offset_seconds"])
        search = (times >= start_time - EPS) & (times <= end_time + EPS)
        condition_scores: dict[str, np.ndarray] = {}
        condition_masks: dict[str, np.ndarray] = {}
        required_masks = []
        required_valid_masks = []
        weighted_scores = []
        weights = []
        for condition in event["query"]["conditions"]:
            series = metrics[condition["metric_id"]]
            values = {"value": series.smoothed, "velocity": series.velocity,
                      "acceleration": series.acceleration}[condition["signal"]]
            score = _operator_score(values, condition["operator"], search)
            threshold = min(1.0, condition["score_threshold"] * threshold_multiplier)
            mask = search & np.isfinite(score) & (score >= threshold)
            minimum = max(1, int(np.ceil(condition["min_duration_seconds"] * fps - EPS)))
            mask = _remove_short(mask, minimum)
            condition_scores[condition["condition_id"]] = score
            condition_masks[condition["condition_id"]] = mask
            weighted_scores.append(score * condition["weight"])
            weights.append(condition["weight"])
            if condition["role"] == "required":
                required_masks.append(mask)
                required_valid_masks.append(search & np.isfinite(score))
        candidate = search.copy()
        for mask in required_masks:
            candidate &= mask
        candidate = _bridge(candidate, int(round(execution["bridge_gap_seconds"] * fps)))
        for valid_mask in required_valid_masks:
            candidate &= valid_mask
        minimum_event_frames = max(
            1,
            int(np.ceil(max(
                condition["min_duration_seconds"]
                for condition in event["query"]["conditions"]
                if condition["role"] == "required"
            ) * fps - EPS)),
        )
        candidate = _remove_short(candidate, minimum_event_frames)
        stack = np.stack(weighted_scores)
        finite_count = np.sum(np.isfinite(stack), axis=0)
        composite = np.nansum(stack, axis=0) / sum(weights)
        composite[finite_count == 0] = np.nan
        available_search = search.copy()
        for valid_mask in required_valid_masks:
            available_search &= valid_mask
        valid_ratio = float(available_search.sum() / max(1, search.sum()))
        runs = _runs(candidate)
        if not runs or valid_ratio < execution["minimum_valid_ratio"]:
            reason = ("insufficient_valid_data" if valid_ratio < execution["minimum_valid_ratio"]
                      else "no_matching_interval")
            outputs[event_id] = {
                **_empty_detection(event, len(times), reason),
                "search_mask": search,
                "candidate_mask": candidate,
                "score": composite,
                "condition_scores": condition_scores,
                "condition_masks": condition_masks,
            }
            outputs[event_id]["result"]["valid_search_ratio"] = valid_ratio
            continue
        strategy = event["query"]["selection"]["strategy"]
        if strategy == "first_match":
            selected = runs[0]
        elif strategy == "last_match":
            selected = runs[-1]
        elif strategy == "longest_match":
            selected = max(
                runs,
                key=lambda item: (item[1] - item[0] + 1,
                                  np.nanmean(composite[item[0]:item[1] + 1])),
            )
        else:
            selected = max(runs, key=lambda item: np.nanmean(composite[item[0]:item[1] + 1]))
        detected_start, detected_end = selected
        interval_start, interval_end = selected
        if event["terminal_extension"] == "move_end":
            endpoint = np.flatnonzero(np.abs(times - move_end) < 0.51 / fps)
            if len(endpoint) != 1 or not available_search[endpoint[0]]:
                missing = _empty_detection(event, len(times), "missing_valid_move_end")
                missing.update(search_mask=search, candidate_mask=candidate, score=composite,
                               condition_scores=condition_scores, condition_masks=condition_masks)
                outputs[event_id] = missing
                continue
            interval_end = int(endpoint[0])
        representative = _representative(
            interval_start, interval_end, composite,
            event["query"]["selection"]["representative_frame"],
        )
        result = {
            "event_id": event_id,
            "label": event["label"],
            "technique": event["technique"],
            "scope": event["query"]["scope"],
            "status": "located",
            "reason": None,
            "interval_start_source_frame_0based": int(frame_ids[interval_start]),
            "interval_end_source_frame_0based": int(frame_ids[interval_end]),
            "representative_source_frame_0based": int(frame_ids[representative]),
            "detected_match_start_source_frame_0based": int(frame_ids[detected_start]),
            "detected_match_end_source_frame_0based": int(frame_ids[detected_end]),
            "detected_match_start_time_seconds": float(times[detected_start]),
            "detected_match_end_time_seconds": float(times[detected_end]),
            "start_time_seconds": float(times[interval_start]),
            "end_time_seconds": float(times[interval_end]),
            "representative_time_seconds": float(times[representative]),
            "representative_score": float(composite[representative]),
            "valid_search_ratio": valid_ratio,
            "terminal_extension": event["terminal_extension"],
            "representative_source": (
                "known_move_end" if event["terminal_extension"] == "move_end"
                and event["query"]["selection"]["representative_frame"] == "end"
                else "signal_query"
            ),
            "selection_rationale": event["query"]["rationale"],
        }
        outputs[event_id] = {
            "result": result,
            "search_mask": search,
            "candidate_mask": candidate,
            "score": composite,
            "condition_scores": condition_scores,
            "condition_masks": condition_masks,
        }
    return outputs


def _empty_detection(event: dict[str, Any], length: int, reason: str) -> dict[str, Any]:
    result = {
        "event_id": event["event_id"], "label": event["label"],
        "technique": event["technique"], "scope": event["query"]["scope"],
        "status": "not_located", "reason": reason,
        "interval_start_source_frame_0based": None,
        "interval_end_source_frame_0based": None,
        "representative_source_frame_0based": None,
        "detected_match_start_source_frame_0based": None,
        "detected_match_end_source_frame_0based": None,
        "detected_match_start_time_seconds": None, "detected_match_end_time_seconds": None,
        "start_time_seconds": None, "end_time_seconds": None,
        "representative_time_seconds": None, "representative_score": None,
        "valid_search_ratio": 0.0, "terminal_extension": event["terminal_extension"],
        "representative_source": None,
        "selection_rationale": event["query"]["rationale"],
    }
    return {
        "result": result,
        "search_mask": np.zeros(length, dtype=bool),
        "candidate_mask": np.zeros(length, dtype=bool),
        "score": np.full(length, np.nan),
        "condition_scores": {}, "condition_masks": {},
    }
