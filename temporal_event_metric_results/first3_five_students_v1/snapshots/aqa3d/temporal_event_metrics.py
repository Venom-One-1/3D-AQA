"""Measured geometry at frozen temporal events, without action-quality labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .endpoint_metrics import ANGLE_IDS, LABELS, MOVE_METRICS
from .pose_measurements import MeasurementQuery, measure_pose


LIMITATIONS = [
    "手腕、头关节是手部和额头位置的代理，不是掌心或额头的精确位置。",
    "torso_v1 的上下方向随躯干转动，不是重力竖直方向。",
    "关节几何不能证明掌心朝向、实际承重、视线、肌肉松紧或动作意图。",
    "教师区间是样本描述；超出区间不等于动作错误，落入区间不等于合格。",
    "短窗变化可能来自自然运动、重建误差或对齐误差，不能直接归因。",
    "过程变化量依赖迁移的事件起止点，不直接评价节奏或协调性。",
]


@dataclass(frozen=True)
class EventMetric:
    metric_id: str
    label: str
    aspect: str
    query: MeasurementQuery
    component: int = 0
    absolute: bool = False
    sign: int = 1

    @property
    def unit(self) -> str:
        return "degree" if self.query.operation.endswith("angle") else "ratio"

    def definition(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id, "label": self.label, "aspect": self.aspect,
            "unit": self.unit, "query": self.query.definition(),
            "component_index": self.component, "absolute_per_frame": self.absolute,
            "sign": self.sign,
        }


def metric_catalog() -> dict[str, EventMetric]:
    result: dict[str, EventMetric] = {}

    def add(mid: str, label: str, aspect: str, operation: str, points: tuple[str, ...],
            normalizer: str | None = None, component: int = 0, absolute: bool = False,
            sign: int = 1) -> None:
        query = MeasurementQuery(mid, operation, points, normalizer,
                                 "torso_v1" if operation == "relative_position" else None)
        result[mid] = EventMetric(mid, label, aspect, query, component, absolute, sign)

    for name, side, zh in (("left", "L", "左"), ("right", "R", "右")):
        for joint, points in (
            ("knee", (f"{side}_Hip", f"{side}_Knee", f"{side}_Ankle")),
            ("elbow", (f"{side}_Shoulder", f"{side}_Elbow", f"{side}_Wrist")),
        ):
            mid = f"{name}_{joint}_angle"
            add(mid, LABELS[mid], "屈膝姿态" if joint == "knee" else "曲臂姿态",
                "joint_angle", points)
        for origin, origin_zh in (("Pelvis", "骨盆"), ("Head", "头关节"), ("Shoulder", f"{zh}肩")):
            for axis, component, axis_zh in (("up", 1, "上方"), ("forward", 2, "前方")):
                mid = f"{name}_wrist_{origin.lower()}_{axis}_torso_ratio"
                point = f"{side}_Shoulder" if origin == "Shoulder" else origin
                add(mid, LABELS.get(mid, f"{zh}腕相对{origin_zh}{axis_zh}位置/躯干长"),
                    "手部位置（腕节点代理）", "relative_position",
                    (f"{side}_Wrist", point), "torso_length", component)
        mid = f"{name}_wrist_hip_distance_torso_ratio"
        add(mid, LABELS[mid], "手落胯旁（腕节点代理）", "distance",
            (f"{side}_Wrist", f"{side}_Hip"), "torso_length")
        for suffix, component, sign, label in (
            ("up", 1, 1, f"{zh}肘相对{zh}肩上方位置/躯干长"),
            ("outward", 0, -1 if side == "L" else 1, f"{zh}肘相对{zh}肩向外位置/躯干长"),
        ):
            add(f"{name}_elbow_shoulder_{suffix}_torso_ratio", label,
                "两肘松垂的空间代理（不测肌肉放松）", "relative_position",
                (f"{side}_Elbow", f"{side}_Shoulder"), "torso_length", component, sign=sign)
        add(f"{name}_elbow_knee_lateral_torso_ratio", f"{zh}肘相对{zh}膝向右偏移/躯干长",
            "肘膝上下相对的左右偏移代理", "relative_position",
            (f"{side}_Elbow", f"{side}_Knee"), "torso_length", 0)
    add("right_wrist_head_right_torso_ratio", LABELS["right_wrist_head_right_torso_ratio"],
        "右手位于头部右前方（腕节点代理）", "relative_position",
        ("R_Wrist", "Head"), "torso_length", 0)
    add("stance_width_shoulder_ratio", LABELS["stance_width_shoulder_ratio"], "开立/弓步站距",
        "relative_position", ("L_Ankle", "R_Ankle"), "shoulder_width", 0, True)
    add("stance_length_leg_ratio", LABELS["stance_length_leg_ratio"], "前后步幅（不测承重）",
        "relative_position", ("L_Ankle", "R_Ankle"), "average_leg_length", 2, True)
    add("ankle_distance_leg_ratio", "双踝距离/腿长", "收脚距离（不证明触地方式）", "distance",
        ("L_Ankle", "R_Ankle"), "average_leg_length")
    add("wrist_distance_torso_ratio", "双腕距离/躯干长", "抱球/分掌的手间距离", "distance",
        ("L_Wrist", "R_Wrist"), "torso_length")
    add("wrist_lateral_width_shoulder_ratio", "双腕左右宽度/肩宽", "两手宽与肩同的空间代理",
        "relative_position", ("L_Wrist", "R_Wrist"), "shoulder_width", 0, True)
    for side, zh in (("L", "左"), ("R", "右")):
        other = "R" if side == "L" else "L"
        add(f"{side}_over_{other}_wrist_height_torso_ratio", f"{zh}腕减另一腕高度/躯干长",
            f"{zh}手在上的有符号上下关系", "relative_position",
            (f"{side}_Wrist", f"{other}_Wrist"), "torso_length", 1)
    return result


def event_metric_plan(event: dict[str, Any]) -> dict[str, Any]:
    """Keep all 26 endpoint slots; add stage-specific geometry and two processes."""
    eid, mid = event["event_id"], event["move_id"]
    extra_elbows = [f"{side}_elbow_shoulder_{axis}_torso_ratio"
                    for side in ("left", "right") for axis in ("up", "outward")]
    if eid == "arms_raised_completion":
        pose = list(ANGLE_IDS) + ["stance_width_shoulder_ratio", "wrist_lateral_width_shoulder_ratio"]
        pose += [f"{side}_wrist_shoulder_{axis}_torso_ratio"
                 for side in ("left", "right") for axis in ("up", "forward")]
        pose += extra_elbows
    elif eid.startswith("hold_ball_") or eid == "gather_hands":
        upper = "L" if eid in ("hold_ball_2", "gather_hands") else "R"
        lower = "R" if upper == "L" else "L"
        pose = list(ANGLE_IDS) + ["ankle_distance_leg_ratio", "wrist_distance_torso_ratio",
                                f"{upper}_over_{lower}_wrist_height_torso_ratio"]
        pose += [f"{side}_wrist_pelvis_{axis}_torso_ratio"
                 for side in ("left", "right") for axis in ("up", "forward")]
    elif eid.startswith("part_mane_"):
        if eid == "part_mane_3_L":
            pose = list(MOVE_METRICS[mid])
        else:
            leading, trailing = ("left", "right") if eid.endswith("_L") else ("right", "left")
            pose = list(ANGLE_IDS) + ["stance_length_leg_ratio",
                f"{leading}_wrist_head_up_torso_ratio", f"{leading}_wrist_head_forward_torso_ratio",
                f"{trailing}_wrist_hip_distance_torso_ratio"]
    elif eid in ("press_down_and_squat_transition", "separate_hands_to_endpoint"):
        pose = list(MOVE_METRICS[mid])
        if mid == 1:
            pose += extra_elbows + [f"{s}_elbow_knee_lateral_torso_ratio" for s in ("left", "right")]
    else:
        raise ValueError(f"No metric plan for {mid}.{eid}")
    if eid == "press_down_and_squat_transition":
        process = ["left_knee_angle", "right_knee_angle", "left_wrist_pelvis_up_torso_ratio",
                   "right_wrist_pelvis_up_torso_ratio", "left_wrist_pelvis_forward_torso_ratio",
                   "right_wrist_pelvis_forward_torso_ratio"]
    elif eid == "separate_hands_to_endpoint":
        process = ["left_wrist_pelvis_up_torso_ratio", "right_wrist_pelvis_up_torso_ratio",
                   "R_over_L_wrist_height_torso_ratio", "wrist_distance_torso_ratio"]
    else:
        process = []
    return {"pose_id": event["pose_id"], "pose_metric_ids": pose,
            "process_metric_ids": process, "process_statistic": "late_median_minus_early_median",
            "contains_f_group_endpoint_metrics": eid in (
                "press_down_and_squat_transition", "part_mane_3_L", "separate_hands_to_endpoint")}


def scalar_series(joints: np.ndarray, frames: np.ndarray, statuses: np.ndarray,
                  metric: EventMetric) -> tuple[np.ndarray, np.ndarray]:
    measured = measure_pose(joints, metric.query, frame_ids=frames, tracking_status=statuses)
    values = measured.values[:, metric.component].copy() * metric.sign
    if metric.absolute:
        values = np.abs(values)
    return values, measured.statuses


def summarize_scalar(values: np.ndarray, statuses: np.ndarray, frames: np.ndarray,
                     center: int) -> dict[str, Any]:
    values, statuses, frames = np.asarray(values), np.asarray(statuses), np.asarray(frames)
    if values.ndim != 1 or values.shape != statuses.shape or values.shape != frames.shape or not len(values):
        raise ValueError("Invalid scalar window shapes")
    if frames.dtype.kind not in "iu" or np.any(np.diff(frames) <= 0):
        raise ValueError("Frames must be strictly increasing integers")
    if isinstance(center, bool) or not isinstance(center, (int, np.integer)):
        raise ValueError("Center must be an integer source frame")
    hits = np.flatnonzero(frames == center)
    if len(hits) != 1:
        raise ValueError("Center must occur exactly once")
    valid = (statuses == "valid") & np.isfinite(values)
    count = int(valid.sum())
    usable = values[valid]
    median = float(np.median(usable)) if count else None
    cv = float(values[hits[0]]) if valid[hits[0]] else None
    enough = count >= 3 and count / len(values) >= 0.7
    return {"status": "valid" if enough else "insufficient_valid_window",
            "value": median if enough else None, "window_median": median, "center_value": cv,
            "center_minus_window_median": cv - median if cv is not None and median is not None else None,
            "window_min": float(usable.min()) if count else None,
            "window_max": float(usable.max()) if count else None,
            "valid_frame_count": count, "requested_frame_count": len(values),
            "window_source_frames_0based": frames.astype(int).tolist(),
            "frame_statuses": statuses.tolist()}


def stability_warnings(stats: dict[str, Any], unit: str) -> list[str]:
    warnings = []
    if stats["status"] != "valid":
        warnings.append(stats["status"])
    tolerance, span = (5.0, 10.0) if unit == "degree" else (0.05, 0.1)
    delta = stats.get("center_minus_window_median")
    if stats.get("center_value") is None:
        warnings.append("missing_center_value")
    if delta is not None and abs(delta) > tolerance:
        warnings.append("center_window_difference")
    lo, hi = stats.get("window_min"), stats.get("window_max")
    if lo is not None and hi is not None and hi - lo > span:
        warnings.append("window_variation")
    if stats["valid_frame_count"] < stats["requested_frame_count"]:
        warnings.append("tracking_or_geometry_gaps")
    return warnings


def process_change(values: np.ndarray, statuses: np.ndarray, frames: np.ndarray,
                   start: int, end: int, fps: float, unit: str) -> dict[str, Any]:
    """One-sided disjoint endpoint windows; no interpolation across tracking gaps."""
    if not np.isfinite(fps) or fps <= 0 or type(start) is not int or type(end) is not int or end < start:
        raise ValueError("Invalid process interval")
    frames, values, statuses = np.asarray(frames), np.asarray(values), np.asarray(statuses)
    if (frames.ndim != 1 or frames.dtype.kind not in "iu" or np.any(np.diff(frames) <= 0)
            or values.shape != frames.shape or statuses.shape != frames.shape):
        raise ValueError("Invalid process series shapes or frame IDs")
    selected = (frames >= start) & (frames <= end)
    f, v, s = frames[selected], values[selected], statuses[selected]
    if not np.array_equal(f, np.arange(start, end + 1)):
        raise ValueError("Process requires dense source frame IDs (missing tracking remains masked)")
    # The cap makes early and late windows disjoint even for short events.
    width = min(int(np.floor(0.2 * fps + 1e-8)) + 1, len(f) // 2)
    valid_ratio = float(np.mean((s == "valid") & np.isfinite(v)))
    result = {"status": "insufficient_process_interval", "value": None,
              "raw_endpoint_change": None, "start": None, "end": None,
              "duration_seconds": (end - start) / fps,
              "interval_valid_ratio": valid_ratio, "warnings": []}
    if width < 3:
        result["warnings"] = ["insufficient_process_interval"]
        return result
    early = summarize_scalar(v[:width], s[:width], f[:width], start)
    late = summarize_scalar(v[-width:], s[-width:], f[-width:], end)
    result.update(start=early, end=late)
    if early["center_value"] is not None and late["center_value"] is not None:
        result["raw_endpoint_change"] = late["center_value"] - early["center_value"]
    if early["status"] == late["status"] == "valid" and valid_ratio >= 0.7:
        result.update(status="valid", value=late["value"] - early["value"])
    for prefix, stats in (("early", early), ("late", late)):
        result["warnings"] += [f"{prefix}:{w}" for w in stability_warnings(stats, unit)]
    if valid_ratio < 1:
        result["warnings"].append("interval_tracking_or_geometry_gaps")
    if result["status"] != "valid":
        result["warnings"].append("insufficient_process_interval")
    return result


def interval_relation(value: float | None, reference: dict[str, Any], method: str) -> str:
    if method not in ("median_2mad", "p10_p90"):
        raise ValueError("Unknown reference method")
    if value is None or not np.isfinite(value):
        return "unavailable"
    if reference["status"] != "valid":
        return "insufficient_teachers"
    keys = ("median_minus_2mad", "median_plus_2mad") if method == "median_2mad" else ("p10", "p90")
    lo, hi = reference[keys[0]], reference[keys[1]]
    if lo is None or hi is None or hi - lo <= 1e-8:
        return "degenerate_reference"
    return "below" if value < lo else "above" if value > hi else "within"


def compare_measurement(row: dict[str, Any], reference: dict[str, Any],
                        unconfirmed_alignment: bool = False) -> dict[str, Any]:
    primary = interval_relation(row["value"], reference, "median_2mad")
    secondary = interval_relation(row["value"], reference, "p10_p90")
    raw = row["center_value"] if row["kind"] == "pose" else row["raw_endpoint_change"]
    center_relation = interval_relation(raw, reference, "median_2mad")
    teacher_warnings = [s["subject_id"] for s in reference["teacher_values"] if s["warnings"]]
    reasons = list(row["warnings"])
    if primary != secondary:
        reasons.append("reference_methods_disagree")
    if center_relation != primary:
        reasons.append("raw_and_robust_relations_disagree")
    if teacher_warnings:
        reasons.append("teacher_window_warnings")
    if unconfirmed_alignment:
        reasons.append("unconfirmed_alignment_warning")
    if primary in ("unavailable", "insufficient_teachers", "degenerate_reference"):
        reasons.append(primary)
    consistent_outside = primary in ("below", "above") and primary == secondary
    category = "review_measurement" if reasons else "difference_evidence" if consistent_outside else "no_interval_difference"
    return {"median_2mad_relation": primary, "p10_p90_relation": secondary,
            "raw_primary_relation": center_relation, "category": category,
            "consistent_outside": consistent_outside,
            "review_reasons": list(dict.fromkeys(reasons)), "teacher_warning_ids": teacher_warnings,
            "is_action_error": None}
