"""Interpretable endpoint geometry and teacher-interval diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .angle_metrics import SMPL_24_JOINTS as J, compute_angle_metrics


ANGLE_IDS = ("left_knee_angle", "right_knee_angle", "left_elbow_angle", "right_elbow_angle")
LABELS = {
    "left_knee_angle": "左膝夹角", "right_knee_angle": "右膝夹角",
    "left_elbow_angle": "左肘夹角", "right_elbow_angle": "右肘夹角",
    "stance_width_shoulder_ratio": "站距/肩宽（身体左右方向）",
    "stance_length_leg_ratio": "步幅/腿长（身体前后方向）",
    "left_wrist_pelvis_up_torso_ratio": "左腕相对骨盆上方位置/躯干长",
    "right_wrist_pelvis_up_torso_ratio": "右腕相对骨盆上方位置/躯干长",
    "left_wrist_pelvis_forward_torso_ratio": "左腕相对骨盆前方位置/躯干长",
    "right_wrist_pelvis_forward_torso_ratio": "右腕相对骨盆前方位置/躯干长",
    "left_wrist_head_up_torso_ratio": "左腕相对头关节上方位置/躯干长",
    "left_wrist_head_forward_torso_ratio": "左腕相对头关节前方位置/躯干长",
    "right_wrist_head_up_torso_ratio": "右腕相对头关节上方位置/躯干长",
    "right_wrist_head_forward_torso_ratio": "右腕相对头关节前方位置/躯干长",
    "right_wrist_head_right_torso_ratio": "右腕相对头关节右方位置/躯干长",
    "left_wrist_hip_distance_torso_ratio": "左腕至左髋距离/躯干长",
    "right_wrist_hip_distance_torso_ratio": "右腕至右髋距离/躯干长",
}

MOVE_METRICS = {
    1: ANGLE_IDS + ("stance_width_shoulder_ratio", "left_wrist_pelvis_up_torso_ratio",
        "right_wrist_pelvis_up_torso_ratio", "left_wrist_pelvis_forward_torso_ratio",
        "right_wrist_pelvis_forward_torso_ratio"),
    2: ANGLE_IDS + ("stance_length_leg_ratio", "left_wrist_head_up_torso_ratio",
        "left_wrist_head_forward_torso_ratio", "right_wrist_hip_distance_torso_ratio"),
    3: ANGLE_IDS + ("stance_length_leg_ratio", "right_wrist_head_up_torso_ratio",
        "right_wrist_head_forward_torso_ratio", "right_wrist_head_right_torso_ratio",
        "left_wrist_hip_distance_torso_ratio"),
}


def _unit(x):
    length = np.linalg.norm(x, axis=-1)
    return x / np.where(length > 1e-8, length, np.nan)[..., None]


def body_axes(joints):
    """Anatomical frame; up follows the torso and is NOT a gravity estimate."""
    up = _unit(joints[:, J["Neck"]] - joints[:, J["Pelvis"]])
    hip = _unit(joints[:, J["L_Hip"]] - joints[:, J["R_Hip"]])
    shoulder = _unit(joints[:, J["L_Shoulder"]] - joints[:, J["R_Shoulder"]])
    left_raw = 0.7 * hip + 0.3 * shoulder
    left = _unit(left_raw - np.sum(left_raw * up, axis=1)[:, None] * up)
    forward = _unit(np.cross(left, up))
    valid = np.isfinite(np.stack((up, left, forward))).all(axis=(0, 2))
    valid &= np.sum(hip * shoulder, axis=1) >= 0.5
    return up, left, forward, valid


def compute_endpoint_metrics(joints, frame_ids) -> pd.DataFrame:
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1:] != (24, 3):
        raise ValueError("Expected native SMPL joints (T,24,3).")
    frame_ids = list(frame_ids)
    angles = compute_angle_metrics(joints, frame_ids=frame_ids)
    rows = angles[angles.metric_id.isin(ANGLE_IDS)].to_dict("records")
    up, left, forward, axis_valid = body_axes(joints)
    torso = np.linalg.norm(joints[:, J["Neck"]] - joints[:, J["Pelvis"]], axis=1)
    shoulders = np.linalg.norm(joints[:, J["L_Shoulder"]] - joints[:, J["R_Shoulder"]], axis=1)
    legs = np.zeros(len(joints))
    for side in ("L", "R"):
        legs += (np.linalg.norm(joints[:, J[side+"_Hip"]] - joints[:, J[side+"_Knee"]], axis=1)
                 + np.linalg.norm(joints[:, J[side+"_Knee"]] - joints[:, J[side+"_Ankle"]], axis=1)) / 2

    def add(metric_id, numerator, denominator, needs_axes=True):
        valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 1e-8)
        if needs_axes:
            valid &= axis_valid
        values = numerator / np.where(denominator > 1e-8, denominator, np.nan)
        for frame, value, ok in zip(frame_ids, values, valid):
            rows.append({"frame_id": int(frame), "metric_id": metric_id,
                         "value": float(value) if ok else np.nan, "unit": "ratio",
                         "status": "valid" if ok else "invalid_geometry_or_body_frame"})

    feet = joints[:, J["L_Ankle"]] - joints[:, J["R_Ankle"]]
    add("stance_width_shoulder_ratio", abs(np.sum(feet * left, axis=1)), shoulders)
    add("stance_length_leg_ratio", abs(np.sum(feet * forward, axis=1)), legs)
    for name, side in (("left", "L"), ("right", "R")):
        for origin in ("Pelvis", "Head"):
            delta = joints[:, J[side+"_Wrist"]] - joints[:, J[origin]]
            for axis_name, axis in (("up", up), ("forward", forward)):
                add(f"{name}_wrist_{origin.lower()}_{axis_name}_torso_ratio",
                    np.sum(delta * axis, axis=1), torso)
            if origin == "Head" and name == "right":
                add("right_wrist_head_right_torso_ratio", -np.sum(delta * left, axis=1), torso)
        delta = joints[:, J[side+"_Wrist"]] - joints[:, J[side+"_Hip"]]
        add(f"{name}_wrist_hip_distance_torso_ratio", np.linalg.norm(delta, axis=1), torso, False)
    return pd.DataFrame(rows)


def endpoint_window(center, start, end, fps, half_window_seconds=0.2):
    if not (0 <= start <= center <= end) or fps <= 0 or half_window_seconds < 0:
        raise ValueError("Invalid endpoint interval or window.")
    radius = int(np.floor(half_window_seconds * fps + 1e-8))
    return np.arange(max(start, center-radius), min(end, center+radius)+1, dtype=np.int64)


def aggregate_window(values, statuses, frames, center, minimum_valid_ratio=0.7):
    values = np.asarray(values, dtype=float)
    valid = (np.asarray(statuses) == "valid") & np.isfinite(values)
    count = int(valid.sum())
    center_indices = np.flatnonzero(np.asarray(frames) == center)
    center_value = None
    if len(center_indices) == 1 and valid[center_indices[0]]:
        center_value = float(values[center_indices[0]])
    enough = count >= 3 and count / max(len(values), 1) >= minimum_valid_ratio
    median = float(np.median(values[valid])) if count else None
    return {"center_value": center_value, "window_median": median,
            "value": median if enough else None,
            "center_minus_window_median": center_value-median if center_value is not None and median is not None else None,
            "window_min": float(np.min(values[valid])) if count else None,
            "window_max": float(np.max(values[valid])) if count else None,
            "valid_frame_count": count, "requested_frame_count": len(values),
            "status": "valid" if enough else "insufficient_valid_window"}


def teacher_interval(values, minimum_count=7):
    valid = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    n = len(valid)
    if not n:
        return {"valid_teacher_count": 0, "status": "insufficient_teachers", **{
            k: None for k in ("p10", "p90", "median", "mad", "median_minus_2mad", "median_plus_2mad")}}
    median = float(np.median(valid)); mad = float(np.median(abs(valid-median)))
    return {"valid_teacher_count": n, "status": "valid" if n >= minimum_count else "insufficient_teachers",
            "p10": float(np.percentile(valid, 10)), "p90": float(np.percentile(valid, 90)),
            "median": median, "mad": mad,
            "median_minus_2mad": median-2*mad, "median_plus_2mad": median+2*mad}


def compare_interval(value, reference, method):
    names = ("p10", "p90") if method == "p10_p90" else ("median_minus_2mad", "median_plus_2mad")
    lo, hi = (reference[k] for k in names)
    if value is None or not np.isfinite(value):
        return "invalid_student", None
    if reference["status"] != "valid":
        return "insufficient_teachers", None
    if hi-lo <= 1e-8:
        return "degenerate_reference_interval", None
    if value < lo:
        return "below", value-lo
    if value > hi:
        return "above", value-hi
    return "within", 0.0


def rules_for_manifest(manifest):
    formulas = {m: "native SMPL three-joint angle; 180 degrees = straight" for m in ANGLE_IDS}
    formulas.update({
        "stance_width_shoulder_ratio": "abs(dot(L_Ankle-R_Ankle, left_axis))/shoulder_width",
        "stance_length_leg_ratio": "abs(dot(L_Ankle-R_Ankle, forward_axis))/average_leg_length",
    })
    for name, side in (("left", "L"), ("right", "R")):
        for origin in ("pelvis", "head"):
            for axis in ("up", "forward"):
                formulas[f"{name}_wrist_{origin}_{axis}_torso_ratio"] = f"dot({side}_Wrist-{origin.title()}, {axis}_axis)/torso_length"
        formulas[f"{name}_wrist_hip_distance_torso_ratio"] = f"norm({side}_Wrist-{side}_Hip)/torso_length"
    formulas["right_wrist_head_right_torso_ratio"] = "-dot(R_Wrist-Head,left_axis)/torso_length"
    aspects = {
        1: {"knee": "两腿屈膝松腰下蹲", "elbow": "两肘松垂", "stance": "开立步站距",
            "wrist": "两掌下按至腹前（以腕关节作为手部位置代理）"},
        2: {"knee": "左弓步，右腿自然蹬直", "elbow": "两臂保持自然弧形",
            "stance": "成左弓步", "wrist": "左手高与眼平，右手落于右胯旁（头/腕节点代理）"},
        3: {"knee": "左虚步，重心后坐（膝角仅描述腿形，不能测量承重）", "elbow": "沉肩屈臂，右臂舒展成弧",
            "stance": "左脚稍向前移", "wrist": "右手在头部右前方，左手落于左胯前（头/腕节点代理）"},
    }
    poses = []
    for event in manifest["endpoint_keyposes"][:3]:
        mid = event["move_id"]
        checks = []
        for metric in MOVE_METRICS[mid]:
            category = next(k for k in ("knee", "elbow", "stance", "wrist") if k in metric)
            checks.append({"metric_id": metric, "technique_aspect": aspects[mid][category],
                           "comparison_only": True})
        poses.append({"pose_id": event["pose_id"], "move_id": mid,
                      "move_name_pinyin": event["move_name_pinyin"],
                      "move_name_zh": event["move_name_zh"],
                      "final_technique_step": event["final_technique_step"], "checks": checks})
    return {"schema_version": "1.0", "status": "diagnostic_first3_pending_review",
            "coordinate_frame": "up=unit(Neck-Pelvis); left=projected 0.7*unit(L_Hip-R_Hip)+0.3*unit(L_Shoulder-R_Shoulder); forward=cross(left,up)",
            "body_axis_validity": "finite nonzero axes and hip/shoulder directions within 60 degrees",
            "normalizers": {"torso_length": "norm(Neck-Pelvis)", "shoulder_width": "norm(L_Shoulder-R_Shoulder)",
                "average_leg_length": "mean(left/right hip-knee length + knee-ankle length)"},
            "limitations": ["Torso up is not gravity; cannot judge trunk verticality or ground clearance.",
                "Head is not the eye/forehead; Wrist is not palm center. Compare teacher proxies only.",
                "Cannot infer muscle relaxation, palm orientation or actual weight bearing.",
                "Teacher-derived ranges are descriptive; outside is not automatically an error.",
                "A 0.2s endpoint window is clipped to the move; post-boundary frames are excluded."],
            "metric_definitions": {m: {"label_zh": LABELS[m], "formula": formulas[m],
                "unit": "degree" if m in ANGLE_IDS else "ratio"} for m in LABELS},
            "poses": poses}
