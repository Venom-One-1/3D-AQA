"""Versioned, task-independent measurements of native SMPL-24 joint positions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


VERSION = "1.0"
JOINT_NAMES = (
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar",
    "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
)
JOINTS = {name: index for index, name in enumerate(JOINT_NAMES)}
MIDPOINTS = {
    "HipMidpoint": ("L_Hip", "R_Hip"),
    "ShoulderMidpoint": ("L_Shoulder", "R_Shoulder"),
    "AnkleMidpoint": ("L_Ankle", "R_Ankle"),
    "WristMidpoint": ("L_Wrist", "R_Wrist"),
}
NORMALIZERS = {
    "torso_length": "norm(Neck-Pelvis), evaluated separately in each frame",
    "shoulder_width": "norm(L_Shoulder-R_Shoulder), evaluated separately in each frame",
    "average_leg_length": "mean(norm(Hip-Knee)+norm(Knee-Ankle)) over left/right legs per frame",
}
OPERATIONS = {"joint_angle": 3, "vector_angle": 4, "relative_position": 2, "distance": 2}
EPS = 1e-8


@dataclass(frozen=True)
class MeasurementQuery:
    """Declarative query; no code execution, arbitrary expressions or joint aliases."""

    query_id: str
    operation: str
    points: tuple[str, ...]
    normalizer: str | None = None
    coordinate_frame: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id or len(self.query_id) > 128:
            raise ValueError("query_id must be a nonempty string of at most 128 characters")
        if not isinstance(self.operation, str) or self.operation not in OPERATIONS:
            raise ValueError("unsupported_operation")
        if not isinstance(self.points, (tuple, list)):
            raise ValueError("points must be an ordered sequence of registered names")
        object.__setattr__(self, "points", tuple(self.points))
        if len(self.points) != OPERATIONS[self.operation]:
            raise ValueError("Wrong point count for operation")
        if any(not isinstance(p, str) or p not in (*JOINT_NAMES, *MIDPOINTS) for p in self.points):
            raise ValueError("unsupported_point: use an explicit SMPL joint or registered midpoint")
        if self.operation in ("joint_angle", "vector_angle"):
            if self.normalizer is not None or self.coordinate_frame is not None:
                raise ValueError("Angles are unsigned 3D angles, without a frame or normalizer")
        else:
            if not isinstance(self.normalizer, str) or self.normalizer not in NORMALIZERS:
                raise ValueError("An explicit registered normalizer is required for lengths")
            if self.operation == "relative_position":
                if self.coordinate_frame != "torso_v1":
                    raise ValueError("unsupported_reference_frame: relative_position requires torso_v1")
            elif self.coordinate_frame is not None:
                raise ValueError("Euclidean distance has no coordinate_frame")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MeasurementQuery:
        if not isinstance(value, dict):
            raise ValueError("Query must be a JSON object")
        required = {"query_id", "operation", "points"}
        if not required <= value.keys() or value.keys() - required - {"normalizer", "coordinate_frame"}:
            raise ValueError("Missing or unknown query fields")
        return cls(**value)

    def definition(self) -> dict[str, Any]:
        formulas = {
            "joint_angle": "unsigned_angle(point[0]-point[1], point[2]-point[1]); point[1] is vertex",
            "vector_angle": "unsigned_angle(point[1]-point[0], point[3]-point[2])",
            "relative_position": "dot(point[0]-point[1], axis)/normalizer for right,up,forward",
            "distance": "norm(point[0]-point[1])/normalizer",
        }
        return {**asdict(self), "version": VERSION, "formula": formulas[self.operation],
                "unit": "degree" if self.operation.endswith("angle") else "ratio",
                "components": ["right", "up", "forward"] if self.operation == "relative_position" else ["value"],
                "normalizer_definition": NORMALIZERS.get(self.normalizer),
                "limitations": [
                    "Geometric measurement, not an action-error diagnosis or calibrated confidence.",
                    "Joint angles are unsigned geometric included angles, not clinical joint rotations.",
                    "Head is not the eye/forehead; Wrist is not the palm center; Ankle is not a contact point.",
                    "torso_v1 up follows the torso, not gravity. No ground height, weight bearing or muscle relaxation.",
                ]}


def measurement_contract() -> dict[str, Any]:
    """Machine-readable capabilities and conventions for future query planning."""
    return {"version": VERSION, "joint_order": list(JOINT_NAMES), "derived_points": MIDPOINTS,
            "operations": {op: {"point_count": count} for op, count in OPERATIONS.items()},
            "query_fields": ["query_id", "operation", "points", "normalizer", "coordinate_frame"],
            "normalizers": NORMALIZERS, "angle_range_degrees": [0, 180],
            "coordinate_frames": {"torso_v1": {
                "up": "unit(Neck-Pelvis)",
                "left": "unit(project(0.7*unit(L_Hip-R_Hip)+0.3*unit(L_Shoulder-R_Shoulder), perpendicular_to=up))",
                "forward": "unit(cross(left,up))", "right": "-left",
                "validity": "finite nonzero axes; dot(unit(hip),unit(shoulder)) >= 0.5",
                "meaning": "Approximate anatomical directions, not gravity or a clinical segment frame.",
            }},
            "frame_convention": "original source frames, 0-based; PHALP frame = source frame + 1",
            "unsupported": ["gravity_frame", "ground_clearance", "signed_axial_twist",
                            "palm_orientation", "actual_weight_bearing", "muscle_relaxation"],
            "invalid_queries": "ValueError; never silently substitute points, frames or scales",
            "validity": "valid means computable only; finite SMPL predictions can still be wrong",
            "aggregation": "Component-wise median on the same valid frames; vector medians need not form an observed pose",
            "reference": "Unscaled MAD, P10-P90 and median +/- 2*MAD are descriptive, not pass/fail thresholds"}


@dataclass
class MeasurementResult:
    query: MeasurementQuery
    frame_ids: np.ndarray
    values: np.ndarray
    statuses: np.ndarray

    @property
    def components(self) -> tuple[str, ...]:
        return ("right", "up", "forward") if self.query.operation == "relative_position" else ("value",)


def _frames(frame_ids: Sequence[int], count: int) -> np.ndarray:
    frames = np.asarray(frame_ids)
    if frames.shape != (count,) or frames.dtype.kind not in "iu" or np.any(frames < 0):
        raise ValueError("frame_ids must be nonnegative integers matching the frame count")
    if np.any(frames[1:] <= frames[:-1]):
        raise ValueError("frame_ids must be unique and strictly increasing")
    return frames.astype(np.int64)


def _point(joints: np.ndarray, name: str) -> np.ndarray:
    if name in JOINTS:
        return joints[:, JOINTS[name]]
    first, second = MIDPOINTS[name]
    return (joints[:, JOINTS[first]] + joints[:, JOINTS[second]]) / 2


def _unit(vector: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / np.where(length > EPS, length, np.nan)


def _torso_axes(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    up = _unit(_point(joints, "Neck") - _point(joints, "Pelvis"))
    hip = _unit(_point(joints, "L_Hip") - _point(joints, "R_Hip"))
    shoulder = _unit(_point(joints, "L_Shoulder") - _point(joints, "R_Shoulder"))
    lateral = 0.7 * hip + 0.3 * shoulder
    left = _unit(lateral - np.sum(lateral * up, axis=1)[:, None] * up)
    forward = _unit(np.cross(left, up))
    axes = np.stack((-left, up, forward), axis=1)
    valid = np.isfinite(axes).all(axis=(1, 2)) & (np.sum(hip * shoulder, axis=1) >= 0.5)
    return axes, valid


def _scale(joints: np.ndarray, name: str) -> np.ndarray:
    def length(a: str, b: str) -> np.ndarray:
        return np.linalg.norm(_point(joints, a) - _point(joints, b), axis=1)
    if name == "torso_length":
        return length("Neck", "Pelvis")
    if name == "shoulder_width":
        return length("L_Shoulder", "R_Shoulder")
    return sum(length(s+"_Hip", s+"_Knee") + length(s+"_Knee", s+"_Ankle") for s in ("L", "R")) / 2


def measure_pose(
    joints: np.ndarray, query: MeasurementQuery, *, frame_ids: Sequence[int] | None = None,
    tracking_status: Sequence[str] | None = None,
) -> MeasurementResult:
    """Measure (24,3) or (T,24,3) joints without importing legacy metric code."""
    joints = np.asarray(joints, dtype=np.float64)
    if joints.shape == (24, 3):
        joints = joints[None]
    if joints.ndim != 3 or joints.shape[1:] != (24, 3) or len(joints) == 0:
        raise ValueError("Expected nonempty (T,24,3) or (24,3) native SMPL joints")
    if not isinstance(query, MeasurementQuery):
        raise ValueError("query must be a validated MeasurementQuery")
    count = len(joints)
    frames = _frames(np.arange(count) if frame_ids is None else frame_ids, count)
    status = np.full(count, "valid", dtype=object)
    points = [_point(joints, name) for name in query.points]
    point_valid = np.isfinite(np.stack(points, axis=1)).all(axis=(1, 2))
    if query.operation.endswith("angle"):
        if query.operation == "joint_angle":
            a, b = points[0] - points[1], points[2] - points[1]
        else:
            a, b = points[1] - points[0], points[3] - points[2]
        valid_length = (np.linalg.norm(a, axis=1) > EPS) & (np.linalg.norm(b, axis=1) > EPS)
        values = np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.sum(a*b, axis=1)))[:, None]
        status[~valid_length] = "invalid_zero_length"
    else:
        delta = points[0] - points[1]
        scale = _scale(joints, query.normalizer)
        scale_valid = np.isfinite(scale) & (scale > EPS)
        if query.operation == "relative_position":
            axes, axes_valid = _torso_axes(joints)
            values = np.einsum("tij,tj->ti", axes, delta)
            status[~axes_valid] = "invalid_body_frame"
        else:
            values = np.linalg.norm(delta, axis=1)[:, None]
        values = values / np.where(scale_valid, scale, np.nan)[:, None]
        status[~scale_valid] = "invalid_normalizer"
    status[~point_valid] = "missing_point"
    status[(status == "valid") & ~np.isfinite(values).all(axis=1)] = "invalid_geometry"
    if tracking_status is not None:
        supplied = np.asarray(tracking_status)
        if supplied.shape != (count,) or supplied.dtype.kind not in "US" or np.any(supplied == ""):
            raise ValueError("tracking_status must contain one nonempty string per frame")
        status[supplied != "valid"] = supplied[supplied != "valid"]
    values[status != "valid"] = np.nan
    return MeasurementResult(query, frames, values, status)


def summarize_window(
    result: MeasurementResult, center: int, *, minimum_valid_ratio: float = 0.7,
    minimum_valid_frames: int = 3,
) -> dict[str, Any]:
    """Use exactly the supplied window; no recentering, interpolation or hole filling."""
    if not 0 < minimum_valid_ratio <= 1 or type(minimum_valid_frames) is not int or minimum_valid_frames < 1:
        raise ValueError("Invalid window thresholds")
    if isinstance(center, bool) or not isinstance(center, (int, np.integer)):
        raise ValueError("center must be an integer source frame")
    hits = np.flatnonzero(result.frame_ids == center)
    if len(hits) != 1:
        raise ValueError("center must occur exactly once in the supplied window")
    valid = (result.statuses == "valid") & np.isfinite(result.values).all(axis=1)
    count = int(valid.sum())
    enough = count >= minimum_valid_frames and count / len(valid) >= minimum_valid_ratio
    components = {}
    for i, name in enumerate(result.components):
        values = result.values[valid, i]
        median = float(np.median(values)) if count else None
        cv = float(result.values[hits[0], i]) if valid[hits[0]] else None
        components[name] = {
            "value": median if enough else None, "window_median": median, "center_value": cv,
            "center_minus_window_median": cv-median if cv is not None and median is not None else None,
            "window_min": float(values.min()) if count else None,
            "window_max": float(values.max()) if count else None,
        }
    return {"status": "valid" if enough else "insufficient_valid_window",
            "valid_frame_count": count, "requested_frame_count": len(valid),
            "center_source_frame_0based": int(center),
            "window_source_frames_0based": result.frame_ids.tolist(),
            "frame_statuses": result.statuses.tolist(), "components": components}


def reference_statistics(samples: list[dict[str, Any]], minimum_teachers: int = 7) -> dict[str, Any]:
    """Keep every teacher, including nulls. No trimming, imputation or error labels."""
    if type(minimum_teachers) is not int or minimum_teachers < 1:
        raise ValueError("minimum_teachers must be a positive integer")
    if any(not isinstance(s.get("subject_id"), str) or not s["subject_id"] for s in samples):
        raise ValueError("Each teacher requires a subject_id")
    if len({s["subject_id"] for s in samples}) != len(samples):
        raise ValueError("Duplicate teacher IDs")
    usable = []
    for s in samples:
        v = s["value"]
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)):
            raise ValueError("Teacher values must be finite numbers or null")
        if s["status"] == "valid" and v is not None:
            usable.append(v)
    stats = dict.fromkeys(("median", "mad", "p10", "p90", "median_minus_2mad", "median_plus_2mad"))
    if usable:
        median = float(np.median(usable))
        mad = float(np.median(np.abs(np.asarray(usable) - median)))
        stats.update(median=median, mad=mad, p10=float(np.percentile(usable, 10)),
                     p90=float(np.percentile(usable, 90)), median_minus_2mad=median-2*mad,
                     median_plus_2mad=median+2*mad)
    return {"status": "valid" if len(usable) >= minimum_teachers else "insufficient_teachers",
            "valid_teacher_count": len(usable), "teacher_count": len(samples),
            "mad_scaled": False, **stats, "teacher_values": samples,
            "zero_width_p10_p90": bool(usable and stats["p10"] == stats["p90"]),
            "zero_width_median_2mad": bool(usable and stats["mad"] == 0)}
