"""Angle and simple geometry metrics for SMPL 24-joint poses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch


SMPL_24_JOINTS: dict[str, int] = {
    "Pelvis": 0,
    "L_Hip": 1,
    "R_Hip": 2,
    "Spine1": 3,
    "L_Knee": 4,
    "R_Knee": 5,
    "Spine2": 6,
    "L_Ankle": 7,
    "R_Ankle": 8,
    "Spine3": 9,
    "L_Foot": 10,
    "R_Foot": 11,
    "Neck": 12,
    "L_Collar": 13,
    "R_Collar": 14,
    "Head": 15,
    "L_Shoulder": 16,
    "R_Shoulder": 17,
    "L_Elbow": 18,
    "R_Elbow": 19,
    "L_Wrist": 20,
    "R_Wrist": 21,
    "L_Hand": 22,
    "R_Hand": 23,
}


@dataclass(frozen=True)
class JointAngleMetric:
    metric_id: str
    first: str
    center: str
    third: str


@dataclass(frozen=True)
class VectorAngleMetric:
    metric_id: str
    first_start: str
    first_end: str
    second_start: str
    second_end: str


@dataclass(frozen=True)
class HeightDiffMetric:
    metric_id: str
    upper_joint: str
    lower_joint: str


JOINT_ANGLE_METRICS: tuple[JointAngleMetric, ...] = (
    JointAngleMetric("left_knee_angle", "L_Hip", "L_Knee", "L_Ankle"),
    JointAngleMetric("right_knee_angle", "R_Hip", "R_Knee", "R_Ankle"),
    JointAngleMetric("left_ankle_angle", "L_Knee", "L_Ankle", "L_Foot"),
    JointAngleMetric("right_ankle_angle", "R_Knee", "R_Ankle", "R_Foot"),
    JointAngleMetric("left_hip_angle", "Spine1", "L_Hip", "L_Knee"),
    JointAngleMetric("right_hip_angle", "Spine1", "R_Hip", "R_Knee"),
    JointAngleMetric("left_leg_opening_angle", "R_Hip", "L_Hip", "L_Knee"),
    JointAngleMetric("right_leg_opening_angle", "L_Hip", "R_Hip", "R_Knee"),
    JointAngleMetric("left_elbow_angle", "L_Shoulder", "L_Elbow", "L_Wrist"),
    JointAngleMetric("right_elbow_angle", "R_Shoulder", "R_Elbow", "R_Wrist"),
    JointAngleMetric("left_shoulder_angle_neck", "Neck", "L_Shoulder", "L_Elbow"),
    JointAngleMetric("right_shoulder_angle_neck", "Neck", "R_Shoulder", "R_Elbow"),
    JointAngleMetric("left_shoulder_angle_spine", "Spine3", "L_Shoulder", "L_Elbow"),
    JointAngleMetric("right_shoulder_angle_spine", "Spine3", "R_Shoulder", "R_Elbow"),
    JointAngleMetric("left_wrist_arm_angle", "L_Elbow", "L_Wrist", "L_Hand"),
    JointAngleMetric("right_wrist_arm_angle", "R_Elbow", "R_Wrist", "R_Hand"),
    JointAngleMetric("spine_bend_angle_1", "Pelvis", "Spine1", "Spine2"),
    JointAngleMetric("spine_bend_angle_2", "Spine1", "Spine2", "Spine3"),
    JointAngleMetric("neck_spine_angle", "Spine3", "Neck", "Head"),
    JointAngleMetric("trunk_vector_angle_with_spine", "Pelvis", "Spine3", "Neck"),
)


VECTOR_ANGLE_METRICS: tuple[VectorAngleMetric, ...] = (
    VectorAngleMetric("shoulder_hip_twist_angle", "R_Shoulder", "L_Shoulder", "R_Hip", "L_Hip"),
)


HEIGHT_DIFF_METRICS: tuple[HeightDiffMetric, ...] = (
    HeightDiffMetric("shoulder_height_diff", "L_Shoulder", "R_Shoulder"),
    HeightDiffMetric("hip_height_diff", "L_Hip", "R_Hip"),
    HeightDiffMetric("hand_height_diff", "L_Wrist", "R_Wrist"),
)


def compute_angle_metrics(
    joints: torch.Tensor,
    *,
    frame_ids: Iterable[int] | None = None,
    vertical_axis: int = 1,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """Compute the first-pass metric library from SMPL 24-joint positions.

    Args:
        joints: Tensor with shape ``(T, 24, 3)`` or a single frame ``(24, 3)``.
        frame_ids: Optional IDs to place in the ``frame_id`` output column.
        vertical_axis: Coordinate axis used for height metrics. Defaults to 1.
        eps: Bone/vector length threshold for invalid zero-length measurements.

    Returns:
        A long-form ``DataFrame`` with columns
        ``frame_id, metric_id, value, unit, status``.
    """
    pose = _as_smpl24_tensor(joints)
    if vertical_axis not in (0, 1, 2):
        raise ValueError(f"vertical_axis must be 0, 1, or 2, got {vertical_axis}.")

    frame_labels = _frame_labels(pose.shape[0], frame_ids)
    rows: list[dict[str, object]] = []

    for metric in JOINT_ANGLE_METRICS:
        values, statuses = _joint_angle_degrees(pose, metric, eps)
        _extend_rows(rows, frame_labels, metric.metric_id, values, "degree", statuses)

    for metric in VECTOR_ANGLE_METRICS:
        values, statuses = _vector_angle_degrees(pose, metric, eps)
        _extend_rows(rows, frame_labels, metric.metric_id, values, "degree", statuses)

    for metric in HEIGHT_DIFF_METRICS:
        values, statuses = _height_diff(pose, metric, vertical_axis)
        _extend_rows(rows, frame_labels, metric.metric_id, values, "length", statuses)

    values, statuses = _single_joint_coordinate(pose, "Pelvis", vertical_axis)
    _extend_rows(rows, frame_labels, "pelvis_height", values, "length", statuses)

    values, statuses = _normalized_pelvis_height(pose, vertical_axis, eps)
    _extend_rows(rows, frame_labels, "normalized_pelvis_height", values, "normalized_length", statuses)

    return pd.DataFrame(rows, columns=("frame_id", "metric_id", "value", "unit", "status"))


def _as_smpl24_tensor(joints: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(joints):
        joints = torch.as_tensor(joints)
    if joints.ndim == 2:
        joints = joints.unsqueeze(0)
    if joints.ndim != 3 or tuple(joints.shape[-2:]) != (24, 3):
        raise ValueError(f"Expected joints with shape (T, 24, 3) or (24, 3), got {tuple(joints.shape)}.")
    return joints.detach().to(dtype=torch.float64)


def _frame_labels(frame_count: int, frame_ids: Iterable[int] | None) -> list[int]:
    if frame_ids is None:
        return list(range(frame_count))
    labels = [int(frame_id) for frame_id in frame_ids]
    if len(labels) != frame_count:
        raise ValueError(f"frame_ids length must be {frame_count}, got {len(labels)}.")
    return labels


def _joint_indices(names: Sequence[str]) -> list[int]:
    return [SMPL_24_JOINTS[name] for name in names]


def _joint_angle_degrees(
    joints: torch.Tensor,
    metric: JointAngleMetric,
    eps: float,
) -> tuple[np.ndarray, list[str]]:
    first, center, third = _joint_indices((metric.first, metric.center, metric.third))
    p_first = joints[:, first]
    p_center = joints[:, center]
    p_third = joints[:, third]
    return _angle_between_vectors_degrees(p_first - p_center, p_third - p_center, eps)


def _vector_angle_degrees(
    joints: torch.Tensor,
    metric: VectorAngleMetric,
    eps: float,
) -> tuple[np.ndarray, list[str]]:
    first_start, first_end, second_start, second_end = _joint_indices(
        (metric.first_start, metric.first_end, metric.second_start, metric.second_end)
    )
    first_vector = joints[:, first_end] - joints[:, first_start]
    second_vector = joints[:, second_end] - joints[:, second_start]
    return _angle_between_vectors_degrees(first_vector, second_vector, eps)


def _angle_between_vectors_degrees(
    first_vector: torch.Tensor,
    second_vector: torch.Tensor,
    eps: float,
) -> tuple[np.ndarray, list[str]]:
    finite = torch.isfinite(first_vector).all(dim=1) & torch.isfinite(second_vector).all(dim=1)
    first_length = torch.linalg.norm(first_vector, dim=1)
    second_length = torch.linalg.norm(second_vector, dim=1)
    nonzero = (first_length > eps) & (second_length > eps)
    valid = finite & nonzero

    denominator = torch.clamp(first_length * second_length, min=eps)
    cosine = torch.sum(first_vector * second_vector, dim=1) / denominator
    angles = torch.rad2deg(torch.acos(torch.clamp(cosine, -1.0, 1.0)))
    values = angles.detach().cpu().numpy()
    values[~valid.detach().cpu().numpy()] = np.nan
    return values, _statuses(finite, nonzero)


def _height_diff(
    joints: torch.Tensor,
    metric: HeightDiffMetric,
    vertical_axis: int,
) -> tuple[np.ndarray, list[str]]:
    upper, lower = _joint_indices((metric.upper_joint, metric.lower_joint))
    values = joints[:, upper, vertical_axis] - joints[:, lower, vertical_axis]
    finite = torch.isfinite(joints[:, upper]).all(dim=1) & torch.isfinite(joints[:, lower]).all(dim=1)
    return _values_with_nan(values, finite), ["valid" if item else "missing_joint" for item in finite.cpu().tolist()]


def _single_joint_coordinate(
    joints: torch.Tensor,
    joint_name: str,
    axis: int,
) -> tuple[np.ndarray, list[str]]:
    joint = SMPL_24_JOINTS[joint_name]
    values = joints[:, joint, axis]
    finite = torch.isfinite(joints[:, joint]).all(dim=1)
    return _values_with_nan(values, finite), ["valid" if item else "missing_joint" for item in finite.cpu().tolist()]


def _normalized_pelvis_height(
    joints: torch.Tensor,
    vertical_axis: int,
    eps: float,
) -> tuple[np.ndarray, list[str]]:
    pelvis = joints[:, SMPL_24_JOINTS["Pelvis"]]
    neck = joints[:, SMPL_24_JOINTS["Neck"]]
    body_scale = torch.linalg.norm(neck - pelvis, dim=1)
    finite = torch.isfinite(pelvis).all(dim=1) & torch.isfinite(neck).all(dim=1)
    nonzero = body_scale > eps
    values = pelvis[:, vertical_axis] / torch.clamp(body_scale, min=eps)
    values_np = values.detach().cpu().numpy()
    values_np[~(finite & nonzero).detach().cpu().numpy()] = np.nan
    return values_np, _statuses(finite, nonzero)


def _values_with_nan(values: torch.Tensor, valid: torch.Tensor) -> np.ndarray:
    values_np = values.detach().cpu().numpy()
    values_np[~valid.detach().cpu().numpy()] = np.nan
    return values_np


def _statuses(finite: torch.Tensor, nonzero: torch.Tensor) -> list[str]:
    statuses: list[str] = []
    for is_finite, is_nonzero in zip(finite.cpu().tolist(), nonzero.cpu().tolist()):
        if not is_finite:
            statuses.append("missing_joint")
        elif not is_nonzero:
            statuses.append("invalid_zero_length")
        else:
            statuses.append("valid")
    return statuses


def _extend_rows(
    rows: list[dict[str, object]],
    frame_ids: Sequence[int],
    metric_id: str,
    values: np.ndarray,
    unit: str,
    statuses: Sequence[str],
) -> None:
    for frame_id, value, status in zip(frame_ids, values, statuses):
        rows.append(
            {
                "frame_id": int(frame_id),
                "metric_id": metric_id,
                "value": float(value),
                "unit": unit,
                "status": status,
            }
        )
