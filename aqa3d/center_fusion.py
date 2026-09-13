"""Teacher-calibrated pelvis metrics and explainable score fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pelvis_quality import SIGNAL_NAMES


@dataclass(frozen=True)
class CenterMetricSpec:
    metric_id: str
    weight: float
    minimum_scale: float


@dataclass(frozen=True)
class RobustCalibration:
    median: float
    mad: float
    scale: float
    sample_count: int


MOVE_METRICS: dict[str, tuple[CenterMetricSpec, ...]] = {
    "qishi": (
        CenterMetricSpec("root_vertical_lowering_error", 0.35, 0.01),
        CenterMetricSpec("horizontal_transfer_rmse", 0.30, 0.005),
        CenterMetricSpec("support_endpoint_error", 0.20, 0.005),
        CenterMetricSpec("vertical_trajectory_rmse", 0.15, 0.005),
    ),
    "yemafenzong": (
        CenterMetricSpec("horizontal_transfer_rmse", 0.50, 0.005),
        CenterMetricSpec("support_endpoint_error", 0.30, 0.005),
        CenterMetricSpec("vertical_trajectory_rmse", 0.20, 0.005),
    ),
    "baiheliangchi": (
        CenterMetricSpec("vertical_range_error", 0.50, 0.05),
        CenterMetricSpec("support_endpoint_error", 0.30, 0.005),
        CenterMetricSpec("forward_transfer_rmse", 0.20, 0.005),
    ),
}


def validate_move_metrics(move_metrics: dict[str, tuple[CenterMetricSpec, ...]]) -> None:
    for move_name, specs in move_metrics.items():
        if not specs:
            raise ValueError(f"{move_name} has no center metrics.")
        total = sum(spec.weight for spec in specs)
        if not np.isclose(total, 1.0):
            raise ValueError(f"{move_name} metric weights sum to {total}, expected 1.")
        for spec in specs:
            if spec.weight < 0 or spec.minimum_scale <= 0:
                raise ValueError(f"Invalid metric configuration: {spec}")


def robust_calibration(
    values: np.ndarray,
    *,
    minimum_scale: float,
) -> RobustCalibration:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 3:
        raise ValueError("At least three finite teacher errors are required.")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return RobustCalibration(
        median=median,
        mad=mad,
        scale=max(1.4826 * mad, minimum_scale),
        sample_count=len(finite),
    )


def pelvis_lowering(profile: np.ndarray, *, endpoint_fraction: float = 0.10) -> float:
    """Return start height minus end height; positive means the pelvis lowered."""
    values = np.asarray(profile, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Pelvis height profile must be a one-dimensional sequence.")
    if not 0.0 < endpoint_fraction <= 0.5:
        raise ValueError("endpoint_fraction must be in (0, 0.5].")
    count = max(int(round(len(values) * endpoint_fraction)), 1)
    return float(np.nanmedian(values[:count]) - np.nanmedian(values[-count:]))


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    return (
        float(np.sqrt(np.mean(np.square(left[valid] - right[valid]))))
        if np.any(valid)
        else np.nan
    )


def _range(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.ptp(finite)) if len(finite) else np.nan


def center_metric_error(
    profile: np.ndarray,
    teacher_template: np.ndarray,
    metric_id: str,
) -> tuple[float, dict[str, float]]:
    """Calculate one non-negative pelvis error and its interpretable raw values."""
    student = np.asarray(profile, dtype=np.float64)
    teacher = np.asarray(teacher_template, dtype=np.float64)
    if student.shape != teacher.shape or student.shape[0] != len(SIGNAL_NAMES):
        raise ValueError(
            f"Expected matching profiles shaped ({len(SIGNAL_NAMES)}, P), "
            f"got {student.shape} and {teacher.shape}."
        )
    signal = {name: index for index, name in enumerate(SIGNAL_NAMES)}
    if metric_id == "horizontal_transfer_rmse":
        student_values = student[
            [signal["support_lateral"], signal["support_forward"]]
        ].T
        teacher_values = teacher[
            [signal["support_lateral"], signal["support_forward"]]
        ].T
        return _rmse(student_values, teacher_values), {}
    if metric_id == "support_endpoint_error":
        difference = student[
            [signal["support_lateral"], signal["support_forward"]], -1
        ] - teacher[
            [signal["support_lateral"], signal["support_forward"]], -1
        ]
        return float(np.linalg.norm(difference)), {}
    if metric_id == "vertical_trajectory_rmse":
        return _rmse(
            student[signal["support_height"]],
            teacher[signal["support_height"]],
        ), {}
    if metric_id == "forward_transfer_rmse":
        return _rmse(
            student[signal["support_forward"]],
            teacher[signal["support_forward"]],
        ), {}
    if metric_id == "pelvis_lowering_error":
        student_lowering = pelvis_lowering(student[signal["support_height"]])
        teacher_lowering = pelvis_lowering(teacher[signal["support_height"]])
        return abs(student_lowering - teacher_lowering), {
            "student_pelvis_lowering": student_lowering,
            "teacher_pelvis_lowering": teacher_lowering,
        }
    if metric_id == "root_vertical_lowering_error":
        student_lowering = pelvis_lowering(student[signal["root_vertical"]])
        teacher_lowering = pelvis_lowering(teacher[signal["root_vertical"]])
        return abs(student_lowering - teacher_lowering), {
            "student_root_vertical_lowering": student_lowering,
            "teacher_root_vertical_lowering": teacher_lowering,
        }
    if metric_id == "vertical_range_error":
        student_range = _range(student[signal["support_height"]])
        teacher_range = _range(teacher[signal["support_height"]])
        error = (
            abs(float(np.log(student_range / teacher_range)))
            if student_range > 0 and teacher_range > 0
            else np.nan
        )
        return error, {
            "student_vertical_range": student_range,
            "teacher_vertical_range": teacher_range,
            "vertical_range_ratio": (
                student_range / teacher_range if teacher_range > 0 else np.nan
            ),
        }
    raise KeyError(f"Unknown center metric: {metric_id}")


def calibrated_excess(
    error: float,
    calibration: RobustCalibration,
    *,
    tolerance_scales: float = 1.0,
) -> tuple[float, float]:
    if tolerance_scales < 0:
        raise ValueError("tolerance_scales must be non-negative.")
    z_value = (float(error) - calibration.median) / calibration.scale
    return float(z_value), float(max(0.0, z_value - tolerance_scales))


def center_penalty(
    weighted_excess: float,
    *,
    points_per_scale: float,
    maximum_penalty: float,
) -> float:
    if weighted_excess < 0 or points_per_scale < 0 or maximum_penalty < 0:
        raise ValueError("Penalty inputs must be non-negative.")
    return float(min(points_per_scale * weighted_excess, maximum_penalty))
