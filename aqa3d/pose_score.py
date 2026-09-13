"""Teacher-calibrated pose scores and explainable pause penalties."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DistanceCalibration:
    median_degrees: float
    mad_degrees: float
    scale_degrees: float
    sample_count: int


@dataclass(frozen=True)
class PoseScore:
    keyframe_z: float
    path_z: float
    keyframe_score: float
    path_score: float
    pose_score: float


@dataclass(frozen=True)
class PauseEvent:
    start_index: int
    end_index: int
    duration_seconds: float
    teacher_active_fraction: float


def calibrate_distances(
    distances_degrees: np.ndarray,
    *,
    minimum_scale_degrees: float = 1.0,
) -> DistanceCalibration:
    values = np.asarray(distances_degrees, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        raise ValueError("At least three finite teacher distances are required for calibration.")
    if minimum_scale_degrees <= 0:
        raise ValueError("minimum_scale_degrees must be positive.")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return DistanceCalibration(
        median_degrees=median,
        mad_degrees=mad,
        scale_degrees=max(1.4826 * mad, minimum_scale_degrees),
        sample_count=len(values),
    )


def calculate_pose_score(
    keyframe_distance_degrees: float,
    path_distance_degrees: float,
    keyframe_calibration: DistanceCalibration,
    path_calibration: DistanceCalibration,
    *,
    keyframe_weight: float = 0.40,
    teacher_median_score: float = 95.0,
    points_per_scale: float = 10.0,
) -> PoseScore:
    if not 0.0 <= keyframe_weight <= 1.0:
        raise ValueError("keyframe_weight must be in [0, 1].")
    if points_per_scale <= 0:
        raise ValueError("points_per_scale must be positive.")
    keyframe_z = (
        float(keyframe_distance_degrees) - keyframe_calibration.median_degrees
    ) / keyframe_calibration.scale_degrees
    path_z = (
        float(path_distance_degrees) - path_calibration.median_degrees
    ) / path_calibration.scale_degrees
    path_weight = 1.0 - keyframe_weight
    pose_z = keyframe_weight * keyframe_z + path_weight * path_z

    def component_score(z_value: float) -> float:
        return float(np.clip(teacher_median_score - points_per_scale * z_value, 0.0, 100.0))

    return PoseScore(
        keyframe_z=keyframe_z,
        path_z=path_z,
        keyframe_score=component_score(keyframe_z),
        path_score=component_score(path_z),
        pose_score=component_score(pose_z),
    )


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def unexpected_pause_events(
    pause_mask: np.ndarray,
    teacher_active_mask: np.ndarray,
    fps: float,
    *,
    minimum_teacher_active_fraction: float = 0.70,
) -> list[PauseEvent]:
    pauses = np.asarray(pause_mask, dtype=bool)
    teacher_active = np.asarray(teacher_active_mask, dtype=bool)
    if pauses.shape != teacher_active.shape:
        raise ValueError("pause_mask and teacher_active_mask must have identical shapes.")
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if not 0.0 <= minimum_teacher_active_fraction <= 1.0:
        raise ValueError("minimum_teacher_active_fraction must be in [0, 1].")
    events: list[PauseEvent] = []
    for start, end in _true_runs(pauses):
        active_fraction = float(np.mean(teacher_active[start : end + 1]))
        if active_fraction < minimum_teacher_active_fraction:
            continue
        events.append(
            PauseEvent(
                start_index=start,
                end_index=end,
                duration_seconds=(end - start + 1) / fps,
                teacher_active_fraction=active_fraction,
            )
        )
    return events


def apply_pause_penalty(
    pose_score: float,
    unexpected_pause_count: int,
    *,
    points_per_pause: float = 5.0,
    maximum_penalty: float = 15.0,
) -> tuple[float, float]:
    if unexpected_pause_count < 0:
        raise ValueError("unexpected_pause_count must be non-negative.")
    if points_per_pause < 0 or maximum_penalty < 0:
        raise ValueError("Penalty parameters must be non-negative.")
    penalty = min(points_per_pause * unexpected_pause_count, maximum_penalty)
    return float(np.clip(pose_score - penalty, 0.0, 100.0)), float(penalty)
