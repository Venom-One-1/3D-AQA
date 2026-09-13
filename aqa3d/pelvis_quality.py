"""Pelvis/root trajectory features for first-pass Tai Chi quality diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .angle_metrics import SMPL_24_JOINTS
from .velocity_quality import resample_profile


SIGNAL_NAMES: tuple[str, ...] = (
    "support_height",
    "support_lateral",
    "support_forward",
    "root_vertical",
    "root_lateral",
    "root_forward",
    "body_yaw_change_degrees",
)


@dataclass(frozen=True)
class PelvisSignals:
    frame_numbers: np.ndarray
    valid: np.ndarray
    body_scale: float
    lateral_axes: np.ndarray
    support_height: np.ndarray
    support_lateral: np.ndarray
    support_forward: np.ndarray
    root_vertical: np.ndarray
    root_lateral: np.ndarray
    root_forward: np.ndarray
    body_yaw_change_degrees: np.ndarray
    support_vertical_residual: np.ndarray
    root_vertical_residual: np.ndarray
    excluded_track_switch_frames: int
    median_lateral_visibility: float
    camera_depth_range_body_scales: float

    def signal_matrix(self) -> np.ndarray:
        return np.stack([getattr(self, name) for name in SIGNAL_NAMES], axis=0)


@dataclass(frozen=True)
class TeacherPelvisModel:
    teacher_ids: tuple[str, ...]
    move_ids: np.ndarray
    move_names: tuple[str, ...]
    signal_names: tuple[str, ...]
    progress_grid: np.ndarray
    profiles: np.ndarray
    median: np.ndarray
    p10: np.ndarray
    p90: np.ndarray
    support_vertical_residual_rms: np.ndarray
    root_vertical_residual_rms: np.ndarray
    valid_teacher_counts: np.ndarray

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            Path(path),
            teacher_ids=np.asarray(self.teacher_ids),
            move_ids=self.move_ids,
            move_names=np.asarray(self.move_names),
            signal_names=np.asarray(self.signal_names),
            progress_grid=self.progress_grid,
            profiles=self.profiles,
            median=self.median,
            p10=self.p10,
            p90=self.p90,
            support_vertical_residual_rms=self.support_vertical_residual_rms,
            root_vertical_residual_rms=self.root_vertical_residual_rms,
            valid_teacher_counts=self.valid_teacher_counts,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TeacherPelvisModel":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                teacher_ids=tuple(str(value) for value in data["teacher_ids"]),
                move_ids=data["move_ids"].astype(np.int64),
                move_names=tuple(str(value) for value in data["move_names"]),
                signal_names=tuple(str(value) for value in data["signal_names"]),
                progress_grid=data["progress_grid"].astype(np.float64),
                profiles=data["profiles"].astype(np.float64),
                median=data["median"].astype(np.float64),
                p10=data["p10"].astype(np.float64),
                p90=data["p90"].astype(np.float64),
                support_vertical_residual_rms=data["support_vertical_residual_rms"].astype(np.float64),
                root_vertical_residual_rms=data["root_vertical_residual_rms"].astype(np.float64),
                valid_teacher_counts=data["valid_teacher_counts"].astype(np.int64),
            )

    def move_index(self, move_name: str) -> int:
        try:
            return self.move_names.index(move_name)
        except ValueError as error:
            raise KeyError(f"Teacher pelvis model has no move named {move_name}.") from error


@dataclass(frozen=True)
class PelvisQualityMetrics:
    vertical_trajectory_rmse: float
    vertical_band_outlier_ratio: float
    vertical_oscillation_ratio: float
    vertical_range_ratio: float
    horizontal_transfer_rmse: float
    lateral_transfer_rmse: float
    forward_transfer_rmse: float
    horizontal_band_outlier_ratio: float
    transfer_timing_error: float
    support_endpoint_error: float
    root_vertical_trajectory_rmse: float
    root_vertical_oscillation_ratio: float
    root_horizontal_transfer_rmse: float
    orientation_change_mae_degrees: float
    valid_frame_ratio: float


def _normalize(vectors: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.linalg.norm(vectors, axis=-1)
    valid = np.isfinite(vectors).all(axis=-1) & (lengths > eps)
    result = np.full_like(vectors, np.nan, dtype=np.float64)
    result[valid] = vectors[valid] / lengths[valid, None]
    return result, valid


def _weighted_moving_average(values: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if data.shape[0] != len(mask):
        raise ValueError("values and valid must have the same frame count.")
    window = max(int(window), 1)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float64)
    flat = data.reshape(len(data), -1)
    output = np.full_like(flat, np.nan)
    for column in range(flat.shape[1]):
        finite = mask & np.isfinite(flat[:, column])
        numerator = np.convolve(np.where(finite, flat[:, column], 0.0), kernel, mode="same")
        denominator = np.convolve(finite.astype(np.float64), kernel, mode="same")
        np.divide(numerator, denominator, out=output[:, column], where=denominator > 0)
    return output.reshape(data.shape)


def _track_switch_valid_mask(track_ids: np.ndarray, radius: int = 1) -> tuple[np.ndarray, int]:
    ids = np.asarray(track_ids, dtype=np.int64)
    valid = np.ones(len(ids), dtype=bool)
    switches = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    for switch in switches:
        start = max(int(switch) - radius, 0)
        end = min(int(switch) + radius + 1, len(ids))
        valid[start:end] = False
    return valid, int(np.sum(~valid))


def _body_scale(joints: np.ndarray, valid: np.ndarray) -> float:
    index = SMPL_24_JOINTS
    torso = np.linalg.norm(joints[:, index["Neck"]] - joints[:, index["Pelvis"]], axis=1)
    left_leg = (
        np.linalg.norm(joints[:, index["L_Hip"]] - joints[:, index["L_Knee"]], axis=1)
        + np.linalg.norm(joints[:, index["L_Knee"]] - joints[:, index["L_Ankle"]], axis=1)
    )
    right_leg = (
        np.linalg.norm(joints[:, index["R_Hip"]] - joints[:, index["R_Knee"]], axis=1)
        + np.linalg.norm(joints[:, index["R_Knee"]] - joints[:, index["R_Ankle"]], axis=1)
    )
    frame_scale = torso + 0.5 * (left_leg + right_leg)
    selected = frame_scale[valid & np.isfinite(frame_scale) & (frame_scale > 1e-8)]
    if len(selected) == 0:
        raise ValueError("Cannot estimate a finite positive SMPL body scale.")
    return float(np.median(selected))


def compute_pelvis_signals(
    frame_numbers: np.ndarray,
    joints: np.ndarray,
    camera_translations: np.ndarray,
    track_ids: np.ndarray,
    fps: float,
    *,
    smoothing_seconds: float = 0.20,
    intent_smoothing_seconds: float = 1.00,
    hip_weight: float = 0.70,
) -> PelvisSignals:
    """Build body-centric pelvis and PHALP root-translation trajectories.

    The support-relative signals use the pelvis relative to the ankle midpoint.
    The root signals use PHALP's camera translation and are deliberately kept
    separate because monocular depth and bounding-box scale can introduce drift.
    """
    frame_numbers = np.asarray(frame_numbers, dtype=np.int64)
    joints = np.asarray(joints, dtype=np.float64)
    camera = np.asarray(camera_translations, dtype=np.float64)
    track_ids = np.asarray(track_ids, dtype=np.int64)
    if joints.ndim != 3 or joints.shape[1:] != (24, 3):
        raise ValueError(f"Expected joints shaped (T, 24, 3), got {joints.shape}.")
    if camera.shape != (len(joints), 3):
        raise ValueError(f"Expected camera translations shaped {(len(joints), 3)}, got {camera.shape}.")
    if len(frame_numbers) != len(joints) or len(track_ids) != len(joints):
        raise ValueError("Frame numbers, joints, cameras and track IDs must have equal lengths.")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")
    if not 0.0 <= hip_weight <= 1.0:
        raise ValueError(f"hip_weight must be in [0, 1], got {hip_weight}.")

    finite = np.isfinite(joints).all(axis=(1, 2)) & np.isfinite(camera).all(axis=1)
    switch_valid, excluded = _track_switch_valid_mask(track_ids)
    valid = finite & switch_valid
    index = SMPL_24_JOINTS
    # PHALP/4D-Humans camera-space SMPL output uses negative Y as upward.
    up = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)

    hip = joints[:, index["R_Hip"]] - joints[:, index["L_Hip"]]
    shoulder = joints[:, index["R_Shoulder"]] - joints[:, index["L_Shoulder"]]
    hip[:, 1] = 0.0
    shoulder[:, 1] = 0.0
    hip_axis, hip_valid = _normalize(hip)
    shoulder_axis, shoulder_valid = _normalize(shoulder)
    reverse_shoulder = np.sum(hip_axis * shoulder_axis, axis=1) < 0.0
    shoulder_axis[reverse_shoulder] *= -1.0
    combined = hip_weight * hip_axis + (1.0 - hip_weight) * shoulder_axis
    axis_valid = valid & hip_valid & shoulder_valid
    window = max(int(round(smoothing_seconds * fps)), 1)
    combined = _weighted_moving_average(combined, axis_valid, window)
    lateral, normalized_axis_valid = _normalize(combined)
    valid &= normalized_axis_valid
    forward = np.cross(up[None, :], lateral)
    forward, forward_valid = _normalize(forward)
    valid &= forward_valid

    scale = _body_scale(joints, valid)
    pelvis = joints[:, index["Pelvis"]]
    ankle_midpoint = 0.5 * (
        joints[:, index["L_Ankle"]] + joints[:, index["R_Ankle"]]
    )
    support = pelvis - ankle_midpoint
    support_height = np.sum(support * up[None, :], axis=1) / scale
    support_lateral = np.sum(support * lateral, axis=1) / scale
    support_forward = np.sum(support * forward, axis=1) / scale

    origin_count = max(int(round(0.5 * fps)), 1)
    origin_candidates = camera[:origin_count][valid[:origin_count]]
    if len(origin_candidates) == 0:
        origin_candidates = camera[valid]
    origin = np.median(origin_candidates, axis=0)
    root_delta = camera - origin
    root_vertical = np.sum(root_delta * up[None, :], axis=1) / scale
    root_lateral = np.sum(root_delta * lateral, axis=1) / scale
    root_forward = np.sum(root_delta * forward, axis=1) / scale

    yaw = np.unwrap(np.arctan2(lateral[:, 2], lateral[:, 0]))
    first_valid = np.flatnonzero(valid)
    yaw_origin = yaw[first_valid[0]] if len(first_valid) else 0.0
    yaw_change = np.degrees(yaw - yaw_origin)

    raw_signals = np.stack(
        (
            support_height,
            support_lateral,
            support_forward,
            root_vertical,
            root_lateral,
            root_forward,
            yaw_change,
        ),
        axis=1,
    )
    smoothed = _weighted_moving_average(raw_signals, valid, window)
    smoothed[~valid] = np.nan
    intent_window = max(int(round(intent_smoothing_seconds * fps)), window)
    support_intent = _weighted_moving_average(smoothed[:, 0], valid, intent_window)
    root_intent = _weighted_moving_average(smoothed[:, 3], valid, intent_window)
    support_residual = smoothed[:, 0] - support_intent
    root_residual = smoothed[:, 3] - root_intent
    support_residual[~valid] = np.nan
    root_residual[~valid] = np.nan

    depth_range = np.ptp(camera[valid, 2]) / scale if np.any(valid) else np.nan
    return PelvisSignals(
        frame_numbers=frame_numbers,
        valid=valid,
        body_scale=scale,
        lateral_axes=lateral,
        support_height=smoothed[:, 0],
        support_lateral=smoothed[:, 1],
        support_forward=smoothed[:, 2],
        root_vertical=smoothed[:, 3],
        root_lateral=smoothed[:, 4],
        root_forward=smoothed[:, 5],
        body_yaw_change_degrees=smoothed[:, 6],
        support_vertical_residual=support_residual,
        root_vertical_residual=root_residual,
        excluded_track_switch_frames=excluded,
        median_lateral_visibility=float(np.nanmedian(np.abs(lateral[valid, 0]))),
        camera_depth_range_body_scales=float(depth_range),
    )


def resample_pelvis_signals(
    signals: PelvisSignals,
    progress: np.ndarray,
    progress_grid: np.ndarray,
) -> np.ndarray:
    values = signals.signal_matrix()
    return np.stack(
        [resample_profile(progress, signal, progress_grid) for signal in values],
        axis=0,
    )


def build_teacher_model(
    teacher_ids: list[str],
    move_ids: list[int],
    move_names: list[str],
    progress_grid: np.ndarray,
    profiles: np.ndarray,
    support_residual_rms: np.ndarray,
    root_residual_rms: np.ndarray,
) -> TeacherPelvisModel:
    values = np.asarray(profiles, dtype=np.float64)
    expected = (len(teacher_ids), len(move_ids), len(SIGNAL_NAMES), len(progress_grid))
    if values.shape != expected:
        raise ValueError(f"Expected teacher profiles shaped {expected}, got {values.shape}.")
    valid_counts = np.sum(np.isfinite(values).any(axis=(2, 3)), axis=0)
    return TeacherPelvisModel(
        teacher_ids=tuple(teacher_ids),
        move_ids=np.asarray(move_ids, dtype=np.int64),
        move_names=tuple(move_names),
        signal_names=SIGNAL_NAMES,
        progress_grid=np.asarray(progress_grid, dtype=np.float64),
        profiles=values,
        median=np.nanmedian(values, axis=0),
        p10=np.nanpercentile(values, 10.0, axis=0),
        p90=np.nanpercentile(values, 90.0, axis=0),
        support_vertical_residual_rms=np.asarray(support_residual_rms, dtype=np.float64),
        root_vertical_residual_rms=np.asarray(root_residual_rms, dtype=np.float64),
        valid_teacher_counts=valid_counts.astype(np.int64),
    )


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    return float(np.sqrt(np.mean(np.square(left[valid] - right[valid])))) if np.any(valid) else np.nan


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if np.isfinite(denominator) and denominator > 1e-12 else np.nan


def _range(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.ptp(finite)) if len(finite) else np.nan


def _band_outlier_ratio(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(low) & np.isfinite(high)
    return float(np.mean((values[valid] < low[valid]) | (values[valid] > high[valid]))) if np.any(valid) else np.nan


def _peak_motion_phase(lateral: np.ndarray, forward: np.ndarray, progress_grid: np.ndarray) -> float:
    delta_lateral = np.gradient(lateral, progress_grid)
    delta_forward = np.gradient(forward, progress_grid)
    speed = np.hypot(delta_lateral, delta_forward)
    return float(progress_grid[int(np.nanargmax(speed))]) if np.isfinite(speed).any() else np.nan


def compute_quality_metrics(
    signals: PelvisSignals,
    student_profile: np.ndarray,
    teacher_model: TeacherPelvisModel,
    move_name: str,
) -> PelvisQualityMetrics:
    move_index = teacher_model.move_index(move_name)
    median = teacher_model.median[move_index]
    p10 = teacher_model.p10[move_index]
    p90 = teacher_model.p90[move_index]
    signal_index = {name: index for index, name in enumerate(teacher_model.signal_names)}

    support_height = signal_index["support_height"]
    support_lateral = signal_index["support_lateral"]
    support_forward = signal_index["support_forward"]
    root_vertical = signal_index["root_vertical"]
    root_lateral = signal_index["root_lateral"]
    root_forward = signal_index["root_forward"]
    yaw = signal_index["body_yaw_change_degrees"]

    student_horizontal = np.stack(
        (student_profile[support_lateral], student_profile[support_forward]),
        axis=1,
    )
    teacher_horizontal = np.stack(
        (median[support_lateral], median[support_forward]),
        axis=1,
    )
    root_student_horizontal = np.stack(
        (student_profile[root_lateral], student_profile[root_forward]),
        axis=1,
    )
    root_teacher_horizontal = np.stack(
        (median[root_lateral], median[root_forward]),
        axis=1,
    )
    student_support_residual_rms = float(
        np.sqrt(np.nanmean(np.square(signals.support_vertical_residual)))
    )
    student_root_residual_rms = float(
        np.sqrt(np.nanmean(np.square(signals.root_vertical_residual)))
    )
    teacher_support_residual = float(
        np.nanmedian(teacher_model.support_vertical_residual_rms[:, move_index])
    )
    teacher_root_residual = float(
        np.nanmedian(teacher_model.root_vertical_residual_rms[:, move_index])
    )
    horizontal_outliers = np.nanmean(
        (
            _band_outlier_ratio(
                student_profile[support_lateral],
                p10[support_lateral],
                p90[support_lateral],
            ),
            _band_outlier_ratio(
                student_profile[support_forward],
                p10[support_forward],
                p90[support_forward],
            ),
        )
    )
    student_peak = _peak_motion_phase(
        student_profile[support_lateral],
        student_profile[support_forward],
        teacher_model.progress_grid,
    )
    teacher_peak = _peak_motion_phase(
        median[support_lateral],
        median[support_forward],
        teacher_model.progress_grid,
    )
    endpoint_difference = student_horizontal[-1] - teacher_horizontal[-1]
    return PelvisQualityMetrics(
        vertical_trajectory_rmse=_rmse(student_profile[support_height], median[support_height]),
        vertical_band_outlier_ratio=_band_outlier_ratio(
            student_profile[support_height],
            p10[support_height],
            p90[support_height],
        ),
        vertical_oscillation_ratio=_safe_ratio(
            student_support_residual_rms,
            teacher_support_residual,
        ),
        vertical_range_ratio=_safe_ratio(
            _range(student_profile[support_height]),
            _range(median[support_height]),
        ),
        horizontal_transfer_rmse=_rmse(student_horizontal, teacher_horizontal),
        lateral_transfer_rmse=_rmse(
            student_profile[support_lateral],
            median[support_lateral],
        ),
        forward_transfer_rmse=_rmse(
            student_profile[support_forward],
            median[support_forward],
        ),
        horizontal_band_outlier_ratio=float(horizontal_outliers),
        transfer_timing_error=float(abs(student_peak - teacher_peak)),
        support_endpoint_error=float(np.linalg.norm(endpoint_difference)),
        root_vertical_trajectory_rmse=_rmse(
            student_profile[root_vertical],
            median[root_vertical],
        ),
        root_vertical_oscillation_ratio=_safe_ratio(
            student_root_residual_rms,
            teacher_root_residual,
        ),
        root_horizontal_transfer_rmse=_rmse(
            root_student_horizontal,
            root_teacher_horizontal,
        ),
        orientation_change_mae_degrees=float(
            np.nanmean(np.abs(student_profile[yaw] - median[yaw]))
        ),
        valid_frame_ratio=float(np.mean(signals.valid)),
    )
