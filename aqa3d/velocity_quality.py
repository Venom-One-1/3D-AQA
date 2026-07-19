"""Teacher-distribution velocity and acceleration diagnostics for SMPL motion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import wasserstein_distance

from .motion_quality import (
    bridge_short_false_gaps,
    remove_short_true_runs,
    robust_noise_floor,
)


@dataclass(frozen=True)
class TeacherMotionModel:
    teacher_ids: tuple[str, ...]
    move_ids: np.ndarray
    move_names: tuple[str, ...]
    regions: tuple[str, ...]
    progress_grid: np.ndarray
    speed_profiles: np.ndarray
    acceleration_profiles: np.ndarray
    active_profiles: np.ndarray
    speed_median: np.ndarray
    speed_mad: np.ndarray
    speed_p10: np.ndarray
    speed_p90: np.ndarray
    acceleration_median: np.ndarray
    acceleration_mad: np.ndarray
    acceleration_p10: np.ndarray
    acceleration_p90: np.ndarray
    acceleration_abs_p95: np.ndarray
    active_probability: np.ndarray
    duration_median: np.ndarray
    amplitude_median: np.ndarray
    peak_rate_median: np.ndarray
    fragment_count_median: np.ndarray

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            Path(path),
            teacher_ids=np.asarray(self.teacher_ids),
            move_ids=self.move_ids,
            move_names=np.asarray(self.move_names),
            regions=np.asarray(self.regions),
            progress_grid=self.progress_grid,
            speed_profiles=self.speed_profiles,
            acceleration_profiles=self.acceleration_profiles,
            active_profiles=self.active_profiles,
            speed_median=self.speed_median,
            speed_mad=self.speed_mad,
            speed_p10=self.speed_p10,
            speed_p90=self.speed_p90,
            acceleration_median=self.acceleration_median,
            acceleration_mad=self.acceleration_mad,
            acceleration_p10=self.acceleration_p10,
            acceleration_p90=self.acceleration_p90,
            acceleration_abs_p95=self.acceleration_abs_p95,
            active_probability=self.active_probability,
            duration_median=self.duration_median,
            amplitude_median=self.amplitude_median,
            peak_rate_median=self.peak_rate_median,
            fragment_count_median=self.fragment_count_median,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TeacherMotionModel":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                teacher_ids=tuple(str(value) for value in data["teacher_ids"]),
                move_ids=data["move_ids"].astype(np.int64),
                move_names=tuple(str(value) for value in data["move_names"]),
                regions=tuple(str(value) for value in data["regions"]),
                progress_grid=data["progress_grid"].astype(np.float64),
                speed_profiles=data["speed_profiles"].astype(np.float64),
                acceleration_profiles=data["acceleration_profiles"].astype(np.float64),
                active_profiles=data["active_profiles"].astype(bool),
                speed_median=data["speed_median"].astype(np.float64),
                speed_mad=data["speed_mad"].astype(np.float64),
                speed_p10=data["speed_p10"].astype(np.float64),
                speed_p90=data["speed_p90"].astype(np.float64),
                acceleration_median=data["acceleration_median"].astype(np.float64),
                acceleration_mad=data["acceleration_mad"].astype(np.float64),
                acceleration_p10=data["acceleration_p10"].astype(np.float64),
                acceleration_p90=data["acceleration_p90"].astype(np.float64),
                acceleration_abs_p95=data["acceleration_abs_p95"].astype(np.float64),
                active_probability=data["active_probability"].astype(np.float64),
                duration_median=data["duration_median"].astype(np.float64),
                amplitude_median=data["amplitude_median"].astype(np.float64),
                peak_rate_median=data["peak_rate_median"].astype(np.float64),
                fragment_count_median=data["fragment_count_median"].astype(np.float64),
            )

    def move_region_indices(self, move_name: str, region: str) -> tuple[int, int]:
        try:
            return self.move_names.index(move_name), self.regions.index(region)
        except ValueError as error:
            raise KeyError(f"Teacher model has no move/region {move_name}/{region}.") from error


@dataclass(frozen=True)
class KinematicQualityMetrics:
    active_mean_speed_ratio: float
    speed_variance_ratio: float
    velocity_profile_nmae: float
    velocity_profile_correlation: float
    velocity_outlier_ratio: float
    velocity_wasserstein_distance: float
    acceleration_profile_nmae: float
    acceleration_outlier_ratio: float
    acceleration_peak_count: int
    acceleration_peak_rate: float
    acceleration_peak_excess_ratio: float
    student_fragment_count: int
    motion_fragmentation_ratio: float
    valid_speed_frames: int
    valid_acceleration_frames: int


def resample_profile(
    progress: np.ndarray,
    values: np.ndarray,
    progress_grid: np.ndarray,
) -> np.ndarray:
    """Median-collapse duplicate progress values and interpolate onto a fixed grid."""
    phase = np.asarray(progress, dtype=np.float64)
    signal = np.asarray(values, dtype=np.float64)
    grid = np.asarray(progress_grid, dtype=np.float64)
    valid = np.isfinite(phase) & np.isfinite(signal)
    if np.sum(valid) < 2:
        return np.full(grid.shape, np.nan, dtype=np.float64)
    phase = np.clip(phase[valid], 0.0, 1.0)
    signal = signal[valid]
    order = np.argsort(phase, kind="stable")
    phase = phase[order]
    signal = signal[order]
    unique, inverse = np.unique(phase, return_inverse=True)
    collapsed = np.asarray(
        [np.median(signal[inverse == index]) for index in range(len(unique))],
        dtype=np.float64,
    )
    if len(unique) == 1:
        return np.full(grid.shape, collapsed[0], dtype=np.float64)
    return np.interp(grid, unique, collapsed, left=collapsed[0], right=collapsed[-1])


def profile_statistics(
    profiles: np.ndarray,
    *,
    window_radius: int = 2,
) -> dict[str, np.ndarray]:
    """Compute robust phase-local statistics over teachers and nearby phase points."""
    values = np.asarray(profiles, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError(f"Expected profiles shaped (teachers, progress), got {values.shape}.")
    if window_radius < 0:
        raise ValueError("window_radius must be non-negative.")
    stats = {
        name: np.full(values.shape[1], np.nan, dtype=np.float64)
        for name in ("median", "mad", "p10", "p90", "abs_p95")
    }
    for index in range(values.shape[1]):
        start = max(index - window_radius, 0)
        end = min(index + window_radius + 1, values.shape[1])
        local = values[:, start:end]
        local = local[np.isfinite(local)]
        if len(local) == 0:
            continue
        median = float(np.median(local))
        stats["median"][index] = median
        stats["mad"][index] = float(np.median(np.abs(local - median)))
        stats["p10"][index] = float(np.percentile(local, 10.0))
        stats["p90"][index] = float(np.percentile(local, 90.0))
        stats["abs_p95"][index] = float(np.percentile(np.abs(local), 95.0))
    return stats


def count_acceleration_peaks(
    acceleration: np.ndarray,
    threshold: np.ndarray | float,
    valid: np.ndarray,
    fps: float,
    *,
    minimum_separation_seconds: float = 0.20,
) -> np.ndarray:
    values = np.abs(np.asarray(acceleration, dtype=np.float64))
    valid_mask = np.asarray(valid, dtype=bool) & np.isfinite(values)
    heights = np.broadcast_to(np.asarray(threshold, dtype=np.float64), values.shape)
    candidates = np.where(valid_mask, values, 0.0)
    peaks, _ = find_peaks(
        candidates,
        height=np.where(np.isfinite(heights), heights, np.inf),
        distance=max(int(round(minimum_separation_seconds * fps)), 1),
    )
    return peaks[valid_mask[peaks]]


def motion_fragment_count(
    speed: np.ndarray,
    valid: np.ndarray,
    fps: float,
    *,
    threshold: float,
    minimum_active_seconds: float = 0.25,
    bridge_gap_seconds: float = 0.10,
) -> tuple[int, np.ndarray]:
    active = np.asarray(valid, dtype=bool) & np.isfinite(speed) & (speed > threshold)
    active = bridge_short_false_gaps(
        active,
        max(int(round(bridge_gap_seconds * fps)), 0),
    )
    active = remove_short_true_runs(
        active,
        max(int(round(minimum_active_seconds * fps)), 1),
    )
    starts = active & ~np.concatenate(([False], active[:-1]))
    return int(np.sum(starts)), active


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if np.isfinite(denominator) and denominator > 0 else float("nan")


def _profile_correlation(student: np.ndarray, teacher: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(mask, dtype=bool) & np.isfinite(student) & np.isfinite(teacher)
    if np.sum(selected) < 3:
        return float("nan")
    left = student[selected]
    right = teacher[selected]
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def analyze_against_teacher_model(
    model: TeacherMotionModel,
    move_name: str,
    region: str,
    progress: np.ndarray,
    speed: np.ndarray,
    acceleration: np.ndarray,
    valid_speed: np.ndarray,
    valid_acceleration: np.ndarray,
    fps: float,
    *,
    active_probability_threshold: float = 0.60,
) -> tuple[KinematicQualityMetrics, dict[str, np.ndarray]]:
    move_index, region_index = model.move_region_indices(move_name, region)
    phase = np.clip(np.asarray(progress, dtype=np.float64), 0.0, 1.0)
    student_speed = np.asarray(speed, dtype=np.float64)
    student_acceleration = np.asarray(acceleration, dtype=np.float64)
    speed_valid = np.asarray(valid_speed, dtype=bool) & np.isfinite(student_speed)
    acceleration_valid = np.asarray(valid_acceleration, dtype=bool) & np.isfinite(student_acceleration)
    if not (len(phase) == len(student_speed) == len(student_acceleration)):
        raise ValueError("Progress, speed and acceleration must have identical lengths.")

    grid = model.progress_grid
    speed_median = model.speed_median[move_index, region_index]
    speed_mad = model.speed_mad[move_index, region_index]
    speed_p10 = model.speed_p10[move_index, region_index]
    speed_p90 = model.speed_p90[move_index, region_index]
    acceleration_median = model.acceleration_median[move_index, region_index]
    acceleration_mad = model.acceleration_mad[move_index, region_index]
    acceleration_abs_p95 = model.acceleration_abs_p95[move_index, region_index]
    active_probability = model.active_probability[move_index, region_index]

    expected_speed = np.interp(phase, grid, speed_median)
    expected_speed_mad = np.interp(phase, grid, speed_mad)
    expected_speed_p10 = np.interp(phase, grid, speed_p10)
    expected_speed_p90 = np.interp(phase, grid, speed_p90)
    expected_acceleration = np.interp(phase, grid, acceleration_median)
    expected_acceleration_mad = np.interp(phase, grid, acceleration_mad)
    expected_acceleration_abs_p95 = np.interp(phase, grid, acceleration_abs_p95)
    active_on_student = np.interp(phase, grid, active_probability) >= active_probability_threshold

    active_speed = active_on_student & speed_valid
    active_acceleration = active_on_student & acceleration_valid
    global_speed_scale = max(float(np.nanpercentile(speed_median, 90.0)) * 0.05, 1e-6)
    speed_scale = np.maximum(1.4826 * expected_speed_mad, global_speed_scale)
    global_acceleration_scale = max(
        float(np.nanpercentile(np.abs(acceleration_median), 90.0)) * 0.05,
        1e-6,
    )
    acceleration_scale = np.maximum(
        1.4826 * expected_acceleration_mad,
        global_acceleration_scale,
    )

    student_speed_profile = resample_profile(phase, student_speed, grid)
    student_acceleration_profile = resample_profile(phase, student_acceleration, grid)
    canonical_active = active_probability >= active_probability_threshold
    teacher_speed_samples = model.speed_profiles[:, move_index, region_index][:, canonical_active]
    teacher_speed_samples = teacher_speed_samples[np.isfinite(teacher_speed_samples)]

    teacher_mean_speed = float(np.mean(expected_speed[active_speed])) if np.any(active_speed) else float("nan")
    student_mean_speed = float(np.mean(student_speed[active_speed])) if np.any(active_speed) else float("nan")
    teacher_variance = float(np.var(teacher_speed_samples)) if len(teacher_speed_samples) else float("nan")
    student_variance = float(np.var(student_speed[active_speed])) if np.any(active_speed) else float("nan")
    velocity_nmae = (
        float(np.mean(np.abs(student_speed[active_speed] - expected_speed[active_speed]) / speed_scale[active_speed]))
        if np.any(active_speed)
        else float("nan")
    )
    velocity_outlier = (
        float(
            np.mean(
                (student_speed[active_speed] < expected_speed_p10[active_speed])
                | (student_speed[active_speed] > expected_speed_p90[active_speed])
            )
        )
        if np.any(active_speed)
        else float("nan")
    )
    speed_normalizer = max(float(np.nanmedian(speed_median[canonical_active])), 1e-6)
    velocity_wasserstein = (
        float(wasserstein_distance(student_speed[active_speed], teacher_speed_samples) / speed_normalizer)
        if np.any(active_speed) and len(teacher_speed_samples)
        else float("nan")
    )
    acceleration_nmae = (
        float(
            np.mean(
                np.abs(student_acceleration[active_acceleration] - expected_acceleration[active_acceleration])
                / acceleration_scale[active_acceleration]
            )
        )
        if np.any(active_acceleration)
        else float("nan")
    )
    acceleration_outlier = (
        float(
            np.mean(
                np.abs(student_acceleration[active_acceleration])
                > expected_acceleration_abs_p95[active_acceleration]
            )
        )
        if np.any(active_acceleration)
        else float("nan")
    )
    peaks = count_acceleration_peaks(
        student_acceleration,
        expected_acceleration_abs_p95,
        active_acceleration,
        fps,
    )
    duration = len(student_speed) / fps
    peak_rate = len(peaks) / duration if duration > 0 else float("nan")
    teacher_peak_rate = float(model.peak_rate_median[move_index, region_index])
    peak_ratio = _safe_ratio(peak_rate, teacher_peak_rate)
    peak_excess = max(0.0, peak_ratio - 1.0) if np.isfinite(peak_ratio) else float("nan")

    noise = robust_noise_floor(student_speed[speed_valid])
    activity_threshold = max(noise, global_speed_scale)
    fragment_count, student_active = motion_fragment_count(
        student_speed,
        speed_valid,
        fps,
        threshold=activity_threshold,
    )
    teacher_fragments = float(model.fragment_count_median[move_index, region_index])
    fragmentation_ratio = _safe_ratio(float(fragment_count), teacher_fragments)

    metrics = KinematicQualityMetrics(
        active_mean_speed_ratio=_safe_ratio(student_mean_speed, teacher_mean_speed),
        speed_variance_ratio=_safe_ratio(student_variance, teacher_variance),
        velocity_profile_nmae=velocity_nmae,
        velocity_profile_correlation=_profile_correlation(
            student_speed_profile,
            speed_median,
            canonical_active,
        ),
        velocity_outlier_ratio=velocity_outlier,
        velocity_wasserstein_distance=velocity_wasserstein,
        acceleration_profile_nmae=acceleration_nmae,
        acceleration_outlier_ratio=acceleration_outlier,
        acceleration_peak_count=int(len(peaks)),
        acceleration_peak_rate=float(peak_rate),
        acceleration_peak_excess_ratio=peak_excess,
        student_fragment_count=fragment_count,
        motion_fragmentation_ratio=fragmentation_ratio,
        valid_speed_frames=int(np.sum(active_speed)),
        valid_acceleration_frames=int(np.sum(active_acceleration)),
    )
    arrays = {
        "student_speed": student_speed,
        "student_acceleration": student_acceleration,
        "student_speed_profile": student_speed_profile,
        "student_acceleration_profile": student_acceleration_profile,
        "expected_speed": expected_speed,
        "expected_speed_p10": expected_speed_p10,
        "expected_speed_p90": expected_speed_p90,
        "expected_acceleration": expected_acceleration,
        "expected_acceleration_abs_p95": expected_acceleration_abs_p95,
        "teacher_active_on_student": active_on_student,
        "student_active": student_active,
        "acceleration_peaks": peaks,
    }
    return metrics, arrays
