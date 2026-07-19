"""Full-frame SMPL motion-quality and temporal-fluency diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


# SMPL body_pose excludes Pelvis/global orientation, so these are SMPL-24 IDs minus one.
BODY_REGIONS: dict[str, tuple[int, ...]] = {
    "left_arm": (15, 17, 19),
    "right_arm": (16, 18, 20),
    "left_leg": (0, 3, 6, 9),
    "right_leg": (1, 4, 7, 10),
    "trunk": (2, 5, 8, 11),
}
BODY_REGIONS["whole_body"] = tuple(
    sorted({joint for joints in BODY_REGIONS.values() for joint in joints})
)


@dataclass(frozen=True)
class MotionSignals:
    smoothed_rotations: np.ndarray
    joint_speed_degrees_per_second: np.ndarray
    joint_speed_change_degrees_per_second2: np.ndarray
    region_intensity: dict[str, np.ndarray]
    region_speed_change: dict[str, np.ndarray]
    valid_frames: np.ndarray
    acceleration_valid_frames: np.ndarray
    smoothing_window_frames: int
    excluded_track_switch_frames: int


@dataclass(frozen=True)
class PauseStatistics:
    pause_mask: np.ndarray
    pause_ratio: float
    pause_count: int
    pause_duration_seconds: float
    longest_pause_seconds: float
    threshold_degrees_per_second: float


@dataclass(frozen=True)
class RegionQualityMetrics:
    region: str
    teacher_noise_floor: float
    student_noise_floor: float
    teacher_active_threshold: float
    student_pause_threshold: float
    teacher_pause_ratio: float
    student_pause_ratio: float
    pause_excess_ratio: float
    student_pause_count: int
    student_pause_duration_seconds: float
    student_longest_pause_seconds: float
    aligned_active_pause_ratio: float
    tempo_stall_ratio: float
    duration_ratio: float
    local_tempo_distortion: float
    amplitude_teacher: float
    amplitude_student: float
    amplitude_ratio: float
    aligned_teacher_active_frames: int
    excluded_student_frames: int
    excluded_teacher_frames: int


@dataclass(frozen=True)
class PauseEvent:
    subject: str
    region: str
    start_index: int
    end_index: int
    duration_seconds: float
    teacher_progress_start: float | None
    teacher_progress_end: float | None
    teacher_active_fraction: float | None
    reason: str


@dataclass(frozen=True)
class TempoBin:
    region: str
    bin_index: int
    teacher_progress_start: float
    teacher_progress_end: float
    teacher_duration_seconds: float
    student_duration_seconds: float
    duration_ratio: float


def _validate_rotations(rotations: np.ndarray) -> np.ndarray:
    values = np.asarray(rotations, dtype=np.float64)
    if values.ndim != 4 or values.shape[1:] != (23, 3, 3):
        raise ValueError(f"Expected rotations shaped (T, 23, 3, 3), got {values.shape}.")
    if len(values) < 2:
        raise ValueError("At least two frames are required for motion analysis.")
    if not np.isfinite(values).all():
        raise ValueError("Rotation sequence contains non-finite values.")
    return values


def smoothing_window_frames(fps: float, seconds: float) -> int:
    if fps <= 0 or seconds < 0:
        raise ValueError(f"Expected fps > 0 and seconds >= 0, got fps={fps}, seconds={seconds}.")
    if seconds == 0:
        return 1
    window = max(int(round(fps * seconds)), 1)
    if window % 2 == 0:
        window += 1
    return window


def smooth_rotations_so3(rotations: np.ndarray, window_frames: int) -> np.ndarray:
    """Center-average rotation matrices and project the average back to SO(3)."""
    values = _validate_rotations(rotations)
    if window_frames <= 0 or window_frames % 2 == 0:
        raise ValueError("window_frames must be a positive odd integer.")
    if window_frames == 1:
        return values.copy()

    radius = window_frames // 2
    padded = np.pad(values, ((radius, radius), (0, 0), (0, 0), (0, 0)), mode="edge")
    cumulative = np.concatenate(
        (
            np.zeros((1,) + padded.shape[1:], dtype=np.float64),
            np.cumsum(padded, axis=0),
        ),
        axis=0,
    )
    averages = (cumulative[window_frames:] - cumulative[:-window_frames]) / window_frames
    flat = averages.reshape(-1, 3, 3)
    u, _, vh = np.linalg.svd(flat)
    projected = u @ vh
    negative = np.linalg.det(projected) < 0
    if np.any(negative):
        u[negative, :, -1] *= -1.0
        projected[negative] = u[negative] @ vh[negative]
    return projected.reshape(values.shape)


def track_valid_frame_mask(
    frame_count: int,
    source_track_ids: np.ndarray | None,
) -> tuple[np.ndarray, int]:
    valid = np.ones(frame_count, dtype=bool)
    if source_track_ids is None:
        return valid, 0
    track_ids = np.asarray(source_track_ids)
    if track_ids.shape != (frame_count,):
        raise ValueError(f"source_track_ids must have shape ({frame_count},), got {track_ids.shape}.")
    switches = np.flatnonzero(track_ids[1:] != track_ids[:-1]) + 1
    for switch in switches:
        valid[max(switch - 1, 0) : min(switch + 1, frame_count)] = False
    return valid, int(np.sum(~valid))


def compute_motion_signals(
    rotations: np.ndarray,
    fps: float,
    *,
    source_track_ids: np.ndarray | None = None,
    smoothing_seconds: float = 0.20,
) -> MotionSignals:
    values = _validate_rotations(rotations)
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")
    window = smoothing_window_frames(fps, smoothing_seconds)
    if source_track_ids is None:
        smoothed = smooth_rotations_so3(values, window)
    else:
        track_ids = np.asarray(source_track_ids)
        if track_ids.shape != (len(values),):
            raise ValueError(
                f"source_track_ids must have shape ({len(values)},), got {track_ids.shape}."
            )
        smoothed = np.empty_like(values)
        starts = np.concatenate(([0], np.flatnonzero(track_ids[1:] != track_ids[:-1]) + 1))
        ends = np.concatenate((starts[1:], [len(values)]))
        for start, end in zip(starts, ends):
            if end - start == 1:
                smoothed[start:end] = values[start:end]
            else:
                smoothed[start:end] = smooth_rotations_so3(values[start:end], window)
    relative = smoothed[1:] @ np.swapaxes(smoothed[:-1], -1, -2)
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0
    step_degrees = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    frame_valid, excluded = track_valid_frame_mask(len(values), source_track_ids)
    transition_valid = frame_valid[1:] & frame_valid[:-1]
    speeds = np.full((len(values), 23), np.nan, dtype=np.float64)
    speeds[1:] = step_degrees * fps
    speeds[1:][~transition_valid] = np.nan
    frame_valid = frame_valid & np.isfinite(speeds).all(axis=1)

    intensities: dict[str, np.ndarray] = {}
    for region, joints in BODY_REGIONS.items():
        region_speeds = speeds[:, np.asarray(joints, dtype=np.int64)]
        squared = region_speeds * region_speeds
        finite_count = np.sum(np.isfinite(squared), axis=1)
        mean_squared = np.divide(
            np.nansum(squared, axis=1),
            finite_count,
            out=np.full(len(values), np.nan, dtype=np.float64),
            where=finite_count > 0,
        )
        intensity = np.sqrt(mean_squared)
        intensity[~frame_valid] = np.nan
        intensities[region] = intensity

    speed_changes = np.full_like(speeds, np.nan)
    speed_changes[1:-1] = (speeds[2:] - speeds[:-2]) * (fps / 2.0)
    acceleration_valid = np.zeros(len(values), dtype=bool)
    if len(values) > 2:
        acceleration_valid[1:-1] = (
            frame_valid[:-2]
            & frame_valid[1:-1]
            & frame_valid[2:]
            & np.isfinite(speed_changes[1:-1]).all(axis=1)
        )
    speed_changes[~acceleration_valid] = np.nan

    region_speed_changes: dict[str, np.ndarray] = {}
    for region, intensity in intensities.items():
        changes = np.full(len(values), np.nan, dtype=np.float64)
        changes[1:-1] = (intensity[2:] - intensity[:-2]) * (fps / 2.0)
        changes[~acceleration_valid] = np.nan
        region_speed_changes[region] = changes
    return MotionSignals(
        smoothed_rotations=smoothed,
        joint_speed_degrees_per_second=speeds,
        joint_speed_change_degrees_per_second2=speed_changes,
        region_intensity=intensities,
        region_speed_change=region_speed_changes,
        valid_frames=frame_valid,
        acceleration_valid_frames=acceleration_valid,
        smoothing_window_frames=window,
        excluded_track_switch_frames=excluded,
    )


def robust_noise_floor(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("nan")
    cutoff = float(np.percentile(finite, 25.0))
    upper = float(np.percentile(finite, 90.0))
    # A sequence that moves at nearly constant speed has no identifiable
    # low-speed noise cluster; treating its lower quartile as noise would mark
    # the entire motion as a pause.
    if upper > 0 and cutoff > 0.5 * upper:
        return 0.0
    low = finite[finite <= cutoff]
    median = float(np.median(low))
    mad = float(np.median(np.abs(low - median)))
    return median + 3.0 * 1.4826 * mad


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def remove_short_true_runs(mask: np.ndarray, minimum_frames: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    for start, end in _runs(result):
        if end - start + 1 < minimum_frames:
            result[start : end + 1] = False
    return result


def bridge_short_false_gaps(mask: np.ndarray, maximum_gap_frames: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    if maximum_gap_frames <= 0:
        return result
    for start, end in _runs(~result):
        if (
            start > 0
            and end < len(result) - 1
            and end - start + 1 <= maximum_gap_frames
            and result[start - 1]
            and result[end + 1]
        ):
            result[start : end + 1] = True
    return result


def hysteresis_activity_mask(
    intensity: np.ndarray,
    valid: np.ndarray,
    enter_threshold: float,
    *,
    exit_ratio: float = 0.7,
    minimum_active_seconds: float = 0.25,
    fps: float,
) -> np.ndarray:
    if not 0 < exit_ratio <= 1:
        raise ValueError("exit_ratio must be in (0, 1].")
    values = np.asarray(intensity, dtype=np.float64)
    valid_values = np.asarray(valid, dtype=bool)
    active = np.zeros(len(values), dtype=bool)
    state = False
    exit_threshold = enter_threshold * exit_ratio
    for index, value in enumerate(values):
        if not valid_values[index] or not np.isfinite(value):
            state = False
        elif not state and value > enter_threshold:
            state = True
        elif state and value < exit_threshold:
            state = False
        active[index] = state
    return remove_short_true_runs(active, max(int(round(minimum_active_seconds * fps)), 1))


def detect_pauses(
    intensity: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    fps: float,
    *,
    minimum_pause_seconds: float = 0.40,
    bridge_gap_seconds: float = 0.10,
) -> PauseStatistics:
    values = np.asarray(intensity, dtype=np.float64)
    valid_values = np.asarray(valid, dtype=bool)
    candidates = valid_values & np.isfinite(values) & (values <= threshold)
    candidates = bridge_short_false_gaps(
        candidates,
        max(int(round(bridge_gap_seconds * fps)), 0),
    )
    pause_mask = remove_short_true_runs(
        candidates,
        max(int(round(minimum_pause_seconds * fps)), 1),
    )
    runs = _runs(pause_mask)
    durations = [(end - start + 1) / fps for start, end in runs]
    valid_count = int(np.sum(valid_values))
    return PauseStatistics(
        pause_mask=pause_mask,
        pause_ratio=float(np.sum(pause_mask) / valid_count) if valid_count else float("nan"),
        pause_count=len(runs),
        pause_duration_seconds=float(np.sum(pause_mask) / fps),
        longest_pause_seconds=float(max(durations, default=0.0)),
        threshold_degrees_per_second=float(threshold),
    )


def dilate_mask(mask: np.ndarray, radius_frames: int) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if radius_frames <= 0:
        return values.copy()
    kernel = np.ones(2 * radius_frames + 1, dtype=np.int32)
    return np.convolve(values.astype(np.int32), kernel, mode="same") > 0


def build_reference_progress(
    path: np.ndarray,
    target_sample_indices: np.ndarray,
    target_source_indices: np.ndarray,
    target_source_frames: np.ndarray,
    reference_start_index: int,
    reference_end_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the median DTW reference match onto target source frames."""
    path_values = np.asarray(path, dtype=np.int64)
    sample_indices = np.asarray(target_sample_indices, dtype=np.int64)
    sample_sources = np.asarray(target_source_indices, dtype=np.float64)
    source_frames = np.asarray(target_source_frames, dtype=np.float64)
    if path_values.ndim != 2 or path_values.shape[1] != 2:
        raise ValueError(f"Expected DTW path shaped (P, 2), got {path_values.shape}.")
    if sample_indices.shape != sample_sources.shape:
        raise ValueError("target_sample_indices and target_source_indices must have identical shapes.")
    if reference_end_index <= reference_start_index:
        raise ValueError("Reference interval must contain at least two sampled frames.")

    known_sources: list[float] = []
    known_references: list[float] = []
    for sample_index, source_index in zip(sample_indices, sample_sources):
        candidates = path_values[path_values[:, 0] == sample_index, 1]
        candidates = candidates[
            (candidates >= reference_start_index) & (candidates <= reference_end_index)
        ]
        if len(candidates):
            known_sources.append(float(source_index))
            known_references.append(float(np.median(candidates)))
    if not known_sources:
        raise RuntimeError("DTW path has no matches inside the requested move interval.")

    known_sources_array = np.asarray(known_sources, dtype=np.float64)
    known_references_array = np.maximum.accumulate(
        np.asarray(known_references, dtype=np.float64)
    )
    reference_indices = np.interp(
        source_frames,
        known_sources_array,
        known_references_array,
        left=known_references_array[0],
        right=known_references_array[-1],
    )
    progress = (reference_indices - reference_start_index) / (
        reference_end_index - reference_start_index
    )
    return np.clip(progress, 0.0, 1.0), reference_indices


def map_reference_mask(
    reference_mask: np.ndarray,
    reference_progress: np.ndarray,
    *,
    dilation_frames: int = 0,
) -> np.ndarray:
    mask = dilate_mask(reference_mask, dilation_frames)
    positions = np.rint(
        np.clip(reference_progress, 0.0, 1.0) * (len(mask) - 1)
    ).astype(np.int64)
    return mask[positions]


def tempo_stall_mask(
    reference_progress: np.ndarray,
    teacher_active_on_student: np.ndarray,
    valid_student: np.ndarray,
    student_fps: float,
    student_duration: float,
    teacher_duration: float,
    *,
    window_seconds: float = 0.60,
    speed_ratio_threshold: float = 0.25,
    minimum_stall_seconds: float = 0.40,
) -> tuple[np.ndarray, np.ndarray, float]:
    progress = np.asarray(reference_progress, dtype=np.float64)
    if len(progress) < 2:
        raise ValueError("At least two progress samples are required.")
    window = max(int(round(window_seconds * student_fps)), 1)
    half = max(window // 2, 1)
    left = np.maximum(np.arange(len(progress)) - half, 0)
    right = np.minimum(np.arange(len(progress)) + half, len(progress) - 1)
    elapsed = (right - left) / student_fps
    progress_speed = np.divide(
        progress[right] - progress[left],
        elapsed,
        out=np.zeros_like(progress),
        where=elapsed > 0,
    )
    baseline = 1.0 / student_duration
    # Equivalent to comparing teacher-time speed against teacher_duration/student_duration.
    threshold = baseline * speed_ratio_threshold
    candidates = (
        np.asarray(valid_student, dtype=bool)
        & np.asarray(teacher_active_on_student, dtype=bool)
        & (progress_speed < threshold)
    )
    stall = remove_short_true_runs(
        candidates,
        max(int(round(minimum_stall_seconds * student_fps)), 1),
    )
    return stall, progress_speed, teacher_duration / student_duration


def tempo_bins(
    reference_progress: np.ndarray,
    student_fps: float,
    teacher_duration: float,
    region: str,
    *,
    bin_count: int = 10,
) -> tuple[list[TempoBin], float]:
    if bin_count <= 0:
        raise ValueError("bin_count must be positive.")
    progress = np.asarray(reference_progress, dtype=np.float64)
    rows: list[TempoBin] = []
    ratios: list[float] = []
    teacher_bin_duration = teacher_duration / bin_count
    for bin_index in range(bin_count):
        lower = bin_index / bin_count
        upper = (bin_index + 1) / bin_count
        if bin_index == bin_count - 1:
            selected = (progress >= lower) & (progress <= upper)
        else:
            selected = (progress >= lower) & (progress < upper)
        student_duration = float(np.sum(selected) / student_fps)
        ratio = (
            student_duration / teacher_bin_duration
            if teacher_bin_duration > 0 and student_duration > 0
            else float("nan")
        )
        if np.isfinite(ratio) and ratio > 0:
            ratios.append(ratio)
        rows.append(
            TempoBin(
                region=region,
                bin_index=bin_index,
                teacher_progress_start=lower,
                teacher_progress_end=upper,
                teacher_duration_seconds=teacher_bin_duration,
                student_duration_seconds=student_duration,
                duration_ratio=ratio,
            )
        )
    distortion = float(np.median(np.abs(np.log(ratios)))) if ratios else float("nan")
    return rows, distortion


def integrate_intensity(intensity: np.ndarray, valid: np.ndarray, fps: float) -> float:
    values = np.asarray(intensity, dtype=np.float64)
    selected = np.asarray(valid, dtype=bool) & np.isfinite(values)
    return float(np.sum(values[selected]) / fps)


def analyze_region(
    region: str,
    teacher_signals: MotionSignals,
    student_signals: MotionSignals,
    teacher_progress: np.ndarray,
    teacher_duration: float,
    student_duration: float,
    teacher_fps: float,
    student_fps: float,
    *,
    active_scale: float = 0.10,
    pause_scale: float = 0.05,
    activity_exit_ratio: float = 0.7,
    minimum_active_seconds: float = 0.25,
    minimum_pause_seconds: float = 0.40,
    bridge_gap_seconds: float = 0.10,
    activity_dilation_seconds: float = 0.40,
    stall_window_seconds: float = 0.60,
    stall_speed_ratio: float = 0.25,
    minimum_stall_seconds: float = 0.40,
    tempo_bin_count: int = 10,
) -> tuple[
    RegionQualityMetrics,
    list[PauseEvent],
    list[TempoBin],
    dict[str, np.ndarray],
]:
    teacher_intensity = teacher_signals.region_intensity[region]
    student_intensity = student_signals.region_intensity[region]
    teacher_noise = robust_noise_floor(teacher_intensity)
    student_noise = robust_noise_floor(student_intensity)
    teacher_p90 = float(np.nanpercentile(teacher_intensity, 90.0))
    active_threshold = max(teacher_noise, active_scale * teacher_p90)
    pause_threshold = max(student_noise, pause_scale * teacher_p90)

    teacher_active = hysteresis_activity_mask(
        teacher_intensity,
        teacher_signals.valid_frames,
        active_threshold,
        exit_ratio=activity_exit_ratio,
        minimum_active_seconds=minimum_active_seconds,
        fps=teacher_fps,
    )
    teacher_pause = detect_pauses(
        teacher_intensity,
        teacher_signals.valid_frames,
        active_threshold * activity_exit_ratio,
        teacher_fps,
        minimum_pause_seconds=minimum_pause_seconds,
        bridge_gap_seconds=bridge_gap_seconds,
    )
    student_pause = detect_pauses(
        student_intensity,
        student_signals.valid_frames,
        pause_threshold,
        student_fps,
        minimum_pause_seconds=minimum_pause_seconds,
        bridge_gap_seconds=bridge_gap_seconds,
    )
    mapped_teacher_active = map_reference_mask(
        teacher_active,
        teacher_progress,
        dilation_frames=max(int(round(activity_dilation_seconds * teacher_fps)), 0),
    )
    active_valid = mapped_teacher_active & student_signals.valid_frames
    active_count = int(np.sum(active_valid))
    aligned_pause_count = int(np.sum(active_valid & student_pause.pause_mask))
    aligned_active_pause_ratio = (
        aligned_pause_count / active_count if active_count else float("nan")
    )
    stall_mask, progress_speed, expected_teacher_time_speed = tempo_stall_mask(
        teacher_progress,
        mapped_teacher_active,
        student_signals.valid_frames,
        student_fps,
        student_duration,
        teacher_duration,
        window_seconds=stall_window_seconds,
        speed_ratio_threshold=stall_speed_ratio,
        minimum_stall_seconds=minimum_stall_seconds,
    )
    tempo_stall_ratio = (
        float(np.sum(stall_mask) / active_count) if active_count else float("nan")
    )
    bins, distortion = tempo_bins(
        teacher_progress,
        student_fps,
        teacher_duration,
        region,
        bin_count=tempo_bin_count,
    )
    teacher_amplitude = integrate_intensity(
        teacher_intensity,
        teacher_signals.valid_frames,
        teacher_fps,
    )
    student_amplitude = integrate_intensity(
        student_intensity,
        student_signals.valid_frames,
        student_fps,
    )
    amplitude_ratio = (
        student_amplitude / teacher_amplitude
        if teacher_amplitude > 0
        else float("nan")
    )
    metrics = RegionQualityMetrics(
        region=region,
        teacher_noise_floor=teacher_noise,
        student_noise_floor=student_noise,
        teacher_active_threshold=active_threshold,
        student_pause_threshold=pause_threshold,
        teacher_pause_ratio=teacher_pause.pause_ratio,
        student_pause_ratio=student_pause.pause_ratio,
        pause_excess_ratio=max(0.0, student_pause.pause_ratio - teacher_pause.pause_ratio),
        student_pause_count=student_pause.pause_count,
        student_pause_duration_seconds=student_pause.pause_duration_seconds,
        student_longest_pause_seconds=student_pause.longest_pause_seconds,
        aligned_active_pause_ratio=aligned_active_pause_ratio,
        tempo_stall_ratio=tempo_stall_ratio,
        duration_ratio=student_duration / teacher_duration,
        local_tempo_distortion=distortion,
        amplitude_teacher=teacher_amplitude,
        amplitude_student=student_amplitude,
        amplitude_ratio=amplitude_ratio,
        aligned_teacher_active_frames=active_count,
        excluded_student_frames=int(np.sum(~student_signals.valid_frames)),
        excluded_teacher_frames=int(np.sum(~teacher_signals.valid_frames)),
    )

    events: list[PauseEvent] = []
    for subject, statistics, fps, progress, active_on_subject in (
        ("teacher", teacher_pause, teacher_fps, None, None),
        ("student", student_pause, student_fps, teacher_progress, mapped_teacher_active),
    ):
        for start, end in _runs(statistics.pause_mask):
            if progress is None:
                progress_start = None
                progress_end = None
                active_fraction = None
                reason = "teacher_pause"
            else:
                progress_start = float(progress[start])
                progress_end = float(progress[end])
                active_fraction = float(np.mean(active_on_subject[start : end + 1]))
                reason = (
                    "aligned_active_pause"
                    if active_fraction > 0
                    else "student_pause_teacher_inactive"
                )
            events.append(
                PauseEvent(
                    subject=subject,
                    region=region,
                    start_index=start,
                    end_index=end,
                    duration_seconds=(end - start + 1) / fps,
                    teacher_progress_start=progress_start,
                    teacher_progress_end=progress_end,
                    teacher_active_fraction=active_fraction,
                    reason=reason,
                )
            )
    arrays = {
        "teacher_intensity": teacher_intensity,
        "student_intensity": student_intensity,
        "teacher_active": teacher_active,
        "teacher_active_on_student": mapped_teacher_active,
        "teacher_pause": teacher_pause.pause_mask,
        "student_pause": student_pause.pause_mask,
        "tempo_stall": stall_mask,
        "reference_progress_speed": progress_speed,
        "expected_teacher_time_speed": np.asarray(
            [expected_teacher_time_speed], dtype=np.float64
        ),
    }
    return metrics, events, bins, arrays
