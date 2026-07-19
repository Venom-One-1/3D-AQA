#!/usr/bin/env python
"""Analyze full-frame SMPL motion fluency for full videos and action clips."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

from aqa3d.motion_quality import (
    BODY_REGIONS,
    MotionSignals,
    PauseEvent,
    RegionQualityMetrics,
    TempoBin,
    analyze_region,
    build_reference_progress,
    compute_motion_signals,
)
from aqa3d.smpl_dtw import (
    VideoSampling,
    backtrack_dtw_path,
    dtw_from_cost_matrix,
    inspect_video_sampling,
    pairwise_geodesic_costs,
    uniform_sample_source_indices,
)
from aqa3d.tracking import TrackPoseSequence, load_stitched_primary_track
from run_student_tas_smpl_dtw import sampled_interval_to_source_interval
from run_tas_smpl_dtw_mapping import (
    ReferenceSegment,
    load_reference_segments,
    tracking_path,
    write_dict_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE_VIDEO_ROOT = Path("/home/sqw/VisualSearch/aqa/teach_trimmed")
DEFAULT_REFERENCE_TRACKING_ROOT = Path("/home/sqw/VisualSearch/aqa/Tracking/teach_trimmed")
DEFAULT_REFERENCE_ID = "QxVvRcRn2TA"
DEFAULT_REFERENCE_SEGMENTS = PROJECT_ROOT / "tas_annotations" / "QxVvRcRn2TA_segments_5fps.csv"
DEFAULT_STUDENT_VIDEO_ROOT = Path("/home/sqw/VisualSearch/aqa/student")
DEFAULT_STUDENT_TRACKING_ROOT = Path("/home/sqw/VisualSearch/aqa/Tracking/student_full")
DEFAULT_STUDENT_SEGMENTATION_ROOT = PROJECT_ROOT / "student_segmentation_results"
DEFAULT_CLIP_ROOT = Path("/home/sqw/VisualSearch/aqa/ActionSegments")
DEFAULT_CLIP_TRACKING_ROOT = Path("/home/sqw/VisualSearch/aqa/Tracking")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "motion_quality_results"
DEFAULT_HUMAN_RANKINGS = PROJECT_ROOT / "human_rankings.csv"


@dataclass(frozen=True)
class SourceSegment:
    move_id: int
    move_name: str
    start_frame: int
    end_frame: int
    start_frame_5fps: int
    end_frame_5fps: int

    @property
    def source_start_index(self) -> int:
        return self.start_frame - 1

    @property
    def source_end_index(self) -> int:
        return self.end_frame - 1


@dataclass(frozen=True)
class SegmentAnalysisInput:
    dataset: str
    case_id: str
    student_id: str
    move_id: int
    move_name: str
    teacher_video: Path
    student_video: Path
    teacher_track: TrackPoseSequence
    student_track: TrackPoseSequence
    teacher_segment: SourceSegment
    student_segment: SourceSegment
    teacher_fps: float
    student_fps: float
    path: np.ndarray
    path_local_costs: np.ndarray
    reference_sample_source_indices: np.ndarray
    target_sample_source_indices: np.ndarray


@dataclass(frozen=True)
class AnalysisConfig:
    sample_fps: float
    smoothing_seconds: float
    active_scale: float
    pause_scale: float
    activity_exit_ratio: float
    minimum_active_seconds: float
    minimum_pause_seconds: float
    bridge_gap_seconds: float
    activity_dilation_seconds: float
    stall_window_seconds: float
    stall_speed_ratio: float
    minimum_stall_seconds: float
    tempo_bin_count: int


def inspect_video(path: Path) -> tuple[float, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or frame_count <= 0:
        raise ValueError(f"Invalid video metadata for {path}: fps={fps}, frames={frame_count}.")
    return fps, frame_count


def select_track_segment(
    track: TrackPoseSequence,
    segment: SourceSegment,
) -> tuple[np.ndarray, np.ndarray | None]:
    source_indices = np.arange(
        segment.source_start_index,
        segment.source_end_index + 1,
        dtype=np.int64,
    )
    poses = track.at_source_frames(source_indices)
    if track.source_track_ids is None:
        return poses, None
    id_lookup = {
        int(frame): int(track_id)
        for frame, track_id in zip(track.frame_numbers, track.source_track_ids)
    }
    tracking_frames = source_indices + 1
    track_ids = np.asarray([id_lookup[int(frame)] for frame in tracking_frames], dtype=np.int64)
    return poses, track_ids


def load_student_segments(path: Path) -> list[SourceSegment]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    segments = [
        SourceSegment(
            move_id=int(row["move_id"]),
            move_name=row["move_name"],
            start_frame=int(row["start_frame"]),
            end_frame=int(row["end_frame"]),
            start_frame_5fps=int(row["start_frame_5fps"]),
            end_frame_5fps=int(row["end_frame_5fps"]),
        )
        for row in rows
    ]
    if [segment.move_id for segment in segments] != list(range(1, 25)):
        raise ValueError(f"Expected ordered student moves 1..24 in {path}.")
    return segments


def build_reference_source_segments(
    references: list[ReferenceSegment],
    sampling: VideoSampling,
) -> list[SourceSegment]:
    segments: list[SourceSegment] = []
    for reference in references:
        start_frame, end_frame = sampled_interval_to_source_interval(
            reference.start_frame,
            reference.end_frame,
            sampling,
        )
        segments.append(
            SourceSegment(
                move_id=reference.move_id,
                move_name=reference.move_name,
                start_frame=start_frame,
                end_frame=end_frame,
                start_frame_5fps=reference.start_frame,
                end_frame_5fps=reference.end_frame,
            )
        )
    return segments


def load_saved_dtw_path(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        target = data["target_sample_indices_0based"].astype(np.int64)
        reference = data["reference_sample_indices_0based"].astype(np.int64)
        local_costs = data["local_geodesic_radians"].astype(np.float64)
    return np.stack((target, reference), axis=1), local_costs


def sampled_source_indices_for_segment(
    segment: SourceSegment,
    fps: float,
    sample_fps: float,
) -> np.ndarray:
    frame_count = segment.end_frame - segment.start_frame + 1
    relative = uniform_sample_source_indices(frame_count, fps, sample_fps)
    return relative + segment.source_start_index


def legacy_qishi_source_range(video_path: Path) -> tuple[int, int]:
    fps, frame_count = inspect_video(video_path)
    interval = max(int(fps / 30), 1)
    sampled = np.arange(0, frame_count, interval, dtype=np.int64)
    keep = 15 * int(fps)
    selected = sampled[-keep:]
    return int(selected[0]), int(selected[-1])


def whole_clip_segment(video_path: Path, move_id: int, move_name: str, sample_fps: float) -> SourceSegment:
    fps, frame_count = inspect_video(video_path)
    if move_name == "qishi":
        start_index, end_index = legacy_qishi_source_range(video_path)
    else:
        start_index, end_index = 0, frame_count - 1
    sample_count = len(
        uniform_sample_source_indices(end_index - start_index + 1, fps, sample_fps)
    )
    return SourceSegment(
        move_id=move_id,
        move_name=move_name,
        start_frame=start_index + 1,
        end_frame=end_index + 1,
        start_frame_5fps=1,
        end_frame_5fps=sample_count,
    )


def infer_move(clip: Path) -> tuple[int, str]:
    parts = clip.stem.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Cannot infer student, move ID and move name from {clip.stem}.")
    return int(parts[-2]), parts[-1].lower()


def clip_tracking_path(root: Path, split: str, video: Path) -> Path:
    return root / split / video.stem / "results" / f"demo_{video.stem}.pkl"


def teacher_clip_map(teacher_root: Path, tracking_root: Path) -> dict[str, Path]:
    teachers: dict[str, Path] = {}
    for video in sorted(teacher_root.glob("*.mp4")):
        _, move = infer_move(video)
        if not clip_tracking_path(tracking_root, "teach", video).is_file():
            continue
        if video.stem.startswith(DEFAULT_REFERENCE_ID):
            teachers[move] = video
            continue
        teachers.setdefault(move, video)
    return teachers


def create_clip_analysis_input(
    student_video: Path,
    teacher_video: Path,
    *,
    tracking_root: Path,
    sample_fps: float,
    pairwise_chunk_size: int,
    dtw_coefficient: float,
) -> SegmentAnalysisInput:
    move_id, move_name = infer_move(student_video)
    teacher_move_id, teacher_move_name = infer_move(teacher_video)
    if move_name != teacher_move_name:
        raise ValueError(f"Move mismatch: {student_video} vs {teacher_video}.")
    teacher_fps, _ = inspect_video(teacher_video)
    student_fps, _ = inspect_video(student_video)
    teacher_segment = whole_clip_segment(teacher_video, teacher_move_id, move_name, sample_fps)
    student_segment = whole_clip_segment(student_video, move_id, move_name, sample_fps)
    teacher_track = load_stitched_primary_track(
        clip_tracking_path(tracking_root, "teach", teacher_video)
    )
    student_track = load_stitched_primary_track(
        clip_tracking_path(tracking_root, "student", student_video)
    )
    teacher_samples = sampled_source_indices_for_segment(
        teacher_segment, teacher_fps, sample_fps
    )
    student_samples = sampled_source_indices_for_segment(
        student_segment, student_fps, sample_fps
    )
    teacher_sampled_poses = teacher_track.at_source_frames(teacher_samples)
    student_sampled_poses = student_track.at_source_frames(student_samples)
    costs = pairwise_geodesic_costs(
        student_sampled_poses,
        teacher_sampled_poses,
        chunk_size=pairwise_chunk_size,
    )
    _, _, accumulated = dtw_from_cost_matrix(costs, dtw_coefficient)
    path = backtrack_dtw_path(accumulated)
    local_costs = costs[path[:, 0], path[:, 1]]
    student_id = student_video.stem.split("_", 1)[0]
    return SegmentAnalysisInput(
        dataset="clips",
        case_id=student_video.stem,
        student_id=student_id,
        move_id=move_id,
        move_name=move_name,
        teacher_video=teacher_video,
        student_video=student_video,
        teacher_track=teacher_track,
        student_track=student_track,
        teacher_segment=teacher_segment,
        student_segment=student_segment,
        teacher_fps=teacher_fps,
        student_fps=student_fps,
        path=path,
        path_local_costs=local_costs,
        reference_sample_source_indices=teacher_samples,
        target_sample_source_indices=student_samples,
    )


def _subset_path_cost(
    data: SegmentAnalysisInput,
) -> float:
    target_start = data.student_segment.start_frame_5fps - 1
    target_end = data.student_segment.end_frame_5fps - 1
    reference_start = data.teacher_segment.start_frame_5fps - 1
    reference_end = data.teacher_segment.end_frame_5fps - 1
    selected = (
        (data.path[:, 0] >= target_start)
        & (data.path[:, 0] <= target_end)
        & (data.path[:, 1] >= reference_start)
        & (data.path[:, 1] <= reference_end)
    )
    if not np.any(selected):
        return float("nan")
    return float(np.mean(data.path_local_costs[selected]))


def analyze_segment(
    data: SegmentAnalysisInput,
    config: AnalysisConfig,
) -> tuple[
    list[dict],
    list[dict],
    list[dict],
    dict[str, np.ndarray],
    dict,
]:
    teacher_poses, teacher_ids = select_track_segment(
        data.teacher_track, data.teacher_segment
    )
    student_poses, student_ids = select_track_segment(
        data.student_track, data.student_segment
    )
    teacher_signals = compute_motion_signals(
        teacher_poses,
        data.teacher_fps,
        source_track_ids=teacher_ids,
        smoothing_seconds=config.smoothing_seconds,
    )
    student_signals = compute_motion_signals(
        student_poses,
        data.student_fps,
        source_track_ids=student_ids,
        smoothing_seconds=config.smoothing_seconds,
    )

    target_start = data.student_segment.start_frame_5fps - 1
    target_end = data.student_segment.end_frame_5fps - 1
    reference_start = data.teacher_segment.start_frame_5fps - 1
    reference_end = data.teacher_segment.end_frame_5fps - 1
    target_sample_indices = np.arange(target_start, target_end + 1, dtype=np.int64)
    if data.dataset == "clips":
        # Clip DTW paths and sample arrays are local to this single move.
        target_sample_indices = np.arange(len(data.target_sample_source_indices), dtype=np.int64)
        reference_start = 0
        reference_end = len(data.reference_sample_source_indices) - 1
    student_source_frames = np.arange(
        data.student_segment.source_start_index,
        data.student_segment.source_end_index + 1,
        dtype=np.int64,
    )
    reference_progress, reference_indices = build_reference_progress(
        data.path,
        target_sample_indices,
        data.target_sample_source_indices[
            target_start : target_end + 1
        ] if data.dataset == "full" else data.target_sample_source_indices,
        student_source_frames,
        reference_start,
        reference_end,
    )
    teacher_duration = len(teacher_poses) / data.teacher_fps
    student_duration = len(student_poses) / data.student_fps
    dtw_distance = _subset_path_cost(data)

    metric_rows: list[dict] = []
    event_rows: list[dict] = []
    tempo_rows: list[dict] = []
    signal_arrays: dict[str, np.ndarray] = {
        "teacher_joint_speed_degrees_per_second": teacher_signals.joint_speed_degrees_per_second,
        "student_joint_speed_degrees_per_second": student_signals.joint_speed_degrees_per_second,
        "teacher_valid_frames": teacher_signals.valid_frames,
        "student_valid_frames": student_signals.valid_frames,
        "reference_progress": reference_progress,
        "reference_sample_index": reference_indices,
    }
    for region in BODY_REGIONS:
        metrics, events, bins, arrays = analyze_region(
            region,
            teacher_signals,
            student_signals,
            reference_progress,
            teacher_duration,
            student_duration,
            data.teacher_fps,
            data.student_fps,
            active_scale=config.active_scale,
            pause_scale=config.pause_scale,
            activity_exit_ratio=config.activity_exit_ratio,
            minimum_active_seconds=config.minimum_active_seconds,
            minimum_pause_seconds=config.minimum_pause_seconds,
            bridge_gap_seconds=config.bridge_gap_seconds,
            activity_dilation_seconds=config.activity_dilation_seconds,
            stall_window_seconds=config.stall_window_seconds,
            stall_speed_ratio=config.stall_speed_ratio,
            minimum_stall_seconds=config.minimum_stall_seconds,
            tempo_bin_count=config.tempo_bin_count,
        )
        metric_rows.append(
            {
                "dataset": data.dataset,
                "case_id": data.case_id,
                "student_id": data.student_id,
                "move_id": data.move_id,
                "move_name": data.move_name,
                **asdict(metrics),
                "dtw_path_mean_geodesic_radians": dtw_distance,
                "dtw_path_mean_geodesic_degrees": float(np.degrees(dtw_distance)),
            }
        )
        event_rows.extend(
            event_to_row(event, data)
            for event in events
        )
        if region == "whole_body":
            tempo_rows.extend(tempo_bin_to_row(row, data) for row in bins)
        for name, values in arrays.items():
            signal_arrays[f"{region}_{name}"] = values

    metadata = {
        "dataset": data.dataset,
        "case_id": data.case_id,
        "student_id": data.student_id,
        "move_id": data.move_id,
        "move_name": data.move_name,
        "teacher_video": str(data.teacher_video),
        "student_video": str(data.student_video),
        "teacher_source_frame_range_1based": [
            data.teacher_segment.start_frame,
            data.teacher_segment.end_frame,
        ],
        "student_source_frame_range_1based": [
            data.student_segment.start_frame,
            data.student_segment.end_frame,
        ],
        "teacher_fps": data.teacher_fps,
        "student_fps": data.student_fps,
        "teacher_track_id": data.teacher_track.track_id,
        "student_track_id": data.student_track.track_id,
        "teacher_used_track_ids": list(data.teacher_track.used_track_ids),
        "student_used_track_ids": list(data.student_track.used_track_ids),
        "teacher_smoothing_window_frames": teacher_signals.smoothing_window_frames,
        "student_smoothing_window_frames": student_signals.smoothing_window_frames,
        "teacher_excluded_track_switch_frames": teacher_signals.excluded_track_switch_frames,
        "student_excluded_track_switch_frames": student_signals.excluded_track_switch_frames,
        "dtw_path_mean_geodesic_radians": dtw_distance,
        "dtw_path_mean_geodesic_degrees": float(np.degrees(dtw_distance)),
    }
    return metric_rows, event_rows, tempo_rows, signal_arrays, metadata


def event_to_row(event: PauseEvent, data: SegmentAnalysisInput) -> dict:
    segment = data.teacher_segment if event.subject == "teacher" else data.student_segment
    fps = data.teacher_fps if event.subject == "teacher" else data.student_fps
    start_source = segment.source_start_index + event.start_index
    end_source = segment.source_start_index + event.end_index
    return {
        "dataset": data.dataset,
        "case_id": data.case_id,
        "student_id": data.student_id,
        "move_id": data.move_id,
        "move_name": data.move_name,
        "subject": event.subject,
        "region": event.region,
        "start_frame_1based": start_source + 1,
        "end_frame_1based": end_source + 1,
        "start_time_seconds": start_source / fps,
        "end_time_seconds": (end_source + 1) / fps,
        "duration_seconds": event.duration_seconds,
        "teacher_progress_start": event.teacher_progress_start,
        "teacher_progress_end": event.teacher_progress_end,
        "teacher_active_fraction": event.teacher_active_fraction,
        "reason": event.reason,
    }


def tempo_bin_to_row(row: TempoBin, data: SegmentAnalysisInput) -> dict:
    return {
        "dataset": data.dataset,
        "case_id": data.case_id,
        "student_id": data.student_id,
        "move_id": data.move_id,
        "move_name": data.move_name,
        **asdict(row),
    }


def save_diagnostic(
    output_path: Path,
    data: SegmentAnalysisInput,
    arrays: dict[str, np.ndarray],
    metrics: list[dict],
) -> None:
    regions = ("whole_body", "left_arm", "right_arm", "left_leg", "right_leg", "trunk")
    figure, axes = plt.subplots(3, 2, figsize=(15, 10), constrained_layout=True)
    metric_map = {row["region"]: row for row in metrics}
    student_time = np.arange(len(arrays["reference_progress"])) / data.student_fps
    for axis, region in zip(axes.flat, regions):
        row = metric_map[region]
        student_intensity = arrays[f"{region}_student_intensity"]
        axis.plot(student_time, student_intensity, color="#2563eb", linewidth=1.0, label="student intensity")
        axis.axhline(
            row["student_pause_threshold"],
            color="#dc2626",
            linestyle="--",
            linewidth=0.9,
            label="pause threshold",
        )
        pause = arrays[f"{region}_student_pause"]
        active = arrays[f"{region}_teacher_active_on_student"]
        stall = arrays[f"{region}_tempo_stall"]
        axis.fill_between(student_time, 0, np.nanmax(student_intensity), where=active, color="#22c55e", alpha=0.08)
        axis.fill_between(student_time, 0, np.nanmax(student_intensity), where=pause, color="#ef4444", alpha=0.18)
        axis.fill_between(student_time, 0, np.nanmax(student_intensity), where=stall, color="#f59e0b", alpha=0.18)
        axis.set_title(
            f"{region} | pause={row['aligned_active_pause_ratio']:.3f} "
            f"stall={row['tempo_stall_ratio']:.3f} amp={row['amplitude_ratio']:.3f}"
        )
        axis.set_xlabel("Student segment time (s)")
        axis.set_ylabel("Angular speed RMS (deg/s)")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(loc="upper right", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.suptitle(f"{data.case_id} | {data.move_name}", fontsize=15)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def write_rows(path: Path, rows: list[dict]) -> None:
    if rows:
        write_dict_csv(path, rows)


def process_case(
    inputs: list[SegmentAnalysisInput],
    output_dir: Path,
    config: AnalysisConfig,
    *,
    save_diagnostics: bool,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    started = time.perf_counter()
    all_metrics: list[dict] = []
    all_events: list[dict] = []
    all_bins: list[dict] = []
    metadata: list[dict] = []
    signal_payload: dict[str, np.ndarray] = {}
    for data in inputs:
        metrics, events, bins, arrays, move_metadata = analyze_segment(data, config)
        all_metrics.extend(metrics)
        all_events.extend(events)
        all_bins.extend(bins)
        metadata.append(move_metadata)
        prefix = f"move_{data.move_id:02d}"
        for name, values in arrays.items():
            signal_payload[f"{prefix}_{name}"] = values
        if save_diagnostics:
            save_diagnostic(
                output_dir / "motion_diagnostics" / f"{data.move_id:02d}_{data.move_name}.png",
                data,
                arrays,
                metrics,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "motion_quality_summary.csv", all_metrics)
    write_rows(output_dir / "pause_events.csv", all_events)
    write_rows(output_dir / "tempo_bins.csv", all_bins)
    np.savez_compressed(output_dir / "motion_signals.npz", **signal_payload)
    summary = {
        "status": "ok",
        "case_id": inputs[0].case_id,
        "dataset": inputs[0].dataset,
        "move_count": len(inputs),
        "config": asdict(config),
        "moves": metadata,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[{inputs[0].dataset}:{inputs[0].case_id}] "
        f"{len(inputs)} move(s) -> {output_dir}",
        flush=True,
    )
    return all_metrics, all_events, all_bins, metadata


def full_inputs(
    student_video_id: str,
    args: argparse.Namespace,
) -> list[SegmentAnalysisInput]:
    reference_video = args.reference_video_root / f"{args.reference_video_id}.mp4"
    student_video = args.student_video_root / f"{student_video_id}.mp4"
    reference_sampling = inspect_video_sampling(reference_video, args.sample_fps)
    student_sampling = inspect_video_sampling(student_video, args.sample_fps)
    references = load_reference_segments(args.reference_segments, args.sample_fps)
    teacher_segments = build_reference_source_segments(references, reference_sampling)
    student_segments = load_student_segments(
        args.student_segmentation_root / student_video_id / "segments.csv"
    )
    teacher_track = load_stitched_primary_track(
        tracking_path(args.reference_tracking_root, args.reference_video_id)
    )
    student_track = load_stitched_primary_track(
        tracking_path(args.student_tracking_root, student_video_id)
    )
    path, local_costs = load_saved_dtw_path(
        args.student_segmentation_root / student_video_id / "dtw_path.npz"
    )
    return [
        SegmentAnalysisInput(
            dataset="full",
            case_id=student_video_id,
            student_id=student_video_id,
            move_id=teacher.move_id,
            move_name=teacher.move_name,
            teacher_video=reference_video,
            student_video=student_video,
            teacher_track=teacher_track,
            student_track=student_track,
            teacher_segment=teacher,
            student_segment=student,
            teacher_fps=reference_sampling.source_fps,
            student_fps=student_sampling.source_fps,
            path=path,
            path_local_costs=local_costs,
            reference_sample_source_indices=reference_sampling.source_indices,
            target_sample_source_indices=student_sampling.source_indices,
        )
        for teacher, student in zip(teacher_segments, student_segments)
        if args.move is None or teacher.move_name == args.move
    ]


def discover_full_ids(args: argparse.Namespace) -> list[str]:
    result_ids = {
        path.parent.name
        for path in args.student_segmentation_root.glob("*/segments.csv")
    }
    video_ids = {path.stem for path in args.student_video_root.glob("*.mp4")}
    ids = sorted(result_ids.intersection(video_ids))
    if args.student_video_id:
        requested = set(args.student_video_id)
        ids = [video_id for video_id in ids if video_id in requested]
    return ids


def discover_clips(args: argparse.Namespace) -> list[Path]:
    clips = sorted((args.clip_root / "student").glob("*.mp4"))
    if args.clip:
        requested = set(args.clip)
        clips = [clip for clip in clips if clip.stem in requested]
    if args.move:
        clips = [clip for clip in clips if infer_move(clip)[1] == args.move]
    return clips


def ranking_value(
    metric: str,
    value: float,
    *,
    amplitude_ranking: str = "larger_is_better",
) -> float:
    if metric == "amplitude_ratio":
        if amplitude_ranking == "larger_is_better":
            return -value
        if amplitude_ranking == "absolute_difference":
            return abs(value - 1.0)
        if amplitude_ranking == "teacher_closeness":
            return abs(float(np.log(max(value, 1e-12))))
        raise ValueError(f"Unsupported amplitude ranking rule: {amplitude_ranking}")
    if metric == "duration_ratio":
        return abs(float(np.log(max(value, 1e-12))))
    return value


def save_rankings(
    root: Path,
    rows: list[dict],
    *,
    amplitude_ranking: str = "larger_is_better",
    filename: str = "metric_rankings.csv",
) -> list[dict]:
    metrics = (
        "pause_excess_ratio",
        "aligned_active_pause_ratio",
        "tempo_stall_ratio",
        "local_tempo_distortion",
        "amplitude_ratio",
        "duration_ratio",
        "dtw_path_mean_geodesic_degrees",
    )
    whole_body = [row for row in rows if row["region"] == "whole_body"]
    ranking_rows: list[dict] = []
    groups = sorted({(row["dataset"], row["move_name"]) for row in whole_body})
    for dataset, move_name in groups:
        selected = [
            row for row in whole_body
            if row["dataset"] == dataset and row["move_name"] == move_name
        ]
        for metric in metrics:
            valid = [row for row in selected if np.isfinite(float(row[metric]))]
            scores = np.asarray(
                [
                    ranking_value(
                        metric,
                        float(row[metric]),
                        amplitude_ranking=amplitude_ranking,
                    )
                    for row in valid
                ],
                dtype=np.float64,
            )
            ranks = rankdata(scores, method="average")
            ordered = sorted(
                zip(valid, scores, ranks),
                key=lambda item: (float(item[2]), item[0]["case_id"]),
            )
            for row, score, rank in ordered:
                ranking_rows.append(
                    {
                        "dataset": dataset,
                        "move_name": move_name,
                        "metric": metric,
                        "ranking_rule": (
                            amplitude_ranking if metric == "amplitude_ratio" else "unchanged"
                        ),
                        "rank_best_is_1": float(rank),
                        "case_id": row["case_id"],
                        "student_id": row["student_id"],
                        "raw_value": row[metric],
                        "ranking_value_lower_is_better": float(score),
                    }
                )
    write_rows(root / filename, ranking_rows)
    return ranking_rows


def save_human_correlations(
    root: Path,
    ranking_rows: list[dict],
    human_rankings_path: Path,
    *,
    filename: str = "human_rank_correlations.csv",
) -> list[dict]:
    with human_rankings_path.open("r", encoding="utf-8", newline="") as handle:
        human = {
            (row["move"], row["student_id"]): int(row["rank"])
            for row in csv.DictReader(handle)
        }
    correlation_rows: list[dict] = []
    groups = sorted(
        {
            (row["move_name"], row["metric"])
            for row in ranking_rows
            if row["dataset"] == "clips"
        }
    )
    for move_name, metric in groups:
        selected = [
            row for row in ranking_rows
            if row["dataset"] == "clips"
            and row["move_name"] == move_name
            and row["metric"] == metric
            and (move_name, str(row["student_id"])) in human
        ]
        if len(selected) < 3:
            continue
        predicted = [float(row["ranking_value_lower_is_better"]) for row in selected]
        expected = [human[(move_name, str(row["student_id"]))] for row in selected]
        correlation, p_value = spearmanr(predicted, expected)
        correlation_rows.append(
            {
                "move_name": move_name,
                "metric": metric,
                "ranking_rule": selected[0]["ranking_rule"],
                "student_count": len(selected),
                "spearman_correlation": float(correlation),
                "p_value": float(p_value),
            }
        )
    write_rows(root / filename, correlation_rows)
    return correlation_rows


def save_ranking_experiments(
    root: Path,
    metric_rows: list[dict],
    human_rankings_path: Path,
) -> None:
    closeness_rankings = save_rankings(
        root,
        metric_rows,
        amplitude_ranking="teacher_closeness",
        filename="metric_rankings_amplitude_teacher_closeness.csv",
    )
    larger_rankings = save_rankings(
        root,
        metric_rows,
        amplitude_ranking="larger_is_better",
        filename="metric_rankings_amplitude_larger_is_better.csv",
    )
    absolute_difference_rankings = save_rankings(
        root,
        metric_rows,
        amplitude_ranking="absolute_difference",
        filename="metric_rankings_amplitude_absolute_difference.csv",
    )
    write_rows(root / "metric_rankings.csv", absolute_difference_rankings)
    if not human_rankings_path.is_file():
        return
    save_human_correlations(
        root,
        closeness_rankings,
        human_rankings_path,
        filename="human_rank_correlations_amplitude_teacher_closeness.csv",
    )
    larger_correlations = save_human_correlations(
        root,
        larger_rankings,
        human_rankings_path,
        filename="human_rank_correlations_amplitude_larger_is_better.csv",
    )
    absolute_difference_correlations = save_human_correlations(
        root,
        absolute_difference_rankings,
        human_rankings_path,
        filename="human_rank_correlations_amplitude_absolute_difference.csv",
    )
    write_rows(root / "human_rank_correlations.csv", absolute_difference_correlations)


def load_metric_summary(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Motion-quality summary does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("full", "clips", "both"), default="both")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reference-video-root", type=Path, default=DEFAULT_REFERENCE_VIDEO_ROOT)
    parser.add_argument("--reference-tracking-root", type=Path, default=DEFAULT_REFERENCE_TRACKING_ROOT)
    parser.add_argument("--reference-video-id", default=DEFAULT_REFERENCE_ID)
    parser.add_argument("--reference-segments", type=Path, default=DEFAULT_REFERENCE_SEGMENTS)
    parser.add_argument("--student-video-root", type=Path, default=DEFAULT_STUDENT_VIDEO_ROOT)
    parser.add_argument("--student-tracking-root", type=Path, default=DEFAULT_STUDENT_TRACKING_ROOT)
    parser.add_argument("--student-segmentation-root", type=Path, default=DEFAULT_STUDENT_SEGMENTATION_ROOT)
    parser.add_argument("--clip-root", type=Path, default=DEFAULT_CLIP_ROOT)
    parser.add_argument("--clip-tracking-root", type=Path, default=DEFAULT_CLIP_TRACKING_ROOT)
    parser.add_argument("--human-rankings", type=Path, default=DEFAULT_HUMAN_RANKINGS)
    parser.add_argument(
        "--rankings-only",
        action="store_true",
        help="Rebuild ranking experiments from the existing aggregate metric CSV.",
    )
    parser.add_argument("--student-video-id", action="append")
    parser.add_argument("--clip", action="append", help="Clip stem; repeatable")
    parser.add_argument("--move", choices=("qishi", "yemafenzong", "baiheliangchi"))
    parser.add_argument("--save-diagnostics", action="store_true")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--pairwise-chunk-size", type=int, default=64)
    parser.add_argument("--dtw-coefficient", type=float, default=1.0)
    parser.add_argument("--smoothing-seconds", type=float, default=0.20)
    parser.add_argument("--active-scale", type=float, default=0.10)
    parser.add_argument("--pause-scale", type=float, default=0.05)
    parser.add_argument("--activity-exit-ratio", type=float, default=0.70)
    parser.add_argument("--minimum-active-seconds", type=float, default=0.25)
    parser.add_argument("--minimum-pause-seconds", type=float, default=0.40)
    parser.add_argument("--bridge-gap-seconds", type=float, default=0.10)
    parser.add_argument("--activity-dilation-seconds", type=float, default=0.40)
    parser.add_argument("--stall-window-seconds", type=float, default=0.60)
    parser.add_argument("--stall-speed-ratio", type=float, default=0.25)
    parser.add_argument("--minimum-stall-seconds", type=float, default=0.40)
    parser.add_argument("--tempo-bin-count", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.rankings_only:
        all_metrics = load_metric_summary(
            args.output_root / "all_motion_quality_summary.csv"
        )
        save_ranking_experiments(args.output_root, all_metrics, args.human_rankings)
        print(
            f"Rebuilt all amplitude-ranking experiments from {len(all_metrics)} metric rows.",
            flush=True,
        )
        return

    config = AnalysisConfig(
        sample_fps=args.sample_fps,
        smoothing_seconds=args.smoothing_seconds,
        active_scale=args.active_scale,
        pause_scale=args.pause_scale,
        activity_exit_ratio=args.activity_exit_ratio,
        minimum_active_seconds=args.minimum_active_seconds,
        minimum_pause_seconds=args.minimum_pause_seconds,
        bridge_gap_seconds=args.bridge_gap_seconds,
        activity_dilation_seconds=args.activity_dilation_seconds,
        stall_window_seconds=args.stall_window_seconds,
        stall_speed_ratio=args.stall_speed_ratio,
        minimum_stall_seconds=args.minimum_stall_seconds,
        tempo_bin_count=args.tempo_bin_count,
    )
    all_metrics: list[dict] = []
    failures: list[dict] = []

    if args.dataset in {"full", "both"}:
        for video_id in discover_full_ids(args):
            try:
                inputs = full_inputs(video_id, args)
                metrics, _, _, _ = process_case(
                    inputs,
                    args.output_root / "full" / video_id,
                    config,
                    save_diagnostics=args.save_diagnostics,
                )
                all_metrics.extend(metrics)
            except Exception as error:
                failures.append({"dataset": "full", "case_id": video_id, "error": str(error)})
                print(f"[full:{video_id}] FAILED: {error}", flush=True)

    if args.dataset in {"clips", "both"}:
        teachers = teacher_clip_map(args.clip_root / "teach", args.clip_tracking_root)
        for student_video in discover_clips(args):
            try:
                _, move_name = infer_move(student_video)
                teacher_video = teachers[move_name]
                analysis_input = create_clip_analysis_input(
                    student_video,
                    teacher_video,
                    tracking_root=args.clip_tracking_root,
                    sample_fps=args.sample_fps,
                    pairwise_chunk_size=args.pairwise_chunk_size,
                    dtw_coefficient=args.dtw_coefficient,
                )
                metrics, _, _, _ = process_case(
                    [analysis_input],
                    args.output_root / "clips" / student_video.stem,
                    config,
                    save_diagnostics=args.save_diagnostics,
                )
                all_metrics.extend(metrics)
            except Exception as error:
                failures.append(
                    {"dataset": "clips", "case_id": student_video.stem, "error": str(error)}
                )
                print(f"[clips:{student_video.stem}] FAILED: {error}", flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_root / "all_motion_quality_summary.csv", all_metrics)
    save_ranking_experiments(args.output_root, all_metrics, args.human_rankings)
    (args.output_root / "batch_summary.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "completed_cases": len({(row["dataset"], row["case_id"]) for row in all_metrics}),
                "metric_rows": len(all_metrics),
                "failed_cases": len(failures),
                "failures": failures,
                "config": asdict(config),
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(f"Motion-quality analysis failed for {len(failures)} case(s).")


if __name__ == "__main__":
    main()
