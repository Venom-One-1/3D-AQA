#!/usr/bin/env python
"""Build teacher kinematic templates and analyze student SMPL motion quality."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

from aqa3d.motion_quality import (
    BODY_REGIONS,
    build_reference_progress,
    compute_motion_signals,
    detect_pauses,
    hysteresis_activity_mask,
    integrate_intensity,
    robust_noise_floor,
)
from aqa3d.smpl_dtw import inspect_video_sampling
from aqa3d.tracking import load_stitched_primary_track
from aqa3d.velocity_quality import (
    TeacherMotionModel,
    analyze_against_teacher_model,
    count_acceleration_peaks,
    motion_fragment_count,
    profile_statistics,
    resample_profile,
)
from run_motion_quality_analysis import (
    DEFAULT_CLIP_ROOT,
    DEFAULT_CLIP_TRACKING_ROOT,
    DEFAULT_HUMAN_RANKINGS,
    DEFAULT_REFERENCE_ID,
    DEFAULT_REFERENCE_SEGMENTS,
    DEFAULT_REFERENCE_TRACKING_ROOT,
    DEFAULT_REFERENCE_VIDEO_ROOT,
    DEFAULT_STUDENT_SEGMENTATION_ROOT,
    DEFAULT_STUDENT_TRACKING_ROOT,
    DEFAULT_STUDENT_VIDEO_ROOT,
    SegmentAnalysisInput,
    _subset_path_cost,
    create_clip_analysis_input,
    discover_clips,
    discover_full_ids,
    full_inputs,
    infer_move,
    select_track_segment,
    teacher_clip_map,
    write_rows,
)
from run_student_tas_smpl_dtw import sampled_interval_to_source_interval
from run_tas_smpl_dtw_mapping import tracking_path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TEACHER_GROUND_TRUTH = PROJECT_ROOT / "tas_ground_truth" / "ground_truth_segments.csv"
DEFAULT_TEACHER_DTW_ROOT = PROJECT_ROOT / "tas_smpl_dtw_results"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "velocity_quality_results"
DEFAULT_TEACHER_IDS = (
    "QxVvRcRn2TA",
    "BV1iE411c7Ni_p03",
    "BV1tk4y1r7Yr_p27",
    "an5qNCspzUw",
    "i8kMrJmAfjU",
)


def load_teacher_segments(path: Path, teacher_ids: tuple[str, ...]) -> dict[str, list[dict]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict]] = {teacher_id: [] for teacher_id in teacher_ids}
    for row in rows:
        if row["video_id"] in grouped:
            grouped[row["video_id"]].append(row)
    for teacher_id, selected in grouped.items():
        selected.sort(key=lambda row: int(row["move_id"]))
        if [int(row["move_id"]) for row in selected] != list(range(1, 25)):
            raise ValueError(f"Expected 24 ordered moves for teacher {teacher_id} in {path}.")
    return grouped


def load_path(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return np.stack(
            (
                data["target_sample_indices_0based"].astype(np.int64),
                data["reference_sample_indices_0based"].astype(np.int64),
            ),
            axis=1,
        )


def teacher_reference_progress(
    path: np.ndarray,
    target_sampling,
    target_row: dict,
    reference_row: dict,
    source_start_frame: int,
    source_end_frame: int,
) -> np.ndarray:
    target_start = int(target_row["start_frame_5fps"]) - 1
    target_end = int(target_row["end_frame_5fps"]) - 1
    reference_start = int(reference_row["start_frame_5fps"]) - 1
    reference_end = int(reference_row["end_frame_5fps"]) - 1
    target_indices = np.arange(target_start, target_end + 1, dtype=np.int64)
    source_frames = np.arange(source_start_frame - 1, source_end_frame, dtype=np.int64)
    progress, _ = build_reference_progress(
        path,
        target_indices,
        target_sampling.source_indices[target_start : target_end + 1],
        source_frames,
        reference_start,
        reference_end,
    )
    return progress


def _teacher_activity(speed: np.ndarray, valid: np.ndarray, fps: float) -> tuple[np.ndarray, float]:
    noise = robust_noise_floor(speed[valid])
    p90 = float(np.nanpercentile(speed[valid], 90.0))
    threshold = max(noise, 0.10 * p90)
    active = hysteresis_activity_mask(
        speed,
        valid,
        threshold,
        exit_ratio=0.70,
        minimum_active_seconds=0.25,
        fps=fps,
    )
    return active, threshold


def build_teacher_model(args: argparse.Namespace) -> TeacherMotionModel:
    teacher_ids = tuple(args.teacher_video_id or DEFAULT_TEACHER_IDS)
    grouped = load_teacher_segments(args.teacher_ground_truth, teacher_ids)
    reference_rows = grouped[args.reference_video_id]
    move_ids = np.arange(1, 25, dtype=np.int64)
    move_names = tuple(row["move_name"] for row in reference_rows)
    regions = tuple(BODY_REGIONS)
    grid = np.linspace(0.0, 1.0, args.phase_points)
    shape = (len(teacher_ids), len(move_ids), len(regions), len(grid))
    speed_profiles = np.full(shape, np.nan, dtype=np.float64)
    acceleration_profiles = np.full(shape, np.nan, dtype=np.float64)
    active_profiles = np.zeros(shape, dtype=bool)
    durations = np.full(shape[:3], np.nan, dtype=np.float64)
    amplitudes = np.full(shape[:3], np.nan, dtype=np.float64)
    peak_rates = np.full(shape[:3], np.nan, dtype=np.float64)
    fragment_counts = np.full(shape[:3], np.nan, dtype=np.float64)
    distribution_rows: list[dict] = []

    for teacher_index, teacher_id in enumerate(teacher_ids):
        video = args.teacher_video_root / f"{teacher_id}.mp4"
        sampling = inspect_video_sampling(video, args.sample_fps)
        track = load_stitched_primary_track(tracking_path(args.teacher_tracking_root, teacher_id))
        if teacher_id == args.reference_video_id:
            samples = np.arange(sampling.sample_count, dtype=np.int64)
            path = np.stack((samples, samples), axis=1)
        else:
            path = load_path(args.teacher_dtw_root / teacher_id / "dtw_path.npz")
        print(f"[teacher:{teacher_id}] extracting 24 moves", flush=True)

        for move_index, (row, reference_row) in enumerate(zip(grouped[teacher_id], reference_rows)):
            start_frame, end_frame = sampled_interval_to_source_interval(
                int(row["start_frame_5fps"]),
                int(row["end_frame_5fps"]),
                sampling,
            )
            source_indices = np.arange(start_frame - 1, end_frame, dtype=np.int64)
            poses = track.at_source_frames(source_indices)
            id_lookup = {
                int(frame): int(track_id)
                for frame, track_id in zip(track.frame_numbers, track.source_track_ids)
            }
            track_ids = np.asarray(
                [id_lookup[int(frame + 1)] for frame in source_indices],
                dtype=np.int64,
            )
            signals = compute_motion_signals(
                poses,
                sampling.source_fps,
                source_track_ids=track_ids,
                smoothing_seconds=args.smoothing_seconds,
            )
            progress = teacher_reference_progress(
                path,
                sampling,
                row,
                reference_row,
                start_frame,
                end_frame,
            )
            duration = len(poses) / sampling.source_fps

            for region_index, region in enumerate(regions):
                speed = signals.region_intensity[region]
                acceleration = signals.region_speed_change[region]
                active, threshold = _teacher_activity(
                    speed,
                    signals.valid_frames,
                    sampling.source_fps,
                )
                speed_profiles[teacher_index, move_index, region_index] = resample_profile(
                    progress, speed, grid
                )
                acceleration_profiles[teacher_index, move_index, region_index] = resample_profile(
                    progress, acceleration, grid
                )
                active_profiles[teacher_index, move_index, region_index] = (
                    resample_profile(progress, active.astype(np.float64), grid) >= 0.5
                )
                amplitude = integrate_intensity(
                    speed,
                    signals.valid_frames,
                    sampling.source_fps,
                )
                valid_acceleration = signals.acceleration_valid_frames & np.isfinite(acceleration)
                acceleration_threshold = (
                    float(np.nanpercentile(np.abs(acceleration[valid_acceleration]), 95.0))
                    if np.any(valid_acceleration)
                    else float("nan")
                )
                peaks = count_acceleration_peaks(
                    acceleration,
                    acceleration_threshold,
                    valid_acceleration,
                    sampling.source_fps,
                )
                fragments, _ = motion_fragment_count(
                    speed,
                    signals.valid_frames,
                    sampling.source_fps,
                    threshold=threshold,
                )
                durations[teacher_index, move_index, region_index] = duration
                amplitudes[teacher_index, move_index, region_index] = amplitude
                peak_rates[teacher_index, move_index, region_index] = len(peaks) / duration
                fragment_counts[teacher_index, move_index, region_index] = fragments
                distribution_rows.append(
                    {
                        "teacher_id": teacher_id,
                        "move_id": move_index + 1,
                        "move_name": row["move_name"],
                        "region": region,
                        "duration_seconds": duration,
                        "amplitude_degrees": amplitude,
                        "mean_speed_degrees_per_second": float(np.nanmean(speed)),
                        "speed_variance": float(np.nanvar(speed)),
                        "speed_p90": float(np.nanpercentile(speed, 90.0)),
                        "mean_abs_speed_change_degrees_per_second2": float(np.nanmean(np.abs(acceleration))),
                        "speed_change_p95": float(np.nanpercentile(np.abs(acceleration), 95.0)),
                        "acceleration_peak_rate": len(peaks) / duration,
                        "motion_fragment_count": fragments,
                        "activity_threshold": threshold,
                        "excluded_speed_frames": int(np.sum(~signals.valid_frames)),
                        "excluded_acceleration_frames": int(np.sum(~signals.acceleration_valid_frames)),
                    }
                )

    stat_shape = shape[1:]
    stat_arrays = {
        name: np.full(stat_shape, np.nan, dtype=np.float64)
        for name in (
            "speed_median",
            "speed_mad",
            "speed_p10",
            "speed_p90",
            "acceleration_median",
            "acceleration_mad",
            "acceleration_p10",
            "acceleration_p90",
            "acceleration_abs_p95",
        )
    }
    for move_index in range(len(move_ids)):
        for region_index in range(len(regions)):
            speed_stats = profile_statistics(
                speed_profiles[:, move_index, region_index],
                window_radius=args.phase_window_radius,
            )
            acceleration_stats = profile_statistics(
                acceleration_profiles[:, move_index, region_index],
                window_radius=args.phase_window_radius,
            )
            for key in ("median", "mad", "p10", "p90"):
                stat_arrays[f"speed_{key}"][move_index, region_index] = speed_stats[key]
                stat_arrays[f"acceleration_{key}"][move_index, region_index] = acceleration_stats[key]
            stat_arrays["acceleration_abs_p95"][move_index, region_index] = acceleration_stats["abs_p95"]

    model = TeacherMotionModel(
        teacher_ids=teacher_ids,
        move_ids=move_ids,
        move_names=move_names,
        regions=regions,
        progress_grid=grid,
        speed_profiles=speed_profiles,
        acceleration_profiles=acceleration_profiles,
        active_profiles=active_profiles,
        speed_median=stat_arrays["speed_median"],
        speed_mad=stat_arrays["speed_mad"],
        speed_p10=stat_arrays["speed_p10"],
        speed_p90=stat_arrays["speed_p90"],
        acceleration_median=stat_arrays["acceleration_median"],
        acceleration_mad=stat_arrays["acceleration_mad"],
        acceleration_p10=stat_arrays["acceleration_p10"],
        acceleration_p90=stat_arrays["acceleration_p90"],
        acceleration_abs_p95=stat_arrays["acceleration_abs_p95"],
        active_probability=np.mean(active_profiles, axis=0),
        duration_median=np.nanmedian(durations, axis=0),
        amplitude_median=np.nanmedian(amplitudes, axis=0),
        peak_rate_median=np.nanmedian(peak_rates, axis=0),
        fragment_count_median=np.nanmedian(fragment_counts, axis=0),
    )
    output_dir = args.output_root / "teacher_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "teacher_motion_model.npz")
    write_rows(output_dir / "teacher_distribution_summary.csv", distribution_rows)
    write_rows(output_dir / "teacher_leave_one_out.csv", leave_one_teacher_out(model, args.phase_window_radius))
    save_teacher_diagnostics(model, output_dir / "teacher_model_diagnostics")
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "teacher_ids": list(teacher_ids),
                "reference_video_id": args.reference_video_id,
                "move_count": len(move_ids),
                "regions": list(regions),
                "phase_points": len(grid),
                "phase_window_radius": args.phase_window_radius,
                "smoothing_seconds": args.smoothing_seconds,
                "sample_fps": args.sample_fps,
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Teacher model -> {output_dir}", flush=True)
    return model


def leave_one_teacher_out(model: TeacherMotionModel, window_radius: int) -> list[dict]:
    rows: list[dict] = []
    for held_index, teacher_id in enumerate(model.teacher_ids):
        keep = np.arange(len(model.teacher_ids)) != held_index
        for move_index, move_name in enumerate(model.move_names):
            for region_index, region in enumerate(model.regions):
                speed_stats = profile_statistics(
                    model.speed_profiles[keep, move_index, region_index],
                    window_radius=window_radius,
                )
                acceleration_stats = profile_statistics(
                    model.acceleration_profiles[keep, move_index, region_index],
                    window_radius=window_radius,
                )
                active = np.mean(model.active_profiles[keep, move_index, region_index], axis=0) >= 0.60
                held_speed = model.speed_profiles[held_index, move_index, region_index]
                held_acceleration = model.acceleration_profiles[held_index, move_index, region_index]
                speed_scale = np.maximum(
                    1.4826 * speed_stats["mad"],
                    max(float(np.nanpercentile(speed_stats["median"], 90.0)) * 0.05, 1e-6),
                )
                acceleration_scale = np.maximum(
                    1.4826 * acceleration_stats["mad"],
                    max(float(np.nanpercentile(np.abs(acceleration_stats["median"]), 90.0)) * 0.05, 1e-6),
                )
                valid_speed = active & np.isfinite(held_speed)
                valid_acceleration = active & np.isfinite(held_acceleration)
                rows.append(
                    {
                        "held_out_teacher_id": teacher_id,
                        "move_id": move_index + 1,
                        "move_name": move_name,
                        "region": region,
                        "velocity_profile_nmae": float(
                            np.mean(np.abs(held_speed[valid_speed] - speed_stats["median"][valid_speed]) / speed_scale[valid_speed])
                        ) if np.any(valid_speed) else float("nan"),
                        "velocity_outlier_ratio": float(
                            np.mean(
                                (held_speed[valid_speed] < speed_stats["p10"][valid_speed])
                                | (held_speed[valid_speed] > speed_stats["p90"][valid_speed])
                            )
                        ) if np.any(valid_speed) else float("nan"),
                        "acceleration_profile_nmae": float(
                            np.mean(
                                np.abs(held_acceleration[valid_acceleration] - acceleration_stats["median"][valid_acceleration])
                                / acceleration_scale[valid_acceleration]
                            )
                        ) if np.any(valid_acceleration) else float("nan"),
                        "acceleration_outlier_ratio": float(
                            np.mean(
                                np.abs(held_acceleration[valid_acceleration])
                                > acceleration_stats["abs_p95"][valid_acceleration]
                            )
                        ) if np.any(valid_acceleration) else float("nan"),
                        "active_progress_points": int(np.sum(active)),
                    }
                )
    return rows


def save_teacher_diagnostics(model: TeacherMotionModel, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    region_index = model.regions.index("whole_body")
    for move_index, move_name in enumerate(model.move_names):
        figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for teacher_index, teacher_id in enumerate(model.teacher_ids):
            axes[0].plot(model.progress_grid, model.speed_profiles[teacher_index, move_index, region_index], alpha=0.55, label=teacher_id)
            axes[1].plot(model.progress_grid, model.acceleration_profiles[teacher_index, move_index, region_index], alpha=0.45)
        axes[0].fill_between(model.progress_grid, model.speed_p10[move_index, region_index], model.speed_p90[move_index, region_index], color="tab:blue", alpha=0.18)
        axes[0].plot(model.progress_grid, model.speed_median[move_index, region_index], color="black", linewidth=2, label="Teacher median")
        axes[1].fill_between(model.progress_grid, model.acceleration_p10[move_index, region_index], model.acceleration_p90[move_index, region_index], color="tab:orange", alpha=0.18)
        axes[1].plot(model.progress_grid, model.acceleration_median[move_index, region_index], color="black", linewidth=2)
        axes[0].set_ylabel("Angular speed (deg/s)")
        axes[1].set_ylabel("Speed change (deg/s²)")
        axes[1].set_xlabel("Qx reference progress")
        axes[0].set_title(f"{move_index + 1}. {move_name} | whole_body")
        axes[0].legend(fontsize=7, ncol=3)
        figure.tight_layout()
        figure.savefig(output_dir / f"{move_index + 1:02d}_{move_name}.png", dpi=150)
        plt.close(figure)


def student_reference_progress(data: SegmentAnalysisInput) -> np.ndarray:
    target_start = data.student_segment.start_frame_5fps - 1
    target_end = data.student_segment.end_frame_5fps - 1
    reference_start = data.teacher_segment.start_frame_5fps - 1
    reference_end = data.teacher_segment.end_frame_5fps - 1
    target_indices = np.arange(target_start, target_end + 1, dtype=np.int64)
    target_sources = data.target_sample_source_indices[target_start : target_end + 1]
    if data.dataset == "clips":
        target_indices = np.arange(len(data.target_sample_source_indices), dtype=np.int64)
        target_sources = data.target_sample_source_indices
        reference_start = 0
        reference_end = len(data.reference_sample_source_indices) - 1
    source_frames = np.arange(
        data.student_segment.source_start_index,
        data.student_segment.source_end_index + 1,
        dtype=np.int64,
    )
    progress, _ = build_reference_progress(
        data.path,
        target_indices,
        target_sources,
        source_frames,
        reference_start,
        reference_end,
    )
    return progress


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def analyze_student_case(
    inputs: list[SegmentAnalysisInput],
    model: TeacherMotionModel,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict] = []
    event_rows: list[dict] = []
    signals_payload: dict[str, np.ndarray] = {}
    move_metadata: list[dict] = []
    diagnostics_dir = output_dir / "kinematic_diagnostics"
    if args.save_diagnostics:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

    for data in inputs:
        student_poses, student_ids = select_track_segment(data.student_track, data.student_segment)
        signals = compute_motion_signals(
            student_poses,
            data.student_fps,
            source_track_ids=student_ids,
            smoothing_seconds=args.smoothing_seconds,
        )
        progress = student_reference_progress(data)
        move_index = model.move_names.index(data.move_name)
        duration = len(student_poses) / data.student_fps
        dtw_distance = _subset_path_cost(data)
        whole_body_arrays: dict[str, np.ndarray] | None = None

        signals_payload[f"{data.move_name}_reference_progress"] = progress
        signals_payload[f"{data.move_name}_valid_speed"] = signals.valid_frames
        signals_payload[f"{data.move_name}_valid_acceleration"] = signals.acceleration_valid_frames
        signals_payload[f"{data.move_name}_joint_speed"] = signals.joint_speed_degrees_per_second
        signals_payload[f"{data.move_name}_joint_speed_change"] = signals.joint_speed_change_degrees_per_second2

        for region_index, region in enumerate(model.regions):
            speed = signals.region_intensity[region]
            acceleration = signals.region_speed_change[region]
            metrics, arrays = analyze_against_teacher_model(
                model,
                data.move_name,
                region,
                progress,
                speed,
                acceleration,
                signals.valid_frames,
                signals.acceleration_valid_frames,
                data.student_fps,
            )
            if region == "whole_body":
                whole_body_arrays = arrays
            amplitude = integrate_intensity(speed, signals.valid_frames, data.student_fps)
            teacher_amplitude = float(model.amplitude_median[move_index, region_index])
            teacher_duration = float(model.duration_median[move_index, region_index])
            teacher_p90 = float(np.nanpercentile(model.speed_p90[move_index, region_index], 90.0))
            pause_threshold = max(robust_noise_floor(speed[signals.valid_frames]), 0.05 * teacher_p90)
            pauses = detect_pauses(
                speed,
                signals.valid_frames,
                pause_threshold,
                data.student_fps,
            )
            aligned_active = arrays["teacher_active_on_student"] & signals.valid_frames
            aligned_pause_ratio = (
                float(np.sum(aligned_active & pauses.pause_mask) / np.sum(aligned_active))
                if np.any(aligned_active)
                else float("nan")
            )
            row = {
                "dataset": data.dataset,
                "case_id": data.case_id,
                "student_id": data.student_id,
                "move_id": data.move_id,
                "move_name": data.move_name,
                "region": region,
                **asdict(metrics),
                "aligned_active_pause_ratio": aligned_pause_ratio,
                "rotation_path_length_degrees": amplitude,
                "amplitude_ratio": amplitude / teacher_amplitude if teacher_amplitude > 0 else float("nan"),
                "duration_seconds": duration,
                "duration_ratio": duration / teacher_duration if teacher_duration > 0 else float("nan"),
                "dtw_path_mean_geodesic_degrees": float(np.degrees(dtw_distance)),
            }
            metric_rows.append(row)

            source_offset = data.student_segment.source_start_index
            for peak in arrays["acceleration_peaks"]:
                event_rows.append(
                    {
                        "dataset": data.dataset,
                        "case_id": data.case_id,
                        "student_id": data.student_id,
                        "move_id": data.move_id,
                        "move_name": data.move_name,
                        "region": region,
                        "event_type": "acceleration_peak",
                        "source_frame_1based": source_offset + int(peak) + 1,
                        "end_source_frame_1based": "",
                        "duration_seconds": "",
                        "time_in_move_seconds": int(peak) / data.student_fps,
                        "teacher_progress": float(progress[peak]),
                        "teacher_progress_start": "",
                        "teacher_progress_end": "",
                        "observed_value": float(np.abs(acceleration[peak])),
                        "expected_threshold": float(arrays["expected_acceleration_abs_p95"][peak]),
                    }
                )
            for start, end in _true_runs(arrays["student_active"]):
                event_rows.append(
                    {
                        "dataset": data.dataset,
                        "case_id": data.case_id,
                        "student_id": data.student_id,
                        "move_id": data.move_id,
                        "move_name": data.move_name,
                        "region": region,
                        "event_type": "motion_fragment",
                        "source_frame_1based": source_offset + start + 1,
                        "end_source_frame_1based": source_offset + end + 1,
                        "duration_seconds": (end - start + 1) / data.student_fps,
                        "time_in_move_seconds": start / data.student_fps,
                        "teacher_progress": "",
                        "teacher_progress_start": float(progress[start]),
                        "teacher_progress_end": float(progress[end]),
                        "observed_value": "",
                        "expected_threshold": "",
                    }
                )

        if args.save_diagnostics and whole_body_arrays is not None:
            save_student_diagnostic(
                data,
                model,
                progress,
                whole_body_arrays,
                diagnostics_dir / f"{data.move_id:02d}_{data.move_name}.png",
            )
        move_metadata.append(
            {
                "move_id": data.move_id,
                "move_name": data.move_name,
                "student_source_frame_range_1based": [
                    data.student_segment.start_frame,
                    data.student_segment.end_frame,
                ],
                "student_fps": data.student_fps,
                "duration_seconds": duration,
                "excluded_speed_frames": int(np.sum(~signals.valid_frames)),
                "excluded_acceleration_frames": int(np.sum(~signals.acceleration_valid_frames)),
            }
        )

    write_rows(output_dir / "kinematic_quality_summary.csv", metric_rows)
    write_rows(output_dir / "kinematic_events.csv", event_rows)
    np.savez_compressed(output_dir / "kinematic_signals.npz", **signals_payload)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "dataset": inputs[0].dataset,
                "case_id": inputs[0].case_id,
                "teacher_model": str(args.teacher_model),
                "smoothing_seconds": args.smoothing_seconds,
                "moves": move_metadata,
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[{inputs[0].dataset}:{inputs[0].case_id}] {len(inputs)} move(s) -> {output_dir}", flush=True)
    return metric_rows


def save_student_diagnostic(
    data: SegmentAnalysisInput,
    model: TeacherMotionModel,
    progress: np.ndarray,
    arrays: dict[str, np.ndarray],
    path: Path,
) -> None:
    time_axis = np.arange(len(progress), dtype=np.float64) / data.student_fps
    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(time_axis, arrays["student_speed"], label="Student", color="tab:blue")
    axes[0].plot(time_axis, arrays["expected_speed"], label="Teacher median", color="black")
    axes[0].fill_between(time_axis, arrays["expected_speed_p10"], arrays["expected_speed_p90"], color="tab:blue", alpha=0.15)
    axes[1].plot(time_axis, arrays["student_acceleration"], color="tab:orange", label="Student")
    axes[1].plot(time_axis, arrays["expected_acceleration"], color="black", label="Teacher median")
    threshold = arrays["expected_acceleration_abs_p95"]
    axes[1].fill_between(time_axis, -threshold, threshold, color="tab:orange", alpha=0.12)
    peaks = arrays["acceleration_peaks"]
    axes[1].scatter(time_axis[peaks], arrays["student_acceleration"][peaks], color="red", s=18, label="Outlier peaks")
    axes[2].plot(time_axis, progress, color="tab:green", label="DTW reference progress")
    axes[2].fill_between(time_axis, 0, 1, where=arrays["student_active"], color="tab:green", alpha=0.12, label="Student active")
    axes[0].set_ylabel("Speed (deg/s)")
    axes[1].set_ylabel("Speed change (deg/s²)")
    axes[2].set_ylabel("Progress")
    axes[2].set_xlabel("Student time (s)")
    axes[0].set_title(f"{data.case_id} | {data.move_name} | whole_body")
    for axis in axes:
        axis.legend(fontsize=8, loc="upper right")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def ranking_value(metric: str, value: float) -> float:
    if metric in {
        "active_mean_speed_ratio",
        "speed_variance_ratio",
        "motion_fragmentation_ratio",
        "amplitude_ratio",
    }:
        return abs(value - 1.0)
    if metric == "duration_ratio":
        return abs(float(np.log(max(value, 1e-12))))
    if metric == "velocity_profile_correlation":
        return -value
    return value


def save_rankings_and_correlations(root: Path, rows: list[dict], human_path: Path) -> None:
    metrics = (
        "active_mean_speed_ratio",
        "speed_variance_ratio",
        "velocity_profile_nmae",
        "velocity_profile_correlation",
        "velocity_outlier_ratio",
        "velocity_wasserstein_distance",
        "acceleration_profile_nmae",
        "acceleration_outlier_ratio",
        "acceleration_peak_excess_ratio",
        "motion_fragmentation_ratio",
        "aligned_active_pause_ratio",
        "amplitude_ratio",
        "duration_ratio",
        "dtw_path_mean_geodesic_degrees",
    )
    whole_body = [row for row in rows if row["region"] == "whole_body"]
    ranking_rows: list[dict] = []
    for dataset, move_name in sorted({(row["dataset"], row["move_name"]) for row in whole_body}):
        selected = [row for row in whole_body if row["dataset"] == dataset and row["move_name"] == move_name]
        for metric in metrics:
            valid = [row for row in selected if np.isfinite(float(row[metric]))]
            scores = np.asarray([ranking_value(metric, float(row[metric])) for row in valid])
            ranks = rankdata(scores, method="average")
            for row, score, rank in sorted(zip(valid, scores, ranks), key=lambda item: (item[2], item[0]["case_id"])):
                ranking_rows.append(
                    {
                        "dataset": dataset,
                        "move_name": move_name,
                        "metric": metric,
                        "rank_best_is_1": float(rank),
                        "case_id": row["case_id"],
                        "student_id": row["student_id"],
                        "raw_value": row[metric],
                        "ranking_value_lower_is_better": float(score),
                    }
                )
    write_rows(root / "metric_rankings.csv", ranking_rows)
    if not human_path.is_file():
        return
    with human_path.open("r", encoding="utf-8", newline="") as handle:
        human = {(row["move"], row["student_id"]): int(row["rank"]) for row in csv.DictReader(handle)}
    correlation_rows: list[dict] = []
    for move_name, metric in sorted({(row["move_name"], row["metric"]) for row in ranking_rows if row["dataset"] == "clips"}):
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
        if np.ptp(predicted) <= 1e-12 or np.ptp(expected) <= 1e-12:
            correlation, p_value = float("nan"), float("nan")
        else:
            correlation, p_value = spearmanr(predicted, expected)
        correlation_rows.append(
            {
                "move_name": move_name,
                "metric": metric,
                "student_count": len(selected),
                "spearman_correlation": float(correlation),
                "p_value": float(p_value),
            }
        )
    write_rows(root / "human_rank_correlations.csv", correlation_rows)


def analyze_students(args: argparse.Namespace) -> None:
    model = TeacherMotionModel.load(args.teacher_model)
    all_metrics: list[dict] = []
    failures: list[dict] = []
    started = time.perf_counter()
    if args.dataset in {"full", "both"}:
        for video_id in discover_full_ids(args):
            try:
                inputs = full_inputs(video_id, args)
                all_metrics.extend(
                    analyze_student_case(inputs, model, args.output_root / "full" / video_id, args)
                )
            except Exception as error:
                failures.append({"dataset": "full", "case_id": video_id, "error": str(error)})
                print(f"[full:{video_id}] FAILED: {error}", flush=True)
    if args.dataset in {"clips", "both"}:
        teachers = teacher_clip_map(args.clip_root / "teach", args.clip_tracking_root)
        for student_video in discover_clips(args):
            try:
                _, move_name = infer_move(student_video)
                analysis_input = create_clip_analysis_input(
                    student_video,
                    teachers[move_name],
                    tracking_root=args.clip_tracking_root,
                    sample_fps=args.sample_fps,
                    pairwise_chunk_size=args.pairwise_chunk_size,
                    dtw_coefficient=args.dtw_coefficient,
                )
                all_metrics.extend(
                    analyze_student_case(
                        [analysis_input],
                        model,
                        args.output_root / "clips" / student_video.stem,
                        args,
                    )
                )
            except Exception as error:
                failures.append({"dataset": "clips", "case_id": student_video.stem, "error": str(error)})
                print(f"[clips:{student_video.stem}] FAILED: {error}", flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_root / "all_kinematic_quality_summary.csv", all_metrics)
    save_rankings_and_correlations(args.output_root, all_metrics, args.human_rankings)
    (args.output_root / "batch_summary.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "completed_cases": len({(row["dataset"], row["case_id"]) for row in all_metrics}),
                "metric_rows": len(all_metrics),
                "failed_cases": len(failures),
                "failures": failures,
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(f"Velocity-quality analysis failed for {len(failures)} case(s).")


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reference-video-root", type=Path, default=DEFAULT_REFERENCE_VIDEO_ROOT)
    parser.add_argument("--reference-tracking-root", type=Path, default=DEFAULT_REFERENCE_TRACKING_ROOT)
    parser.add_argument("--reference-video-id", default=DEFAULT_REFERENCE_ID)
    parser.add_argument("--reference-segments", type=Path, default=DEFAULT_REFERENCE_SEGMENTS)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--smoothing-seconds", type=float, default=0.20)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-teacher-model")
    add_common_paths(build)
    build.add_argument("--teacher-video-root", type=Path, default=DEFAULT_REFERENCE_VIDEO_ROOT)
    build.add_argument("--teacher-tracking-root", type=Path, default=DEFAULT_REFERENCE_TRACKING_ROOT)
    build.add_argument("--teacher-ground-truth", type=Path, default=DEFAULT_TEACHER_GROUND_TRUTH)
    build.add_argument("--teacher-dtw-root", type=Path, default=DEFAULT_TEACHER_DTW_ROOT)
    build.add_argument("--teacher-video-id", action="append")
    build.add_argument("--phase-points", type=int, default=101)
    build.add_argument("--phase-window-radius", type=int, default=2)

    analyze = subparsers.add_parser("analyze")
    add_common_paths(analyze)
    analyze.add_argument("--dataset", choices=("full", "clips", "both"), default="both")
    analyze.add_argument("--teacher-model", type=Path, default=DEFAULT_OUTPUT_ROOT / "teacher_model" / "teacher_motion_model.npz")
    analyze.add_argument("--student-video-root", type=Path, default=DEFAULT_STUDENT_VIDEO_ROOT)
    analyze.add_argument("--student-tracking-root", type=Path, default=DEFAULT_STUDENT_TRACKING_ROOT)
    analyze.add_argument("--student-segmentation-root", type=Path, default=DEFAULT_STUDENT_SEGMENTATION_ROOT)
    analyze.add_argument("--clip-root", type=Path, default=DEFAULT_CLIP_ROOT)
    analyze.add_argument("--clip-tracking-root", type=Path, default=DEFAULT_CLIP_TRACKING_ROOT)
    analyze.add_argument("--human-rankings", type=Path, default=DEFAULT_HUMAN_RANKINGS)
    analyze.add_argument("--student-video-id", action="append")
    analyze.add_argument("--clip", action="append")
    analyze.add_argument("--move")
    analyze.add_argument("--save-diagnostics", action="store_true")
    analyze.add_argument("--pairwise-chunk-size", type=int, default=64)
    analyze.add_argument("--dtw-coefficient", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "build-teacher-model":
        build_teacher_model(args)
    else:
        analyze_students(args)


if __name__ == "__main__":
    main()
