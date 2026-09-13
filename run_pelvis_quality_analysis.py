#!/usr/bin/env python
"""Run the first pelvis/root-trajectory Tai Chi quality experiment."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

from aqa3d.motion_quality import build_reference_progress
from aqa3d.pelvis_quality import (
    SIGNAL_NAMES,
    PelvisSignals,
    TeacherPelvisModel,
    build_teacher_model,
    compute_pelvis_signals,
    compute_quality_metrics,
    resample_pelvis_signals,
)
from aqa3d.smpl_dtw import (
    backtrack_dtw_path,
    dtw_from_cost_matrix,
    inspect_video_sampling,
    pairwise_geodesic_costs,
    uniform_sample_source_indices,
)
from aqa3d.tracking import TrackRootSequence, load_stitched_root_track
from run_motion_quality_analysis import (
    clip_tracking_path,
    infer_move,
    inspect_video,
    teacher_clip_map,
    whole_clip_segment,
)
from run_student_tas_smpl_dtw import sampled_interval_to_source_interval
from run_tas_smpl_dtw_mapping import tracking_path


PROJECT_ROOT = Path(__file__).resolve().parent
REFERENCE_ID = "QxVvRcRn2TA"
DEFAULT_TEACHER_IDS = (
    REFERENCE_ID,
    "BV1iE411c7Ni_p03",
    "BV1tk4y1r7Yr_p27",
    "an5qNCspzUw",
    "i8kMrJmAfjU",
    "BV1vShdzLEsn_p1",
    "420p2gYFTa",
    "BV1svYmz6EFB_pNA",
    "Bg3kJjFReAQ",
    "BV1MFZKYnEdM_pNA",
)
MOVE_NAMES = ("qishi", "yemafenzong", "baiheliangchi")
DEFAULT_TEACHER_VIDEO_ROOT = Path("/home/sqw/VisualSearch/aqa/teach_trimmed")
DEFAULT_TEACHER_TRACKING_ROOT = Path("/home/sqw/VisualSearch/aqa/Tracking/teach_trimmed")
DEFAULT_REFERENCE_SEGMENTS = PROJECT_ROOT / "tas_annotations" / "QxVvRcRn2TA_segments_5fps.csv"
DEFAULT_TEACHER_DTW_ROOT = PROJECT_ROOT / "tas_smpl_dtw_results"
DEFAULT_CLIP_ROOT = Path("/home/sqw/VisualSearch/aqa/ActionSegments")
DEFAULT_CLIP_TRACKING_ROOT = Path("/home/sqw/VisualSearch/aqa/Tracking")
DEFAULT_HUMAN_RANKINGS = PROJECT_ROOT / "human_rankings.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "pelvis_quality_results"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def teacher_segments(
    teacher_id: str,
    reference_segments_path: Path,
    teacher_dtw_root: Path,
) -> list[dict]:
    path = (
        reference_segments_path
        if teacher_id == REFERENCE_ID
        else teacher_dtw_root / teacher_id / "segments_5fps.csv"
    )
    rows = read_rows(path)
    selected = sorted(
        (row for row in rows if int(row["move_id"]) <= len(MOVE_NAMES)),
        key=lambda row: int(row["move_id"]),
    )
    if len(selected) != len(MOVE_NAMES):
        raise ValueError(f"Expected first three moves in {path}, got {len(selected)}.")
    normalized: list[dict] = []
    for row in selected:
        normalized.append(
            {
                "move_id": int(row["move_id"]),
                "move_name": row["move_name"],
                "start_frame_5fps": int(row["start_frame"]),
                "end_frame_5fps": int(row["end_frame"]),
            }
        )
    return normalized


def load_path(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return np.stack(
            (
                data["target_sample_indices_0based"].astype(np.int64),
                data["reference_sample_indices_0based"].astype(np.int64),
            ),
            axis=1,
        )


def subset_root_track(
    track: TrackRootSequence,
    start_frame_1based: int,
    end_frame_1based: int,
) -> TrackRootSequence:
    selected = (track.frame_numbers >= start_frame_1based) & (
        track.frame_numbers <= end_frame_1based
    )
    if np.sum(selected) < 2:
        raise ValueError(
            f"Only {np.sum(selected)} tracked frames in source interval "
            f"[{start_frame_1based}, {end_frame_1based}]."
        )
    indices = np.flatnonzero(selected)
    return TrackRootSequence(
        frame_numbers=track.frame_numbers[indices],
        global_orients=track.global_orients[indices],
        body_poses=track.body_poses[indices],
        betas=track.betas[indices],
        joints=track.joints[indices],
        camera_translations=track.camera_translations[indices],
        track_id=track.track_id,
        source_track_ids=track.source_track_ids[indices],
    )


def signal_summary(
    subject_id: str,
    move_id: int,
    move_name: str,
    signals: PelvisSignals,
    expected_frame_count: int,
) -> dict:
    return {
        "subject_id": subject_id,
        "move_id": move_id,
        "move_name": move_name,
        "tracked_frame_count": len(signals.frame_numbers),
        "expected_frame_count": expected_frame_count,
        "tracking_coverage_ratio": len(signals.frame_numbers) / expected_frame_count,
        "valid_frame_ratio": float(np.mean(signals.valid)),
        "body_scale": signals.body_scale,
        "used_track_switch_excluded_frames": signals.excluded_track_switch_frames,
        "median_lateral_visibility": signals.median_lateral_visibility,
        "camera_depth_range_body_scales": signals.camera_depth_range_body_scales,
        "support_vertical_residual_rms": float(
            np.sqrt(np.nanmean(np.square(signals.support_vertical_residual)))
        ),
        "root_vertical_residual_rms": float(
            np.sqrt(np.nanmean(np.square(signals.root_vertical_residual)))
        ),
    }


def teacher_progress(
    teacher_id: str,
    path: np.ndarray,
    sampling,
    target_row: dict,
    reference_row: dict,
    source_frames_0based: np.ndarray,
) -> np.ndarray:
    if teacher_id == REFERENCE_ID:
        start = source_frames_0based[0]
        end = source_frames_0based[-1]
        return (
            (source_frames_0based - start) / max(end - start, 1)
        ).astype(np.float64)
    target_start = target_row["start_frame_5fps"] - 1
    target_end = target_row["end_frame_5fps"] - 1
    reference_start = reference_row["start_frame_5fps"] - 1
    reference_end = reference_row["end_frame_5fps"] - 1
    progress, _ = build_reference_progress(
        path,
        np.arange(target_start, target_end + 1, dtype=np.int64),
        sampling.source_indices[target_start : target_end + 1],
        source_frames_0based,
        reference_start,
        reference_end,
    )
    return progress


def save_teacher_template_csv(path: Path, model: TeacherPelvisModel) -> None:
    rows: list[dict] = []
    for move_index, (move_id, move_name) in enumerate(
        zip(model.move_ids, model.move_names)
    ):
        for signal_index, signal_name in enumerate(model.signal_names):
            for point_index, progress in enumerate(model.progress_grid):
                rows.append(
                    {
                        "move_id": int(move_id),
                        "move_name": move_name,
                        "signal": signal_name,
                        "progress": float(progress),
                        "median": model.median[move_index, signal_index, point_index],
                        "p10": model.p10[move_index, signal_index, point_index],
                        "p90": model.p90[move_index, signal_index, point_index],
                        "teacher_count": int(model.valid_teacher_counts[move_index]),
                    }
                )
    write_rows(path, rows)


def build_ten_teacher_model(args: argparse.Namespace) -> tuple[TeacherPelvisModel, list[dict]]:
    teacher_ids = tuple(args.teacher_video_id or DEFAULT_TEACHER_IDS)
    reference_rows = teacher_segments(
        REFERENCE_ID,
        args.reference_segments,
        args.teacher_dtw_root,
    )
    grid = np.linspace(0.0, 1.0, args.phase_points)
    shape = (len(teacher_ids), len(MOVE_NAMES), len(SIGNAL_NAMES), len(grid))
    profiles = np.full(shape, np.nan, dtype=np.float64)
    support_residual = np.full((len(teacher_ids), len(MOVE_NAMES)), np.nan)
    root_residual = np.full_like(support_residual, np.nan)
    summary_rows: list[dict] = []

    for teacher_index, teacher_id in enumerate(teacher_ids):
        video = args.teacher_video_root / f"{teacher_id}.mp4"
        sampling = inspect_video_sampling(video, args.sample_fps)
        rows = teacher_segments(
            teacher_id,
            args.reference_segments,
            args.teacher_dtw_root,
        )
        track = load_stitched_root_track(
            tracking_path(args.teacher_tracking_root, teacher_id)
        )
        if teacher_id == REFERENCE_ID:
            samples = np.arange(sampling.sample_count, dtype=np.int64)
            path = np.stack((samples, samples), axis=1)
        else:
            path = load_path(args.teacher_dtw_root / teacher_id / "dtw_path.npz")
        print(f"[teacher:{teacher_id}] extracting pelvis trajectories", flush=True)
        for move_index, (row, reference_row) in enumerate(zip(rows, reference_rows)):
            start_frame, end_frame = sampled_interval_to_source_interval(
                row["start_frame_5fps"],
                row["end_frame_5fps"],
                sampling,
            )
            segment = subset_root_track(track, start_frame, end_frame)
            native_joints = segment.to_smpl24_joints(
                batch_size=args.smpl_batch_size,
                device=args.smpl_device,
            )
            signals = compute_pelvis_signals(
                segment.frame_numbers,
                native_joints,
                segment.camera_translations,
                segment.source_track_ids,
                sampling.source_fps,
                smoothing_seconds=args.smoothing_seconds,
                intent_smoothing_seconds=args.intent_smoothing_seconds,
                hip_weight=args.hip_weight,
            )
            progress = teacher_progress(
                teacher_id,
                path,
                sampling,
                row,
                reference_row,
                segment.frame_numbers - 1,
            )
            profiles[teacher_index, move_index] = resample_pelvis_signals(
                signals,
                progress,
                grid,
            )
            support_residual[teacher_index, move_index] = np.sqrt(
                np.nanmean(np.square(signals.support_vertical_residual))
            )
            root_residual[teacher_index, move_index] = np.sqrt(
                np.nanmean(np.square(signals.root_vertical_residual))
            )
            summary_rows.append(
                signal_summary(
                    teacher_id,
                    row["move_id"],
                    row["move_name"],
                    signals,
                    end_frame - start_frame + 1,
                )
            )
        del track

    model = build_teacher_model(
        list(teacher_ids),
        list(range(1, len(MOVE_NAMES) + 1)),
        list(MOVE_NAMES),
        grid,
        profiles,
        support_residual,
        root_residual,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    model.save(args.output_root / "teacher_pelvis_model.npz")
    save_teacher_template_csv(args.output_root / "teacher_trajectory_template.csv", model)
    write_rows(args.output_root / "teacher_signal_summary.csv", summary_rows)
    return model, summary_rows


def sampled_clip_indices(start_index: int, end_index: int, fps: float, sample_fps: float) -> np.ndarray:
    return (
        uniform_sample_source_indices(end_index - start_index + 1, fps, sample_fps)
        + start_index
    )


def clip_alignment(
    student_track: TrackRootSequence,
    teacher_track: TrackRootSequence,
    student_source_indices: np.ndarray,
    teacher_source_indices: np.ndarray,
    pairwise_chunk_size: int,
) -> tuple[np.ndarray, float]:
    student = student_track.at_source_frames(student_source_indices)
    teacher = teacher_track.at_source_frames(teacher_source_indices)
    costs = pairwise_geodesic_costs(
        student.body_poses,
        teacher.body_poses,
        chunk_size=pairwise_chunk_size,
    )
    distance, _, accumulated = dtw_from_cost_matrix(costs)
    return backtrack_dtw_path(accumulated), distance


def save_case_trajectory(
    path: Path,
    signals: PelvisSignals,
    progress: np.ndarray,
) -> None:
    matrix = signals.signal_matrix()
    rows: list[dict] = []
    for index, frame in enumerate(signals.frame_numbers):
        row = {
            "source_frame_1based": int(frame),
            "reference_progress": float(progress[index]),
            "valid": bool(signals.valid[index]),
            "support_vertical_residual": signals.support_vertical_residual[index],
            "root_vertical_residual": signals.root_vertical_residual[index],
        }
        row.update(
            {
                signal_name: matrix[signal_index, index]
                for signal_index, signal_name in enumerate(SIGNAL_NAMES)
            }
        )
        rows.append(row)
    write_rows(path, rows)


def plot_case(
    path: Path,
    case_id: str,
    move_name: str,
    profile: np.ndarray,
    model: TeacherPelvisModel,
) -> None:
    move_index = model.move_index(move_name)
    indices = {name: index for index, name in enumerate(model.signal_names)}
    panels = (
        ("support_height", "Pelvis height above ankle midpoint"),
        ("support_lateral", "Pelvis lateral shift within support"),
        ("support_forward", "Pelvis forward shift within support"),
        ("root_vertical", "PHALP root vertical translation"),
        ("root_lateral", "PHALP root lateral translation"),
        ("body_yaw_change_degrees", "Body yaw change"),
    )
    figure, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    for axis, (signal_name, title) in zip(axes.flat, panels):
        signal_index = indices[signal_name]
        axis.fill_between(
            model.progress_grid,
            model.p10[move_index, signal_index],
            model.p90[move_index, signal_index],
            color="#93c5fd",
            alpha=0.45,
            label="teacher P10-P90",
        )
        axis.plot(
            model.progress_grid,
            model.median[move_index, signal_index],
            color="#1d4ed8",
            linewidth=1.8,
            label="teacher median",
        )
        axis.plot(
            model.progress_grid,
            profile[signal_index],
            color="#dc2626",
            linewidth=1.4,
            label="student",
        )
        axis.set_title(title)
        axis.set_xlabel("Reference action progress")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(loc="best", fontsize=8)
    figure.suptitle(f"{case_id} | {move_name}", fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def analyze_clips(
    args: argparse.Namespace,
    model: TeacherPelvisModel,
) -> tuple[list[dict], list[dict]]:
    teachers = teacher_clip_map(
        args.clip_root / "teach",
        args.clip_tracking_root,
    )
    student_ids = set(args.student_id or ("1", "2", "3", "4", "10"))
    moves = set(args.move or MOVE_NAMES)
    clips = [
        path
        for path in sorted((args.clip_root / "student").glob("*.mp4"))
        if path.stem.split("_", 1)[0] in student_ids and infer_move(path)[1] in moves
    ]
    metric_rows: list[dict] = []
    quality_rows: list[dict] = []
    for clip in clips:
        move_id, move_name = infer_move(clip)
        teacher_video = teachers[move_name]
        student_fps, _ = inspect_video(clip)
        teacher_fps, _ = inspect_video(teacher_video)
        student_segment = whole_clip_segment(clip, move_id, move_name, args.sample_fps)
        teacher_move_id, _ = infer_move(teacher_video)
        teacher_segment = whole_clip_segment(
            teacher_video,
            teacher_move_id,
            move_name,
            args.sample_fps,
        )
        student_track = load_stitched_root_track(
            clip_tracking_path(args.clip_tracking_root, "student", clip)
        )
        teacher_track = load_stitched_root_track(
            clip_tracking_path(args.clip_tracking_root, "teach", teacher_video)
        )
        student_samples = sampled_clip_indices(
            student_segment.source_start_index,
            student_segment.source_end_index,
            student_fps,
            args.sample_fps,
        )
        teacher_samples = sampled_clip_indices(
            teacher_segment.source_start_index,
            teacher_segment.source_end_index,
            teacher_fps,
            args.sample_fps,
        )
        path, dtw_distance = clip_alignment(
            student_track,
            teacher_track,
            student_samples,
            teacher_samples,
            args.pairwise_chunk_size,
        )
        student_root = subset_root_track(
            student_track,
            student_segment.start_frame,
            student_segment.end_frame,
        )
        native_joints = student_root.to_smpl24_joints(
            batch_size=args.smpl_batch_size,
            device=args.smpl_device,
        )
        signals = compute_pelvis_signals(
            student_root.frame_numbers,
            native_joints,
            student_root.camera_translations,
            student_root.source_track_ids,
            student_fps,
            smoothing_seconds=args.smoothing_seconds,
            intent_smoothing_seconds=args.intent_smoothing_seconds,
            hip_weight=args.hip_weight,
        )
        progress, reference_indices = build_reference_progress(
            path,
            np.arange(len(student_samples), dtype=np.int64),
            student_samples,
            student_root.frame_numbers - 1,
            0,
            len(teacher_samples) - 1,
        )
        profile = resample_pelvis_signals(signals, progress, model.progress_grid)
        metrics = compute_quality_metrics(signals, profile, model, move_name)
        case_id = clip.stem
        case_dir = args.output_root / "clips" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "case_id": case_id,
            "student_id": case_id.split("_", 1)[0],
            "move_id": move_id,
            "move_name": move_name,
            **asdict(metrics),
            "dtw_path_mean_geodesic_degrees": float(np.degrees(dtw_distance)),
            "student_frame_count": len(signals.frame_numbers),
            "student_fps": student_fps,
            "body_scale": signals.body_scale,
            "median_lateral_visibility": signals.median_lateral_visibility,
            "camera_depth_range_body_scales": signals.camera_depth_range_body_scales,
            "excluded_track_switch_frames": signals.excluded_track_switch_frames,
        }
        metric_rows.append(row)
        quality_rows.append(
            signal_summary(
                case_id,
                move_id,
                move_name,
                signals,
                student_segment.end_frame - student_segment.start_frame + 1,
            )
        )
        save_case_trajectory(case_dir / "pelvis_trajectory.csv", signals, progress)
        np.savez_compressed(
            case_dir / "pelvis_signals.npz",
            source_frames_1based=signals.frame_numbers,
            valid=signals.valid,
            reference_progress=progress,
            reference_sample_index=reference_indices,
            profile=profile,
            progress_grid=model.progress_grid,
            dtw_path=path,
        )
        plot_case(
            case_dir / "pelvis_diagnostics.png",
            case_id,
            move_name,
            profile,
            model,
        )
        (case_dir / "summary.json").write_text(
            json.dumps(
                {
                    **row,
                    "student_video": str(clip),
                    "teacher_clip": str(teacher_video),
                    "student_tracking": str(
                        clip_tracking_path(args.clip_tracking_root, "student", clip)
                    ),
                    "teacher_tracking": str(
                        clip_tracking_path(args.clip_tracking_root, "teach", teacher_video)
                    ),
                    "coordinate_system": {
                        "up": "negative camera-space Y",
                        "lateral": f"{args.hip_weight:.2f} hip + {1.0 - args.hip_weight:.2f} shoulder, projected to ground",
                        "forward": "cross(up, lateral)",
                        "origin": "ankle midpoint for support signals; first 0.5 s median for root signals",
                        "normalization": "SMPL torso plus mean leg length",
                    },
                    "warning": "PHALP root translation is a monocular camera-space proxy, not metric center of mass.",
                },
                indent=2,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[clip:{case_id}] done -> {case_dir}", flush=True)
        del student_track, teacher_track
    return metric_rows, quality_rows


def ranking_value(metric: str, value: float) -> float:
    if metric in {"vertical_oscillation_ratio", "root_vertical_oscillation_ratio"}:
        return max(value - 1.0, 0.0)
    if metric == "vertical_range_ratio":
        return abs(value - 1.0)
    return value


def save_rankings(
    output_root: Path,
    metric_rows: list[dict],
    human_rankings: Path,
) -> tuple[list[dict], list[dict]]:
    identity = {
        "case_id",
        "student_id",
        "move_id",
        "move_name",
        "student_frame_count",
        "student_fps",
        "body_scale",
        "median_lateral_visibility",
        "camera_depth_range_body_scales",
        "excluded_track_switch_frames",
        "valid_frame_ratio",
    }
    metrics = [name for name in metric_rows[0] if name not in identity]
    human = {
        (row["move"], row["student_id"]): int(row["rank"])
        for row in read_rows(human_rankings)
    }
    ranking_rows: list[dict] = []
    correlation_rows: list[dict] = []
    for move_name in MOVE_NAMES:
        selected = [row for row in metric_rows if row["move_name"] == move_name]
        for metric in metrics:
            values = np.asarray(
                [ranking_value(metric, float(row[metric])) for row in selected],
                dtype=np.float64,
            )
            ranks = rankdata(values, method="average")
            human_values = np.asarray(
                [human[(move_name, row["student_id"])] for row in selected],
                dtype=np.float64,
            )
            correlation = (
                float(spearmanr(ranks, human_values).statistic)
                if np.unique(ranks).size > 1
                else np.nan
            )
            correlation_rows.append(
                {
                    "move_name": move_name,
                    "metric": metric,
                    "spearman_correlation": correlation,
                    "ranking_policy": (
                        "excess_above_1_lower_is_better"
                        if "oscillation_ratio" in metric
                        else "absolute_difference_from_1_lower_is_better"
                        if metric == "vertical_range_ratio"
                        else "lower_is_better"
                    ),
                }
            )
            for row, value, rank in zip(selected, values, ranks):
                ranking_rows.append(
                    {
                        "move_name": move_name,
                        "metric": metric,
                        "student_id": row["student_id"],
                        "raw_value": row[metric],
                        "ranking_value": float(value),
                        "predicted_rank": float(rank),
                        "human_rank": human[(move_name, row["student_id"])],
                    }
                )
    write_rows(output_root / "metric_rankings.csv", ranking_rows)
    write_rows(output_root / "human_rank_correlations.csv", correlation_rows)
    return ranking_rows, correlation_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-video-root", type=Path, default=DEFAULT_TEACHER_VIDEO_ROOT)
    parser.add_argument("--teacher-tracking-root", type=Path, default=DEFAULT_TEACHER_TRACKING_ROOT)
    parser.add_argument("--teacher-dtw-root", type=Path, default=DEFAULT_TEACHER_DTW_ROOT)
    parser.add_argument("--reference-segments", type=Path, default=DEFAULT_REFERENCE_SEGMENTS)
    parser.add_argument("--teacher-video-id", action="append")
    parser.add_argument("--clip-root", type=Path, default=DEFAULT_CLIP_ROOT)
    parser.add_argument("--clip-tracking-root", type=Path, default=DEFAULT_CLIP_TRACKING_ROOT)
    parser.add_argument("--human-rankings", type=Path, default=DEFAULT_HUMAN_RANKINGS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--student-id", action="append")
    parser.add_argument("--move", action="append", choices=MOVE_NAMES)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--phase-points", type=int, default=201)
    parser.add_argument("--pairwise-chunk-size", type=int, default=32)
    parser.add_argument("--smoothing-seconds", type=float, default=0.20)
    parser.add_argument("--intent-smoothing-seconds", type=float, default=1.00)
    parser.add_argument("--hip-weight", type=float, default=0.70)
    parser.add_argument("--smpl-device", default="cpu")
    parser.add_argument("--smpl-batch-size", type=int, default=256)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.phase_points < 2:
        raise ValueError("--phase-points must be at least 2.")
    if args.sample_fps <= 0:
        raise ValueError("--sample-fps must be positive.")
    model, teacher_rows = build_ten_teacher_model(args)
    metric_rows, quality_rows = analyze_clips(args, model)
    write_rows(args.output_root / "all_pelvis_quality_summary.csv", metric_rows)
    write_rows(args.output_root / "student_signal_summary.csv", quality_rows)
    _, correlations = save_rankings(
        args.output_root,
        metric_rows,
        args.human_rankings,
    )
    payload = {
        "status": "ok",
        "teacher_ids": list(model.teacher_ids),
        "teacher_count": len(model.teacher_ids),
        "moves": list(MOVE_NAMES),
        "student_case_count": len(metric_rows),
        "teacher_signal_rows": len(teacher_rows),
        "phase_points": args.phase_points,
        "sample_fps_for_dtw": args.sample_fps,
        "smoothing_seconds": args.smoothing_seconds,
        "intent_smoothing_seconds": args.intent_smoothing_seconds,
        "hip_weight": args.hip_weight,
        "coordinate_note": "Pelvis relative to ankle midpoint is primary; PHALP camera/root translation is auxiliary.",
        "root_translation_reliability": (
            "Monocular camera depth varies by several body scales in these data; "
            "do not use root_forward/root_lateral as primary quality evidence."
        ),
        "composite_score_created": False,
        "correlations": correlations,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Completed {len(metric_rows)} clips -> {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
