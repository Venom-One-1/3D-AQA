#!/usr/bin/env python
"""Evaluate a teacher-calibrated pose score with unexpected-pause penalties."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import argrelextrema
from scipy.stats import rankdata, spearmanr

from aqa3d.motion_quality import (
    compute_motion_signals,
    detect_pauses,
    robust_noise_floor,
)
from aqa3d.pose_score import (
    DistanceCalibration,
    apply_pause_penalty,
    calculate_pose_score,
    calibrate_distances,
    unexpected_pause_events,
)
from aqa3d.smpl_dtw import (
    backtrack_dtw_path,
    dtw_from_cost_matrix,
    inspect_video_sampling,
    pairwise_geodesic_costs,
    select_reference_frame_matches,
)
from aqa3d.tracking import TrackPoseSequence, load_stitched_primary_track
from aqa3d.velocity_quality import TeacherMotionModel
from run_motion_quality_analysis import (
    DEFAULT_CLIP_ROOT,
    DEFAULT_CLIP_TRACKING_ROOT,
    DEFAULT_HUMAN_RANKINGS,
    SegmentAnalysisInput,
    create_clip_analysis_input,
    infer_move,
    select_track_segment,
    teacher_clip_map,
)
from run_student_tas_smpl_dtw import sampled_interval_to_source_interval
from run_tas_smpl_dtw_mapping import tracking_path
from run_velocity_quality_analysis import (
    DEFAULT_TEACHER_GROUND_TRUTH,
    DEFAULT_TEACHER_IDS,
    student_reference_progress,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "pose_score_experiment_results"
DEFAULT_TEACHER_VIDEO_ROOT = Path("/home/sqw/VisualSearch/aqa/teach_trimmed")
DEFAULT_TEACHER_TRACKING_ROOT = Path("/home/sqw/VisualSearch/aqa/Tracking/teach_trimmed")
DEFAULT_TEACHER_MODEL = (
    PROJECT_ROOT / "velocity_quality_results" / "teacher_model" / "teacher_motion_model.npz"
)
DEFAULT_MOVES = ("qishi", "yemafenzong", "baiheliangchi")


@dataclass(frozen=True)
class TeacherMoveData:
    teacher_id: str
    move_id: int
    move_name: str
    video_path: Path
    sampled_source_indices: np.ndarray
    sampled_poses: np.ndarray
    keyframe_sample_indices: np.ndarray


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_teacher_rows(path: Path, teacher_ids: tuple[str, ...]) -> dict[str, dict[str, dict]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, dict[str, dict]] = {teacher_id: {} for teacher_id in teacher_ids}
    for row in rows:
        if row["video_id"] in grouped:
            grouped[row["video_id"]][row["move_name"]] = row
    return grouped


def visual_difference_keyframes(
    video_path: Path,
    start_index: int,
    end_index: int,
    source_fps: float,
    *,
    smooth_kernel_size: int,
    keyframe_order: int,
) -> np.ndarray:
    """Extract visual-motion peaks without retaining decoded video frames."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)
    ok, first = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError(f"Cannot decode source frame {start_index} from {video_path}.")
    previous = cv2.cvtColor(first, cv2.COLOR_BGR2LUV).astype(np.int32)
    differences: list[float] = []
    for _ in range(end_index - start_index):
        ok, frame = capture.read()
        if not ok:
            break
        current = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV).astype(np.int32)
        differences.append(float(np.abs(current - previous).mean()))
        previous = current
    capture.release()
    expected = end_index - start_index
    if len(differences) != expected:
        raise RuntimeError(
            f"Decoded {len(differences) + 1} of {expected + 1} requested frames from {video_path}."
        )
    if smooth_kernel_size < 3 or smooth_kernel_size % 2 == 0:
        raise ValueError("smooth_kernel_size must be an odd integer of at least 3.")
    if len(differences) < smooth_kernel_size + 1:
        raise ValueError(f"Video interval is too short for keyframe extraction: {video_path}.")
    values = np.asarray(differences, dtype=np.float64)
    padding = smooth_kernel_size // 2
    smooth = np.convolve(
        np.pad(values, (padding, padding), mode="reflect"),
        np.ones(smooth_kernel_size, dtype=np.float64) / smooth_kernel_size,
        mode="valid",
    )
    keyframes = argrelextrema(smooth, np.greater, order=keyframe_order)[0]
    last = expected - 1
    endpoint_gap = max(int(round(source_fps)), 1)
    if keyframes.size == 0:
        keyframes = np.asarray([last], dtype=np.int64)
    elif last not in keyframes:
        if last - keyframes[-1] < endpoint_gap:
            keyframes[-1] = last
        else:
            keyframes = np.append(keyframes, last)
    if keyframes[0] + 1 > endpoint_gap:
        keyframes = np.concatenate((np.asarray([-1], dtype=np.int64), keyframes))
    return keyframes + 1


def visual_keyframes_on_sample_grid(
    video_path: Path,
    source_start_index: int,
    source_end_index: int,
    sampled_source_indices: np.ndarray,
    source_fps: float,
    *,
    smooth_kernel_size: int,
    keyframe_order: int,
) -> np.ndarray:
    relative = visual_difference_keyframes(
        video_path,
        source_start_index,
        source_end_index,
        source_fps,
        smooth_kernel_size=smooth_kernel_size,
        keyframe_order=keyframe_order,
    )
    source_keyframes = source_start_index + relative
    mapped = [int(np.argmin(np.abs(sampled_source_indices - frame))) for frame in source_keyframes]
    return np.asarray(list(dict.fromkeys(mapped)), dtype=np.int64)


def load_teacher_data(args: argparse.Namespace) -> dict[tuple[str, str], TeacherMoveData]:
    teacher_ids = tuple(args.teacher_video_id or DEFAULT_TEACHER_IDS)
    rows = load_teacher_rows(args.teacher_ground_truth, teacher_ids)
    result: dict[tuple[str, str], TeacherMoveData] = {}
    for teacher_id in teacher_ids:
        video_path = args.teacher_video_root / f"{teacher_id}.mp4"
        sampling = inspect_video_sampling(video_path, args.sample_fps)
        track = load_stitched_primary_track(tracking_path(args.teacher_tracking_root, teacher_id))
        for move_name in args.move:
            if move_name not in rows[teacher_id]:
                raise KeyError(f"No {move_name} boundary for teacher {teacher_id}.")
            row = rows[teacher_id][move_name]
            sample_start = int(row["start_frame_5fps"]) - 1
            sample_end = int(row["end_frame_5fps"]) - 1
            sampled_sources = sampling.source_indices[sample_start : sample_end + 1]
            source_start, source_end = sampled_interval_to_source_interval(
                sample_start + 1,
                sample_end + 1,
                sampling,
            )
            keyframes = visual_keyframes_on_sample_grid(
                video_path,
                source_start - 1,
                source_end - 1,
                sampled_sources,
                sampling.source_fps,
                smooth_kernel_size=args.smooth_kernel_size,
                keyframe_order=args.keyframe_order,
            )
            result[(teacher_id, move_name)] = TeacherMoveData(
                teacher_id=teacher_id,
                move_id=int(row["move_id"]),
                move_name=move_name,
                video_path=video_path,
                sampled_source_indices=sampled_sources,
                sampled_poses=track.at_source_frames(sampled_sources),
                keyframe_sample_indices=keyframes,
            )
            print(
                f"[teacher:{teacher_id}:{move_name}] "
                f"{len(sampled_sources)} samples, {len(keyframes)} keyframes",
                flush=True,
            )
    return result


def aligned_distances(
    target_poses: np.ndarray,
    reference_poses: np.ndarray,
    reference_keyframes: np.ndarray,
    *,
    chunk_size: int,
    coefficient: float,
) -> tuple[float, float, int, int]:
    costs = pairwise_geodesic_costs(target_poses, reference_poses, chunk_size=chunk_size)
    _, _, accumulated = dtw_from_cost_matrix(costs, coefficient)
    path = backtrack_dtw_path(accumulated)
    path_distance = float(np.mean(costs[path[:, 0], path[:, 1]]))
    matches = select_reference_frame_matches(costs, path, reference_keyframes)
    keyframe_distance = float(np.mean([match.local_cost for match in matches]))
    return keyframe_distance, path_distance, len(path), len(matches)


def calibrate_teachers(
    data: dict[tuple[str, str], TeacherMoveData],
    args: argparse.Namespace,
) -> tuple[dict[str, tuple[DistanceCalibration, DistanceCalibration]], list[dict], list[dict]]:
    teacher_ids = tuple(args.teacher_video_id or DEFAULT_TEACHER_IDS)
    pair_rows: list[dict] = []
    for move_name in args.move:
        for target_id in teacher_ids:
            for reference_id in teacher_ids:
                if target_id == reference_id:
                    continue
                target = data[(target_id, move_name)]
                reference = data[(reference_id, move_name)]
                key_distance, path_distance, path_length, keyframe_count = aligned_distances(
                    target.sampled_poses,
                    reference.sampled_poses,
                    reference.keyframe_sample_indices,
                    chunk_size=args.pairwise_chunk_size,
                    coefficient=args.dtw_coefficient,
                )
                pair_rows.append(
                    {
                        "move_id": reference.move_id,
                        "move_name": move_name,
                        "target_teacher_id": target_id,
                        "reference_teacher_id": reference_id,
                        "reference_keyframe_count": keyframe_count,
                        "dtw_path_length": path_length,
                        "keyframe_geodesic_degrees": float(np.degrees(key_distance)),
                        "dtw_path_mean_geodesic_degrees": float(np.degrees(path_distance)),
                    }
                )
        print(f"[calibration:{move_name}] 20 ordered teacher pairs complete", flush=True)

    calibrations: dict[str, tuple[DistanceCalibration, DistanceCalibration]] = {}
    calibration_rows: list[dict] = []
    for move_name in args.move:
        selected = [row for row in pair_rows if row["move_name"] == move_name]
        key_calibration = calibrate_distances(
            np.asarray([row["keyframe_geodesic_degrees"] for row in selected]),
            minimum_scale_degrees=args.minimum_scale_degrees,
        )
        path_calibration = calibrate_distances(
            np.asarray([row["dtw_path_mean_geodesic_degrees"] for row in selected]),
            minimum_scale_degrees=args.minimum_scale_degrees,
        )
        calibrations[move_name] = (key_calibration, path_calibration)
        for distance_type, calibration in (
            ("keyframe_geodesic", key_calibration),
            ("dtw_path_mean_geodesic", path_calibration),
        ):
            calibration_rows.append(
                {
                    "move_name": move_name,
                    "distance_type": distance_type,
                    **asdict(calibration),
                }
            )
    return calibrations, pair_rows, calibration_rows


def dilate_activity_profile(
    profile: np.ndarray,
    duration_seconds: float,
    dilation_seconds: float,
) -> np.ndarray:
    values = np.asarray(profile, dtype=bool)
    if dilation_seconds <= 0 or duration_seconds <= 0:
        return values.copy()
    radius = int(np.ceil(dilation_seconds / duration_seconds * max(len(values) - 1, 1)))
    if radius == 0:
        return values.copy()
    kernel = np.ones(2 * radius + 1, dtype=np.int16)
    return np.convolve(values.astype(np.int16), kernel, mode="same") > 0


def pause_diagnostics(
    data: SegmentAnalysisInput,
    model: TeacherMotionModel,
    args: argparse.Namespace,
) -> tuple[list, dict]:
    poses, track_ids = select_track_segment(data.student_track, data.student_segment)
    signals = compute_motion_signals(
        poses,
        data.student_fps,
        source_track_ids=track_ids,
        smoothing_seconds=args.smoothing_seconds,
    )
    move_index, region_index = model.move_region_indices(data.move_name, "whole_body")
    speed = signals.region_intensity["whole_body"]
    teacher_p90 = float(np.nanpercentile(model.speed_p90[move_index, region_index], 90.0))
    pause_threshold = max(
        robust_noise_floor(speed[signals.valid_frames]),
        args.pause_scale * teacher_p90,
    )
    pauses = detect_pauses(
        speed,
        signals.valid_frames,
        pause_threshold,
        data.student_fps,
        minimum_pause_seconds=args.minimum_pause_seconds,
        bridge_gap_seconds=args.bridge_gap_seconds,
    )
    progress = student_reference_progress(data)
    teacher_active_profile = model.active_probability[move_index, region_index] >= args.teacher_active_probability
    teacher_duration = float(model.duration_median[move_index, region_index])
    teacher_active_profile = dilate_activity_profile(
        teacher_active_profile,
        teacher_duration,
        args.activity_dilation_seconds,
    )
    teacher_active = np.interp(
        progress,
        model.progress_grid,
        teacher_active_profile.astype(np.float64),
    ) >= 0.5
    events = unexpected_pause_events(
        pauses.pause_mask,
        teacher_active,
        data.student_fps,
        minimum_teacher_active_fraction=args.minimum_teacher_active_fraction,
    )
    return events, {
        "pause_threshold_degrees_per_second": pause_threshold,
        "all_pause_count": pauses.pause_count,
        "all_pause_ratio": pauses.pause_ratio,
        "all_pause_duration_seconds": pauses.pause_duration_seconds,
        "longest_pause_seconds": pauses.longest_pause_seconds,
        "teacher_active_student_frame_count": int(np.sum(teacher_active & signals.valid_frames)),
        "excluded_student_frame_count": int(np.sum(~signals.valid_frames)),
    }


def analyze_students(
    teacher_data: dict[tuple[str, str], TeacherMoveData],
    calibrations: dict[str, tuple[DistanceCalibration, DistanceCalibration]],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    model = TeacherMotionModel.load(args.teacher_model)
    teachers = teacher_clip_map(args.clip_root / "teach", args.clip_tracking_root)
    score_rows: list[dict] = []
    event_rows: list[dict] = []
    student_videos = sorted((args.clip_root / "student").glob("*.mp4"))
    if args.clip:
        requested = set(args.clip)
        student_videos = [video for video in student_videos if video.stem in requested]
    for student_video in student_videos:
        _, move_name = infer_move(student_video)
        if move_name not in args.move:
            continue
        data = create_clip_analysis_input(
            student_video,
            teachers[move_name],
            tracking_root=args.clip_tracking_root,
            sample_fps=args.sample_fps,
            pairwise_chunk_size=args.pairwise_chunk_size,
            dtw_coefficient=args.dtw_coefficient,
        )
        reference = teacher_data[(args.reference_video_id, move_name)]
        target_poses = data.student_track.at_source_frames(data.target_sample_source_indices)
        costs = pairwise_geodesic_costs(
            target_poses,
            reference.sampled_poses,
            chunk_size=args.pairwise_chunk_size,
        )
        _, _, accumulated = dtw_from_cost_matrix(costs, args.dtw_coefficient)
        path = backtrack_dtw_path(accumulated)
        path_local_costs = costs[path[:, 0], path[:, 1]]
        data = replace(
            data,
            path=path,
            path_local_costs=path_local_costs,
            reference_sample_source_indices=reference.sampled_source_indices,
        )
        matches = select_reference_frame_matches(
            costs,
            path,
            reference.keyframe_sample_indices,
        )
        key_distance = float(np.mean([match.local_cost for match in matches]))
        path_distance = float(np.mean(path_local_costs))
        key_degrees = float(np.degrees(key_distance))
        path_degrees = float(np.degrees(path_distance))
        key_calibration, path_calibration = calibrations[move_name]
        pose = calculate_pose_score(
            key_degrees,
            path_degrees,
            key_calibration,
            path_calibration,
            keyframe_weight=args.keyframe_weight,
            teacher_median_score=args.teacher_median_score,
            points_per_scale=args.points_per_scale,
        )
        events, pause_summary = pause_diagnostics(data, model, args)
        final_score, penalty = apply_pause_penalty(
            pose.pose_score,
            len(events),
            points_per_pause=args.points_per_pause,
            maximum_penalty=args.maximum_pause_penalty,
        )
        score_rows.append(
            {
                "case_id": data.case_id,
                "student_id": data.student_id,
                "move_id": data.move_id,
                "move_name": data.move_name,
                "keyframe_geodesic_degrees": key_degrees,
                "dtw_path_mean_geodesic_degrees": path_degrees,
                "keyframe_z": pose.keyframe_z,
                "path_z": pose.path_z,
                "keyframe_score": pose.keyframe_score,
                "path_score": pose.path_score,
                "pose_score": pose.pose_score,
                "unexpected_pause_count": len(events),
                "pause_penalty": penalty,
                "final_score": final_score,
                **pause_summary,
            }
        )
        for event_index, event in enumerate(events, start=1):
            source_offset = data.student_segment.source_start_index
            event_rows.append(
                {
                    "case_id": data.case_id,
                    "student_id": data.student_id,
                    "move_id": data.move_id,
                    "move_name": data.move_name,
                    "event_index": event_index,
                    "start_frame_1based": source_offset + event.start_index + 1,
                    "end_frame_1based": source_offset + event.end_index + 1,
                    "start_time_in_move_seconds": event.start_index / data.student_fps,
                    "end_time_in_move_seconds": (event.end_index + 1) / data.student_fps,
                    "duration_seconds": event.duration_seconds,
                    "teacher_active_fraction": event.teacher_active_fraction,
                }
            )
        print(
            f"[student:{data.case_id}] pose={pose.pose_score:.2f}, "
            f"pauses={len(events)}, final={final_score:.2f}",
            flush=True,
        )
    return score_rows, event_rows


def load_human_ranks(path: Path) -> dict[tuple[str, str], int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            (row["move"], row["student_id"]): int(row["rank"])
            for row in csv.DictReader(handle)
        }


def save_rankings(
    output_root: Path,
    score_rows: list[dict],
    human_path: Path,
    args: argparse.Namespace,
) -> None:
    human = load_human_ranks(human_path)
    metrics = {
        "keyframe_geodesic_degrees": False,
        "dtw_path_mean_geodesic_degrees": False,
        "pose_score": True,
        "unexpected_pause_count": False,
        "final_score": True,
    }
    ranking_rows: list[dict] = []
    correlation_rows: list[dict] = []
    for move_name in args.move:
        selected = [row for row in score_rows if row["move_name"] == move_name]
        for metric, higher_is_better in metrics.items():
            values = np.asarray([float(row[metric]) for row in selected])
            ranking_values = -values if higher_is_better else values
            predicted_ranks = rankdata(ranking_values, method="average")
            human_ranks = np.asarray(
                [human[(move_name, str(row["student_id"]))] for row in selected],
                dtype=np.float64,
            )
            correlation, p_value = spearmanr(predicted_ranks, human_ranks)
            correlation_rows.append(
                {
                    "move_name": move_name,
                    "metric": metric,
                    "student_count": len(selected),
                    "spearman_correlation": float(correlation),
                    "p_value": float(p_value),
                }
            )
            for row, predicted_rank, human_rank in sorted(
                zip(selected, predicted_ranks, human_ranks), key=lambda item: item[1]
            ):
                ranking_rows.append(
                    {
                        "move_name": move_name,
                        "metric": metric,
                        "student_id": row["student_id"],
                        "raw_value": row[metric],
                        "predicted_rank": float(predicted_rank),
                        "human_rank": int(human_rank),
                    }
                )
    write_rows(output_root / "score_rankings.csv", ranking_rows)
    write_rows(output_root / "human_rank_correlations.csv", correlation_rows)

    sensitivity_rows: list[dict] = []
    sensitivity_correlation_rows: list[dict] = []
    for points_per_pause in args.sensitivity_points_per_pause:
        scored_by_move: dict[str, list[tuple[dict, float]]] = {
            move_name: [] for move_name in args.move
        }
        for row in score_rows:
            score, penalty = apply_pause_penalty(
                float(row["pose_score"]),
                int(row["unexpected_pause_count"]),
                points_per_pause=points_per_pause,
                maximum_penalty=args.maximum_pause_penalty,
            )
            sensitivity_rows.append(
                {
                    "move_name": row["move_name"],
                    "student_id": row["student_id"],
                    "points_per_pause": points_per_pause,
                    "pose_score": row["pose_score"],
                    "unexpected_pause_count": row["unexpected_pause_count"],
                    "pause_penalty": penalty,
                    "final_score": score,
                }
            )
            scored_by_move[row["move_name"]].append((row, score))
        for move_name, scored in scored_by_move.items():
            values = np.asarray([-score for _, score in scored], dtype=np.float64)
            predicted_ranks = rankdata(values, method="average")
            human_ranks = np.asarray(
                [human[(move_name, str(row["student_id"]))] for row, _ in scored],
                dtype=np.float64,
            )
            correlation, p_value = spearmanr(predicted_ranks, human_ranks)
            sensitivity_correlation_rows.append(
                {
                    "move_name": move_name,
                    "points_per_pause": points_per_pause,
                    "maximum_pause_penalty": args.maximum_pause_penalty,
                    "student_count": len(scored),
                    "spearman_correlation": float(correlation),
                    "p_value": float(p_value),
                }
            )
    write_rows(output_root / "penalty_sensitivity.csv", sensitivity_rows)
    write_rows(
        output_root / "penalty_sensitivity_correlations.csv",
        sensitivity_correlation_rows,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--teacher-video-root", type=Path, default=DEFAULT_TEACHER_VIDEO_ROOT)
    parser.add_argument("--teacher-tracking-root", type=Path, default=DEFAULT_TEACHER_TRACKING_ROOT)
    parser.add_argument("--teacher-ground-truth", type=Path, default=DEFAULT_TEACHER_GROUND_TRUTH)
    parser.add_argument("--teacher-model", type=Path, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--teacher-video-id", action="append")
    parser.add_argument("--reference-video-id", default="QxVvRcRn2TA")
    parser.add_argument("--clip-root", type=Path, default=DEFAULT_CLIP_ROOT)
    parser.add_argument("--clip-tracking-root", type=Path, default=DEFAULT_CLIP_TRACKING_ROOT)
    parser.add_argument("--human-rankings", type=Path, default=DEFAULT_HUMAN_RANKINGS)
    parser.add_argument("--move", action="append", choices=DEFAULT_MOVES)
    parser.add_argument("--clip", action="append")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--pairwise-chunk-size", type=int, default=64)
    parser.add_argument("--dtw-coefficient", type=float, default=1.0)
    parser.add_argument("--smooth-kernel-size", type=int, default=25)
    parser.add_argument("--keyframe-order", type=int, default=15)
    parser.add_argument("--minimum-scale-degrees", type=float, default=1.0)
    parser.add_argument("--keyframe-weight", type=float, default=0.40)
    parser.add_argument("--teacher-median-score", type=float, default=95.0)
    parser.add_argument("--points-per-scale", type=float, default=10.0)
    parser.add_argument("--smoothing-seconds", type=float, default=0.20)
    parser.add_argument("--pause-scale", type=float, default=0.05)
    parser.add_argument("--minimum-pause-seconds", type=float, default=0.40)
    parser.add_argument("--bridge-gap-seconds", type=float, default=0.10)
    parser.add_argument("--teacher-active-probability", type=float, default=0.60)
    parser.add_argument("--activity-dilation-seconds", type=float, default=0.40)
    parser.add_argument("--minimum-teacher-active-fraction", type=float, default=0.70)
    parser.add_argument("--points-per-pause", type=float, default=5.0)
    parser.add_argument("--maximum-pause-penalty", type=float, default=15.0)
    parser.add_argument(
        "--sensitivity-points-per-pause",
        type=float,
        nargs="+",
        default=(0.0, 2.0, 5.0, 8.0),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.move = tuple(args.move or DEFAULT_MOVES)
    started = time.perf_counter()
    args.output_root.mkdir(parents=True, exist_ok=True)
    teacher_data = load_teacher_data(args)
    calibrations, pair_rows, calibration_rows = calibrate_teachers(teacher_data, args)
    write_rows(args.output_root / "teacher_pairwise_distances.csv", pair_rows)
    write_rows(args.output_root / "teacher_calibration.csv", calibration_rows)
    score_rows, event_rows = analyze_students(teacher_data, calibrations, args)
    write_rows(args.output_root / "student_pose_scores.csv", score_rows)
    write_rows(args.output_root / "unexpected_pause_events.csv", event_rows)
    save_rankings(args.output_root, score_rows, args.human_rankings, args)
    teacher_ids = tuple(args.teacher_video_id or DEFAULT_TEACHER_IDS)
    summary = {
        "status": "ok",
        "moves": list(args.move),
        "teacher_ids": list(teacher_ids),
        "teacher_ordered_pair_count_per_move": len(teacher_ids) * (len(teacher_ids) - 1),
        "student_case_count": len(score_rows),
        "pose_score": {
            "keyframe_weight": args.keyframe_weight,
            "path_weight": 1.0 - args.keyframe_weight,
            "teacher_median_score": args.teacher_median_score,
            "points_per_robust_scale": args.points_per_scale,
        },
        "pause_penalty": {
            "points_per_unexpected_pause": args.points_per_pause,
            "maximum_penalty": args.maximum_pause_penalty,
            "teacher_active_probability": args.teacher_active_probability,
            "minimum_teacher_active_fraction": args.minimum_teacher_active_fraction,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Completed pose-score experiment -> {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
