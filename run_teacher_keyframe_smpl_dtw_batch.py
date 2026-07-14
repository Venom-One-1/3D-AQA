#!/usr/bin/env python
"""Score teacher-keyframe anchors after DTW with SMPL geodesic local costs.

This script is separate from ``run_teacher_keyframe_batch.py``. It keeps the
same teacher motion-peak anchors, but replaces the YOLO 2D body-vector DTW
local cost with the mean geodesic distance between SMPL local joint rotations.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aqa3d.alignment import extract_keyframes, sample_video, trim_qishi_student_video
from aqa3d.geodesic import geodesic_distance
from aqa3d.smpl_dtw import dtw_from_cost_matrix, pairwise_geodesic_costs
from aqa3d.tracking import TrackPoseSequence, load_primary_track


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ACTION_ROOT = Path("/home/sqw/VisualSearch/aqa/ActionSegments")
DEFAULT_TRACKING_ROOT = Path("/home/sqw/VisualSearch/aqa/Tracking")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "teacher_keyframe_smpl_dtw_results"


@dataclass(frozen=True)
class TeacherReference:
    video_path: Path
    sampled_source_indices: np.ndarray
    keyframe_indices: np.ndarray
    track: TrackPoseSequence
    sampled_body_poses: np.ndarray


def infer_move(video_path: Path) -> str:
    return video_path.stem.rsplit("_", 1)[-1].lower()


def tracking_path_for_video(tracking_root: Path, split: str, video: Path) -> Path:
    return tracking_root / split / video.stem / "results" / f"demo_{video.stem}.pkl"


def build_teacher_map(teacher_root: Path, tracking_root: Path) -> dict[str, Path]:
    """Choose one teacher video per move, requiring an existing tracking file."""
    teachers: dict[str, Path] = {}
    skipped: dict[str, list[str]] = {}
    for video in sorted(teacher_root.glob("*.mp4")):
        move = infer_move(video)
        tracking_path = tracking_path_for_video(tracking_root, "teach", video)
        if not tracking_path.is_file():
            skipped.setdefault(move, []).append(video.stem)
            continue
        teachers.setdefault(move, video)
    missing = sorted(set(skipped) - set(teachers))
    if missing:
        details = "; ".join(f"{move}: {', '.join(skipped[move])}" for move in missing)
        raise FileNotFoundError(f"No teacher tracking file exists for move(s): {details}")
    return teachers


def load_teacher_reference(
    teacher_video: Path,
    teacher_tracking: Path,
    *,
    smooth_kernel_size: int,
    keyframe_order: int,
) -> TeacherReference:
    sampled = sample_video(teacher_video)
    keyframes = extract_keyframes(sampled.frames, smooth_kernel_size, keyframe_order)
    track = load_primary_track(teacher_tracking)
    sampled_body_poses = track.at_source_frames(sampled.source_indices)
    return TeacherReference(
        video_path=teacher_video,
        sampled_source_indices=sampled.source_indices,
        keyframe_indices=keyframes,
        track=track,
        sampled_body_poses=sampled_body_poses,
    )


def select_student_matches(
    local_costs: np.ndarray,
    teacher: TeacherReference,
    matching: dict[int, list[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map teacher anchors to students using minimum local SMPL geodesic cost."""
    inverse_matching: dict[int, list[int]] = {}
    for student_index, teacher_indices in matching.items():
        for teacher_index in teacher_indices:
            inverse_matching.setdefault(int(teacher_index), []).append(int(student_index))

    selected: list[int] = []
    candidate_counts: list[int] = []
    local_distances: list[float] = []
    for teacher_index in teacher.keyframe_indices:
        candidates = inverse_matching.get(int(teacher_index), [])
        if not candidates:
            raise RuntimeError(f"DTW path does not contain teacher keyframe {teacher_index}.")
        candidate_array = np.asarray(candidates, dtype=np.int64)
        candidate_costs = local_costs[candidate_array, int(teacher_index)]
        best_position = int(np.argmin(candidate_costs))
        selected.append(int(candidate_array[best_position]))
        candidate_counts.append(len(candidates))
        local_distances.append(float(candidate_costs[best_position]))
    return (
        np.asarray(selected, dtype=np.int64),
        np.asarray(candidate_counts, dtype=np.int64),
        np.asarray(local_distances, dtype=np.float64),
    )


def save_result(
    output_dir: Path,
    *,
    student_video: Path,
    teacher: TeacherReference,
    student_tracking: Path,
    student_track_id: int,
    student_sample_indices: np.ndarray,
    student_source_indices: np.ndarray,
    teacher_source_indices: np.ndarray,
    candidate_counts: np.ndarray,
    local_distances: np.ndarray,
    errors_radians: np.ndarray,
    distance_radians: float,
    dtw_distance: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    errors_degrees = np.degrees(errors_radians)
    np.savez_compressed(
        output_dir / "geodesic_errors.npz",
        errors_radians=errors_radians,
        errors_degrees=errors_degrees,
        teacher_keyframe_indices=teacher.keyframe_indices,
        student_match_indices=student_sample_indices,
        teacher_source_frame_indices=teacher_source_indices,
        student_source_frame_indices=student_source_indices,
        candidate_counts=candidate_counts,
        selected_local_smpl_geodesic_distances_radians=local_distances,
        selected_local_smpl_geodesic_distances_degrees=np.degrees(local_distances),
    )
    with (output_dir / "teacher_anchored_keyframes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "anchor_order", "teacher_sample_index", "student_sample_index",
                "teacher_source_frame_0based", "student_source_frame_0based",
                "teacher_phalp_frame_1based", "student_phalp_frame_1based",
                "student_dtw_candidate_count",
                "selected_local_smpl_geodesic_distance_radians",
                "selected_local_smpl_geodesic_distance_degrees",
                "mean_geodesic_error_radians", "mean_geodesic_error_degrees",
            )
        )
        for anchor_order, values in enumerate(
            zip(
                teacher.keyframe_indices, student_sample_indices, teacher_source_indices, student_source_indices,
                candidate_counts, local_distances, errors_radians,
            )
        ):
            teacher_sample, student_sample, teacher_source, student_source, count, local_distance, errors = values
            writer.writerow(
                (
                    anchor_order, int(teacher_sample), int(student_sample), int(teacher_source), int(student_source),
                    int(teacher_source) + 1, int(student_source) + 1, int(count),
                    float(local_distance), float(np.degrees(local_distance)),
                    float(np.mean(errors)), float(np.degrees(np.mean(errors))),
                )
            )
    summary = {
        "anchor_strategy": "teacher_motion_peaks",
        "dtw_local_cost": "mean local-SMPL geodesic distance across 23 joints",
        "anchor_candidate_selection": "minimum local SMPL geodesic distance, not accumulated DTW cost",
        "weighting": "uniform across fixed teacher keyframes and 23 local SMPL joints",
        "global_orientation_included": False,
        "student_video": str(student_video),
        "teacher_video": str(teacher.video_path),
        "student_tracking": str(student_tracking),
        "teacher_track_id": teacher.track.track_id,
        "student_track_id": student_track_id,
        "teacher_keyframe_count": int(len(teacher.keyframe_indices)),
        "unique_matched_student_frame_count": int(len(np.unique(student_sample_indices))),
        "dtw_distance_smpl_geodesic_radians": float(dtw_distance),
        "dtw_distance_smpl_geodesic_degrees": float(np.degrees(dtw_distance)),
        "mean_geodesic_distance_radians": float(distance_radians),
        "mean_geodesic_distance_degrees": float(np.degrees(distance_radians)),
        "score": None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def process_student(
    student_video: Path,
    student_tracking: Path,
    teacher: TeacherReference,
    *,
    coefficient: float,
    pairwise_chunk_size: int,
    output_root: Path,
) -> float:
    student = sample_video(student_video)
    if infer_move(student_video) == "qishi":
        student = trim_qishi_student_video(student)

    student_track = load_primary_track(student_tracking)
    student_sampled_poses = student_track.at_source_frames(student.source_indices)
    local_costs = pairwise_geodesic_costs(
        student_sampled_poses,
        teacher.sampled_body_poses,
        chunk_size=pairwise_chunk_size,
    )
    dtw_distance, matching, _ = dtw_from_cost_matrix(local_costs, coefficient)
    student_samples, candidate_counts, local_distances = select_student_matches(local_costs, teacher, matching)
    student_sources = student.source_indices[student_samples]
    teacher_sources = teacher.sampled_source_indices[teacher.keyframe_indices]

    student_poses = student_track.at_source_frames(student_sources)
    teacher_poses = teacher.track.at_source_frames(teacher_sources)
    errors, distance = geodesic_distance(student_poses, teacher_poses)
    save_result(
        output_root / student_video.stem,
        student_video=student_video,
        teacher=teacher,
        student_tracking=student_tracking,
        student_track_id=student_track.track_id,
        student_sample_indices=student_samples,
        student_source_indices=student_sources,
        teacher_source_indices=teacher_sources,
        candidate_counts=candidate_counts,
        local_distances=local_distances,
        errors_radians=errors,
        distance_radians=float(distance),
        dtw_distance=dtw_distance,
    )
    return float(distance)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-root", type=Path, default=DEFAULT_ACTION_ROOT / "student")
    parser.add_argument("--teacher-root", type=Path, default=DEFAULT_ACTION_ROOT / "teach")
    parser.add_argument("--tracking-root", type=Path, default=DEFAULT_TRACKING_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--student", action="append", help="Student clip stem to process; repeatable")
    parser.add_argument("--move", choices=("qishi", "yemafenzong", "baiheliangchi"))
    parser.add_argument("--smooth-kernel-size", type=int, default=25)
    parser.add_argument("--keyframe-order", type=int, default=15)
    parser.add_argument("--dtw-coefficient", type=float, default=1.0)
    parser.add_argument("--pairwise-chunk-size", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    clips = sorted(args.student_root.glob("*.mp4"))
    if args.student:
        requested = set(args.student)
        clips = [clip for clip in clips if clip.stem in requested]
    if args.move:
        clips = [clip for clip in clips if infer_move(clip) == args.move]
    if not clips:
        raise ValueError("No student clips matched the requested filters.")

    teachers = build_teacher_map(args.teacher_root, args.tracking_root)
    references: dict[str, TeacherReference] = {}
    failures: list[str] = []
    for student_video in clips:
        move = infer_move(student_video)
        teacher_video = teachers.get(move)
        if teacher_video is None:
            failures.append(f"{student_video.stem}: teacher video for {move} not found")
            continue
        try:
            if move not in references:
                teacher_tracking = tracking_path_for_video(args.tracking_root, "teach", teacher_video)
                references[move] = load_teacher_reference(
                    teacher_video,
                    teacher_tracking,
                    smooth_kernel_size=args.smooth_kernel_size,
                    keyframe_order=args.keyframe_order,
                )
            student_tracking = tracking_path_for_video(args.tracking_root, "student", student_video)
            distance = process_student(
                student_video,
                student_tracking,
                references[move],
                coefficient=args.dtw_coefficient,
                pairwise_chunk_size=args.pairwise_chunk_size,
                output_root=args.output_root,
            )
            print(f"{student_video.stem}: {distance:.6f} rad -> {args.output_root / student_video.stem}")
        except Exception as error:
            failures.append(f"{student_video.stem}: {error}")
            print(f"FAILED {failures[-1]}")
    if failures:
        raise RuntimeError("SMPL-geodesic DTW batch finished with failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
