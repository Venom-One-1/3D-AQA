#!/usr/bin/env python
"""Score all students using motion-peak keyframes from each teacher video.

This is intentionally separate from ``run_batch.py``: the existing pipeline
continues to use student keyframes for diagnostic analysis, while this script
uses a fixed teacher-keyframe set for within-move student ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aqa3d.alignment import body_vectors, dtw_alignment, extract_keyframes, sample_video, trim_qishi_student_video
from aqa3d.geodesic import geodesic_distance
from aqa3d.tracking import TrackPoseSequence, load_primary_track


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ACTION_ROOT = Path("/home/sqw/VisualSearch/aqa/ActionSegments")
DEFAULT_TRACKING_ROOT = Path("/home/sqw/VisualSearch/aqa/Tracking")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "teacher_keyframe_results"
DEFAULT_YOLO_WEIGHTS = PROJECT_ROOT.parent / "aqa" / "model_weights" / "yolo11m-pose.pt"


@dataclass(frozen=True)
class TeacherReference:
    video_path: Path
    sampled_source_indices: np.ndarray
    vectors: np.ndarray
    keyframe_indices: np.ndarray
    track: TrackPoseSequence


def infer_move(video_path: Path) -> str:
    return video_path.stem.rsplit("_", 1)[-1].lower()


def load_yolo(weights: Path, device: str):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise ImportError("Install ultralytics in the active environment before running this script.") from error
    model = YOLO(str(weights))
    if device != "auto":
        model.to(device)
    return model


def load_teacher_reference(
    teacher_video: Path,
    teacher_tracking: Path,
    model,
    *,
    smooth_kernel_size: int,
    keyframe_order: int,
    yolo_batch_size: int,
) -> TeacherReference:
    sampled = sample_video(teacher_video)
    vectors = body_vectors(model, sampled.frames, yolo_batch_size)
    keyframes = extract_keyframes(sampled.frames, smooth_kernel_size, keyframe_order)
    return TeacherReference(
        video_path=teacher_video,
        sampled_source_indices=sampled.source_indices,
        vectors=vectors,
        keyframe_indices=keyframes,
        track=load_primary_track(teacher_tracking),
    )


def select_student_matches(
    student_vectors: np.ndarray,
    teacher: TeacherReference,
    matching: dict[int, list[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map fixed teacher anchors to students using minimum local 2D vector error."""
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
        # This is the raw 13-vector distance, not a cumulative DTW cost.
        local_distances_for_candidates = np.linalg.norm(
            student_vectors[candidate_array] - teacher.vectors[int(teacher_index)], axis=-1
        ).mean(axis=-1)
        best_position = int(np.argmin(local_distances_for_candidates))
        selected.append(int(candidate_array[best_position]))
        candidate_counts.append(len(candidates))
        local_distances.append(float(local_distances_for_candidates[best_position]))
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
        selected_local_2d_distances=local_distances,
    )
    with (output_dir / "teacher_anchored_keyframes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "anchor_order", "teacher_sample_index", "student_sample_index",
                "teacher_source_frame_0based", "student_source_frame_0based",
                "teacher_phalp_frame_1based", "student_phalp_frame_1based",
                "student_dtw_candidate_count", "selected_local_2d_distance",
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
                    int(teacher_source) + 1, int(student_source) + 1, int(count), float(local_distance),
                    float(np.mean(errors)), float(np.degrees(np.mean(errors))),
                )
            )
    summary = {
        "anchor_strategy": "teacher_motion_peaks",
        "weighting": "uniform across fixed teacher keyframes and 23 local SMPL joints",
        "global_orientation_included": False,
        "student_video": str(student_video),
        "teacher_video": str(teacher.video_path),
        "student_tracking": str(student_tracking),
        "teacher_track_id": teacher.track.track_id,
        "student_track_id": student_track_id,
        "teacher_keyframe_count": int(len(teacher.keyframe_indices)),
        "unique_matched_student_frame_count": int(len(np.unique(student_sample_indices))),
        "dtw_distance_2d": float(dtw_distance),
        "mean_geodesic_distance_radians": float(distance_radians),
        "mean_geodesic_distance_degrees": float(np.degrees(distance_radians)),
        "score": None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def process_student(
    student_video: Path,
    student_tracking: Path,
    teacher: TeacherReference,
    model,
    *,
    smooth_kernel_size: int,
    keyframe_order: int,
    coefficient: float,
    yolo_batch_size: int,
    output_root: Path,
) -> float:
    student = sample_video(student_video)
    if infer_move(student_video) == "qishi":
        student = trim_qishi_student_video(student)
    student_vectors = body_vectors(model, student.frames, yolo_batch_size)
    dtw_distance, matching, _ = dtw_alignment(student_vectors, teacher.vectors, coefficient)
    student_samples, candidate_counts, local_distances = select_student_matches(student_vectors, teacher, matching)
    student_sources = student.source_indices[student_samples]
    teacher_sources = teacher.sampled_source_indices[teacher.keyframe_indices]

    student_track = load_primary_track(student_tracking)
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
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--smooth-kernel-size", type=int, default=25)
    parser.add_argument("--keyframe-order", type=int, default=15)
    parser.add_argument("--dtw-coefficient", type=float, default=1.0)
    parser.add_argument("--yolo-batch-size", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.yolo_weights.is_file():
        raise FileNotFoundError(args.yolo_weights)
    clips = sorted(args.student_root.glob("*.mp4"))
    if args.student:
        requested = set(args.student)
        clips = [clip for clip in clips if clip.stem in requested]
    if args.move:
        clips = [clip for clip in clips if infer_move(clip) == args.move]
    if not clips:
        raise ValueError("No student clips matched the requested filters.")

    teachers = {infer_move(path): path for path in args.teacher_root.glob("*.mp4")}
    model = load_yolo(args.yolo_weights, args.device)
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
                teacher_tracking = args.tracking_root / "teach" / teacher_video.stem / "results" / f"demo_{teacher_video.stem}.pkl"
                references[move] = load_teacher_reference(
                    teacher_video, teacher_tracking, model,
                    smooth_kernel_size=args.smooth_kernel_size,
                    keyframe_order=args.keyframe_order,
                    yolo_batch_size=args.yolo_batch_size,
                )
            student_tracking = args.tracking_root / "student" / student_video.stem / "results" / f"demo_{student_video.stem}.pkl"
            distance = process_student(
                student_video, student_tracking, references[move], model,
                smooth_kernel_size=args.smooth_kernel_size,
                keyframe_order=args.keyframe_order,
                coefficient=args.dtw_coefficient,
                yolo_batch_size=args.yolo_batch_size,
                output_root=args.output_root,
            )
            print(f"{student_video.stem}: {distance:.6f} rad -> {args.output_root / student_video.stem}")
        except Exception as error:
            failures.append(f"{student_video.stem}: {error}")
            print(f"FAILED {failures[-1]}")
    if failures:
        raise RuntimeError("Teacher-keyframe batch finished with failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
