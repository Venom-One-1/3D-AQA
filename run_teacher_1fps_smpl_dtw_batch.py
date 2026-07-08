#!/usr/bin/env python
"""Score students with 1FPS teacher anchors and SMPL-geodesic DTW alignment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from aqa3d.alignment import sample_video
from aqa3d.tracking import load_primary_track
from run_teacher_keyframe_smpl_dtw_batch import (
    DEFAULT_ACTION_ROOT,
    DEFAULT_TRACKING_ROOT,
    TeacherReference,
    build_teacher_map,
    infer_move,
    process_student,
    tracking_path_for_video,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "teacher_1fps_smpl_dtw_results"


def uniform_1fps_keyframes(sampled_source_indices: np.ndarray, source_fps: float) -> np.ndarray:
    """Choose teacher sampled-frame indices at 1FPS, including first and last."""
    if sampled_source_indices.ndim != 1 or sampled_source_indices.size == 0:
        raise ValueError("sampled_source_indices must be a non-empty 1D array.")
    if source_fps <= 0:
        raise ValueError(f"source_fps must be positive, got {source_fps}.")

    last_source = int(sampled_source_indices[-1])
    step = max(int(round(source_fps)), 1)
    target_sources = list(range(0, last_source + 1, step))
    target_sources.extend((0, last_source))
    keyframes: list[int] = []
    for target in sorted(set(target_sources)):
        index = int(np.argmin(np.abs(sampled_source_indices - target)))
        keyframes.append(index)
    return np.asarray(sorted(set(keyframes)), dtype=np.int64)


def load_teacher_reference_1fps(teacher_video: Path, teacher_tracking: Path) -> TeacherReference:
    sampled = sample_video(teacher_video)
    keyframes = uniform_1fps_keyframes(sampled.source_indices, sampled.source_fps)
    track = load_primary_track(teacher_tracking)
    sampled_body_poses = track.at_source_frames(sampled.source_indices)
    return TeacherReference(
        video_path=teacher_video,
        sampled_source_indices=sampled.source_indices,
        keyframe_indices=keyframes,
        track=track,
        sampled_body_poses=sampled_body_poses,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-root", type=Path, default=DEFAULT_ACTION_ROOT / "student")
    parser.add_argument("--teacher-root", type=Path, default=DEFAULT_ACTION_ROOT / "teach")
    parser.add_argument("--tracking-root", type=Path, default=DEFAULT_TRACKING_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--student", action="append", help="Student clip stem to process; repeatable")
    parser.add_argument("--move", choices=("qishi", "yemafenzong", "baiheliangchi"))
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
                references[move] = load_teacher_reference_1fps(teacher_video, teacher_tracking)
            student_tracking = tracking_path_for_video(args.tracking_root, "student", student_video)
            distance = process_student(
                student_video,
                student_tracking,
                references[move],
                coefficient=args.dtw_coefficient,
                pairwise_chunk_size=args.pairwise_chunk_size,
                output_root=args.output_root,
            )
            summary_path = args.output_root / student_video.stem / "summary.json"
            _patch_summary_for_1fps(summary_path)
            print(f"{student_video.stem}: {distance:.6f} rad -> {args.output_root / student_video.stem}")
        except Exception as error:
            failures.append(f"{student_video.stem}: {error}")
            print(f"FAILED {failures[-1]}")
    if failures:
        raise RuntimeError("1FPS teacher-anchor SMPL-DTW batch finished with failures:\n" + "\n".join(failures))


def _patch_summary_for_1fps(summary_path: Path) -> None:
    import json

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["anchor_strategy"] = "teacher_uniform_1fps_include_first_last"
    summary["teacher_keyframe_sampling_fps"] = 1.0
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
