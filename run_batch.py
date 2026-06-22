#!/usr/bin/env python
"""Batch-run 3D-AQA for the tracked student action segments."""

from __future__ import annotations

import argparse
from pathlib import Path

from aqa3d.pipeline import run_pair, save_pair_result
from run_3d_aqa import DEFAULT_YOLO_WEIGHTS, infer_move, load_yolo


DEFAULT_ACTION_ROOT = Path("/home/sqw/VisualSearch/aqa/ActionSegments")
DEFAULT_TRACKING_ROOT = Path("/home/sqw/VisualSearch/aqa/Tracking")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-root", type=Path, default=DEFAULT_ACTION_ROOT / "student")
    parser.add_argument("--teacher-root", type=Path, default=DEFAULT_ACTION_ROOT / "teach")
    parser.add_argument("--tracking-root", type=Path, default=DEFAULT_TRACKING_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--student", action="append", help="Student clip stem to process; repeatable")
    parser.add_argument("--move", choices=("qishi", "yemafenzong", "baiheliangchi"))
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smooth-kernel-size", type=int, default=25)
    parser.add_argument("--keyframe-order", type=int, default=15)
    parser.add_argument("--dtw-coefficient", type=float, default=1.0)
    parser.add_argument("--yolo-batch-size", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    teachers = {infer_move(path): path for path in args.teacher_root.glob("*.mp4")}
    clips = sorted(args.student_root.glob("*.mp4"))
    if args.student:
        requested = set(args.student)
        clips = [path for path in clips if path.stem in requested]
    if args.move:
        clips = [path for path in clips if infer_move(path) == args.move]
    if not clips:
        raise ValueError("No student clips matched the requested filters.")

    model = load_yolo(args.yolo_weights, args.device)
    failures: list[str] = []
    for student_video in clips:
        move = infer_move(student_video)
        teacher_video = teachers.get(move)
        if teacher_video is None:
            failures.append(f"{student_video.stem}: teacher video for {move} not found")
            continue
        student_tracking = args.tracking_root / "student" / student_video.stem / "results" / f"demo_{student_video.stem}.pkl"
        teacher_tracking = args.tracking_root / "teach" / teacher_video.stem / "results" / f"demo_{teacher_video.stem}.pkl"
        try:
            result = run_pair(
                student_video, teacher_video, student_tracking, teacher_tracking, model,
                move=move,
                smooth_kernel_size=args.smooth_kernel_size,
                keyframe_order=args.keyframe_order,
                dtw_coefficient=args.dtw_coefficient,
                yolo_batch_size=args.yolo_batch_size,
            )
            output_dir = args.output_root / student_video.stem
            save_pair_result(
                result, output_dir,
                student_video_path=student_video,
                teacher_video_path=teacher_video,
                student_tracking_path=student_tracking,
                teacher_tracking_path=teacher_tracking,
                move=move,
            )
            print(f"{student_video.stem}: {result.distance_radians:.6f} rad -> {output_dir}")
        except Exception as error:
            failures.append(f"{student_video.stem}: {error}")
            print(f"FAILED {failures[-1]}")
    if failures:
        raise RuntimeError("Batch finished with failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
