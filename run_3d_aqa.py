#!/usr/bin/env python
"""Run YOLO+DTW alignment and score matched frames with SMPL geodesic distance."""

from __future__ import annotations

import argparse
from pathlib import Path

from aqa3d.pipeline import run_pair, save_pair_result


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_YOLO_WEIGHTS = PROJECT_ROOT.parent / "aqa" / "model_weights" / "yolo11m-pose.pt"


def infer_move(video_path: str | Path) -> str:
    return Path(video_path).stem.rsplit("_", 1)[-1].lower()


def load_yolo(weights: Path, device: str):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise ImportError(
            "The 4d-humans environment needs ultralytics for the existing YOLO alignment stage. "
            "Install it with: python -m pip install ultralytics"
        ) from error
    model = YOLO(str(weights))
    if device != "auto":
        model.to(device)
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-video", required=True, type=Path)
    parser.add_argument("--teacher-video", required=True, type=Path)
    parser.add_argument("--student-tracking", required=True, type=Path)
    parser.add_argument("--teacher-tracking", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--move", choices=("qishi", "yemafenzong", "baiheliangchi"))
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--device", default="auto", help="YOLO device, for example cuda:2; default: auto")
    parser.add_argument("--smooth-kernel-size", type=int, default=25)
    parser.add_argument("--keyframe-order", type=int, default=15)
    parser.add_argument("--dtw-coefficient", type=float, default=1.0)
    parser.add_argument("--yolo-batch-size", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for path in (args.student_video, args.teacher_video, args.student_tracking, args.teacher_tracking, args.yolo_weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    move = args.move or infer_move(args.student_video)
    model = load_yolo(args.yolo_weights, args.device)
    result = run_pair(
        args.student_video, args.teacher_video, args.student_tracking, args.teacher_tracking, model,
        move=move,
        smooth_kernel_size=args.smooth_kernel_size,
        keyframe_order=args.keyframe_order,
        dtw_coefficient=args.dtw_coefficient,
        yolo_batch_size=args.yolo_batch_size,
    )
    save_pair_result(
        result, args.output_dir,
        student_video_path=args.student_video,
        teacher_video_path=args.teacher_video,
        student_tracking_path=args.student_tracking,
        teacher_tracking_path=args.teacher_tracking,
        move=move,
    )
    print(f"Saved {result.errors_radians.shape[0]} matched keyframes to {args.output_dir}")
    print(f"Mean local-SMPL geodesic distance: {result.distance_radians:.6f} rad ({result.distance_radians * 180.0 / 3.141592653589793:.3f} deg)")


if __name__ == "__main__":
    main()
