#!/usr/bin/env python
"""Export generic SMPL-24 angle metrics for teacher-keyframe matched results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from aqa3d.angle_metrics import compute_angle_metrics
from aqa3d.tracking import DEFAULT_SMPL_MODEL_PATH, load_primary_smpl_track


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "teacher_keyframe_results"
DEFAULT_OUTPUT_ROOT = DEFAULT_RESULT_ROOT / "angle_metrics"
DEFAULT_TRACKING_ROOT = Path("/home/sqw/VisualSearch/aqa/Tracking")


def infer_move(stem: str) -> str:
    return stem.rsplit("_", 1)[-1].lower()


def infer_student_id(stem: str) -> str:
    return stem.split("_", 1)[0]


def teacher_tracking_path(teacher_video: str, tracking_root: Path) -> Path:
    teacher_stem = Path(teacher_video).stem
    return tracking_root / "teach" / teacher_stem / "results" / f"demo_{teacher_stem}.pkl"


def load_keyframe_rows(path: Path) -> list[dict[str, int]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [
            {
                "anchor_order": int(row["anchor_order"]),
                "teacher_source_frame_0based": int(row["teacher_source_frame_0based"]),
                "student_source_frame_0based": int(row["student_source_frame_0based"]),
            }
            for row in csv.DictReader(handle)
        ]


def make_subject_metrics(
    *,
    subject: str,
    move: str,
    student_id: str,
    clip: str,
    tracking_path: Path,
    source_frames: list[int],
    anchor_orders: list[int],
    smpl_model_path: Path,
    smpl_batch_size: int,
    device: str,
) -> pd.DataFrame:
    track = load_primary_smpl_track(tracking_path).at_source_frames(source_frames)
    joints = track.to_smpl24_joints(model_path=smpl_model_path, batch_size=smpl_batch_size, device=device)
    metrics = compute_angle_metrics(joints, frame_ids=anchor_orders)
    frame_metadata = pd.DataFrame(
        {
            "frame_id": anchor_orders,
            "source_frame_0based": source_frames,
            "phalp_frame_1based": [frame + 1 for frame in source_frames],
        }
    )
    metrics = metrics.merge(frame_metadata, on="frame_id", how="left")
    metrics = metrics.rename(columns={"frame_id": "anchor_order"})
    metrics.insert(0, "subject", subject)
    metrics.insert(1, "move", move)
    metrics.insert(2, "student_id", student_id)
    metrics.insert(3, "clip", clip)
    return metrics[
        [
            "subject", "move", "student_id", "clip", "anchor_order",
            "source_frame_0based", "phalp_frame_1based",
            "metric_id", "value", "unit", "status",
        ]
    ]


def make_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    key_columns = ["move", "student_id", "clip", "anchor_order", "metric_id", "unit"]
    teacher = metrics[metrics["subject"] == "teacher"][
        key_columns + ["value", "status", "source_frame_0based", "phalp_frame_1based"]
    ].rename(
        columns={
            "value": "teacher_value",
            "status": "teacher_status",
            "source_frame_0based": "teacher_source_frame_0based",
            "phalp_frame_1based": "teacher_phalp_frame_1based",
        }
    )
    student = metrics[metrics["subject"] == "student"][
        key_columns + ["value", "status", "source_frame_0based", "phalp_frame_1based"]
    ].rename(
        columns={
            "value": "student_value",
            "status": "student_status",
            "source_frame_0based": "student_source_frame_0based",
            "phalp_frame_1based": "student_phalp_frame_1based",
        }
    )
    comparison = teacher.merge(student, on=key_columns, how="inner")
    comparison["diff_student_minus_teacher"] = comparison["student_value"] - comparison["teacher_value"]
    comparison["abs_diff"] = comparison["diff_student_minus_teacher"].abs()
    return comparison[
        [
            "move", "student_id", "clip", "anchor_order", "metric_id", "unit",
            "teacher_source_frame_0based", "student_source_frame_0based",
            "teacher_phalp_frame_1based", "student_phalp_frame_1based",
            "teacher_value", "student_value", "diff_student_minus_teacher", "abs_diff",
            "teacher_status", "student_status",
        ]
    ]


def process_result_dir(
    result_dir: Path,
    *,
    output_root: Path,
    tracking_root: Path,
    smpl_model_path: Path,
    smpl_batch_size: int,
    device: str,
) -> tuple[Path, Path]:
    summary_path = result_dir / "summary.json"
    keyframes_path = result_dir / "teacher_anchored_keyframes.csv"
    if not summary_path.is_file() or not keyframes_path.is_file():
        raise FileNotFoundError(f"Missing summary/keyframe files in {result_dir}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = load_keyframe_rows(keyframes_path)
    anchor_orders = [row["anchor_order"] for row in rows]
    teacher_frames = [row["teacher_source_frame_0based"] for row in rows]
    student_frames = [row["student_source_frame_0based"] for row in rows]
    move = infer_move(result_dir.name)
    student_id = infer_student_id(result_dir.name)
    teacher_tracking = teacher_tracking_path(summary["teacher_video"], tracking_root)
    student_tracking = Path(summary["student_tracking"])

    teacher_metrics = make_subject_metrics(
        subject="teacher",
        move=move,
        student_id=student_id,
        clip=result_dir.name,
        tracking_path=teacher_tracking,
        source_frames=teacher_frames,
        anchor_orders=anchor_orders,
        smpl_model_path=smpl_model_path,
        smpl_batch_size=smpl_batch_size,
        device=device,
    )
    student_metrics = make_subject_metrics(
        subject="student",
        move=move,
        student_id=student_id,
        clip=result_dir.name,
        tracking_path=student_tracking,
        source_frames=student_frames,
        anchor_orders=anchor_orders,
        smpl_model_path=smpl_model_path,
        smpl_batch_size=smpl_batch_size,
        device=device,
    )
    metrics = pd.concat([teacher_metrics, student_metrics], ignore_index=True)
    comparison = make_comparison(metrics)

    output_dir = output_root / result_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "angle_metrics_long.csv"
    comparison_path = output_dir / "angle_metric_comparison.csv"
    metrics.to_csv(metrics_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    return metrics_path, comparison_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tracking-root", type=Path, default=DEFAULT_TRACKING_ROOT)
    parser.add_argument("--smpl-model-path", type=Path, default=DEFAULT_SMPL_MODEL_PATH)
    parser.add_argument("--smpl-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clip", action="append", help="Process only this result directory name; repeatable.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result_dirs = sorted(path for path in args.result_root.iterdir() if path.is_dir())
    result_dirs = [path for path in result_dirs if (path / "summary.json").is_file()]
    if args.clip:
        requested = set(args.clip)
        result_dirs = [path for path in result_dirs if path.name in requested]
    if not result_dirs:
        raise ValueError("No teacher-keyframe result directories matched the request.")

    written: list[tuple[Path, Path]] = []
    for result_dir in result_dirs:
        metrics_path, comparison_path = process_result_dir(
            result_dir,
            output_root=args.output_root,
            tracking_root=args.tracking_root,
            smpl_model_path=args.smpl_model_path,
            smpl_batch_size=args.smpl_batch_size,
            device=args.device,
        )
        written.append((metrics_path, comparison_path))
        print(f"{result_dir.name}: wrote {metrics_path} and {comparison_path}")
    print(f"Processed {len(written)} result directories.")


if __name__ == "__main__":
    main()
