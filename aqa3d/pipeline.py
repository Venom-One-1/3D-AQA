"""End-to-end 2D alignment followed by local-SMPL geodesic scoring."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .alignment import KeyframeAlignment, align_keyframes, sample_video, trim_qishi_student_video
from .geodesic import geodesic_distance
from .tracking import load_primary_track


@dataclass(frozen=True)
class PairResult:
    alignment: KeyframeAlignment
    errors_radians: np.ndarray
    distance_radians: float
    student_track_id: int
    teacher_track_id: int


def run_pair(
    student_video_path: str | Path,
    teacher_video_path: str | Path,
    student_tracking_path: str | Path,
    teacher_tracking_path: str | Path,
    model,
    *,
    move: str,
    smooth_kernel_size: int = 25,
    keyframe_order: int = 15,
    dtw_coefficient: float = 1.0,
    yolo_batch_size: int = 8,
) -> PairResult:
    student_video = sample_video(student_video_path)
    if move.lower() == "qishi":
        student_video = trim_qishi_student_video(student_video)
    teacher_video = sample_video(teacher_video_path)
    alignment = align_keyframes(
        student_video,
        teacher_video,
        model,
        smooth_kernel_size=smooth_kernel_size,
        keyframe_order=keyframe_order,
        dtw_coefficient=dtw_coefficient,
        yolo_batch_size=yolo_batch_size,
    )


    # 计算每个关键帧的局部-SMPL关节误差
    # 读取.pkl文件，并且选择出现次数最多的track，并且返回该track的SMPL body_pose
    student_track = load_primary_track(student_tracking_path)
    teacher_track = load_primary_track(teacher_tracking_path)

    # 根据关键帧的索引，获取对应的SMPL body_pose
    student_poses = student_track.at_source_frames(alignment.student_source_indices)
    teacher_poses = teacher_track.at_source_frames(alignment.teacher_source_indices)
    errors, distance = geodesic_distance(student_poses, teacher_poses)
    return PairResult(
        alignment=alignment,
        errors_radians=errors,
        distance_radians=float(distance),
        student_track_id=student_track.track_id,
        teacher_track_id=teacher_track.track_id,
    )


def save_pair_result(
    result: PairResult,
    output_dir: str | Path,
    *,
    student_video_path: str | Path,
    teacher_video_path: str | Path,
    student_tracking_path: str | Path,
    teacher_tracking_path: str | Path,
    move: str,
) -> None:
    """Save the raw error map, a per-keyframe table, and a compact summary."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    errors_degrees = np.degrees(result.errors_radians)
    alignment = result.alignment
    np.savez_compressed(
        directory / "geodesic_errors.npz",
        errors_radians=result.errors_radians,
        errors_degrees=errors_degrees,
        student_keyframe_indices=alignment.student_keyframe_indices,
        teacher_match_indices=alignment.teacher_match_indices,
        student_source_frame_indices=alignment.student_source_indices,
        teacher_source_frame_indices=alignment.teacher_source_indices,
    )

    with (directory / "matched_keyframes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "keyframe_order", "student_sample_index", "teacher_sample_index",
                "student_source_frame_0based", "teacher_source_frame_0based",
                "student_phalp_frame_1based", "teacher_phalp_frame_1based",
                "mean_error_radians", "mean_error_degrees",
            ]
        )
        for index, (student_sample, teacher_sample, student_source, teacher_source, row_errors) in enumerate(
            zip(
                alignment.student_keyframe_indices,
                alignment.teacher_match_indices,
                alignment.student_source_indices,
                alignment.teacher_source_indices,
                result.errors_radians,
            )
        ):
            writer.writerow(
                [
                    index, int(student_sample), int(teacher_sample), int(student_source), int(teacher_source),
                    int(student_source) + 1, int(teacher_source) + 1,
                    float(np.mean(row_errors)), float(np.degrees(np.mean(row_errors))),
                ]
            )

    summary = {
        "move": move,
        "student_video": str(student_video_path),
        "teacher_video": str(teacher_video_path),
        "student_tracking": str(student_tracking_path),
        "teacher_tracking": str(teacher_tracking_path),
        "student_track_id": result.student_track_id,
        "teacher_track_id": result.teacher_track_id,
        "keyframe_count": int(result.errors_radians.shape[0]),
        "joint_count": int(result.errors_radians.shape[1]),
        "dtw_distance_2d": result.alignment.dtw_distance,
        "mean_geodesic_distance_radians": result.distance_radians,
        "mean_geodesic_distance_degrees": float(np.degrees(result.distance_radians)),
        "score": None,
        "notes": "Uniform frame and joint weights. SMPL global orientation is excluded.",
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
