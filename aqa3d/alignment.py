"""The existing YOLO-pose + DTW alignment stage, kept separate from 3D scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import argrelextrema


@dataclass(frozen=True)
class SampledVideo:
    frames: list[np.ndarray]
    source_indices: np.ndarray # 下标从 '0' 开始
    source_fps: float


@dataclass(frozen=True)
class KeyframeAlignment:
    student_keyframe_indices: np.ndarray
    teacher_match_indices: np.ndarray
    student_source_indices: np.ndarray
    teacher_source_indices: np.ndarray
    dtw_distance: float


def sample_video(video_path: str | Path, target_fps: int = 30) -> SampledVideo:
    """Sample with the same integer-interval rule used in the legacy AQA code."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    interval = max(int(source_fps / target_fps), 1)
    frames: list[np.ndarray] = []
    indices: list[int] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % interval == 0:
            frames.append(frame)
            indices.append(frame_index)
        frame_index += 1
    capture.release()
    if len(frames) < 3:
        raise ValueError(f"Video has too few frames for alignment: {video_path}")
    return SampledVideo(frames=frames, source_indices=np.asarray(indices), source_fps=source_fps)


def trim_qishi_student_video(video: SampledVideo, seconds: int = 15) -> SampledVideo:
    """Match the old pipeline's qishi rule while retaining original frame numbers."""
    keep = seconds * int(video.source_fps)
    if keep <= 0:
        raise ValueError(f"Invalid source FPS for qishi trimming: {video.source_fps}")
    return SampledVideo(
        frames=video.frames[-keep:],
        source_indices=video.source_indices[-keep:],
        source_fps=video.source_fps,
    )


def body_vectors(model, frames: list[np.ndarray], batch_size: int = 8) -> np.ndarray:
    """Construct the legacy 13 two-dimensional body vectors from YOLO keypoints."""
    if batch_size <= 0:
        raise ValueError("YOLO batch_size must be positive.")
    vectors: list[np.ndarray] = []
    # Explicitly chunk the Python list. Recent Ultralytics releases ignore their
    # ``batch`` option for this source type and otherwise combine every frame.
    for chunk_start in range(0, len(frames), batch_size):
        chunk = frames[chunk_start : chunk_start + batch_size]
        for offset, result in enumerate(model(chunk, stream=True, verbose=False)):
            index = chunk_start + offset
            if result.keypoints is None or result.keypoints.xy is None or len(result.keypoints.xy) == 0:
                raise ValueError(f"YOLO did not detect a person in sampled frame {index}.")
            points = result.keypoints.xy[0].detach().cpu().numpy().astype(np.float64)
            if points.shape != (17, 2):
                raise ValueError(f"Expected 17 COCO keypoints in frame {index}, got {points.shape}.")
            chest = 0.5 * (points[5] + points[6])
            vectors.append(
                np.stack(
                    (
                        chest - points[0], points[5] - points[6], points[8] - points[6],
                        points[10] - points[8], points[7] - points[5], points[9] - points[7],
                        points[12] - points[6], points[11] - points[12], points[11] - points[5],
                        points[14] - points[12], points[16] - points[14], points[13] - points[11],
                        points[15] - points[13],
                    ),
                    axis=0,
                )
            )
    return np.stack(vectors, axis=0)


def extract_keyframes(
    frames: list[np.ndarray], smooth_kernel_size: int = 25, order: int = 15, fps: int = 30
) -> np.ndarray:
    """Port of the project's frame-difference keyframe selection."""
    if smooth_kernel_size < 3 or smooth_kernel_size % 2 == 0:
        raise ValueError("smooth_kernel_size must be an odd integer of at least 3.")
    if len(frames) < smooth_kernel_size + 2:
        raise ValueError("Video segment is too short for the requested smoothing kernel.")

    previous = cv2.cvtColor(frames[0], cv2.COLOR_BGR2LUV).astype(np.int32)
    differences = []
    for frame in frames[1:]:
        current = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV).astype(np.int32)
        differences.append(np.abs(current - previous).mean())
        previous = current
    diff_values = np.asarray(differences, dtype=np.float64)
    padding = smooth_kernel_size // 2
    smooth = np.convolve(
        np.pad(diff_values, (padding, padding), mode="reflect"),
        np.ones(smooth_kernel_size, dtype=np.float64) / smooth_kernel_size,
        mode="valid",
    )
    keyframes = argrelextrema(smooth, np.greater, order=order)[0]
    last = len(frames) - 2
    if keyframes.size == 0:
        keyframes = np.asarray([last], dtype=np.int64)
    elif last not in keyframes:
        if last - keyframes[-1] < fps:
            keyframes[-1] = last
        else:
            keyframes = np.append(keyframes, last) # 保证在视频结尾'1s'内采样一个关键帧
    if keyframes[0] + 1 > fps:
        keyframes = np.concatenate((np.asarray([-1], dtype=np.int64), keyframes))
    return keyframes + 1


def _pairwise_vector_distance(student: np.ndarray, teacher: np.ndarray) -> np.ndarray:
    return np.linalg.norm(student[:, None] - teacher[None, :], axis=-1).mean(axis=-1)


def dtw_alignment(student: np.ndarray, teacher: np.ndarray, coefficient: float = 1.0) -> tuple[float, dict[int, list[int]], np.ndarray]:
    """Dynamic time warping matching the recurrence used in the original project."""
    if coefficient <= 0:
        raise ValueError("DTW coefficient must be positive.")
    accumulated = _pairwise_vector_distance(student, teacher)
    rows, cols = accumulated.shape
    accumulated[0, 1:] = np.cumsum(accumulated[0, 1:]) + accumulated[0, 0]
    accumulated[1:, 0] = np.cumsum(accumulated[1:, 0]) + accumulated[0, 0]
    accumulated[0, 0] *= coefficient
    for row in range(1, rows):
        for col in range(1, cols):
            accumulated[row, col] = min(
                accumulated[row, col] + accumulated[row - 1, col],
                accumulated[row, col] + accumulated[row, col - 1],
                coefficient * accumulated[row, col] + accumulated[row - 1, col - 1],
            )

    # 回溯最佳路径
    row, col = rows - 1, cols - 1
    path = [(row, col)]
    while row > 0 or col > 0:
        if row == 0:
            col -= 1
        elif col == 0:
            row -= 1
        else:
            # Prefer diagonal moves for exact ties, as conventional DTW does.
            choice = int(np.argmin((accumulated[row - 1, col - 1], accumulated[row - 1, col], accumulated[row, col - 1])))
            if choice == 0:
                row, col = row - 1, col - 1
            elif choice == 1:
                row -= 1
            else:
                col -= 1
        path.append((row, col))
    path.reverse()
    matching: dict[int, list[int]] = {}
    for student_index, teacher_index in path:
        matching.setdefault(student_index, []).append(teacher_index)
    return float(accumulated[-1, -1] / len(path)), matching, accumulated


def align_keyframes(
    student_video: SampledVideo,
    teacher_video: SampledVideo,
    model,
    *,
    smooth_kernel_size: int = 25,
    keyframe_order: int = 15,
    dtw_coefficient: float = 1.0,
    yolo_batch_size: int = 8,
) -> KeyframeAlignment:
    # 构建肢体段向量
    student_vectors = body_vectors(model, student_video.frames, yolo_batch_size)
    teacher_vectors = body_vectors(model, teacher_video.frames, yolo_batch_size)

    # 计算DTW距离和匹配
    distance, matching, costs = dtw_alignment(student_vectors, teacher_vectors, dtw_coefficient)
    keyframes = extract_keyframes(student_video.frames, smooth_kernel_size, keyframe_order)
    teacher_matches = np.asarray(
        [min(matching[int(index)], key=lambda candidate: costs[int(index), candidate]) for index in keyframes],
        dtype=np.int64,
    )
    return KeyframeAlignment(
        student_keyframe_indices=keyframes,
        teacher_match_indices=teacher_matches,
        student_source_indices=student_video.source_indices[keyframes],
        teacher_source_indices=teacher_video.source_indices[teacher_matches],
        dtw_distance=distance,
    )
