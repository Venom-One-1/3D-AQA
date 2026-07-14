"""Reusable SMPL-geodesic dynamic time warping utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoSampling:
    """Uniform sampled-frame metadata without decoding the video frames."""

    video_path: Path
    source_fps: float
    source_frame_count: int
    sample_fps: float
    source_indices: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(len(self.source_indices))

    @property
    def sample_times(self) -> np.ndarray:
        return np.arange(self.sample_count, dtype=np.float64) / self.sample_fps


@dataclass(frozen=True)
class ReferenceFrameMatch:
    """One reference sample frame mapped to its best DTW-path target frame."""

    reference_index: int
    target_index: int
    candidate_count: int
    local_cost: float


def uniform_sample_source_indices(
    source_frame_count: int,
    source_fps: float,
    sample_fps: float,
) -> np.ndarray:
    """Return source-frame indices for a uniform sequence starting at time zero.

    The sample count is the nearest integer to ``duration * sample_fps``. This
    matches the project's interval interpretation: a 311-second video has
    exactly 1555 samples at 5 FPS, numbered 1 through 1555.
    """
    if source_frame_count <= 0:
        raise ValueError(f"source_frame_count must be positive, got {source_frame_count}.")
    if source_fps <= 0 or sample_fps <= 0:
        raise ValueError(f"FPS values must be positive, got source={source_fps}, sample={sample_fps}.")

    duration = source_frame_count / source_fps
    sample_count = max(int(np.floor(duration * sample_fps + 0.5)), 1)
    indices = np.floor(np.arange(sample_count, dtype=np.float64) * source_fps / sample_fps).astype(np.int64)
    indices = np.clip(indices, 0, source_frame_count - 1)
    if np.any(np.diff(indices) <= 0):
        raise ValueError(
            "The requested sample FPS produces duplicate source frames; "
            f"source_fps={source_fps}, sample_fps={sample_fps}."
        )
    return indices


def inspect_video_sampling(video_path: str | Path, sample_fps: float = 5.0) -> VideoSampling:
    """Read video timing metadata and construct a uniform sampled sequence."""
    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    source_indices = uniform_sample_source_indices(source_frame_count, source_fps, sample_fps)
    return VideoSampling(
        video_path=path,
        source_fps=source_fps,
        source_frame_count=source_frame_count,
        sample_fps=sample_fps,
        source_indices=source_indices,
    )


def pairwise_geodesic_costs(
    target_poses: np.ndarray,
    reference_poses: np.ndarray,
    *,
    chunk_size: int = 32,
) -> np.ndarray:
    """Return frame-pair mean geodesic distances over 23 local SMPL joints."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    target = np.asarray(target_poses, dtype=np.float64)
    reference = np.asarray(reference_poses, dtype=np.float64)
    if target.ndim != 4 or target.shape[1:] != (23, 3, 3):
        raise ValueError(f"Expected target poses shaped (T, 23, 3, 3), got {target.shape}.")
    if reference.ndim != 4 or reference.shape[1:] != (23, 3, 3):
        raise ValueError(f"Expected reference poses shaped (R, 23, 3, 3), got {reference.shape}.")

    costs = np.empty((target.shape[0], reference.shape[0]), dtype=np.float64)
    reference_t = np.swapaxes(reference, -1, -2)
    for start in range(0, target.shape[0], chunk_size):
        end = min(start + chunk_size, target.shape[0])
        relative = target[start:end, None] @ reference_t[None]
        cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0
        costs[start:end] = np.arccos(np.clip(cosine, -1.0, 1.0)).mean(axis=-1)
    return costs


def backtrack_dtw_path(accumulated_costs: np.ndarray) -> np.ndarray:
    """Backtrack a complete ``(target_index, reference_index)`` DTW path."""
    accumulated = np.asarray(accumulated_costs)
    if accumulated.ndim != 2 or min(accumulated.shape) == 0:
        raise ValueError(f"accumulated_costs must be a non-empty 2D matrix, got {accumulated.shape}.")

    row, col = accumulated.shape[0] - 1, accumulated.shape[1] - 1
    path = [(row, col)]
    while row > 0 or col > 0:
        if row == 0:
            col -= 1
        elif col == 0:
            row -= 1
        else:
            choice = int(
                np.argmin(
                    (
                        accumulated[row - 1, col - 1],
                        accumulated[row - 1, col],
                        accumulated[row, col - 1],
                    )
                )
            )
            if choice == 0:
                row, col = row - 1, col - 1
            elif choice == 1:
                row -= 1
            else:
                col -= 1
        path.append((row, col))
    path.reverse()
    return np.asarray(path, dtype=np.int64)


def dtw_from_cost_matrix(
    local_costs: np.ndarray,
    coefficient: float = 1.0,
) -> tuple[float, dict[int, list[int]], np.ndarray]:
    """Run the same DTW recurrence used by ``aqa3d.alignment``."""
    if coefficient <= 0:
        raise ValueError("DTW coefficient must be positive.")
    accumulated = np.asarray(local_costs, dtype=np.float64).copy()
    if accumulated.ndim != 2 or min(accumulated.shape) == 0:
        raise ValueError(f"local_costs must be a non-empty 2D matrix, got {accumulated.shape}.")

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

    path = backtrack_dtw_path(accumulated)
    matching: dict[int, list[int]] = {}
    for target_index, reference_index in path:
        matching.setdefault(int(target_index), []).append(int(reference_index))
    return float(accumulated[-1, -1] / len(path)), matching, accumulated


def select_reference_frame_matches(
    local_costs: np.ndarray,
    path: np.ndarray,
    reference_indices: Iterable[int],
) -> list[ReferenceFrameMatch]:
    """Select the minimum-local-cost target candidate for each reference frame."""
    costs = np.asarray(local_costs, dtype=np.float64)
    path_array = np.asarray(path, dtype=np.int64)
    if path_array.ndim != 2 or path_array.shape[1] != 2:
        raise ValueError(f"Expected path shaped (P, 2), got {path_array.shape}.")

    matches: list[ReferenceFrameMatch] = []
    for reference_index_value in reference_indices:
        reference_index = int(reference_index_value)
        candidates = path_array[path_array[:, 1] == reference_index, 0]
        if len(candidates) == 0:
            raise RuntimeError(f"DTW path does not contain reference frame {reference_index}.")
        candidate_costs = costs[candidates, reference_index]
        best = int(np.argmin(candidate_costs))
        matches.append(
            ReferenceFrameMatch(
                reference_index=reference_index,
                target_index=int(candidates[best]),
                candidate_count=int(len(candidates)),
                local_cost=float(candidate_costs[best]),
            )
        )
    return matches


def require_strictly_increasing_boundaries(matches: Iterable[ReferenceFrameMatch]) -> None:
    """Reject duplicate or reversed mapped boundaries without silently changing them."""
    values = list(matches)
    for previous, current in zip(values, values[1:]):
        if current.target_index <= previous.target_index:
            raise ValueError(
                "Mapped target boundaries are not strictly increasing: "
                f"reference {previous.reference_index}->{previous.target_index}, "
                f"reference {current.reference_index}->{current.target_index}."
            )
