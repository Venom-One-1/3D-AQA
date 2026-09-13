"""Utilities for aligning sparse KeyPose images to a tracked reference video."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .smpl_dtw import (
    ReferenceFrameMatch,
    backtrack_dtw_path,
    dtw_from_cost_matrix,
    pairwise_geodesic_costs,
    select_reference_frame_matches,
)


MOVE_DIR_PATTERN = re.compile(r"^(?P<move_id>\d+)_(?P<move_name>.+)$")


@dataclass(frozen=True)
class KeyPoseImage:
    move_id: int
    move_name: str
    keypose_order: int
    source_frame_number: int
    image_path: Path
    relative_path: Path


@dataclass(frozen=True)
class MoveBoundary:
    move_id: int
    move_name: str
    trimmed_start_time: float
    trimmed_end_time: float
    original_start_time: float
    original_end_time: float


@dataclass(frozen=True)
class KeyPoseAlignment:
    local_costs: np.ndarray
    accumulated_costs: np.ndarray
    path: np.ndarray
    matches: tuple[ReferenceFrameMatch, ...]


def discover_keypose_images(
    root: str | Path,
    *,
    expected_move_count: int | None = 24,
) -> list[KeyPoseImage]:
    """Discover KeyPose JPG files in move and source-frame order."""
    keypose_root = Path(root).expanduser().resolve()
    if not keypose_root.is_dir():
        raise FileNotFoundError(f"KeyPose root does not exist: {keypose_root}")

    move_dirs: list[tuple[int, str, Path]] = []
    for path in keypose_root.iterdir():
        if not path.is_dir():
            continue
        match = MOVE_DIR_PATTERN.match(path.name)
        if match:
            move_dirs.append(
                (int(match.group("move_id")), match.group("move_name"), path)
            )
    move_dirs.sort(key=lambda item: item[0])

    move_ids = [item[0] for item in move_dirs]
    if expected_move_count is not None:
        expected = list(range(1, expected_move_count + 1))
        if move_ids != expected:
            raise ValueError(
                f"Expected KeyPose move directories {expected}, got {move_ids}."
            )

    records: list[KeyPoseImage] = []
    for move_id, move_name, move_dir in move_dirs:
        image_paths = sorted(
            move_dir.glob("*.jpg"),
            key=lambda path: int(path.stem),
        )
        if not image_paths:
            raise ValueError(f"No JPG KeyPose images found in {move_dir}.")
        for order, image_path in enumerate(image_paths, start=1):
            try:
                source_frame_number = int(image_path.stem)
            except ValueError as error:
                raise ValueError(
                    f"KeyPose filename must be a numeric source frame: {image_path}"
                ) from error
            records.append(
                KeyPoseImage(
                    move_id=move_id,
                    move_name=move_name,
                    keypose_order=order,
                    source_frame_number=source_frame_number,
                    image_path=image_path.resolve(),
                    relative_path=image_path.resolve().relative_to(keypose_root),
                )
            )
    return records


def load_move_boundaries(
    path: str | Path,
    video_id: str,
) -> list[MoveBoundary]:
    """Load the 24 ground-truth intervals exported by the TAS workflow."""
    csv_path = Path(path).expanduser()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("video_id") == video_id
        ]
    if len(rows) != 24:
        raise ValueError(
            f"Expected 24 ground-truth rows for {video_id} in {csv_path}, got {len(rows)}."
        )

    boundaries = [
        MoveBoundary(
            move_id=int(row["move_id"]),
            move_name=row["move_name"],
            trimmed_start_time=float(row["ground_truth_start_time"]),
            trimmed_end_time=float(row["ground_truth_end_time"]),
            original_start_time=float(row["source_annotated_first_active_time"]),
            original_end_time=float(row["source_boundary_end_time"]),
        )
        for row in rows
    ]
    boundaries.sort(key=lambda item: item.move_id)
    if [item.move_id for item in boundaries] != list(range(1, 25)):
        raise ValueError("Ground-truth move IDs must be exactly 1 through 24.")
    return boundaries


def time_interval_to_inclusive_frames(
    start_time: float,
    end_time: float,
    fps: float,
    frame_count: int,
) -> tuple[int, int]:
    """Convert a continuous interval to clipped, inclusive zero-based frames."""
    if fps <= 0 or frame_count <= 0:
        raise ValueError("FPS and frame count must be positive.")
    if end_time < start_time:
        raise ValueError(
            f"End time {end_time} precedes start time {start_time}."
        )
    start = int(round(start_time * fps))
    end = int(round(end_time * fps))
    start = int(np.clip(start, 0, frame_count - 1))
    end = int(np.clip(end, 0, frame_count - 1))
    if end < start:
        raise ValueError(f"Clipped frame interval is empty: {start}..{end}.")
    return start, end


def align_keyposes_to_reference_segment(
    reference_segment_poses: np.ndarray,
    keypose_poses: np.ndarray,
    *,
    chunk_size: int = 32,
    coefficient: float = 1.0,
) -> KeyPoseAlignment:
    """Align every sparse KeyPose to a dense reference-video move segment."""
    local_costs = pairwise_geodesic_costs(
        reference_segment_poses,
        keypose_poses,
        chunk_size=chunk_size,
    )
    _, _, accumulated = dtw_from_cost_matrix(local_costs, coefficient=coefficient)
    path = backtrack_dtw_path(accumulated)
    matches = tuple(
        select_reference_frame_matches(
            local_costs,
            path,
            range(len(keypose_poses)),
        )
    )
    target_indices = np.asarray(
        [match.target_index for match in matches],
        dtype=np.int64,
    )
    if np.any(np.diff(target_indices) < 0):
        raise RuntimeError(
            "Minimum-local-cost KeyPose matches are not temporally monotonic."
        )
    return KeyPoseAlignment(
        local_costs=local_costs,
        accumulated_costs=accumulated,
        path=path,
        matches=matches,
    )


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    if seconds < 0:
        raise ValueError("Timestamp cannot be negative.")
    milliseconds = int(round(seconds * 1000.0))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
