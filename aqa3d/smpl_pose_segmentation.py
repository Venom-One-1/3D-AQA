"""Transfer 24-form gold boundaries through full-sequence SMPL Pose DTW."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np

from .smpl_dtw import ReferenceFrameMatch, VideoSampling, require_strictly_increasing_boundaries


@dataclass(frozen=True)
class GoldBoundary:
    """One reference-video end boundary on the fixed sampled-time grid."""

    move_id: int
    move_name: str
    end_time_seconds: float
    sample_index_0based: int


@dataclass(frozen=True)
class MappedBoundary:
    """One gold boundary mapped to a target sampled frame through DTW."""

    video_id: str
    move_id: int
    move_name: str
    reference_end_time_seconds: float
    reference_sample_index_0based: int
    reference_source_frame_0based: int
    target_end_time_seconds: float
    target_sample_index_0based: int
    target_sample_frame_1based: int
    target_source_frame_0based: int
    target_source_frame_1based: int
    target_phalp_frame_1based: int
    candidate_count: int
    local_geodesic_radians: float
    local_geodesic_degrees: float
    strictly_after_previous: bool
    boundary_policy: str


@dataclass(frozen=True)
class MappedSegment:
    """A non-overlapping target interval bounded by mapped sampled frames."""

    video_id: str
    move_id: int
    move_name: str
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    start_frame_5fps: int
    end_frame_5fps: int
    source_interval_start_time_seconds: float
    source_interval_end_time_exclusive_seconds: float
    source_fps: float
    sample_fps: float
    boundary_policy: str


def load_gold_boundaries(
    path: str | Path,
    sample_fps: float,
    *,
    expected_count: int = 24,
    expected_url_id: str | None = None,
) -> list[GoldBoundary]:
    """Load and validate end boundaries that lie exactly on a sampled grid."""
    boundary_path = Path(path).expanduser()
    if sample_fps <= 0:
        raise ValueError(f"sample_fps must be positive, got {sample_fps}.")
    with boundary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"URLID", "TagID", "Tag", "End"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{boundary_path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if len(rows) != expected_count:
        raise ValueError(
            f"Expected {expected_count} gold boundaries in {boundary_path}, got {len(rows)}."
        )

    fps_decimal = Decimal(str(sample_fps))
    boundaries: list[GoldBoundary] = []
    for expected_move_id, row in enumerate(rows, start=1):
        move_id = int(row["TagID"])
        if move_id != expected_move_id:
            raise ValueError(
                f"Expected TagID={expected_move_id} at row {expected_move_id}, got {move_id}."
            )
        if expected_url_id is not None and row["URLID"] != expected_url_id:
            raise ValueError(
                f"Expected URLID={expected_url_id!r}, got {row['URLID']!r} for move {move_id}."
            )
        end_decimal = Decimal(row["End"])
        grid_position = end_decimal * fps_decimal
        integer_position = grid_position.to_integral_value()
        if grid_position != integer_position:
            raise ValueError(
                f"Move {move_id} boundary {end_decimal}s is not on the {sample_fps} FPS grid."
            )
        boundaries.append(
            GoldBoundary(
                move_id=move_id,
                move_name=row["Tag"],
                end_time_seconds=float(end_decimal),
                sample_index_0based=int(integer_position),
            )
        )

    indices = np.asarray([item.sample_index_0based for item in boundaries])
    if np.any(np.diff(indices) <= 0):
        raise ValueError("Gold boundary sample indices must be strictly increasing.")
    return boundaries


def validate_gold_boundaries_against_sampling(
    boundaries: list[GoldBoundary],
    sampling: VideoSampling,
) -> None:
    """Ensure every gold boundary addresses a valid reference sample."""
    if not boundaries:
        raise ValueError("Gold boundary list cannot be empty.")
    last_index = boundaries[-1].sample_index_0based
    if last_index >= sampling.sample_count:
        raise ValueError(
            f"Last gold boundary uses sample index {last_index}, but reference video "
            f"has only {sampling.sample_count} samples."
        )
    for boundary in boundaries:
        sampled_time = boundary.sample_index_0based / sampling.sample_fps
        if not np.isclose(sampled_time, boundary.end_time_seconds, atol=1e-9):
            raise ValueError(
                f"Move {boundary.move_id} boundary time {boundary.end_time_seconds} "
                f"does not match sampled time {sampled_time}."
            )


def build_mapped_boundaries(
    video_id: str,
    gold_boundaries: list[GoldBoundary],
    matches: list[ReferenceFrameMatch],
    reference_sampling: VideoSampling,
    target_sampling: VideoSampling,
) -> list[MappedBoundary]:
    """Convert selected DTW matches into explicit target boundary coordinates."""
    if len(gold_boundaries) != len(matches):
        raise ValueError("Each gold boundary must have exactly one DTW match.")
    require_strictly_increasing_boundaries(matches)

    mapped: list[MappedBoundary] = []
    previous_target = -1
    for boundary, match in zip(gold_boundaries, matches):
        if match.reference_index != boundary.sample_index_0based:
            raise ValueError(
                f"Move {boundary.move_id} expected reference sample "
                f"{boundary.sample_index_0based}, got {match.reference_index}."
            )
        target_source = int(target_sampling.source_indices[match.target_index])
        reference_source = int(reference_sampling.source_indices[match.reference_index])
        mapped.append(
            MappedBoundary(
                video_id=video_id,
                move_id=boundary.move_id,
                move_name=boundary.move_name,
                reference_end_time_seconds=boundary.end_time_seconds,
                reference_sample_index_0based=match.reference_index,
                reference_source_frame_0based=reference_source,
                target_end_time_seconds=match.target_index / target_sampling.sample_fps,
                target_sample_index_0based=match.target_index,
                target_sample_frame_1based=match.target_index + 1,
                target_source_frame_0based=target_source,
                target_source_frame_1based=target_source + 1,
                target_phalp_frame_1based=target_source + 1,
                candidate_count=match.candidate_count,
                local_geodesic_radians=match.local_cost,
                local_geodesic_degrees=float(np.degrees(match.local_cost)),
                strictly_after_previous=match.target_index > previous_target,
                boundary_policy="minimum_local_geodesic_among_dtw_path_candidates",
            )
        )
        previous_target = match.target_index
    return mapped


def build_mapped_segments(
    video_id: str,
    boundaries: list[MappedBoundary],
    sampling: VideoSampling,
) -> list[MappedSegment]:
    """Build 24 non-overlapping frame intervals from mapped end boundaries."""
    if not boundaries:
        raise ValueError("Mapped boundary list cannot be empty.")
    target_indices = np.asarray([item.target_sample_index_0based for item in boundaries])
    if np.any(np.diff(target_indices) <= 0):
        raise ValueError("Mapped target boundaries must be strictly increasing.")

    segments: list[MappedSegment] = []
    previous_sample_end = -1
    previous_source_end = 0
    previous_boundary_time = 0.0
    for boundary in boundaries:
        start_sample = previous_sample_end + 1
        end_sample = boundary.target_sample_index_0based
        start_source = previous_source_end + 1
        end_source = boundary.target_source_frame_1based
        if start_sample > end_sample or start_source > end_source:
            raise ValueError(f"Mapped move {boundary.move_id} has an empty frame interval.")
        segments.append(
            MappedSegment(
                video_id=video_id,
                move_id=boundary.move_id,
                move_name=boundary.move_name,
                start_time=previous_boundary_time,
                end_time=boundary.target_end_time_seconds,
                start_frame=start_source,
                end_frame=end_source,
                start_frame_5fps=start_sample + 1,
                end_frame_5fps=end_sample + 1,
                source_interval_start_time_seconds=(start_source - 1) / sampling.source_fps,
                source_interval_end_time_exclusive_seconds=end_source / sampling.source_fps,
                source_fps=sampling.source_fps,
                sample_fps=sampling.sample_fps,
                boundary_policy=(
                    "continuous_boundary_times_with_non_overlapping_closed_sampled_"
                    "and_source_frame_intervals"
                ),
            )
        )
        previous_sample_end = end_sample
        previous_source_end = end_source
        previous_boundary_time = boundary.target_end_time_seconds
    return segments
