"""Reference-manifest utilities for one endpoint KeyPose per Tai Chi form."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np

from .smpl_dtw import VideoSampling
from .smpl_pose_segmentation import GoldBoundary
from .tracking import TrackPoseSequence


MOVE_NAMES_PINYIN = (
    "qishi",
    "yemafenzong",
    "baiheliangchi",
    "louxiaobu",
    "shouhuipipa",
    "daojuangong",
    "zuolanquewei",
    "youlanquewei",
    "danbian",
    "yunshou",
    "danbian_2",
    "gaotanma",
    "youdengtui",
    "shuangfengguaner",
    "zhuanshenzuodengtui",
    "zuoxiashiduli",
    "youxiashiduli",
    "zuoyouchuansuo",
    "haidizhen",
    "shantongbi",
    "zhuanshenbanlanchui",
    "rufengsibi",
    "shizishou",
    "shoushi",
)


@dataclass(frozen=True)
class EndpointKeypose:
    """One Gold reference event located at a form-end boundary."""

    reference_video_id: str
    reference_sample_sequence_id: str
    move_id: int
    move_name_pinyin: str
    move_name_zh: str
    pose_id: str
    keypose_role: str
    boundary_time_seconds: float
    sample_index_0based: int
    sample_frame_1based: int
    source_frame_index_0based: int
    source_frame_1based: int
    phalp_frame_1based: int
    source_frame_time_seconds: float
    segment_sample_start_index_0based: int
    segment_sample_end_index_0based: int
    segment_source_start_index_0based: int
    segment_source_end_index_0based: int
    final_technique_step: str
    boundary_annotation_status: str = "gold"
    technique_binding_status: str = "provisional_last_step"
    tracking_pose_available: bool = False
    tracking_source_track_id: int | None = None


MOVE_HEADING_PATTERN = re.compile(r"^##\s+第(?P<move_id>\d+)式\s+(?P<name>.+?)\s*$")
NUMBERED_STEP_PATTERN = re.compile(r"^\s*\d+\.\s*(?P<text>.*\S)?\s*$")


def load_final_technique_steps(path: str | Path) -> dict[int, str]:
    """Extract the final numbered item under each move's action-principle section."""
    markdown_path = Path(path).expanduser()
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    steps_by_move: dict[int, list[str]] = {}
    current_move: int | None = None
    in_action_principles = False
    current_step: list[str] | None = None

    def finish_step() -> None:
        nonlocal current_step
        if current_move is not None and current_step:
            text = " ".join(part.strip() for part in current_step if part.strip()).strip()
            if text:
                steps_by_move.setdefault(current_move, []).append(text)
        current_step = None

    for line in lines:
        move_match = MOVE_HEADING_PATTERN.match(line)
        if move_match:
            finish_step()
            current_move = int(move_match.group("move_id"))
            in_action_principles = False
            continue
        if current_move is None:
            continue
        if line.strip() == "### 动作要领":
            finish_step()
            in_action_principles = True
            continue
        if line.startswith("### "):
            finish_step()
            in_action_principles = False
            continue
        if not in_action_principles:
            continue

        step_match = NUMBERED_STEP_PATTERN.match(line)
        if step_match:
            finish_step()
            current_step = [step_match.group("text") or ""]
        elif current_step is not None and line.strip():
            current_step.append(line.strip())
    finish_step()

    expected_ids = list(range(1, 25))
    actual_ids = sorted(steps_by_move)
    if actual_ids != expected_ids:
        raise ValueError(
            f"Expected action principles for moves 1..24 in {markdown_path}, got {actual_ids}."
        )
    return {move_id: steps_by_move[move_id][-1] for move_id in expected_ids}


def build_endpoint_keyposes(
    boundaries: Iterable[GoldBoundary],
    sampling: VideoSampling,
    final_technique_steps: dict[int, str],
    *,
    reference_video_id: str,
    reference_sample_sequence_id: str,
) -> list[EndpointKeypose]:
    """Combine Gold boundaries, video indices and provisional technique bindings."""
    items = list(boundaries)
    move_ids = [item.move_id for item in items]
    if move_ids != list(range(1, 25)):
        raise ValueError(f"Expected ordered move IDs 1..24, got {move_ids}.")
    if sorted(final_technique_steps) != list(range(1, 25)):
        raise ValueError("Final technique steps must contain exactly move IDs 1..24.")
    if not reference_video_id or not reference_sample_sequence_id:
        raise ValueError("Reference video and sampled-sequence IDs must be non-empty.")

    keyposes: list[EndpointKeypose] = []
    previous_sample_index = -1
    previous_source_index = -1
    for boundary, move_name_pinyin in zip(items, MOVE_NAMES_PINYIN):
        sample_index = int(boundary.sample_index_0based)
        if not 0 <= sample_index < sampling.sample_count:
            raise IndexError(
                f"Move {boundary.move_id} boundary sample {sample_index} is outside "
                f"0..{sampling.sample_count - 1}."
            )
        expected_time = sample_index / sampling.sample_fps
        if not np.isclose(expected_time, boundary.end_time_seconds, atol=1e-9):
            raise ValueError(
                f"Move {boundary.move_id} boundary time {boundary.end_time_seconds} does not "
                f"match sample index {sample_index} at {sampling.sample_fps} FPS."
            )
        source_index = int(sampling.source_indices[sample_index])
        if sample_index <= previous_sample_index or source_index <= previous_source_index:
            raise ValueError("Boundary sample and source indices must be strictly increasing.")
        keyposes.append(
            EndpointKeypose(
                reference_video_id=reference_video_id,
                reference_sample_sequence_id=reference_sample_sequence_id,
                move_id=boundary.move_id,
                move_name_pinyin=move_name_pinyin,
                move_name_zh=boundary.move_name,
                pose_id=f"{boundary.move_id}.end",
                keypose_role="form_end_boundary_event",
                boundary_time_seconds=float(boundary.end_time_seconds),
                sample_index_0based=sample_index,
                sample_frame_1based=sample_index + 1,
                source_frame_index_0based=source_index,
                source_frame_1based=source_index + 1,
                phalp_frame_1based=source_index + 1,
                source_frame_time_seconds=source_index / sampling.source_fps,
                segment_sample_start_index_0based=previous_sample_index + 1,
                segment_sample_end_index_0based=sample_index,
                segment_source_start_index_0based=previous_source_index + 1,
                segment_source_end_index_0based=source_index,
                final_technique_step=final_technique_steps[boundary.move_id],
            )
        )
        previous_sample_index = sample_index
        previous_source_index = source_index
    return keyposes


def attach_tracking_availability(
    keyposes: Iterable[EndpointKeypose],
    track: TrackPoseSequence,
) -> list[EndpointKeypose]:
    """Record whether each exact boundary event has a stitched SMPL pose."""
    track_lookup: dict[int, int | None] = {}
    source_ids = track.source_track_ids
    for index, phalp_frame in enumerate(track.frame_numbers):
        source_track_id = int(source_ids[index]) if source_ids is not None else int(track.track_id)
        track_lookup[int(phalp_frame)] = source_track_id
    return [
        replace(
            keypose,
            tracking_pose_available=keypose.phalp_frame_1based in track_lookup,
            tracking_source_track_id=track_lookup.get(keypose.phalp_frame_1based),
        )
        for keypose in keyposes
    ]


def endpoint_keypose_rows(keyposes: Iterable[EndpointKeypose]) -> list[dict]:
    """Return serialization-ready rows in the dataclass field order."""
    return [asdict(item) for item in keyposes]


def build_reference_manifest(
    *,
    keyposes: Iterable[EndpointKeypose],
    sampling: VideoSampling,
    reference_video_id: str,
    reference_sample_sequence_id: str,
    reference_video_path: Path,
    reference_tracking_path: Path,
    gold_boundary_path: Path,
    technique_path: Path,
    output_dir: Path,
    metric_window_seconds: float,
    tracking_used_track_ids: Iterable[int],
) -> dict:
    """Build the canonical machine-readable contract for the reference sequence."""
    rows = endpoint_keypose_rows(keyposes)
    if len(rows) != 24:
        raise ValueError(f"Expected 24 endpoint KeyPoses, got {len(rows)}.")
    if metric_window_seconds < 0:
        raise ValueError("metric_window_seconds must be non-negative.")
    tracking_available_count = sum(bool(row["tracking_pose_available"]) for row in rows)
    last_endpoint = rows[-1]
    remaining_sample_count = (
        sampling.sample_count - int(last_endpoint["sample_index_0based"]) - 1
    )
    remaining_source_frame_count = (
        sampling.source_frame_count
        - int(last_endpoint["source_frame_index_0based"])
        - 1
    )
    return {
        "schema_version": "1.0",
        "artifact_type": "tai_chi_24_form_endpoint_keypose_reference",
        "reference": {
            "video_id": reference_video_id,
            "sample_sequence_id": reference_sample_sequence_id,
            "video_path": str(reference_video_path),
            "tracking_path": str(reference_tracking_path),
            "gold_boundary_path": str(gold_boundary_path),
            "technique_path": str(technique_path),
            "source_fps": sampling.source_fps,
            "source_frame_count": sampling.source_frame_count,
            "source_duration_seconds": sampling.source_frame_count / sampling.source_fps,
            "sample_fps": sampling.sample_fps,
            "sample_count": sampling.sample_count,
            "time_origin": "trimmed_video_start",
            "tail_after_last_endpoint": {
                "sample_count": remaining_sample_count,
                "source_frame_count": remaining_source_frame_count,
                "source_duration_seconds": remaining_source_frame_count / sampling.source_fps,
            },
        },
        "move_registry": {
            "count": 24,
            "join_key": "move_id",
            "program_name_field": "move_name_pinyin",
            "display_name_field": "move_name_zh",
        },
        "frame_index_conventions": {
            "source_frame_index_0based": "OpenCV source-video frame index; canonical internal index.",
            "source_frame_1based": "source_frame_index_0based + 1; display/export only.",
            "sample_index_0based": "Uniform 5 FPS time-point index beginning at t=0.",
            "sample_frame_1based": "sample_index_0based + 1; display/export only.",
            "phalp_frame_1based": "PHALP image-frame key; source_frame_index_0based + 1.",
            "boundary_event_membership": "The endpoint event belongs to the form it completes.",
            "sampled_segment_interval": "Closed [previous_boundary + 1, current_boundary]; move 1 starts at 0.",
            "source_segment_interval": "Closed [previous_boundary_source + 1, current_boundary_source]; move 1 starts at 0.",
        },
        "keypose_policy": {
            "name": "one_form_end_boundary_event",
            "count": 24,
            "pose_id_format": "{move_id}.end",
            "technique_binding": "Last numbered action principle is a provisional endpoint description.",
            "visualization_frame": "Exact Gold boundary source frame.",
            "metric_window_seconds_each_side": metric_window_seconds,
            "metric_window_aggregation": "Per-metric median without best-frame reselection.",
            "scope": "Endpoint posture only; temporal technique requirements remain out of scope.",
        },
        "tracking_validation": {
            "loader": "load_stitched_primary_track",
            "used_track_ids": sorted(int(value) for value in tracking_used_track_ids),
            "exact_endpoint_pose_count": tracking_available_count,
            "missing_endpoint_pose_count": len(rows) - tracking_available_count,
        },
        "artifacts": {
            "endpoint_keyposes_csv": str(output_dir / "endpoint_keyposes.csv"),
            "endpoint_keyposes_json": str(output_dir / "endpoint_keyposes.json"),
            "overview_image": str(output_dir / "endpoint_keypose_overview.jpg"),
        },
        "endpoint_keyposes": rows,
    }
