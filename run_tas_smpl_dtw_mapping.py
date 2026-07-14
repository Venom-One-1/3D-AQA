#!/usr/bin/env python
"""Map 24-form boundaries to teaching videos with full-sequence SMPL DTW."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aqa3d.smpl_dtw import (
    ReferenceFrameMatch,
    VideoSampling,
    backtrack_dtw_path,
    dtw_from_cost_matrix,
    inspect_video_sampling,
    pairwise_geodesic_costs,
    require_strictly_increasing_boundaries,
    select_reference_frame_matches,
)
from aqa3d.tracking import TrackPoseSequence, load_stitched_primary_track


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO_ROOT = Path("/home/sqw/VisualSearch/aqa/teach_trimmed")
DEFAULT_TRACKING_ROOT = Path("/home/sqw/VisualSearch/aqa/Tracking/teach_trimmed")
DEFAULT_REFERENCE_ID = "QxVvRcRn2TA"
DEFAULT_REFERENCE_SEGMENTS = PROJECT_ROOT / "tas_annotations" / "QxVvRcRn2TA_segments_5fps.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "tas_smpl_dtw_results"


@dataclass(frozen=True)
class ReferenceSegment:
    move_id: int
    move_name: str
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class MappedSegment:
    video_id: str
    move_id: int
    move_name: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    sample_fps: float
    reference_end_frame: int
    boundary_candidate_count: int
    boundary_local_geodesic_radians: float
    boundary_local_geodesic_degrees: float
    boundary_policy: str


def tracking_path(tracking_root: Path, video_id: str) -> Path:
    return tracking_root / video_id / "results" / f"demo_{video_id}.pkl"


def load_reference_segments(path: Path, sample_fps: float) -> list[ReferenceSegment]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 24:
        raise ValueError(f"Expected 24 reference segments in {path}, got {len(rows)}.")

    segments = [
        ReferenceSegment(
            move_id=int(row["move_id"]),
            move_name=row["move_name"],
            start_frame=int(row["start_frame"]),
            end_frame=int(row["end_frame"]),
        )
        for row in rows
    ]
    for expected_move_id, segment in enumerate(segments, start=1):
        if segment.move_id != expected_move_id:
            raise ValueError(f"Expected move_id={expected_move_id}, got {segment.move_id}.")
        expected_start = 1 if expected_move_id == 1 else segments[expected_move_id - 2].end_frame + 1
        if segment.start_frame != expected_start:
            raise ValueError(
                f"Reference move {segment.move_id} starts at {segment.start_frame}; expected {expected_start}."
            )
    annotation_fps = float(rows[0]["sample_fps"])
    if not np.isclose(annotation_fps, sample_fps):
        raise ValueError(f"Annotation FPS is {annotation_fps}, but requested sample FPS is {sample_fps}.")
    return segments


def build_mapped_segments(
    video_id: str,
    reference_segments: list[ReferenceSegment],
    matches: list[ReferenceFrameMatch],
    sample_fps: float,
) -> list[MappedSegment]:
    if len(reference_segments) != len(matches):
        raise ValueError("Each reference segment must have exactly one mapped end boundary.")
    require_strictly_increasing_boundaries(matches)

    mapped: list[MappedSegment] = []
    previous_end = 0
    for reference, match in zip(reference_segments, matches):
        expected_reference_index = reference.end_frame - 1
        if match.reference_index != expected_reference_index:
            raise ValueError(
                f"Move {reference.move_id} expected reference index {expected_reference_index}, "
                f"got {match.reference_index}."
            )
        end_frame = match.target_index + 1
        start_frame = previous_end + 1
        if end_frame < start_frame:
            raise ValueError(f"Mapped move {reference.move_id} has an empty frame interval.")
        mapped.append(
            MappedSegment(
                video_id=video_id,
                move_id=reference.move_id,
                move_name=reference.move_name,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=(start_frame - 1) / sample_fps,
                end_time=end_frame / sample_fps,
                sample_fps=sample_fps,
                reference_end_frame=reference.end_frame,
                boundary_candidate_count=match.candidate_count,
                boundary_local_geodesic_radians=match.local_cost,
                boundary_local_geodesic_degrees=float(np.degrees(match.local_cost)),
                boundary_policy="dtw_mapped_non_overlapping_closed_1based_sampled_frames",
            )
        )
        previous_end = end_frame
    return mapped


def write_dict_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_boundary_mapping(
    path: Path,
    reference_segments: list[ReferenceSegment],
    matches: list[ReferenceFrameMatch],
    reference_sampling: VideoSampling,
    target_sampling: VideoSampling,
) -> None:
    rows: list[dict] = []
    previous_target = -1
    for segment, match in zip(reference_segments, matches):
        target_source_index = int(target_sampling.source_indices[match.target_index])
        reference_source_index = int(reference_sampling.source_indices[match.reference_index])
        rows.append(
            {
                "move_id": segment.move_id,
                "move_name": segment.move_name,
                "reference_end_frame_1based": segment.end_frame,
                "reference_sample_index_0based": match.reference_index,
                "reference_source_frame_0based": reference_source_index,
                "reference_boundary_time_seconds": segment.end_frame / reference_sampling.sample_fps,
                "target_end_frame_1based": match.target_index + 1,
                "target_sample_index_0based": match.target_index,
                "target_source_frame_0based": target_source_index,
                "target_phalp_frame_1based": target_source_index + 1,
                "target_boundary_time_seconds": (match.target_index + 1) / target_sampling.sample_fps,
                "candidate_count": match.candidate_count,
                "selected_local_geodesic_radians": match.local_cost,
                "selected_local_geodesic_degrees": float(np.degrees(match.local_cost)),
                "strictly_after_previous": match.target_index > previous_target,
            }
        )
        previous_target = match.target_index
    write_dict_csv(path, rows)


def save_dtw_path(
    output_dir: Path,
    path: np.ndarray,
    local_costs: np.ndarray,
    reference_sampling: VideoSampling,
    target_sampling: VideoSampling,
) -> None:
    target_indices = path[:, 0]
    reference_indices = path[:, 1]
    path_costs = local_costs[target_indices, reference_indices]
    np.savez_compressed(
        output_dir / "dtw_path.npz",
        target_sample_indices_0based=target_indices.astype(np.int32),
        reference_sample_indices_0based=reference_indices.astype(np.int32),
        target_source_frame_indices_0based=target_sampling.source_indices[target_indices].astype(np.int32),
        reference_source_frame_indices_0based=reference_sampling.source_indices[reference_indices].astype(np.int32),
        local_geodesic_radians=path_costs.astype(np.float32),
        local_geodesic_degrees=np.degrees(path_costs).astype(np.float32),
    )
    with (output_dir / "dtw_path.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "path_order",
                "reference_sample_frame_1based",
                "target_sample_frame_1based",
                "reference_source_frame_0based",
                "target_source_frame_0based",
                "reference_time_seconds",
                "target_time_seconds",
                "local_geodesic_radians",
                "local_geodesic_degrees",
            )
        )
        for order, (target_index, reference_index, cost) in enumerate(
            zip(target_indices, reference_indices, path_costs)
        ):
            writer.writerow(
                (
                    order,
                    int(reference_index) + 1,
                    int(target_index) + 1,
                    int(reference_sampling.source_indices[reference_index]),
                    int(target_sampling.source_indices[target_index]),
                    float(reference_index / reference_sampling.sample_fps),
                    float(target_index / target_sampling.sample_fps),
                    float(cost),
                    float(np.degrees(cost)),
                )
            )


def plot_diagnostics(
    output_path: Path,
    local_costs: np.ndarray,
    path: np.ndarray,
    reference_segments: list[ReferenceSegment],
    matches: list[ReferenceFrameMatch],
    reference_sampling: VideoSampling,
    target_sampling: VideoSampling,
    target_video_id: str,
) -> None:
    reference_times = path[:, 1] / reference_sampling.sample_fps
    target_times = path[:, 0] / target_sampling.sample_fps
    boundary_reference_times = np.asarray([segment.end_frame for segment in reference_segments]) / reference_sampling.sample_fps
    boundary_target_times = np.asarray([match.target_index + 1 for match in matches]) / target_sampling.sample_fps
    costs_degrees = np.degrees(local_costs)
    color_max = float(np.percentile(costs_degrees, 95.0))

    figure, axes = plt.subplots(2, 1, figsize=(15, 11), constrained_layout=True)
    image = axes[0].imshow(
        costs_degrees,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(
            0.0,
            reference_sampling.sample_count / reference_sampling.sample_fps,
            0.0,
            target_sampling.sample_count / target_sampling.sample_fps,
        ),
        cmap="viridis",
        vmin=0.0,
        vmax=color_max,
    )
    axes[0].plot(reference_times, target_times, color="white", linewidth=1.2, label="DTW path")
    axes[0].scatter(
        boundary_reference_times,
        boundary_target_times,
        s=28,
        color="#ef4444",
        edgecolor="white",
        linewidth=0.5,
        label="Mapped boundaries",
        zorder=3,
    )
    for move_id, x_value, y_value in zip(range(1, 25), boundary_reference_times, boundary_target_times):
        axes[0].text(x_value, y_value, str(move_id), fontsize=6, color="white", ha="left", va="bottom")
    axes[0].set_title(f"5 FPS local SMPL geodesic costs and global DTW path: {target_video_id}")
    axes[0].set_xlabel("Reference time (s)")
    axes[0].set_ylabel("Target time (s)")
    axes[0].legend(loc="upper left")
    figure.colorbar(image, ax=axes[0], label="Mean local-joint geodesic distance (deg)")

    axes[1].plot(reference_times, target_times, color="#2563eb", linewidth=1.5, label="DTW path")
    axes[1].plot(
        (0.0, reference_sampling.sample_count / reference_sampling.sample_fps),
        (0.0, target_sampling.sample_count / target_sampling.sample_fps),
        color="#6b7280",
        linestyle="--",
        linewidth=1.0,
        label="Uniform-speed baseline",
    )
    axes[1].scatter(boundary_reference_times, boundary_target_times, s=34, color="#dc2626", zorder=3)
    for move_id, x_value, y_value in zip(range(1, 25), boundary_reference_times, boundary_target_times):
        axes[1].annotate(str(move_id), (x_value, y_value), xytext=(3, 3), textcoords="offset points", fontsize=7)
    axes[1].set_title("Temporal alignment and transferred 24-form boundaries")
    axes[1].set_xlabel("Reference time (s)")
    axes[1].set_ylabel("Target time (s)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper left")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def load_sampled_track(
    video_path: Path,
    tracking_file: Path,
    sample_fps: float,
) -> tuple[VideoSampling, TrackPoseSequence, np.ndarray]:
    sampling = inspect_video_sampling(video_path, sample_fps)
    track = load_stitched_primary_track(tracking_file)
    poses = track.at_source_frames(sampling.source_indices)
    return sampling, track, poses


def process_target(
    target_video_id: str,
    *,
    video_root: Path,
    tracking_root: Path,
    output_root: Path,
    reference_video_id: str,
    reference_segments: list[ReferenceSegment],
    reference_sampling: VideoSampling,
    reference_track: TrackPoseSequence,
    reference_poses: np.ndarray,
    sample_fps: float,
    dtw_coefficient: float,
    pairwise_chunk_size: int,
    save_local_costs: bool,
) -> dict:
    started = time.perf_counter()
    target_video = video_root / f"{target_video_id}.mp4"
    target_tracking = tracking_path(tracking_root, target_video_id)
    target_sampling, target_track, target_poses = load_sampled_track(target_video, target_tracking, sample_fps)
    print(
        f"[{target_video_id}] loaded {target_sampling.sample_count} target samples; "
        f"computing {target_sampling.sample_count} x {reference_sampling.sample_count} local costs",
        flush=True,
    )
    local_costs = pairwise_geodesic_costs(target_poses, reference_poses, chunk_size=pairwise_chunk_size)
    dtw_distance, _, accumulated = dtw_from_cost_matrix(local_costs, dtw_coefficient)
    path = backtrack_dtw_path(accumulated)
    reference_boundary_indices = [segment.end_frame - 1 for segment in reference_segments]
    matches = select_reference_frame_matches(local_costs, path, reference_boundary_indices)
    mapped_segments = build_mapped_segments(target_video_id, reference_segments, matches, sample_fps)

    output_dir = output_root / target_video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    segment_rows = [asdict(segment) for segment in mapped_segments]
    write_dict_csv(output_dir / "segments_5fps.csv", segment_rows)
    save_boundary_mapping(
        output_dir / "boundary_mapping.csv",
        reference_segments,
        matches,
        reference_sampling,
        target_sampling,
    )
    save_dtw_path(output_dir, path, local_costs, reference_sampling, target_sampling)
    if save_local_costs:
        np.savez_compressed(output_dir / "local_geodesic_costs.npz", local_costs_radians=local_costs.astype(np.float32))
    plot_diagnostics(
        output_dir / "dtw_diagnostics.png",
        local_costs,
        path,
        reference_segments,
        matches,
        reference_sampling,
        target_sampling,
        target_video_id,
    )

    elapsed = time.perf_counter() - started
    summary = {
        "status": "ok",
        "reference_video_id": reference_video_id,
        "target_video_id": target_video_id,
        "reference_video": str(reference_sampling.video_path),
        "target_video": str(target_video),
        "reference_tracking": str(tracking_path(tracking_root, reference_video_id)),
        "target_tracking": str(target_tracking),
        "reference_track_id": reference_track.track_id,
        "target_track_id": target_track.track_id,
        "reference_used_track_ids": list(reference_track.used_track_ids),
        "target_used_track_ids": list(target_track.used_track_ids),
        "sample_fps": sample_fps,
        "reference_sample_count": reference_sampling.sample_count,
        "target_sample_count": target_sampling.sample_count,
        "dtw_path_length": int(len(path)),
        "dtw_local_cost": "mean geodesic distance over 23 local SMPL body_pose joints",
        "global_orientation_included": False,
        "boundary_candidate_selection": "minimum local geodesic distance among DTW-path candidates",
        "mapped_boundaries_strictly_increasing": True,
        "dtw_distance_radians": dtw_distance,
        "dtw_distance_degrees": float(np.degrees(dtw_distance)),
        "mapped_last_end_frame": mapped_segments[-1].end_frame,
        "unmapped_target_tail_frames": target_sampling.sample_count - mapped_segments[-1].end_frame,
        "elapsed_seconds": elapsed,
    }
    (output_dir / "segments_5fps.json").write_text(
        json.dumps({**summary, "segments": segment_rows}, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"[{target_video_id}] done in {elapsed:.1f}s -> {output_dir}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--tracking-root", type=Path, default=DEFAULT_TRACKING_ROOT)
    parser.add_argument("--reference-video-id", default=DEFAULT_REFERENCE_ID)
    parser.add_argument("--reference-segments", type=Path, default=DEFAULT_REFERENCE_SEGMENTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-video-id", action="append", help="Target video ID; repeatable")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--dtw-coefficient", type=float, default=1.0)
    parser.add_argument("--pairwise-chunk-size", type=int, default=32)
    parser.add_argument("--save-local-costs", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reference_video = args.video_root / f"{args.reference_video_id}.mp4"
    reference_tracking = tracking_path(args.tracking_root, args.reference_video_id)
    reference_segments = load_reference_segments(args.reference_segments, args.sample_fps)
    reference_sampling, reference_track, reference_poses = load_sampled_track(
        reference_video,
        reference_tracking,
        args.sample_fps,
    )
    if reference_segments[-1].end_frame > reference_sampling.sample_count:
        raise ValueError(
            f"Last reference boundary is frame {reference_segments[-1].end_frame}, "
            f"but the sampled reference has only {reference_sampling.sample_count} frames."
        )

    target_ids = args.target_video_id or [
        path.stem for path in sorted(args.video_root.glob("*.mp4")) if path.stem != args.reference_video_id
    ]
    if not target_ids:
        raise ValueError("No target teaching videos were found.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    failures: list[dict] = []
    for target_video_id in target_ids:
        try:
            summaries.append(
                process_target(
                    target_video_id,
                    video_root=args.video_root,
                    tracking_root=args.tracking_root,
                    output_root=args.output_root,
                    reference_video_id=args.reference_video_id,
                    reference_segments=reference_segments,
                    reference_sampling=reference_sampling,
                    reference_track=reference_track,
                    reference_poses=reference_poses,
                    sample_fps=args.sample_fps,
                    dtw_coefficient=args.dtw_coefficient,
                    pairwise_chunk_size=args.pairwise_chunk_size,
                    save_local_costs=args.save_local_costs,
                )
            )
        except Exception as error:
            failure = {"target_video_id": target_video_id, "status": "failed", "error": str(error)}
            failures.append(failure)
            print(f"[{target_video_id}] FAILED: {error}", flush=True)

    batch_payload = {
        "reference_video_id": args.reference_video_id,
        "sample_fps": args.sample_fps,
        "completed": len(summaries),
        "failed": len(failures),
        "results": summaries,
        "failures": failures,
    }
    (args.output_root / "mapping_summary.json").write_text(
        json.dumps(batch_payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    if summaries:
        write_dict_csv(args.output_root / "mapping_summary.csv", summaries)
        all_segments: list[dict] = []
        for summary in summaries:
            segment_path = args.output_root / summary["target_video_id"] / "segments_5fps.csv"
            with segment_path.open("r", encoding="utf-8", newline="") as handle:
                all_segments.extend(csv.DictReader(handle))
        write_dict_csv(args.output_root / "all_mapped_segments_5fps.csv", all_segments)
    if failures:
        raise RuntimeError(f"Boundary mapping failed for {len(failures)} target video(s).")


if __name__ == "__main__":
    main()
