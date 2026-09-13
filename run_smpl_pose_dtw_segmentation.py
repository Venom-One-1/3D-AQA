#!/usr/bin/env python
"""Segment a trimmed 24-form video using SMPL Pose DTW and gold boundaries."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aqa3d.smpl_dtw import (
    VideoSampling,
    backtrack_dtw_path,
    dtw_from_cost_matrix,
    inspect_video_sampling,
    pairwise_geodesic_costs,
    select_reference_frame_matches,
)
from aqa3d.smpl_pose_segmentation import (
    GoldBoundary,
    MappedBoundary,
    build_mapped_boundaries,
    build_mapped_segments,
    load_gold_boundaries,
    validate_gold_boundaries_against_sampling,
)
from aqa3d.tracking import TrackPoseSequence, load_stitched_primary_track


DEFAULT_REFERENCE_VIDEO = Path(
    "/home/sqw/VisualSearch/aqa/teach_trimmed/BV1WE411W7JB.mp4"
)
DEFAULT_REFERENCE_TRACKING = Path(
    "/home/sqw/VisualSearch/aqa/Tracking/teach_trimmed/"
    "BV1WE411W7JB/results/demo_BV1WE411W7JB.pkl"
)
DEFAULT_GOLD_BOUNDARIES = Path(
    "/home/sqw/VisualSearch/aqa/teach_trimmed/BV1WE411W7JB_5FPS_Boundary_ft.txt"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/sqw/Projects/3D-AQA/smpl_pose_dtw_segmentation_results"
)
DEFAULT_GOLD_URL_ID = "BV1WE411W7JB_5FPS"


def write_dict_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_sampled_track(
    video_path: Path,
    tracking_path: Path,
    sample_fps: float,
    max_tracking_gap_seconds: float,
) -> tuple[VideoSampling, TrackPoseSequence, np.ndarray]:
    """Read video timing and fetch matching local SMPL poses at the sample times."""
    sampling = inspect_video_sampling(video_path, sample_fps)
    track = load_stitched_primary_track(tracking_path)
    max_distance_frames = int(round(max_tracking_gap_seconds * sampling.source_fps))
    poses = track.at_source_frames(
        sampling.source_indices,
        nearest_max_distance_frames=max_distance_frames,
    )
    return sampling, track, poses


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
    target_source = target_sampling.source_indices[target_indices]
    reference_source = reference_sampling.source_indices[reference_indices]

    np.savez_compressed(
        output_dir / "dtw_path.npz",
        target_sample_indices_0based=target_indices.astype(np.int32),
        reference_sample_indices_0based=reference_indices.astype(np.int32),
        target_source_frame_indices_0based=target_source.astype(np.int32),
        reference_source_frame_indices_0based=reference_source.astype(np.int32),
        target_phalp_frames_1based=(target_source + 1).astype(np.int32),
        reference_phalp_frames_1based=(reference_source + 1).astype(np.int32),
        target_times_seconds=(target_indices / target_sampling.sample_fps).astype(np.float64),
        reference_times_seconds=(reference_indices / reference_sampling.sample_fps).astype(np.float64),
        local_geodesic_radians=path_costs.astype(np.float32),
        local_geodesic_degrees=np.degrees(path_costs).astype(np.float32),
    )

    rows = []
    for order, (target_index, reference_index, target_frame, reference_frame, cost) in enumerate(
        zip(target_indices, reference_indices, target_source, reference_source, path_costs)
    ):
        rows.append(
            {
                "path_order": order,
                "reference_sample_index_0based": int(reference_index),
                "target_sample_index_0based": int(target_index),
                "reference_source_frame_0based": int(reference_frame),
                "target_source_frame_0based": int(target_frame),
                "reference_phalp_frame_1based": int(reference_frame) + 1,
                "target_phalp_frame_1based": int(target_frame) + 1,
                "reference_time_seconds": float(reference_index / reference_sampling.sample_fps),
                "target_time_seconds": float(target_index / target_sampling.sample_fps),
                "local_geodesic_radians": float(cost),
                "local_geodesic_degrees": float(np.degrees(cost)),
            }
        )
    write_dict_csv(output_dir / "dtw_path.csv", rows)


def plot_diagnostics(
    output_path: Path,
    video_id: str,
    local_costs: np.ndarray,
    path: np.ndarray,
    gold_boundaries: list[GoldBoundary],
    mapped_boundaries: list[MappedBoundary],
    reference_sampling: VideoSampling,
    target_sampling: VideoSampling,
) -> None:
    reference_path_times = path[:, 1] / reference_sampling.sample_fps
    target_path_times = path[:, 0] / target_sampling.sample_fps
    reference_boundary_times = np.asarray(
        [boundary.end_time_seconds for boundary in gold_boundaries], dtype=np.float64
    )
    target_boundary_times = np.asarray(
        [boundary.target_end_time_seconds for boundary in mapped_boundaries], dtype=np.float64
    )
    costs_degrees = np.degrees(local_costs)
    color_max = max(float(np.percentile(costs_degrees, 95.0)), 1e-6)
    reference_duration = (reference_sampling.sample_count - 1) / reference_sampling.sample_fps
    target_duration = (target_sampling.sample_count - 1) / target_sampling.sample_fps

    figure, axes = plt.subplots(2, 1, figsize=(15, 11), constrained_layout=True)
    image = axes[0].imshow(
        costs_degrees,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(0.0, reference_duration, 0.0, target_duration),
        cmap="viridis",
        vmin=0.0,
        vmax=color_max,
    )
    axes[0].plot(reference_path_times, target_path_times, color="white", linewidth=1.1)
    axes[0].scatter(
        reference_boundary_times,
        target_boundary_times,
        s=28,
        color="#ef4444",
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    for boundary in mapped_boundaries:
        axes[0].annotate(
            str(boundary.move_id),
            (boundary.reference_end_time_seconds, boundary.target_end_time_seconds),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
            color="white",
        )
    axes[0].set_title(f"5 FPS SMPL local-pose costs and global DTW path: {video_id}")
    axes[0].set_xlabel("Gold reference time (s)")
    axes[0].set_ylabel("Input video time (s)")
    figure.colorbar(image, ax=axes[0], label="Mean 23-joint geodesic distance (deg)")

    axes[1].plot(reference_path_times, target_path_times, color="#2563eb", linewidth=1.4, label="DTW path")
    axes[1].plot(
        (0.0, reference_duration),
        (0.0, target_duration),
        color="#6b7280",
        linestyle="--",
        linewidth=1.0,
        label="Uniform progress",
    )
    axes[1].scatter(reference_boundary_times, target_boundary_times, s=34, color="#dc2626", zorder=3)
    for boundary in mapped_boundaries:
        axes[1].annotate(
            str(boundary.move_id),
            (boundary.reference_end_time_seconds, boundary.target_end_time_seconds),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
    axes[1].set_title("Transferred 24-form end boundaries")
    axes[1].set_xlabel("Gold reference time (s)")
    axes[1].set_ylabel("Input video time (s)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper left")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_segmentation(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    input_video = args.input_video.expanduser().absolute()
    input_tracking = args.input_tracking.expanduser().absolute()
    reference_video = args.reference_video.expanduser().absolute()
    reference_tracking = args.reference_tracking.expanduser().absolute()
    gold_file = args.gold_boundaries.expanduser().absolute()
    for path in (input_video, input_tracking, reference_video, reference_tracking, gold_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    video_id = args.video_id or input_video.stem
    if not video_id.strip() or Path(video_id).name != video_id:
        raise ValueError("--video-id must be a non-empty directory-safe name.")

    gold_boundaries = load_gold_boundaries(
        gold_file,
        args.sample_fps,
        expected_url_id=args.gold_url_id,
    )
    reference_sampling, reference_track, reference_poses = load_sampled_track(
        reference_video,
        reference_tracking,
        args.sample_fps,
        args.max_tracking_gap_seconds,
    )
    validate_gold_boundaries_against_sampling(gold_boundaries, reference_sampling)
    target_sampling, target_track, target_poses = load_sampled_track(
        input_video,
        input_tracking,
        args.sample_fps,
        args.max_tracking_gap_seconds,
    )
    print(
        f"[{video_id}] reference={reference_sampling.sample_count}, "
        f"input={target_sampling.sample_count} samples; computing local costs",
        flush=True,
    )
    local_costs = pairwise_geodesic_costs(
        target_poses,
        reference_poses,
        chunk_size=args.pairwise_chunk_size,
    )
    dtw_distance, _, accumulated = dtw_from_cost_matrix(local_costs, args.dtw_coefficient)
    path = backtrack_dtw_path(accumulated)
    reference_indices = [item.sample_index_0based for item in gold_boundaries]
    matches = select_reference_frame_matches(local_costs, path, reference_indices)
    mapped_boundaries = build_mapped_boundaries(
        video_id,
        gold_boundaries,
        matches,
        reference_sampling,
        target_sampling,
    )
    mapped_segments = build_mapped_segments(video_id, mapped_boundaries, target_sampling)

    output_dir = args.output_root.expanduser().absolute() / video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    boundary_rows = [asdict(item) for item in mapped_boundaries]
    segment_rows = [asdict(item) for item in mapped_segments]
    write_dict_csv(output_dir / "boundaries.csv", boundary_rows)
    write_dict_csv(output_dir / "segments.csv", segment_rows)
    save_dtw_path(output_dir, path, local_costs, reference_sampling, target_sampling)
    if args.save_local_costs:
        np.savez_compressed(
            output_dir / "local_geodesic_costs.npz",
            local_costs_radians=local_costs.astype(np.float32),
            local_costs_degrees=np.degrees(local_costs).astype(np.float32),
        )
    plot_diagnostics(
        output_dir / "dtw_diagnostics.png",
        video_id,
        local_costs,
        path,
        gold_boundaries,
        mapped_boundaries,
        reference_sampling,
        target_sampling,
    )

    path_costs = local_costs[path[:, 0], path[:, 1]]
    last_boundary = mapped_boundaries[-1]
    elapsed = time.perf_counter() - started
    summary = {
        "status": "ok",
        "method": "full_sequence_5fps_smpl_local_pose_geodesic_dtw",
        "reference_video_id": reference_video.stem,
        "video_id": video_id,
        "reference_video": str(reference_video),
        "reference_tracking": str(reference_tracking),
        "gold_boundaries": str(gold_file),
        "input_video": str(input_video),
        "input_tracking": str(input_tracking),
        "input_assumed_trimmed_to_24_form": True,
        "sample_fps": args.sample_fps,
        "reference_source_fps": reference_sampling.source_fps,
        "reference_source_frame_count": reference_sampling.source_frame_count,
        "reference_sample_count": reference_sampling.sample_count,
        "input_source_fps": target_sampling.source_fps,
        "input_source_frame_count": target_sampling.source_frame_count,
        "input_sample_count": target_sampling.sample_count,
        "reference_track_id": reference_track.track_id,
        "input_track_id": target_track.track_id,
        "reference_used_track_ids": list(reference_track.used_track_ids),
        "input_used_track_ids": list(target_track.used_track_ids),
        "global_orientation_included": False,
        "local_cost": "mean SO(3) geodesic distance over 23 SMPL body_pose joints",
        "boundary_candidate_selection": "minimum local cost among DTW-path candidates",
        "mapped_boundary_count": len(mapped_boundaries),
        "mapped_boundaries_strictly_increasing": True,
        "segment_frame_indexing": "1-based closed non-overlapping intervals",
        "dtw_coefficient": args.dtw_coefficient,
        "dtw_path_length": int(len(path)),
        "dtw_distance_radians": dtw_distance,
        "dtw_distance_degrees": float(np.degrees(dtw_distance)),
        "dtw_path_mean_geodesic_radians": float(np.mean(path_costs)),
        "dtw_path_mean_geodesic_degrees": float(np.degrees(np.mean(path_costs))),
        "last_boundary_time_seconds": last_boundary.target_end_time_seconds,
        "last_boundary_sample_index_0based": last_boundary.target_sample_index_0based,
        "last_boundary_source_frame_1based": last_boundary.target_source_frame_1based,
        "unmapped_tail_sample_count": (
            target_sampling.sample_count - last_boundary.target_sample_index_0based - 1
        ),
        "unmapped_tail_source_frame_count": (
            target_sampling.source_frame_count - last_boundary.target_source_frame_1based
        ),
        "max_tracking_gap_seconds": args.max_tracking_gap_seconds,
        "elapsed_seconds": elapsed,
    }
    (output_dir / "segmentation.json").write_text(
        json.dumps(
            {"summary": summary, "boundaries": boundary_rows, "segments": segment_rows},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[{video_id}] completed in {elapsed:.1f}s -> {output_dir}", flush=True)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", type=Path, required=True, help="Trimmed complete 24-form MP4.")
    parser.add_argument("--input-tracking", type=Path, required=True, help="Matching PHALP tracking PKL.")
    parser.add_argument("--video-id", help="Output directory name; defaults to the MP4 stem.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reference-video", type=Path, default=DEFAULT_REFERENCE_VIDEO)
    parser.add_argument("--reference-tracking", type=Path, default=DEFAULT_REFERENCE_TRACKING)
    parser.add_argument("--gold-boundaries", type=Path, default=DEFAULT_GOLD_BOUNDARIES)
    parser.add_argument("--gold-url-id", default=DEFAULT_GOLD_URL_ID)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--dtw-coefficient", type=float, default=1.0)
    parser.add_argument("--pairwise-chunk-size", type=int, default=32)
    parser.add_argument("--save-local-costs", action="store_true")
    parser.add_argument(
        "--max-tracking-gap-seconds",
        type=float,
        default=0.0,
        help="Allow nearest tracked poses for short missing-frame gaps; default is strict.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sample_fps <= 0:
        raise ValueError("--sample-fps must be positive.")
    if args.dtw_coefficient <= 0:
        raise ValueError("--dtw-coefficient must be positive.")
    if args.pairwise_chunk_size <= 0:
        raise ValueError("--pairwise-chunk-size must be positive.")
    if args.max_tracking_gap_seconds < 0:
        raise ValueError("--max-tracking-gap-seconds must be non-negative.")
    run_segmentation(args)


if __name__ == "__main__":
    main()
