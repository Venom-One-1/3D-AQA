#!/usr/bin/env python
"""Transfer reference 24-form boundaries to full student videos with SMPL DTW."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from aqa3d.smpl_dtw import (
    ReferenceFrameMatch,
    VideoSampling,
    backtrack_dtw_path,
    dtw_from_cost_matrix,
    pairwise_geodesic_costs,
    require_strictly_increasing_boundaries,
    select_reference_frame_matches,
)
from aqa3d.tracking import TrackPoseSequence
from run_tas_smpl_dtw_mapping import (
    ReferenceSegment,
    load_reference_segments,
    load_sampled_track,
    plot_diagnostics,
    save_boundary_mapping,
    save_dtw_path,
    tracking_path,
    write_dict_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE_VIDEO_ROOT = Path("/home/sqw/VisualSearch/aqa/teach_trimmed")
DEFAULT_REFERENCE_TRACKING_ROOT = Path("/home/sqw/VisualSearch/aqa/Tracking/teach_trimmed")
DEFAULT_REFERENCE_ID = "QxVvRcRn2TA"
DEFAULT_REFERENCE_SEGMENTS = PROJECT_ROOT / "tas_annotations" / "QxVvRcRn2TA_segments_5fps.csv"
DEFAULT_STUDENT_VIDEO_ROOT = Path("/home/sqw/VisualSearch/aqa/student")
DEFAULT_STUDENT_TRACKING_ROOT = Path("/home/sqw/VisualSearch/aqa/Tracking/student_full")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "student_segmentation_results"


@dataclass(frozen=True)
class StudentSegment:
    video_id: str
    move_id: int
    move_name: str
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    source_fps: float
    start_frame_5fps: int
    end_frame_5fps: int
    sample_fps: float
    reference_end_frame_5fps: int
    boundary_candidate_count: int
    boundary_local_geodesic_radians: float
    boundary_local_geodesic_degrees: float
    boundary_policy: str


def sampled_interval_to_source_interval(
    start_frame_5fps: int,
    end_frame_5fps: int,
    sampling: VideoSampling,
) -> tuple[int, int]:
    """Convert a 1-based closed sampled interval to a 1-based source interval."""
    if not 1 <= start_frame_5fps <= end_frame_5fps <= sampling.sample_count:
        raise ValueError(
            "Invalid sampled interval: "
            f"[{start_frame_5fps}, {end_frame_5fps}] for {sampling.sample_count} samples."
        )

    start_source_frame = int(sampling.source_indices[start_frame_5fps - 1]) + 1
    if end_frame_5fps < sampling.sample_count:
        # The next sampled pose marks the exclusive end in zero-based source coordinates.
        end_source_frame = int(sampling.source_indices[end_frame_5fps])
    else:
        end_source_frame = sampling.source_frame_count
    if end_source_frame < start_source_frame:
        raise ValueError(
            f"Mapped source interval is empty: [{start_source_frame}, {end_source_frame}]."
        )
    return start_source_frame, end_source_frame


def build_student_segments(
    video_id: str,
    reference_segments: list[ReferenceSegment],
    matches: list[ReferenceFrameMatch],
    sampling: VideoSampling,
) -> list[StudentSegment]:
    if len(reference_segments) != len(matches):
        raise ValueError("Each reference segment must have exactly one mapped end boundary.")
    require_strictly_increasing_boundaries(matches)

    segments: list[StudentSegment] = []
    previous_end_5fps = 0
    for reference, match in zip(reference_segments, matches):
        expected_reference_index = reference.end_frame - 1
        if match.reference_index != expected_reference_index:
            raise ValueError(
                f"Move {reference.move_id} expected reference index {expected_reference_index}, "
                f"got {match.reference_index}."
            )

        start_frame_5fps = previous_end_5fps + 1
        end_frame_5fps = match.target_index + 1
        start_frame, end_frame = sampled_interval_to_source_interval(
            start_frame_5fps,
            end_frame_5fps,
            sampling,
        )
        segments.append(
            StudentSegment(
                video_id=video_id,
                move_id=reference.move_id,
                move_name=reference.move_name,
                start_time=(start_frame - 1) / sampling.source_fps,
                end_time=end_frame / sampling.source_fps,
                start_frame=start_frame,
                end_frame=end_frame,
                source_fps=sampling.source_fps,
                start_frame_5fps=start_frame_5fps,
                end_frame_5fps=end_frame_5fps,
                sample_fps=sampling.sample_fps,
                reference_end_frame_5fps=reference.end_frame,
                boundary_candidate_count=match.candidate_count,
                boundary_local_geodesic_radians=match.local_cost,
                boundary_local_geodesic_degrees=float(np.degrees(match.local_cost)),
                boundary_policy=(
                    "dtw_mapped_non_overlapping_closed_1based_source_and_sampled_frames"
                ),
            )
        )
        previous_end_5fps = end_frame_5fps
    return segments


def discover_student_ids(video_root: Path, tracking_root: Path) -> list[str]:
    video_ids = {path.stem for path in video_root.glob("*.mp4")}
    tracked_ids = {
        path.parent.parent.name
        for path in tracking_root.glob("*/results/demo_*.pkl")
    }
    return sorted(video_ids.intersection(tracked_ids))


def process_student(
    student_video_id: str,
    *,
    student_video_root: Path,
    student_tracking_root: Path,
    output_root: Path,
    reference_video_id: str,
    reference_tracking_file: Path,
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
    student_video = student_video_root / f"{student_video_id}.mp4"
    student_tracking = tracking_path(student_tracking_root, student_video_id)
    student_sampling, student_track, student_poses = load_sampled_track(
        student_video,
        student_tracking,
        sample_fps,
    )
    print(
        f"[{student_video_id}] loaded {student_sampling.sample_count} student samples; "
        f"computing {student_sampling.sample_count} x "
        f"{reference_sampling.sample_count} local costs",
        flush=True,
    )

    local_costs = pairwise_geodesic_costs(
        student_poses,
        reference_poses,
        chunk_size=pairwise_chunk_size,
    )
    dtw_distance, _, accumulated = dtw_from_cost_matrix(local_costs, dtw_coefficient)
    path = backtrack_dtw_path(accumulated)
    reference_boundary_indices = [segment.end_frame - 1 for segment in reference_segments]
    matches = select_reference_frame_matches(local_costs, path, reference_boundary_indices)
    student_segments = build_student_segments(
        student_video_id,
        reference_segments,
        matches,
        student_sampling,
    )

    output_dir = output_root / student_video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    segment_rows = [asdict(segment) for segment in student_segments]
    write_dict_csv(output_dir / "segments.csv", segment_rows)
    save_boundary_mapping(
        output_dir / "boundary_mapping.csv",
        reference_segments,
        matches,
        reference_sampling,
        student_sampling,
    )
    save_dtw_path(output_dir, path, local_costs, reference_sampling, student_sampling)
    if save_local_costs:
        np.savez_compressed(
            output_dir / "local_geodesic_costs.npz",
            local_costs_radians=local_costs.astype(np.float32),
        )
    plot_diagnostics(
        output_dir / "dtw_diagnostics.png",
        local_costs,
        path,
        reference_segments,
        matches,
        reference_sampling,
        student_sampling,
        student_video_id,
    )

    elapsed = time.perf_counter() - started
    summary = {
        "status": "ok",
        "reference_video_id": reference_video_id,
        "student_video_id": student_video_id,
        "reference_video": str(reference_sampling.video_path),
        "student_video": str(student_video),
        "reference_tracking": str(reference_tracking_file),
        "student_tracking": str(student_tracking),
        "reference_track_id": reference_track.track_id,
        "student_track_id": student_track.track_id,
        "reference_used_track_ids": list(reference_track.used_track_ids),
        "student_used_track_ids": list(student_track.used_track_ids),
        "source_fps": student_sampling.source_fps,
        "source_frame_count": student_sampling.source_frame_count,
        "sample_fps": sample_fps,
        "reference_sample_count": reference_sampling.sample_count,
        "student_sample_count": student_sampling.sample_count,
        "dtw_path_length": int(len(path)),
        "dtw_local_cost": "mean geodesic distance over 23 local SMPL body_pose joints",
        "global_orientation_included": False,
        "boundary_candidate_selection": (
            "minimum local geodesic distance among DTW-path candidates"
        ),
        "mapped_boundaries_strictly_increasing": True,
        "dtw_distance_radians": dtw_distance,
        "dtw_distance_degrees": float(np.degrees(dtw_distance)),
        "mapped_last_end_frame": student_segments[-1].end_frame,
        "mapped_last_end_frame_5fps": student_segments[-1].end_frame_5fps,
        "unmapped_student_tail_frames": (
            student_sampling.source_frame_count - student_segments[-1].end_frame
        ),
        "elapsed_seconds": elapsed,
    }
    (output_dir / "segments.json").write_text(
        json.dumps({**summary, "segments": segment_rows}, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"[{student_video_id}] done in {elapsed:.1f}s -> {output_dir}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-video-root", type=Path, default=DEFAULT_REFERENCE_VIDEO_ROOT)
    parser.add_argument(
        "--reference-tracking-root",
        type=Path,
        default=DEFAULT_REFERENCE_TRACKING_ROOT,
    )
    parser.add_argument("--reference-video-id", default=DEFAULT_REFERENCE_ID)
    parser.add_argument("--reference-segments", type=Path, default=DEFAULT_REFERENCE_SEGMENTS)
    parser.add_argument("--student-video-root", type=Path, default=DEFAULT_STUDENT_VIDEO_ROOT)
    parser.add_argument(
        "--student-tracking-root",
        type=Path,
        default=DEFAULT_STUDENT_TRACKING_ROOT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--student-video-id",
        action="append",
        help="Student video ID; repeatable. By default, process all available tracking results.",
    )
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--dtw-coefficient", type=float, default=1.0)
    parser.add_argument("--pairwise-chunk-size", type=int, default=32)
    parser.add_argument("--save-local-costs", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reference_video = args.reference_video_root / f"{args.reference_video_id}.mp4"
    reference_tracking = tracking_path(
        args.reference_tracking_root,
        args.reference_video_id,
    )
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

    student_ids = args.student_video_id or discover_student_ids(
        args.student_video_root,
        args.student_tracking_root,
    )
    if not student_ids:
        raise ValueError("No student videos with tracking results were found.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    failures: list[dict] = []
    for student_video_id in student_ids:
        try:
            summaries.append(
                process_student(
                    student_video_id,
                    student_video_root=args.student_video_root,
                    student_tracking_root=args.student_tracking_root,
                    output_root=args.output_root,
                    reference_video_id=args.reference_video_id,
                    reference_tracking_file=reference_tracking,
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
            failure = {
                "student_video_id": student_video_id,
                "status": "failed",
                "error": str(error),
            }
            failures.append(failure)
            print(f"[{student_video_id}] FAILED: {error}", flush=True)

    batch_payload = {
        "reference_video_id": args.reference_video_id,
        "reference_segments": str(args.reference_segments),
        "sample_fps": args.sample_fps,
        "completed": len(summaries),
        "failed": len(failures),
        "results": summaries,
        "failures": failures,
    }
    (args.output_root / "segmentation_summary.json").write_text(
        json.dumps(batch_payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    if summaries:
        write_dict_csv(args.output_root / "segmentation_summary.csv", summaries)
        all_segments: list[dict] = []
        for summary in summaries:
            segment_path = args.output_root / summary["student_video_id"] / "segments.csv"
            with segment_path.open("r", encoding="utf-8", newline="") as handle:
                all_segments.extend(csv.DictReader(handle))
        write_dict_csv(args.output_root / "all_student_segments.csv", all_segments)
    if failures:
        raise RuntimeError(f"Segmentation failed for {len(failures)} student video(s).")


if __name__ == "__main__":
    main()
