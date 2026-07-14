#!/usr/bin/env python
"""Export five-video TAS ground truth and compare it with DTW boundaries."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from export_tas_reference_annotations import MOVE_NAMES, load_point_labels


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ANNOTATION_PATH = (
    Path("/home/sqw/Projects/annotation-tool/annotations") / "instruction_2026-07-08_22.42.55.txt"
)
DEFAULT_TRIM_MANIFEST = Path("/home/sqw/VisualSearch/aqa/teach_trimmed/trim_manifest.csv")
DEFAULT_DTW_ROOT = PROJECT_ROOT / "tas_smpl_dtw_results"
DEFAULT_REFERENCE_SEGMENTS = PROJECT_ROOT / "tas_annotations" / "QxVvRcRn2TA_segments_5fps.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tas_ground_truth"
DEFAULT_REFERENCE_ID = "QxVvRcRn2TA"
DEFAULT_VIDEO_IDS = (
    "QxVvRcRn2TA",
    "BV1iE411c7Ni_p03",
    "BV1tk4y1r7Yr_p27",
    "an5qNCspzUw",
    "i8kMrJmAfjU",
)


@dataclass(frozen=True)
class GroundTruthSegment:
    video_id: str
    move_id: int
    move_name: str
    source_annotated_first_active_time: float
    source_last_matching_label_time: float
    source_boundary_end_time: float
    trim_start_time: float
    annotated_first_active_time: float
    ground_truth_start_time: float
    ground_truth_end_time: float
    ground_truth_duration: float
    start_frame_5fps: int
    end_frame_5fps: int
    sample_fps: float
    annotation_anomaly_count: int
    time_policy: str


@dataclass(frozen=True)
class BoundaryComparison:
    video_id: str
    move_id: int
    move_name: str
    dtw_boundary_role: str
    ground_truth_start_time: float
    ground_truth_end_time: float
    ground_truth_duration: float
    dtw_start_time: float
    dtw_end_time: float
    dtw_duration: float
    start_error_seconds: float
    start_absolute_error_seconds: float
    end_error_seconds: float
    end_absolute_error_seconds: float
    duration_error_seconds: float
    duration_absolute_error_seconds: float
    temporal_iou: float


def load_trim_offsets(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    offsets = {row["video_id"]: float(row["start_time"]) for row in rows}
    if not offsets:
        raise ValueError(f"No trim jobs found in {path}.")
    return offsets


def build_ground_truth_segments(
    annotation_path: Path,
    video_id: str,
    trim_start_time: float,
    sample_fps: float,
) -> list[GroundTruthSegment]:
    labels = [
        label
        for label in load_point_labels(annotation_path, video_id)
        if label.state == 1 and label.tag_id != 0
    ]
    by_move = {
        move_id: [label for label in labels if label.tag_id == move_id]
        for move_id in range(1, 25)
    }
    missing = [move_id for move_id, move_labels in by_move.items() if not move_labels]
    if missing:
        raise ValueError(f"{video_id} is missing move labels: {missing}.")
    first_active_times = {
        move_id: min(label.start_second for label in move_labels)
        for move_id, move_labels in by_move.items()
    }
    if any(first_active_times[move_id] >= first_active_times[move_id + 1] for move_id in range(1, 24)):
        raise ValueError(f"{video_id} move start times are not strictly increasing.")

    segments: list[GroundTruthSegment] = []
    previous_end_time: float | None = None
    for move_id in range(1, 25):
        move_labels = by_move[move_id]
        source_first_active = float(first_active_times[move_id])
        source_last_matching = float(max(label.start_second for label in move_labels))
        source_end = (
            float(first_active_times[move_id + 1] - 1.0)
            if move_id < 24
            else source_last_matching
        )
        anomaly_count = sum(
            label.tag_id != move_id
            for label in labels
            if source_first_active <= label.start_second <= source_end
        )
        first_active = source_first_active - trim_start_time
        end_time = source_end - trim_start_time
        start_time = first_active if previous_end_time is None else previous_end_time
        if start_time < -1e-9 or end_time <= start_time:
            raise ValueError(
                f"{video_id} move {move_id} has invalid trimmed interval {start_time}..{end_time}."
            )
        start_frame = int(round(start_time * sample_fps)) + 1
        end_frame = int(round(end_time * sample_fps))
        segments.append(
            GroundTruthSegment(
                video_id=video_id,
                move_id=move_id,
                move_name=MOVE_NAMES[move_id],
                source_annotated_first_active_time=source_first_active,
                source_last_matching_label_time=source_last_matching,
                source_boundary_end_time=source_end,
                trim_start_time=trim_start_time,
                annotated_first_active_time=first_active,
                ground_truth_start_time=start_time,
                ground_truth_end_time=end_time,
                ground_truth_duration=end_time - start_time,
                start_frame_5fps=start_frame,
                end_frame_5fps=end_frame,
                sample_fps=sample_fps,
                annotation_anomaly_count=anomaly_count,
                time_policy="trim_relative_continuous_intervals_using_next_move_first_second_minus_one",
            )
        )
        previous_end_time = end_time
    return segments


def load_dtw_ranges(
    video_id: str,
    reference_video_id: str,
    dtw_root: Path,
    reference_segments_path: Path,
) -> tuple[str, dict[int, tuple[float, float]]]:
    if video_id == reference_video_id:
        with reference_segments_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        ranges = {
            int(row["move_id"]): (
                float(row["frame_start_boundary_time"]),
                float(row["frame_end_boundary_time"]),
            )
            for row in rows
        }
        return "reference_input", ranges

    path = dtw_root / video_id / "segments_5fps.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ranges = {
        int(row["move_id"]): (float(row["start_time"]), float(row["end_time"]))
        for row in rows
    }
    return "dtw_prediction", ranges


def compare_segments(
    ground_truth: list[GroundTruthSegment],
    dtw_role: str,
    dtw_ranges: dict[int, tuple[float, float]],
) -> list[BoundaryComparison]:
    comparisons: list[BoundaryComparison] = []
    for segment in ground_truth:
        if segment.move_id not in dtw_ranges:
            raise KeyError(f"Missing DTW range for {segment.video_id} move {segment.move_id}.")
        dtw_start, dtw_end = dtw_ranges[segment.move_id]
        dtw_duration = dtw_end - dtw_start
        overlap = max(
            min(segment.ground_truth_end_time, dtw_end)
            - max(segment.ground_truth_start_time, dtw_start),
            0.0,
        )
        union = max(segment.ground_truth_end_time, dtw_end) - min(segment.ground_truth_start_time, dtw_start)
        start_error = dtw_start - segment.ground_truth_start_time
        end_error = dtw_end - segment.ground_truth_end_time
        duration_error = dtw_duration - segment.ground_truth_duration
        comparisons.append(
            BoundaryComparison(
                video_id=segment.video_id,
                move_id=segment.move_id,
                move_name=segment.move_name,
                dtw_boundary_role=dtw_role,
                ground_truth_start_time=segment.ground_truth_start_time,
                ground_truth_end_time=segment.ground_truth_end_time,
                ground_truth_duration=segment.ground_truth_duration,
                dtw_start_time=dtw_start,
                dtw_end_time=dtw_end,
                dtw_duration=dtw_duration,
                start_error_seconds=start_error,
                start_absolute_error_seconds=abs(start_error),
                end_error_seconds=end_error,
                end_absolute_error_seconds=abs(end_error),
                duration_error_seconds=duration_error,
                duration_absolute_error_seconds=abs(duration_error),
                temporal_iou=overlap / union if union > 0 else 1.0,
            )
        )
    return comparisons


def summarize_comparisons(comparisons: list[BoundaryComparison]) -> list[dict]:
    video_ids = list(dict.fromkeys(row.video_id for row in comparisons))
    summaries: list[dict] = []
    for video_id in video_ids:
        rows = [row for row in comparisons if row.video_id == video_id]
        end_errors = np.asarray([row.end_absolute_error_seconds for row in rows])
        summaries.append(
            {
                "video_id": video_id,
                "dtw_boundary_role": rows[0].dtw_boundary_role,
                "move_count": len(rows),
                "mean_start_absolute_error_seconds": float(
                    np.mean([row.start_absolute_error_seconds for row in rows])
                ),
                "mean_end_absolute_error_seconds": float(np.mean(end_errors)),
                "median_end_absolute_error_seconds": float(np.median(end_errors)),
                "max_end_absolute_error_seconds": float(np.max(end_errors)),
                "mean_duration_absolute_error_seconds": float(
                    np.mean([row.duration_absolute_error_seconds for row in rows])
                ),
                "mean_temporal_iou": float(np.mean([row.temporal_iou for row in rows])),
            }
        )
    predicted = [row for row in comparisons if row.dtw_boundary_role == "dtw_prediction"]
    if predicted:
        end_errors = np.asarray([row.end_absolute_error_seconds for row in predicted])
        summaries.append(
            {
                "video_id": "all_dtw_targets",
                "dtw_boundary_role": "aggregate_dtw_prediction",
                "move_count": len(predicted),
                "mean_start_absolute_error_seconds": float(
                    np.mean([row.start_absolute_error_seconds for row in predicted])
                ),
                "mean_end_absolute_error_seconds": float(np.mean(end_errors)),
                "median_end_absolute_error_seconds": float(np.median(end_errors)),
                "max_end_absolute_error_seconds": float(np.max(end_errors)),
                "mean_duration_absolute_error_seconds": float(
                    np.mean([row.duration_absolute_error_seconds for row in predicted])
                ),
                "mean_temporal_iou": float(np.mean([row.temporal_iou for row in predicted])),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {
                key: _rounded_csv_value(value)
                for key, value in row.items()
            }
            for row in rows
        )


def _rounded_csv_value(value: object) -> object:
    if not isinstance(value, (float, np.floating)):
        return value
    rounded = round(float(value), 6)
    return 0.0 if rounded == 0.0 else rounded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-path", type=Path, default=DEFAULT_ANNOTATION_PATH)
    parser.add_argument("--trim-manifest", type=Path, default=DEFAULT_TRIM_MANIFEST)
    parser.add_argument("--dtw-root", type=Path, default=DEFAULT_DTW_ROOT)
    parser.add_argument("--reference-segments", type=Path, default=DEFAULT_REFERENCE_SEGMENTS)
    parser.add_argument("--reference-video-id", default=DEFAULT_REFERENCE_ID)
    parser.add_argument("--video-id", action="append", help="Video ID to export; repeatable")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    video_ids = tuple(args.video_id or DEFAULT_VIDEO_IDS)
    trim_offsets = load_trim_offsets(args.trim_manifest)
    missing_offsets = sorted(set(video_ids) - set(trim_offsets))
    if missing_offsets:
        raise KeyError(f"Missing trim offsets for: {', '.join(missing_offsets)}")

    ground_truth: list[GroundTruthSegment] = []
    comparisons: list[BoundaryComparison] = []
    for video_id in video_ids:
        segments = build_ground_truth_segments(
            args.annotation_path,
            video_id,
            trim_offsets[video_id],
            args.sample_fps,
        )
        role, ranges = load_dtw_ranges(
            video_id,
            args.reference_video_id,
            args.dtw_root,
            args.reference_segments,
        )
        ground_truth.extend(segments)
        comparisons.extend(compare_segments(segments, role, ranges))

    summaries = summarize_comparisons(comparisons)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_rows = [asdict(row) for row in ground_truth]
    comparison_rows = [asdict(row) for row in comparisons]
    write_csv(args.output_dir / "ground_truth_segments.csv", ground_truth_rows)
    write_csv(args.output_dir / "ground_truth_vs_dtw.csv", comparison_rows)
    write_csv(args.output_dir / "ground_truth_vs_dtw_summary.csv", summaries)
    payload = {
        "annotation_path": str(args.annotation_path),
        "trim_manifest": str(args.trim_manifest),
        "dtw_root": str(args.dtw_root),
        "reference_video_id": args.reference_video_id,
        "video_ids": list(video_ids),
        "ground_truth_segment_count": len(ground_truth),
        "comparison_count": len(comparisons),
        "time_policy": "trim-relative continuous intervals; each move after the first starts at the previous move end",
        "summaries": summaries,
    }
    (args.output_dir / "ground_truth_vs_dtw_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Ground truth: {args.output_dir / 'ground_truth_segments.csv'}")
    print(f"Comparison:   {args.output_dir / 'ground_truth_vs_dtw.csv'}")
    print(f"Summary:      {args.output_dir / 'ground_truth_vs_dtw_summary.csv'}")


if __name__ == "__main__":
    main()
