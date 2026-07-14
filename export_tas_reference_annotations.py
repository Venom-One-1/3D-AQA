#!/usr/bin/env python
"""Export a clean 5FPS TAS boundary table from annotation-tool labels."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ANNOTATION_DIR = Path("/home/sqw/Projects/annotation-tool/annotations")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tas_annotations"
DEFAULT_VIDEO_ID = "QxVvRcRn2TA"
DEFAULT_SAMPLE_FPS = 5.0

MOVE_NAMES = {
    1: "qishi",
    2: "yemafenzong",
    3: "baiheliangchi",
    4: "louxiaobu",
    5: "shouhuipipa",
    6: "daojuangong",
    7: "zuolanquewei",
    8: "youlanquewei",
    9: "danbian",
    10: "yunshou",
    11: "danbian_2",
    12: "gaotanma",
    13: "youdengtui",
    14: "shuangfengguaner",
    15: "zhuanshenzuodengtui",
    16: "zuoxiashiduli",
    17: "youxiashiduli",
    18: "zuoyouchuansuo",
    19: "haidizhen",
    20: "shantongbi",
    21: "zhuanshenbanlanchui",
    22: "rufengsibi",
    23: "shizishou",
    24: "shoushi",
}


@dataclass(frozen=True)
class PointLabel:
    video_id: str
    url: str
    tag_id: int
    start_second: float
    end_second: float
    state: int


@dataclass(frozen=True)
class TasSegment:
    video_id: str
    move_id: int
    move_name: str
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    frame_start_boundary_time: float
    frame_end_boundary_time: float
    sample_fps: float
    source_point_count: int
    boundary_policy: str


def latest_annotation_file(annotation_dir: Path) -> Path:
    candidates = sorted(annotation_dir.glob("instruction_*.txt"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No instruction_*.txt files found under {annotation_dir}")
    return candidates[-1]


def load_point_labels(annotation_path: Path, video_id: str) -> list[PointLabel]:
    labels: list[PointLabel] = []
    with annotation_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"URLID", "URL", "TagID", "Start", "End", "State"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{annotation_path} is missing columns: {sorted(missing)}")
        for row in reader:
            if row["URLID"] != video_id:
                continue
            labels.append(
                PointLabel(
                    video_id=row["URLID"],
                    url=row["URL"],
                    tag_id=int(row["TagID"]),
                    start_second=float(row["Start"]),
                    end_second=float(row["End"]),
                    state=int(row["State"]),
                )
            )
    if not labels:
        raise ValueError(f"No labels for video_id={video_id!r} in {annotation_path}")
    return sorted(labels, key=lambda item: (item.start_second, item.end_second, item.tag_id))


def compress_point_labels(labels: list[PointLabel], include_background: bool = False) -> list[list[PointLabel]]:
    active = [label for label in labels if label.state == 1 and (include_background or label.tag_id != 0)]
    if not active:
        raise ValueError("No active non-background labels to export.")

    groups: list[list[PointLabel]] = []
    current: list[PointLabel] = []
    for label in active:
        if current and label.tag_id != current[-1].tag_id:
            groups.append(current)
            current = []
        current.append(label)
    if current:
        groups.append(current)
    return groups


def seconds_to_frame_index(second: float, sample_fps: float) -> int:
    return int(round(second * sample_fps))


def build_segments(groups: list[list[PointLabel]], sample_fps: float) -> list[TasSegment]:
    if sample_fps <= 0:
        raise ValueError(f"sample_fps must be positive, got {sample_fps}.")

    segments: list[TasSegment] = []
    previous_end: float | None = None
    for group in groups:
        tag_id = group[0].tag_id
        if tag_id not in MOVE_NAMES:
            raise ValueError(f"Unsupported move tag_id={tag_id}.")
        label_start = group[0].start_second
        label_end = group[-1].start_second
        frame_start_boundary = label_start if previous_end is None else previous_end
        frame_end_boundary = label_end
        start_frame = seconds_to_frame_index(frame_start_boundary, sample_fps) + 1
        end_frame = seconds_to_frame_index(frame_end_boundary, sample_fps)
        if end_frame < start_frame:
            raise ValueError(
                f"Invalid frame range for move {tag_id}: start_frame={start_frame}, "
                f"end_frame={end_frame}."
            )
        segments.append(
            TasSegment(
                video_id=group[0].video_id,
                move_id=tag_id,
                move_name=MOVE_NAMES[tag_id],
                start_time=label_start,
                end_time=label_end,
                start_frame=start_frame,
                end_frame=end_frame,
                frame_start_boundary_time=frame_start_boundary,
                frame_end_boundary_time=frame_end_boundary,
                sample_fps=sample_fps,
                source_point_count=len(group),
                boundary_policy="non_overlapping_closed_1based_sampled_frames",
            )
        )
        previous_end = label_end
    return segments


def write_csv(path: Path, segments: list[TasSegment]) -> None:
    fieldnames = list(asdict(segments[0]).keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for segment in segments:
            writer.writerow(asdict(segment))


def write_json(path: Path, annotation_path: Path, segments: list[TasSegment]) -> None:
    payload = {
        "annotation_path": str(annotation_path),
        "video_id": segments[0].video_id,
        "sample_fps": segments[0].sample_fps,
        "boundary_policy": segments[0].boundary_policy,
        "segments": [asdict(segment) for segment in segments],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-path", type=Path, help="annotation-tool instruction_*.txt file")
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--video-id", default=DEFAULT_VIDEO_ID)
    parser.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    annotation_path = args.annotation_path or latest_annotation_file(args.annotation_dir)
    labels = load_point_labels(annotation_path, args.video_id)
    groups = compress_point_labels(labels)
    segments = build_segments(groups, args.sample_fps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.video_id}_segments_5fps.csv"
    json_path = args.output_dir / f"{args.video_id}_segments_5fps.json"
    write_csv(csv_path, segments)
    write_json(json_path, annotation_path, segments)

    print(f"Exported {len(segments)} TAS segments")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
