#!/usr/bin/env python
"""Extract one video at a fixed FPS and append it to COIN input manifests."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aqa3d.smpl_dtw import inspect_video_sampling


DEFAULT_VIDEO = Path(
    "/home/sqw/VisualSearch/aqa/teach_trimmed/BV1WE411W7JB.mp4"
)
DEFAULT_OUTPUT = Path(
    "/home/sqw/VisualSearch/aqa/FrameData/teach/BV1WE411W7JB_5FPS"
)
DEFAULT_FRAME_MANIFEST = Path(
    "/home/sqw/Projects/annotation-tool/input/frame.txt"
)
DEFAULT_VIDEO_MANIFEST = Path(
    "/home/sqw/Projects/annotation-tool/input/video.txt"
)
DEFAULT_URL_ID = "BV1WE411W7JB_5FPS"


def format_time(seconds: float) -> str:
    """Keep one decimal place for the 0.2-second COIN frame grid."""
    return f"{seconds:.1f}"


def build_frame_rows(
    url_id: str,
    sample_count: int,
    sample_fps: float,
) -> list[dict[str, str]]:
    if sample_count <= 0 or sample_fps <= 0:
        raise ValueError("Sample count and FPS must be positive.")
    return [
        {
            "URLID": url_id,
            "Frame": f"{index:05d}.jpg",
            "Time": format_time(index / sample_fps),
        }
        for index in range(sample_count)
    ]


def _read_csv(path: Path, expected_fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise ValueError(
                f"{path} fields are {reader.fieldnames}; expected {expected_fields}."
            )
        return [dict(row) for row in reader]


def merge_unique_url_rows(
    existing_rows: list[dict[str, str]],
    new_rows: list[dict[str, str]],
    url_id: str,
) -> list[dict[str, str]]:
    """Preserve other videos and replace this URLID at the end without duplicates."""
    retained = [row for row in existing_rows if row["URLID"] != url_id]
    return retained + new_rows


def _atomic_write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def extract_sampled_frames(
    video_path: Path,
    output_dir: Path,
    sample_fps: float,
) -> tuple[int, float, int, int]:
    """Decode sequentially and save the exact uniform source-frame indices."""
    sampling = inspect_video_sampling(video_path, sample_fps)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Remove or move it before intentionally extracting again."
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".tmp",
            dir=output_dir.parent,
        )
    )
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        next_sample = 0
        source_index = 0
        while next_sample < sampling.sample_count:
            ok, frame = capture.read()
            if not ok:
                break
            expected_source = int(sampling.source_indices[next_sample])
            if source_index == expected_source:
                output_path = temporary_dir / f"{next_sample:05d}.jpg"
                if not cv2.imwrite(str(output_path), frame):
                    raise IOError(f"Failed to write frame: {output_path}")
                next_sample += 1
            source_index += 1
        if next_sample != sampling.sample_count:
            raise RuntimeError(
                f"Decoded {next_sample} of {sampling.sample_count} requested samples."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        for frame_path in sorted(temporary_dir.glob("*.jpg")):
            os.replace(frame_path, output_dir / frame_path.name)
    finally:
        capture.release()
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)

    return (
        sampling.sample_count,
        sampling.source_fps,
        sampling.source_frame_count,
        int(sampling.source_indices[-1]),
    )


def update_coin_manifests(
    frame_manifest: Path,
    video_manifest: Path,
    url_id: str,
    video_name: str,
    frame_rows: list[dict[str, str]],
) -> tuple[int, int]:
    existing_frames = _read_csv(
        frame_manifest,
        ("URLID", "Frame", "Time"),
    )
    existing_videos = _read_csv(
        video_manifest,
        ("URLID", "URL"),
    )
    merged_frames = merge_unique_url_rows(existing_frames, frame_rows, url_id)
    merged_videos = merge_unique_url_rows(
        existing_videos,
        [{"URLID": url_id, "URL": video_name}],
        url_id,
    )
    _atomic_write_csv(
        frame_manifest,
        ("URLID", "Frame", "Time"),
        merged_frames,
    )
    _atomic_write_csv(
        video_manifest,
        ("URLID", "URL"),
        merged_videos,
    )
    return len(existing_frames), len(existing_videos)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--url-id", default=DEFAULT_URL_ID)
    parser.add_argument("--frame-manifest", type=Path, default=DEFAULT_FRAME_MANIFEST)
    parser.add_argument("--video-manifest", type=Path, default=DEFAULT_VIDEO_MANIFEST)
    parser.add_argument(
        "--video-name",
        default=f"{DEFAULT_URL_ID}.mp4",
        help="Display URL stored in video.txt; frame mode resolves images by URLID.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sample_count, source_fps, source_count, last_source_index = (
        extract_sampled_frames(
            args.video,
            args.output_dir,
            args.sample_fps,
        )
    )
    frame_rows = build_frame_rows(args.url_id, sample_count, args.sample_fps)
    old_frame_count, old_video_count = update_coin_manifests(
        args.frame_manifest,
        args.video_manifest,
        args.url_id,
        args.video_name,
        frame_rows,
    )
    print(f"Extracted {sample_count} frames to {args.output_dir}")
    print(
        f"Source: {source_count} frames at {source_fps:.6f} FPS; "
        f"last sampled source frame: {last_source_index}"
    )
    print(
        f"Preserved {old_frame_count} existing frame rows and "
        f"{old_video_count} existing video rows; appended URLID={args.url_id}"
    )


if __name__ == "__main__":
    main()
