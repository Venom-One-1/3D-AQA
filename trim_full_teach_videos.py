#!/usr/bin/env python
"""Trim fully annotated 24-form teaching videos before PHALP tracking."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ANNOTATION_DIR = Path("/home/sqw/Projects/annotation-tool/annotations")
DEFAULT_ANNOTATION_PATH = DEFAULT_ANNOTATION_DIR / "instruction_2026-07-08_22.42.55.txt"
DEFAULT_VIDEO_ROOT = Path("/home/sqw/VisualSearch/aqa/teach")
DEFAULT_OUTPUT_ROOT = Path("/home/sqw/VisualSearch/aqa/teach_trimmed")
REQUIRED_MOVE_IDS = set(range(1, 25))


@dataclass(frozen=True)
class TrimJob:
    video_id: str
    source_video: Path
    output_video: Path
    start_time: float
    end_time: float
    duration: float
    move_count: int
    source_duration: float | None
    output_duration: float | None = None
    status: str = "pending"
    message: str = ""


def latest_annotation_file(annotation_dir: Path) -> Path:
    candidates = sorted(annotation_dir.glob("instruction_*.txt"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No instruction_*.txt files found under {annotation_dir}")
    return candidates[-1]


def read_rows(annotation_path: Path) -> list[dict[str, str]]:
    with annotation_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"URLID", "URL", "TagID", "Start", "End", "State"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{annotation_path} is missing columns: {sorted(missing)}")
        return list(reader)


def find_trim_jobs(
    rows: list[dict[str, str]],
    video_root: Path,
    output_root: Path,
) -> list[TrimJob]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if int(row["State"]) != 1:
            continue
        grouped.setdefault(row["URLID"], []).append(row)

    jobs: list[TrimJob] = []
    for video_id, labels in sorted(grouped.items()):
        move_ids = {int(row["TagID"]) for row in labels if int(row["TagID"]) != 0}
        if not REQUIRED_MOVE_IDS.issubset(move_ids):
            continue

        first_move = [row for row in labels if int(row["TagID"]) == 1]
        last_move = [row for row in labels if int(row["TagID"]) == 24]
        if not first_move or not last_move:
            continue

        source_name = labels[0]["URL"]
        source_video = video_root / source_name
        output_video = output_root / f"{video_id}.mp4"
        start_time = min(_row_start(row) for row in first_move)
        end_time = max(max(_row_start(row), _row_end(row)) for row in last_move)
        if end_time <= start_time:
            raise ValueError(f"Invalid trim range for {video_id}: {start_time}..{end_time}")

        source_duration = probe_duration(source_video) if source_video.exists() else None
        jobs.append(
            TrimJob(
                video_id=video_id,
                source_video=source_video,
                output_video=output_video,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                move_count=len(move_ids),
                source_duration=source_duration,
            )
        )
    return jobs


def _row_start(row: dict[str, str]) -> float:
    return float(row["Start"])


def _row_end(row: dict[str, str]) -> float:
    return float(row["End"])


def probe_duration(video_path: Path) -> float | None:
    if not video_path.exists():
        return None
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    value = result.stdout.strip()
    return float(value) if value else None


def trim_video(job: TrimJob, overwrite: bool = False) -> TrimJob:
    if not job.source_video.exists():
        return _replace(job, status="missing_source", message=f"Missing source video: {job.source_video}")
    if job.output_video.exists() and not overwrite:
        output_duration = probe_duration(job.output_video)
        return _replace(job, output_duration=output_duration, status="skipped", message="Output exists")

    job.output_video.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        _format_time(job.start_time),
        "-i",
        str(job.source_video),
        "-t",
        _format_time(job.duration),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(job.output_video),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return _replace(job, status="failed", message=result.stderr.strip())
    output_duration = probe_duration(job.output_video)
    return _replace(job, output_duration=output_duration, status="trimmed", message="OK")


def _format_time(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _replace(job: TrimJob, **updates: object) -> TrimJob:
    payload = asdict(job)
    payload.update(updates)
    return TrimJob(**payload)


def write_manifest(output_root: Path, annotation_path: Path, jobs: list[TrimJob]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [_manifest_row(job) for job in jobs]
    csv_path = output_root / "trim_manifest.csv"
    json_path = output_root / "trim_manifest.json"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    payload = {
        "annotation_path": str(annotation_path),
        "required_move_ids": sorted(REQUIRED_MOVE_IDS),
        "jobs": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _manifest_row(job: TrimJob) -> dict[str, object]:
    row = asdict(job)
    row["source_video"] = str(job.source_video)
    row["output_video"] = str(job.output_video)
    row["trimmed_background_seconds"] = (
        None if job.source_duration is None else max(job.source_duration - job.duration, 0.0)
    )
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-path", type=Path, help="annotation-tool instruction_*.txt file")
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--video-id", action="append", help="Only trim this video id; repeatable")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    annotation_path = args.annotation_path or (
        DEFAULT_ANNOTATION_PATH if DEFAULT_ANNOTATION_PATH.exists() else latest_annotation_file(args.annotation_dir)
    )
    rows = read_rows(annotation_path)
    jobs = find_trim_jobs(rows, args.video_root, args.output_root)
    if args.video_id:
        requested = set(args.video_id)
        jobs = [job for job in jobs if job.video_id in requested]
    if not jobs:
        raise ValueError("No fully annotated 24-form teaching videos found.")

    if args.dry_run:
        results = [_replace(job, status="dry_run", message="Not trimmed") for job in jobs]
    else:
        results = [trim_video(job, overwrite=args.overwrite) for job in jobs]
    write_manifest(args.output_root, annotation_path, results)

    for job in results:
        print(
            f"{job.status:14s} {job.video_id:18s} "
            f"{_format_time(job.start_time)}..{_format_time(job.end_time)}s -> {job.output_video}"
        )
        if job.status == "failed":
            print(job.message)
    failures = [job for job in results if job.status in {"failed", "missing_source"}]
    if failures:
        raise RuntimeError(f"{len(failures)} trim jobs failed.")


if __name__ == "__main__":
    main()
