#!/usr/bin/env python
"""Split a complete 24-form video into frame-exact, non-overlapping clips."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from export_tas_reference_annotations import MOVE_NAMES


DEFAULT_INPUT_VIDEO = Path(
    "/home/sqw/VisualSearch/aqa/teach_trimmed/BV1WE411W7JB.mp4"
)
DEFAULT_BOUNDARIES = Path(
    "/home/sqw/VisualSearch/aqa/teach_trimmed/BV1WE411W7JB_5FPS_Boundary.txt"
)
DEFAULT_OUTPUT_ROOT = Path("/home/sqw/VisualSearch/aqa/ActionSegments/teach")
DEFAULT_SAMPLE_FPS = 5.0


@dataclass(frozen=True)
class Boundary:
    move_id: int
    label: str
    end_time_seconds: float
    sample_index_0based: int


@dataclass(frozen=True)
class VideoInfo:
    fps_numerator: int
    fps_denominator: int
    frame_count: int
    width: int
    height: int
    has_audio: bool

    @property
    def fps(self) -> Fraction:
        return Fraction(self.fps_numerator, self.fps_denominator)

    @property
    def duration_seconds(self) -> float:
        return float(Fraction(self.frame_count, 1) / self.fps)


@dataclass(frozen=True)
class Segment:
    video_id: str
    move_id: int
    move_name: str
    source_label: str
    output_file: str
    start_frame_0based: int
    end_frame_0based: int
    frame_count: int
    start_time_seconds: float
    end_time_exclusive_seconds: float
    annotated_end_time_seconds: float
    end_policy: str


def load_boundaries(
    path: Path,
    sample_fps: float,
    expected_count: int = 24,
) -> list[Boundary]:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive.")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"TagID", "Tag", "End"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if len(rows) != expected_count:
        raise ValueError(
            f"Expected {expected_count} boundaries in {path}, got {len(rows)}."
        )

    fps_decimal = Decimal(str(sample_fps))
    boundaries: list[Boundary] = []
    for expected_move_id, row in enumerate(rows, start=1):
        move_id = int(row["TagID"])
        if move_id != expected_move_id:
            raise ValueError(
                f"Expected TagID={expected_move_id} at row {expected_move_id}, "
                f"got {move_id}."
            )
        end_decimal = Decimal(row["End"])
        grid_position = end_decimal * fps_decimal
        sample_index = grid_position.to_integral_value()
        if grid_position != sample_index:
            raise ValueError(
                f"Move {move_id} boundary {end_decimal}s is not on the "
                f"{sample_fps:g} FPS grid."
            )
        boundaries.append(
            Boundary(
                move_id=move_id,
                label=row["Tag"].strip(),
                end_time_seconds=float(end_decimal),
                sample_index_0based=int(sample_index),
            )
        )
    if any(
        current.sample_index_0based >= following.sample_index_0based
        for current, following in zip(boundaries, boundaries[1:])
    ):
        raise ValueError("Boundary sample indices must be strictly increasing.")
    return boundaries


def probe_video(path: Path, ffprobe: str = "ffprobe") -> VideoInfo:
    command = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        "stream=codec_type,avg_frame_rate,nb_frames,nb_read_frames,width,height",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    video_streams = [
        stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"
    ]
    if len(video_streams) != 1:
        raise ValueError(f"Expected one video stream in {path}, got {len(video_streams)}.")
    stream = video_streams[0]
    rate = Fraction(stream["avg_frame_rate"])
    if rate <= 0:
        raise ValueError(f"Invalid average frame rate in {path}: {rate}.")
    raw_frame_count = stream.get("nb_frames") or stream.get("nb_read_frames")
    if raw_frame_count in (None, "N/A"):
        raise ValueError(f"ffprobe could not determine frame count for {path}.")
    frame_count = int(raw_frame_count)
    if frame_count <= 0:
        raise ValueError(f"Invalid frame count in {path}: {frame_count}.")
    return VideoInfo(
        fps_numerator=rate.numerator,
        fps_denominator=rate.denominator,
        frame_count=frame_count,
        width=int(stream["width"]),
        height=int(stream["height"]),
        has_audio=any(
            item.get("codec_type") == "audio" for item in payload.get("streams", [])
        ),
    )


def sampled_index_to_source_frame(
    sample_index: int,
    source_fps: Fraction,
    sample_fps: float,
) -> int:
    """Match the project's floor-based uniform 5 FPS source-frame sampling."""
    return int(Fraction(sample_index, 1) * source_fps / Fraction(str(sample_fps)))


def safe_move_name(move_id: int, label: str) -> str:
    preferred = MOVE_NAMES.get(move_id, "")
    if preferred:
        return preferred
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_").lower()
    return cleaned or f"move_{move_id:02d}"


def build_segments(
    video_id: str,
    boundaries: list[Boundary],
    video_info: VideoInfo,
    sample_fps: float,
) -> list[Segment]:
    if len(boundaries) != 24:
        raise ValueError(f"Expected 24 boundaries, got {len(boundaries)}.")
    source_boundary_frames = [
        sampled_index_to_source_frame(
            boundary.sample_index_0based,
            video_info.fps,
            sample_fps,
        )
        for boundary in boundaries
    ]
    if source_boundary_frames[22] >= video_info.frame_count - 1:
        raise ValueError(
            "Move 23 boundary must leave at least one source frame for move 24."
        )
    if source_boundary_frames[23] >= video_info.frame_count:
        raise ValueError(
            f"Move 24 annotation maps outside the video: frame "
            f"{source_boundary_frames[23]} >= {video_info.frame_count}."
        )

    segments: list[Segment] = []
    previous_end = -1
    for index, boundary in enumerate(boundaries):
        start_frame = previous_end + 1
        if boundary.move_id == 24:
            end_frame = video_info.frame_count - 1
            end_policy = "video_last_frame"
        else:
            end_frame = source_boundary_frames[index]
            end_policy = "annotated_boundary_frame_inclusive"
        if end_frame < start_frame:
            raise ValueError(
                f"Move {boundary.move_id} has an empty range: "
                f"{start_frame}..{end_frame}."
            )
        move_name = safe_move_name(boundary.move_id, boundary.label)
        segments.append(
            Segment(
                video_id=video_id,
                move_id=boundary.move_id,
                move_name=move_name,
                source_label=boundary.label,
                output_file=f"{boundary.move_id:02d}_{move_name}.mp4",
                start_frame_0based=start_frame,
                end_frame_0based=end_frame,
                frame_count=end_frame - start_frame + 1,
                start_time_seconds=float(Fraction(start_frame, 1) / video_info.fps),
                end_time_exclusive_seconds=float(
                    Fraction(end_frame + 1, 1) / video_info.fps
                ),
                annotated_end_time_seconds=boundary.end_time_seconds,
                end_policy=end_policy,
            )
        )
        previous_end = end_frame

    if sum(segment.frame_count for segment in segments) != video_info.frame_count:
        raise AssertionError("Segment frame ranges do not cover the source video exactly once.")
    return segments


def seconds_text(frame_index: int, fps: Fraction) -> str:
    value = Fraction(frame_index, 1) / fps
    return f"{float(value):.9f}".rstrip("0").rstrip(".")


def build_ffmpeg_command(
    *,
    ffmpeg: str,
    input_video: Path,
    output_dir: Path,
    segments: list[Segment],
    video_info: VideoInfo,
    crf: int,
    preset: str,
) -> list[str]:
    count = len(segments)
    video_inputs = "".join(f"[vin{index}]" for index in range(count))
    filters = [f"[0:v:0]split={count}{video_inputs}"]
    for index, segment in enumerate(segments):
        filters.append(
            f"[vin{index}]trim=start_frame={segment.start_frame_0based}:"
            f"end_frame={segment.end_frame_0based + 1},"
            f"setpts=PTS-STARTPTS[v{index}]"
        )

    if video_info.has_audio:
        audio_inputs = "".join(f"[ain{index}]" for index in range(count))
        filters.append(f"[0:a:0]asplit={count}{audio_inputs}")
        for index, segment in enumerate(segments):
            start = seconds_text(segment.start_frame_0based, video_info.fps)
            end = seconds_text(segment.end_frame_0based + 1, video_info.fps)
            filters.append(
                f"[ain{index}]atrim=start={start}:end={end},"
                f"asetpts=PTS-STARTPTS[a{index}]"
            )

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostdin",
        "-y",
        "-i",
        str(input_video),
        "-filter_complex",
        ";".join(filters),
    ]
    for index, segment in enumerate(segments):
        command.extend(["-map", f"[v{index}]"])
        if video_info.has_audio:
            command.extend(["-map", f"[a{index}]"])
        command.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
            ]
        )
        if video_info.has_audio:
            command.extend(["-c:a", "aac", "-b:a", "192k"])
        command.extend(["-movflags", "+faststart", str(output_dir / segment.output_file)])
    return command


def validate_outputs(
    output_dir: Path,
    segments: list[Segment],
    ffprobe: str,
) -> None:
    actual_total = 0
    for segment in segments:
        path = output_dir / segment.output_file
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output clip: {path}")
        info = probe_video(path, ffprobe)
        if info.frame_count != segment.frame_count:
            raise RuntimeError(
                f"{path.name}: expected {segment.frame_count} frames, "
                f"got {info.frame_count}."
            )
        actual_total += info.frame_count
    if actual_total != sum(segment.frame_count for segment in segments):
        raise AssertionError("Validated output frame total does not match the manifest.")


def write_manifests(
    output_dir: Path,
    *,
    input_video: Path,
    boundary_file: Path,
    sample_fps: float,
    video_info: VideoInfo,
    segments: list[Segment],
) -> None:
    rows = [asdict(segment) for segment in segments]
    with (output_dir / "segments.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "input_video": str(input_video),
        "boundary_file": str(boundary_file),
        "boundary_sample_fps": sample_fps,
        "source_video": asdict(video_info),
        "frame_coverage_policy": (
            "moves 1-23 include their boundary frame; each next move starts at "
            "the following source frame; move 24 ends at the video last frame"
        ),
        "segments": rows,
    }
    (output_dir / "segments.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def split_video(args: argparse.Namespace) -> Path:
    input_video = args.input_video.expanduser().absolute()
    boundary_file = args.boundary_file.expanduser().absolute()
    for path in (input_video, boundary_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    video_id = args.video_id or input_video.stem
    if not video_id or Path(video_id).name != video_id:
        raise ValueError("--video-id must be a non-empty directory-safe name.")
    if not 0 <= args.crf <= 51:
        raise ValueError("--crf must be between 0 and 51.")

    output_dir = args.output_root.expanduser().absolute() / video_id
    boundaries = load_boundaries(boundary_file, args.sample_fps)
    video_info = probe_video(input_video, args.ffprobe)
    segments = build_segments(video_id, boundaries, video_info, args.sample_fps)
    expected_outputs = [output_dir / segment.output_file for segment in segments]
    expected_outputs.extend([output_dir / "segments.csv", output_dir / "segments.json"])
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{len(existing)} output files already exist under {output_dir}; "
            "use --overwrite to replace them."
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{video_id}_split_", dir=output_dir.parent)
    )
    try:
        print(
            f"[{video_id}] splitting {video_info.frame_count} frames at "
            f"{float(video_info.fps):.6f} FPS into 24 clips",
            flush=True,
        )
        command = build_ffmpeg_command(
            ffmpeg=args.ffmpeg,
            input_video=input_video,
            output_dir=temporary_dir,
            segments=segments,
            video_info=video_info,
            crf=args.crf,
            preset=args.preset,
        )
        subprocess.run(command, check=True)
        validate_outputs(temporary_dir, segments, args.ffprobe)
        write_manifests(
            temporary_dir,
            input_video=input_video,
            boundary_file=boundary_file,
            sample_fps=args.sample_fps,
            video_info=video_info,
            segments=segments,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        for temporary_path in temporary_dir.iterdir():
            os.replace(temporary_path, output_dir / temporary_path.name)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)

    print(f"[{video_id}] saved 24 clips and manifests to {output_dir}", flush=True)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", type=Path, default=DEFAULT_INPUT_VIDEO)
    parser.add_argument("--boundary-file", type=Path, default=DEFAULT_BOUNDARIES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--video-id", help="Defaults to the input video stem.")
    parser.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    split_video(build_parser().parse_args())


if __name__ == "__main__":
    main()
