#!/usr/bin/env python
"""Render reference GT and student DTW end-boundary frames in one 24-row image."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from visualize_tas_boundary_frames import (
    VideoMetadata,
    boundary_time_to_source_frame,
    draw_centered_text,
    fit_frame,
    inspect_video,
    load_font,
    read_video_frames,
    require_24_moves,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE_VIDEO_ROOT = Path("/home/sqw/VisualSearch/aqa/teach_trimmed")
DEFAULT_REFERENCE_ID = "QxVvRcRn2TA"
DEFAULT_REFERENCE_SEGMENTS = PROJECT_ROOT / "tas_annotations" / "QxVvRcRn2TA_segments_5fps.csv"
DEFAULT_STUDENT_VIDEO_ROOT = Path("/home/sqw/VisualSearch/aqa/student")
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "student_segmentation_results"


@dataclass(frozen=True)
class ReferenceBoundary:
    move_id: int
    move_name: str
    end_time: float


@dataclass(frozen=True)
class StudentBoundary:
    move_id: int
    move_name: str
    source_frame_index: int
    end_time: float
    local_geodesic_degrees: float


@dataclass(frozen=True)
class Layout:
    row_label_width: int = 210
    cell_width: int = 292
    frame_width: int = 272
    frame_height: int = 153
    header_height: int = 88
    row_height: int = 180
    outer_margin: int = 12

    @property
    def canvas_width(self) -> int:
        return self.outer_margin * 2 + self.row_label_width + self.cell_width * 2

    @property
    def canvas_height(self) -> int:
        return self.outer_margin * 2 + self.header_height + self.row_height * 24


def load_reference_boundaries(path: Path, reference_video_id: str) -> list[ReferenceBoundary]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row.get("video_id") == reference_video_id]
    boundaries = [
        ReferenceBoundary(
            move_id=int(row["move_id"]),
            move_name=row["move_name"],
            end_time=float(row["end_time"]),
        )
        for row in selected
    ]
    boundaries.sort(key=lambda item: item.move_id)
    require_24_moves(reference_video_id, boundaries)
    return boundaries


def load_student_boundaries(result_root: Path, student_video_id: str) -> list[StudentBoundary]:
    path = result_root / student_video_id / "boundary_mapping.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    boundaries = [
        StudentBoundary(
            move_id=int(row["move_id"]),
            move_name=row["move_name"],
            source_frame_index=int(row["target_source_frame_0based"]),
            end_time=float(row["target_boundary_time_seconds"]),
            local_geodesic_degrees=float(row["selected_local_geodesic_degrees"]),
        )
        for row in rows
    ]
    boundaries.sort(key=lambda item: item.move_id)
    require_24_moves(student_video_id, boundaries)
    return boundaries


def discover_student_ids(result_root: Path, video_root: Path) -> list[str]:
    video_ids = {path.stem for path in video_root.glob("*.mp4")}
    result_ids = {
        path.parent.name
        for path in result_root.glob("*/boundary_mapping.csv")
    }
    return sorted(video_ids.intersection(result_ids))


def render_student_grid(
    output_path: Path,
    *,
    reference_video_id: str,
    student_video_id: str,
    reference_boundaries: list[ReferenceBoundary],
    student_boundaries: list[StudentBoundary],
    reference_metadata: VideoMetadata,
    reference_frames: dict[int, Image.Image],
    student_frames: dict[int, Image.Image],
) -> None:
    require_24_moves(reference_video_id, reference_boundaries)
    require_24_moves(student_video_id, student_boundaries)

    layout = Layout()
    canvas = Image.new("RGB", (layout.canvas_width, layout.canvas_height), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(23)
    header_font = load_font(18)
    row_font = load_font(15)
    small_font = load_font(12)

    draw.text(
        (layout.outer_margin, layout.outer_margin),
        f"24-form boundaries | Student: {student_video_id}",
        font=title_font,
        fill="#111827",
    )
    header_top = layout.outer_margin + 38
    first_column_x = layout.outer_margin + layout.row_label_width
    columns = (
        f"Reference GT | {reference_video_id}",
        f"Student DTW | {student_video_id}",
    )
    for column_index, title in enumerate(columns):
        left = first_column_x + column_index * layout.cell_width
        draw_centered_text(
            draw,
            (left, header_top, left + layout.cell_width, layout.outer_margin + layout.header_height),
            title,
            header_font,
            "#1f2937",
        )

    for row_index, (reference, student) in enumerate(
        zip(reference_boundaries, student_boundaries)
    ):
        if reference.move_id != student.move_id:
            raise ValueError(
                f"Move mismatch at row {row_index}: "
                f"reference={reference.move_id}, student={student.move_id}."
            )
        row_top = layout.outer_margin + layout.header_height + row_index * layout.row_height
        row_bottom = row_top + layout.row_height
        background = "#f8fafc" if row_index % 2 == 0 else "#ffffff"
        draw.rectangle(
            (layout.outer_margin, row_top, layout.canvas_width - layout.outer_margin, row_bottom),
            fill=background,
        )
        draw.text(
            (layout.outer_margin + 9, row_top + 62),
            f"{reference.move_id:02d}  {reference.move_name}",
            font=row_font,
            fill="#111827",
        )

        reference_index = boundary_time_to_source_frame(
            reference.end_time,
            reference_metadata,
        )
        cell_frames = (
            reference_frames[reference_index],
            student_frames[student.source_frame_index],
        )
        footer_texts = (
            f"GT {reference.end_time:.1f}s | frame {reference_index + 1}",
            f"DTW {student.end_time:.1f}s | frame {student.source_frame_index + 1}",
        )
        for column_index, (frame, footer) in enumerate(zip(cell_frames, footer_texts)):
            cell_left = first_column_x + column_index * layout.cell_width
            frame_left = cell_left + (layout.cell_width - layout.frame_width) // 2
            frame_top = row_top + 4
            fitted = fit_frame(frame, layout.frame_width, layout.frame_height)
            canvas.paste(fitted, (frame_left, frame_top))
            draw.rectangle(
                (
                    frame_left,
                    frame_top,
                    frame_left + layout.frame_width,
                    frame_top + layout.frame_height,
                ),
                outline="#94a3b8",
                width=1,
            )
            draw_centered_text(
                draw,
                (
                    cell_left,
                    frame_top + layout.frame_height + 1,
                    cell_left + layout.cell_width,
                    row_bottom - 1,
                ),
                footer,
                small_font,
                "#334155",
            )
        draw.line(
            (layout.outer_margin, row_bottom, layout.canvas_width - layout.outer_margin, row_bottom),
            fill="#cbd5e1",
            width=1,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=92, subsampling=0, optimize=True)


def visualize_student(
    student_video_id: str,
    *,
    student_video_root: Path,
    result_root: Path,
    reference_video_id: str,
    reference_boundaries: list[ReferenceBoundary],
    reference_metadata: VideoMetadata,
    reference_frames: dict[int, Image.Image],
) -> Path:
    student_video = student_video_root / f"{student_video_id}.mp4"
    student_boundaries = load_student_boundaries(result_root, student_video_id)
    student_metadata = inspect_video(student_video)
    invalid = [
        boundary.source_frame_index
        for boundary in student_boundaries
        if not 0 <= boundary.source_frame_index < student_metadata.frame_count
    ]
    if invalid:
        raise IndexError(
            f"Student {student_video_id} has predicted frames outside "
            f"0..{student_metadata.frame_count - 1}: {invalid[:10]}"
        )
    student_frames = read_video_frames(
        student_video,
        [boundary.source_frame_index for boundary in student_boundaries],
    )
    output_path = result_root / student_video_id / "boundary_frames.jpg"
    render_student_grid(
        output_path,
        reference_video_id=reference_video_id,
        student_video_id=student_video_id,
        reference_boundaries=reference_boundaries,
        student_boundaries=student_boundaries,
        reference_metadata=reference_metadata,
        reference_frames=reference_frames,
        student_frames=student_frames,
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-video-root", type=Path, default=DEFAULT_REFERENCE_VIDEO_ROOT)
    parser.add_argument("--reference-video-id", default=DEFAULT_REFERENCE_ID)
    parser.add_argument("--reference-segments", type=Path, default=DEFAULT_REFERENCE_SEGMENTS)
    parser.add_argument("--student-video-root", type=Path, default=DEFAULT_STUDENT_VIDEO_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument(
        "--student-video-id",
        action="append",
        help="Student video ID; repeatable. Defaults to every completed segmentation result.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reference_video = args.reference_video_root / f"{args.reference_video_id}.mp4"
    reference_metadata = inspect_video(reference_video)
    reference_boundaries = load_reference_boundaries(
        args.reference_segments,
        args.reference_video_id,
    )
    reference_indices = [
        boundary_time_to_source_frame(boundary.end_time, reference_metadata)
        for boundary in reference_boundaries
    ]
    reference_frames = read_video_frames(reference_video, reference_indices)

    student_ids = args.student_video_id or discover_student_ids(
        args.result_root,
        args.student_video_root,
    )
    if not student_ids:
        raise ValueError("No completed student segmentation results were found.")

    failures: list[tuple[str, str]] = []
    for student_video_id in student_ids:
        try:
            output_path = visualize_student(
                student_video_id,
                student_video_root=args.student_video_root,
                result_root=args.result_root,
                reference_video_id=args.reference_video_id,
                reference_boundaries=reference_boundaries,
                reference_metadata=reference_metadata,
                reference_frames=reference_frames,
            )
            print(f"[{student_video_id}] saved {output_path}", flush=True)
        except Exception as error:
            failures.append((student_video_id, str(error)))
            print(f"[{student_video_id}] FAILED: {error}", flush=True)
    if failures:
        details = "; ".join(f"{video_id}: {error}" for video_id, error in failures)
        raise RuntimeError(f"Visualization failed for {len(failures)} student(s): {details}")


if __name__ == "__main__":
    main()
