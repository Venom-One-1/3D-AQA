#!/usr/bin/env python
"""Render Ground Truth and SMPL Pose DTW end boundaries in one 24-row image."""

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


DEFAULT_VIDEO_ID = "QxVvRcRn2TA"
DEFAULT_INPUT_VIDEO = Path(
    "/home/sqw/VisualSearch/aqa/teach_trimmed/QxVvRcRn2TA.mp4"
)
DEFAULT_GROUND_TRUTH = Path(
    "/home/sqw/Projects/3D-AQA/tas_annotations/"
    "QxVvRcRn2TA_segments_5fps.csv"
)
DEFAULT_PREDICTED_BOUNDARIES = Path(
    "/home/sqw/Projects/3D-AQA/smpl_pose_dtw_segmentation_results/"
    "QxVvRcRn2TA/boundaries.csv"
)


@dataclass(frozen=True)
class GroundTruthBoundary:
    move_id: int
    move_name: str
    end_time: float


@dataclass(frozen=True)
class PredictedBoundary:
    move_id: int
    move_name: str
    end_time: float
    source_frame_index_0based: int
    local_geodesic_degrees: float


@dataclass(frozen=True)
class Layout:
    row_label_width: int = 190
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


def load_ground_truth_boundaries(
    path: Path,
    video_id: str,
) -> list[GroundTruthBoundary]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if not row.get("video_id") or row["video_id"] == video_id]
    boundaries = [
        GroundTruthBoundary(
            move_id=int(row["move_id"]),
            move_name=row["move_name"],
            end_time=float(row.get("frame_end_boundary_time") or row["end_time"]),
        )
        for row in selected
    ]
    boundaries.sort(key=lambda item: item.move_id)
    require_24_moves(video_id, boundaries)
    return boundaries


def load_predicted_boundaries(path: Path, video_id: str) -> list[PredictedBoundary]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    boundaries = [
        PredictedBoundary(
            move_id=int(row["move_id"]),
            move_name=row["move_name"],
            end_time=float(row["target_end_time_seconds"]),
            source_frame_index_0based=int(row["target_source_frame_0based"]),
            local_geodesic_degrees=float(row["local_geodesic_degrees"]),
        )
        for row in rows
    ]
    boundaries.sort(key=lambda item: item.move_id)
    require_24_moves(video_id, boundaries)
    return boundaries


def render_boundary_grid(
    output_path: Path,
    *,
    video_id: str,
    ground_truth: list[GroundTruthBoundary],
    predictions: list[PredictedBoundary],
    video_metadata: VideoMetadata,
    frames: dict[int, Image.Image],
) -> None:
    require_24_moves(video_id, ground_truth)
    require_24_moves(video_id, predictions)
    layout = Layout()
    canvas = Image.new("RGB", (layout.canvas_width, layout.canvas_height), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(23)
    header_font = load_font(18)
    row_font = load_font(15)
    detail_font = load_font(12)

    draw.text(
        (layout.outer_margin, layout.outer_margin),
        f"24-form boundary comparison | {video_id}",
        font=title_font,
        fill="#111827",
    )
    first_column_x = layout.outer_margin + layout.row_label_width
    header_top = layout.outer_margin + 38
    for column_index, title in enumerate(("Ground Truth boundary", "SMPL Pose DTW boundary")):
        cell_left = first_column_x + column_index * layout.cell_width
        draw_centered_text(
            draw,
            (
                cell_left,
                header_top,
                cell_left + layout.cell_width,
                layout.outer_margin + layout.header_height,
            ),
            title,
            header_font,
            "#1f2937",
        )

    for row_index, (truth, prediction) in enumerate(zip(ground_truth, predictions)):
        if truth.move_id != prediction.move_id:
            raise ValueError(
                f"Move mismatch at row {row_index}: GT={truth.move_id}, "
                f"prediction={prediction.move_id}."
            )
        row_top = layout.outer_margin + layout.header_height + row_index * layout.row_height
        row_bottom = row_top + layout.row_height
        background = "#f8fafc" if row_index % 2 == 0 else "#ffffff"
        draw.rectangle(
            (layout.outer_margin, row_top, layout.canvas_width - layout.outer_margin, row_bottom),
            fill=background,
        )

        error_seconds = prediction.end_time - truth.end_time
        error_color = "#15803d" if abs(error_seconds) <= 0.4 + 1e-9 else "#b91c1c"
        draw.text(
            (layout.outer_margin + 8, row_top + 52),
            f"{truth.move_id:02d}  {truth.move_name}",
            font=row_font,
            fill="#111827",
        )
        draw.text(
            (layout.outer_margin + 8, row_top + 79),
            f"error {error_seconds:+.1f}s",
            font=row_font,
            fill=error_color,
        )

        truth_frame_index = boundary_time_to_source_frame(truth.end_time, video_metadata)
        cell_frames = (
            frames[truth_frame_index],
            frames[prediction.source_frame_index_0based],
        )
        footer_texts = (
            f"GT {truth.end_time:.1f}s | frame {truth_frame_index + 1}",
            (
                f"DTW {prediction.end_time:.1f}s | frame "
                f"{prediction.source_frame_index_0based + 1} | "
                f"geo {prediction.local_geodesic_degrees:.1f} deg"
            ),
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
                detail_font,
                "#334155",
            )
        draw.line(
            (layout.outer_margin, row_bottom, layout.canvas_width - layout.outer_margin, row_bottom),
            fill="#cbd5e1",
            width=1,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=92, subsampling=0, optimize=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", default=DEFAULT_VIDEO_ID)
    parser.add_argument("--input-video", type=Path, default=DEFAULT_INPUT_VIDEO)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument(
        "--predicted-boundaries",
        type=Path,
        default=DEFAULT_PREDICTED_BOUNDARIES,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to boundary_frames.jpg beside the predicted boundaries CSV.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_video = args.input_video.expanduser().absolute()
    ground_truth_path = args.ground_truth.expanduser().absolute()
    predicted_path = args.predicted_boundaries.expanduser().absolute()
    for path in (input_video, ground_truth_path, predicted_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = inspect_video(input_video)
    ground_truth = load_ground_truth_boundaries(ground_truth_path, args.video_id)
    predictions = load_predicted_boundaries(predicted_path, args.video_id)
    truth_indices = [
        boundary_time_to_source_frame(boundary.end_time, metadata)
        for boundary in ground_truth
    ]
    predicted_indices = [boundary.source_frame_index_0based for boundary in predictions]
    invalid = [index for index in predicted_indices if not 0 <= index < metadata.frame_count]
    if invalid:
        raise IndexError(
            f"Predicted frames outside 0..{metadata.frame_count - 1}: {invalid[:10]}"
        )
    frames = read_video_frames(input_video, truth_indices + predicted_indices)
    output_path = (
        args.output.expanduser().absolute()
        if args.output is not None
        else predicted_path.parent / "boundary_frames.jpg"
    )
    render_boundary_grid(
        output_path,
        video_id=args.video_id,
        ground_truth=ground_truth,
        predictions=predictions,
        video_metadata=metadata,
        frames=frames,
    )
    print(f"[{args.video_id}] saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
