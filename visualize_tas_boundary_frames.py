#!/usr/bin/env python
"""Render 24-form Ground Truth and DTW boundary-frame comparison grids."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO_ROOT = Path("/home/sqw/VisualSearch/aqa/teach_trimmed")
DEFAULT_GROUND_TRUTH = PROJECT_ROOT / "tas_ground_truth" / "ground_truth_segments.csv"
DEFAULT_COMPARISON = PROJECT_ROOT / "tas_ground_truth" / "ground_truth_vs_dtw.csv"
DEFAULT_DTW_ROOT = PROJECT_ROOT / "tas_smpl_dtw_results"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tas_boundary_visualizations"
DEFAULT_REFERENCE_ID = "QxVvRcRn2TA"
DEFAULT_TARGET_IDS = (
    "BV1iE411c7Ni_p03",
    "BV1tk4y1r7Yr_p27",
    "an5qNCspzUw",
    "i8kMrJmAfjU",
)
DEFAULT_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


@dataclass(frozen=True)
class GroundTruthBoundary:
    video_id: str
    move_id: int
    move_name: str
    end_time: float


@dataclass(frozen=True)
class PredictedBoundary:
    video_id: str
    move_id: int
    move_name: str
    source_frame_index: int
    end_time: float
    error_seconds: float


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    frame_count: int


@dataclass(frozen=True)
class Layout:
    row_label_width: int = 190
    cell_width: int = 330
    frame_width: int = 306
    frame_height: int = 172
    header_height: int = 82
    row_height: int = 216
    footer_height: int = 30
    outer_margin: int = 12

    @property
    def canvas_width(self) -> int:
        return self.outer_margin * 2 + self.row_label_width + self.cell_width * 3

    @property
    def canvas_height(self) -> int:
        return self.outer_margin * 2 + self.header_height + self.row_height * 24


def load_ground_truth(path: Path) -> dict[str, list[GroundTruthBoundary]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[GroundTruthBoundary]] = {}
    for row in rows:
        boundary = GroundTruthBoundary(
            video_id=row["video_id"],
            move_id=int(row["move_id"]),
            move_name=row["move_name"],
            end_time=float(row["ground_truth_end_time"]),
        )
        grouped.setdefault(boundary.video_id, []).append(boundary)
    for video_id, boundaries in grouped.items():
        boundaries.sort(key=lambda item: item.move_id)
        require_24_moves(video_id, boundaries)
    return grouped


def load_end_errors(path: Path) -> dict[tuple[str, int], float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            (row["video_id"], int(row["move_id"])): float(row["end_error_seconds"])
            for row in csv.DictReader(handle)
        }


def load_predicted_boundaries(
    dtw_root: Path,
    video_id: str,
    errors: dict[tuple[str, int], float],
) -> list[PredictedBoundary]:
    path = dtw_root / video_id / "boundary_mapping.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    boundaries = [
        PredictedBoundary(
            video_id=video_id,
            move_id=int(row["move_id"]),
            move_name=row["move_name"],
            source_frame_index=int(row["target_source_frame_0based"]),
            end_time=float(row["target_boundary_time_seconds"]),
            error_seconds=errors[(video_id, int(row["move_id"]))],
        )
        for row in rows
    ]
    boundaries.sort(key=lambda item: item.move_id)
    require_24_moves(video_id, boundaries)
    return boundaries


def require_24_moves(video_id: str, boundaries: list) -> None:
    move_ids = [item.move_id for item in boundaries]
    if move_ids != list(range(1, 25)):
        raise ValueError(f"{video_id} must contain ordered move IDs 1..24, got {move_ids}.")


def inspect_video(video_path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    metadata = VideoMetadata(
        fps=float(capture.get(cv2.CAP_PROP_FPS)),
        frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    capture.release()
    if metadata.fps <= 0 or metadata.frame_count <= 0:
        raise ValueError(f"Invalid video metadata for {video_path}: {metadata}")
    return metadata


def boundary_time_to_source_frame(end_time: float, metadata: VideoMetadata) -> int:
    """Use the video frame at the annotated boundary timestamp, clamped at EOF."""
    return min(max(int(round(end_time * metadata.fps)), 0), metadata.frame_count - 1)


def read_video_frames(video_path: Path, source_indices: list[int]) -> dict[int, Image.Image]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    requested = sorted(set(source_indices))
    invalid = [index for index in requested if index < 0 or index >= frame_count]
    if invalid:
        capture.release()
        raise IndexError(f"Frame indices outside 0..{frame_count - 1} for {video_path}: {invalid[:10]}")

    frames: dict[int, Image.Image] = {}
    for source_index in requested:
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Cannot decode source frame {source_index} from {video_path}.")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames[source_index] = Image.fromarray(rgb)
    capture.release()
    return frames


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if DEFAULT_FONT_PATH.is_file():
        return ImageFont.truetype(str(DEFAULT_FONT_PATH), size=size)
    return ImageFont.load_default()


def fit_frame(frame: Image.Image, width: int, height: int) -> Image.Image:
    fitted = Image.new("RGB", (width, height), "#111827")
    resized = frame.copy()
    resized.thumbnail((width, height), Image.Resampling.LANCZOS)
    x = (width - resized.width) // 2
    y = (height - resized.height) // 2
    fitted.paste(resized, (x, y))
    return fitted


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    left, top, right, bottom = bounds
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2 - box[1]),
        text,
        font=font,
        fill=fill,
    )


def render_target_grid(
    output_path: Path,
    *,
    reference_video_id: str,
    target_video_id: str,
    reference_gt: list[GroundTruthBoundary],
    target_gt: list[GroundTruthBoundary],
    predictions: list[PredictedBoundary],
    reference_metadata: VideoMetadata,
    target_metadata: VideoMetadata,
    reference_frames: dict[int, Image.Image],
    target_frames: dict[int, Image.Image],
) -> None:
    layout = Layout()
    canvas = Image.new("RGB", (layout.canvas_width, layout.canvas_height), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(25)
    header_font = load_font(20)
    row_font = load_font(16)
    detail_font = load_font(15)
    small_font = load_font(13)

    draw.text(
        (layout.outer_margin, layout.outer_margin),
        f"24-form boundary comparison | Target: {target_video_id}",
        font=title_font,
        fill="#111827",
    )
    header_top = layout.outer_margin + 38
    columns = (
        f"Reference GT | {reference_video_id}",
        f"Target GT | {target_video_id}",
        "Target DTW prediction",
    )
    first_column_x = layout.outer_margin + layout.row_label_width
    for column_index, title in enumerate(columns):
        left = first_column_x + column_index * layout.cell_width
        draw_centered_text(
            draw,
            (left, header_top, left + layout.cell_width, layout.outer_margin + layout.header_height),
            title,
            header_font,
            "#1f2937",
        )

    for row_index, (reference, target, prediction) in enumerate(zip(reference_gt, target_gt, predictions)):
        if not (reference.move_id == target.move_id == prediction.move_id):
            raise ValueError(f"Move mismatch at row {row_index} for {target_video_id}.")
        row_top = layout.outer_margin + layout.header_height + row_index * layout.row_height
        row_bottom = row_top + layout.row_height
        background = "#f8fafc" if row_index % 2 == 0 else "#ffffff"
        draw.rectangle((layout.outer_margin, row_top, layout.canvas_width - layout.outer_margin, row_bottom), fill=background)

        label = f"{reference.move_id:02d}  {reference.move_name}"
        draw.text((layout.outer_margin + 10, row_top + 72), label, font=row_font, fill="#111827")

        reference_index = boundary_time_to_source_frame(reference.end_time, reference_metadata)
        target_gt_index = boundary_time_to_source_frame(target.end_time, target_metadata)
        cell_frames = (
            reference_frames[reference_index],
            target_frames[target_gt_index],
            target_frames[prediction.source_frame_index],
        )
        footer_texts = (
            f"GT {reference.end_time:.1f}s | src {reference_index}",
            f"GT {target.end_time:.1f}s | src {target_gt_index}",
            f"DTW {prediction.end_time:.1f}s | src {prediction.source_frame_index}",
        )
        for column_index, (frame, footer) in enumerate(zip(cell_frames, footer_texts)):
            cell_left = first_column_x + column_index * layout.cell_width
            frame_left = cell_left + (layout.cell_width - layout.frame_width) // 2
            frame_top = row_top + 7
            fitted = fit_frame(frame, layout.frame_width, layout.frame_height)
            canvas.paste(fitted, (frame_left, frame_top))
            draw.rectangle(
                (frame_left, frame_top, frame_left + layout.frame_width, frame_top + layout.frame_height),
                outline="#94a3b8",
                width=1,
            )
            footer_top = frame_top + layout.frame_height + 2
            draw_centered_text(
                draw,
                (cell_left, footer_top, cell_left + layout.cell_width, row_bottom - 2),
                footer,
                small_font,
                "#334155",
            )

        error_color = "#15803d" if abs(prediction.error_seconds) <= 0.5 else "#b91c1c"
        error_text = f"error {prediction.error_seconds:+.1f}s"
        error_box = draw.textbbox((0, 0), error_text, font=detail_font)
        error_width = error_box[2] - error_box[0]
        draw.text(
            (layout.outer_margin + layout.row_label_width - error_width - 10, row_top + 98),
            error_text,
            font=detail_font,
            fill=error_color,
        )
        draw.line(
            (layout.outer_margin, row_bottom, layout.canvas_width - layout.outer_margin, row_bottom),
            fill="#cbd5e1",
            width=1,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=93, subsampling=0, optimize=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--dtw-root", type=Path, default=DEFAULT_DTW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-video-id", default=DEFAULT_REFERENCE_ID)
    parser.add_argument("--target-video-id", action="append", help="Target video ID; repeatable")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ground_truth = load_ground_truth(args.ground_truth)
    errors = load_end_errors(args.comparison)
    target_ids = tuple(args.target_video_id or DEFAULT_TARGET_IDS)
    if args.reference_video_id not in ground_truth:
        raise KeyError(f"Ground Truth is missing reference video {args.reference_video_id}.")

    reference_video = args.video_root / f"{args.reference_video_id}.mp4"
    reference_metadata = inspect_video(reference_video)
    reference_gt = ground_truth[args.reference_video_id]
    reference_indices = [
        boundary_time_to_source_frame(boundary.end_time, reference_metadata)
        for boundary in reference_gt
    ]
    reference_frames = read_video_frames(reference_video, reference_indices)

    for target_video_id in target_ids:
        if target_video_id not in ground_truth:
            raise KeyError(f"Ground Truth is missing target video {target_video_id}.")
        target_video = args.video_root / f"{target_video_id}.mp4"
        target_metadata = inspect_video(target_video)
        target_gt = ground_truth[target_video_id]
        predictions = load_predicted_boundaries(args.dtw_root, target_video_id, errors)
        target_gt_indices = [
            boundary_time_to_source_frame(boundary.end_time, target_metadata)
            for boundary in target_gt
        ]
        target_indices = target_gt_indices + [boundary.source_frame_index for boundary in predictions]
        target_frames = read_video_frames(target_video, target_indices)
        output_path = args.output_dir / f"{target_video_id}_boundary_comparison.jpg"
        render_target_grid(
            output_path,
            reference_video_id=args.reference_video_id,
            target_video_id=target_video_id,
            reference_gt=reference_gt,
            target_gt=target_gt,
            predictions=predictions,
            reference_metadata=reference_metadata,
            target_metadata=target_metadata,
            reference_frames=reference_frames,
            target_frames=target_frames,
        )
        print(f"{target_video_id}: {output_path}")


if __name__ == "__main__":
    main()
