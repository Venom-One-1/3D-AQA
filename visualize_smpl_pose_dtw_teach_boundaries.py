#!/usr/bin/env python
"""Visualize reference, optional target GT, and SMPL-DTW end boundaries."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from aqa3d.smpl_pose_segmentation import load_gold_boundaries
from export_tas_ground_truth_comparison import (
    build_ground_truth_segments,
    load_trim_offsets,
)
from export_tas_reference_annotations import load_point_labels
from visualize_smpl_pose_dtw_boundaries import (
    GroundTruthBoundary,
    PredictedBoundary,
    load_predicted_boundaries,
)
from visualize_tas_boundary_frames import (
    VideoMetadata,
    boundary_time_to_source_frame,
    draw_centered_text,
    fit_frame,
    inspect_video,
    read_video_frames,
    require_24_moves,
)


DEFAULT_REFERENCE_ID = "BV1WE411W7JB"
DEFAULT_REFERENCE_VIDEO = Path(
    "/home/sqw/VisualSearch/aqa/teach_trimmed/BV1WE411W7JB.mp4"
)
DEFAULT_REFERENCE_BOUNDARIES = Path(
    "/home/sqw/VisualSearch/aqa/teach_trimmed/BV1WE411W7JB_5FPS_Boundary.txt"
)
DEFAULT_ANNOTATION_PATH = Path(
    "/home/sqw/Projects/annotation-tool/annotations/"
    "instruction_2026-07-28_13.35.36.txt"
)
DEFAULT_TRIM_MANIFEST = Path(
    "/home/sqw/VisualSearch/aqa/teach_trimmed/trim_manifest.csv"
)
DEFAULT_SAMPLE_FPS = 5.0
CHINESE_FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
FALLBACK_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


@dataclass(frozen=True)
class Layout:
    column_count: int
    row_label_width: int = 190
    cell_width: int = 286
    frame_width: int = 266
    frame_height: int = 150
    header_height: int = 88
    row_height: int = 180
    outer_margin: int = 12

    @property
    def canvas_width(self) -> int:
        return (
            self.outer_margin * 2
            + self.row_label_width
            + self.cell_width * self.column_count
        )

    @property
    def canvas_height(self) -> int:
        return self.outer_margin * 2 + self.header_height + self.row_height * 24


def load_cjk_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (CHINESE_FONT_PATH, FALLBACK_FONT_PATH):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def load_reference_boundaries(
    path: Path,
    sample_fps: float,
) -> list[GroundTruthBoundary]:
    gold = load_gold_boundaries(path, sample_fps)
    return [
        GroundTruthBoundary(
            move_id=item.move_id,
            move_name=item.move_name,
            end_time=item.end_time_seconds,
        )
        for item in gold
    ]


def load_optional_target_ground_truth(
    annotation_path: Path,
    trim_manifest: Path,
    video_id: str,
    sample_fps: float,
) -> list[GroundTruthBoundary] | None:
    """Return trimmed-timeline GT only when all 24 move IDs are annotated."""
    try:
        labels = load_point_labels(annotation_path, video_id)
    except ValueError:
        return None
    move_ids = {
        label.tag_id
        for label in labels
        if label.state == 1 and 1 <= label.tag_id <= 24
    }
    if move_ids != set(range(1, 25)):
        return None

    offsets = load_trim_offsets(trim_manifest)
    if video_id not in offsets:
        raise KeyError(f"Missing trim offset for fully annotated video {video_id}.")
    segments = build_ground_truth_segments(
        annotation_path,
        video_id,
        offsets[video_id],
        sample_fps,
    )
    return [
        GroundTruthBoundary(
            move_id=item.move_id,
            move_name=item.move_name,
            end_time=item.ground_truth_end_time,
        )
        for item in segments
    ]


def render_boundary_grid(
    output_path: Path,
    *,
    reference_video_id: str,
    target_video_id: str,
    reference_boundaries: list[GroundTruthBoundary],
    target_ground_truth: list[GroundTruthBoundary] | None,
    predictions: list[PredictedBoundary],
    reference_metadata: VideoMetadata,
    target_metadata: VideoMetadata,
    reference_frames: dict[int, Image.Image],
    target_frames: dict[int, Image.Image],
) -> None:
    require_24_moves(reference_video_id, reference_boundaries)
    require_24_moves(target_video_id, predictions)
    if target_ground_truth is not None:
        require_24_moves(target_video_id, target_ground_truth)

    column_count = 3 if target_ground_truth is not None else 2
    layout = Layout(column_count=column_count)
    canvas = Image.new("RGB", (layout.canvas_width, layout.canvas_height), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    title_font = load_cjk_font(22)
    header_font = load_cjk_font(16)
    row_font = load_cjk_font(14)
    detail_font = load_cjk_font(11)

    draw.text(
        (layout.outer_margin, layout.outer_margin),
        f"24-form boundary comparison | target: {target_video_id}",
        font=title_font,
        fill="#111827",
    )
    headers = [f"Reference GT | {reference_video_id}"]
    if target_ground_truth is not None:
        headers.append(f"Target GT | {target_video_id}")
    headers.append(f"SMPL Pose DTW | {target_video_id}")
    first_column_x = layout.outer_margin + layout.row_label_width
    header_top = layout.outer_margin + 38
    for column_index, header in enumerate(headers):
        cell_left = first_column_x + column_index * layout.cell_width
        draw_centered_text(
            draw,
            (
                cell_left,
                header_top,
                cell_left + layout.cell_width,
                layout.outer_margin + layout.header_height,
            ),
            header,
            header_font,
            "#1f2937",
        )

    target_gt_by_move = (
        {item.move_id: item for item in target_ground_truth}
        if target_ground_truth is not None
        else {}
    )
    for row_index, (reference, prediction) in enumerate(
        zip(reference_boundaries, predictions)
    ):
        if reference.move_id != prediction.move_id:
            raise ValueError(
                f"Move mismatch at row {row_index}: reference={reference.move_id}, "
                f"prediction={prediction.move_id}."
            )
        target_truth = target_gt_by_move.get(reference.move_id)
        row_top = layout.outer_margin + layout.header_height + row_index * layout.row_height
        row_bottom = row_top + layout.row_height
        background = "#f8fafc" if row_index % 2 == 0 else "#ffffff"
        draw.rectangle(
            (layout.outer_margin, row_top, layout.canvas_width - layout.outer_margin, row_bottom),
            fill=background,
        )
        draw.text(
            (layout.outer_margin + 8, row_top + 54),
            f"{reference.move_id:02d}  {reference.move_name}",
            font=row_font,
            fill="#111827",
        )
        if target_truth is not None:
            error_seconds = prediction.end_time - target_truth.end_time
            error_color = "#15803d" if abs(error_seconds) <= 0.4 + 1e-9 else "#b91c1c"
            draw.text(
                (layout.outer_margin + 8, row_top + 82),
                f"error {error_seconds:+.1f}s",
                font=row_font,
                fill=error_color,
            )

        reference_index = boundary_time_to_source_frame(
            reference.end_time, reference_metadata
        )
        cells: list[tuple[Image.Image, str]] = [
            (
                reference_frames[reference_index],
                f"GT {reference.end_time:.1f}s | frame {reference_index + 1}",
            )
        ]
        if target_truth is not None:
            target_gt_index = boundary_time_to_source_frame(
                target_truth.end_time, target_metadata
            )
            cells.append(
                (
                    target_frames[target_gt_index],
                    f"GT {target_truth.end_time:.1f}s | frame {target_gt_index + 1}",
                )
            )
        cells.append(
            (
                target_frames[prediction.source_frame_index_0based],
                (
                    f"DTW {prediction.end_time:.1f}s | frame "
                    f"{prediction.source_frame_index_0based + 1} | "
                    f"geo {prediction.local_geodesic_degrees:.1f} deg"
                ),
            )
        )

        for column_index, (frame, footer) in enumerate(cells):
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


def visualize_target(
    *,
    video_id: str,
    target_video: Path,
    predicted_boundaries_path: Path,
    output_path: Path,
    reference_video: Path = DEFAULT_REFERENCE_VIDEO,
    reference_boundaries_path: Path = DEFAULT_REFERENCE_BOUNDARIES,
    annotation_path: Path = DEFAULT_ANNOTATION_PATH,
    trim_manifest: Path = DEFAULT_TRIM_MANIFEST,
    reference_video_id: str = DEFAULT_REFERENCE_ID,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
) -> bool:
    for path in (
        target_video,
        predicted_boundaries_path,
        reference_video,
        reference_boundaries_path,
        annotation_path,
        trim_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    reference_boundaries = load_reference_boundaries(
        reference_boundaries_path, sample_fps
    )
    target_ground_truth = load_optional_target_ground_truth(
        annotation_path,
        trim_manifest,
        video_id,
        sample_fps,
    )
    predictions = load_predicted_boundaries(predicted_boundaries_path, video_id)
    reference_metadata = inspect_video(reference_video)
    target_metadata = inspect_video(target_video)

    reference_indices = [
        boundary_time_to_source_frame(item.end_time, reference_metadata)
        for item in reference_boundaries
    ]
    target_indices = [item.source_frame_index_0based for item in predictions]
    if target_ground_truth is not None:
        target_indices.extend(
            boundary_time_to_source_frame(item.end_time, target_metadata)
            for item in target_ground_truth
        )
    invalid = [index for index in target_indices if not 0 <= index < target_metadata.frame_count]
    if invalid:
        raise IndexError(
            f"Target frames outside 0..{target_metadata.frame_count - 1}: {invalid[:10]}"
        )

    render_boundary_grid(
        output_path,
        reference_video_id=reference_video_id,
        target_video_id=video_id,
        reference_boundaries=reference_boundaries,
        target_ground_truth=target_ground_truth,
        predictions=predictions,
        reference_metadata=reference_metadata,
        target_metadata=target_metadata,
        reference_frames=read_video_frames(reference_video, reference_indices),
        target_frames=read_video_frames(target_video, target_indices),
    )
    return target_ground_truth is not None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--target-video", type=Path, required=True)
    parser.add_argument("--predicted-boundaries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-video-id", default=DEFAULT_REFERENCE_ID)
    parser.add_argument("--reference-video", type=Path, default=DEFAULT_REFERENCE_VIDEO)
    parser.add_argument(
        "--reference-boundaries", type=Path, default=DEFAULT_REFERENCE_BOUNDARIES
    )
    parser.add_argument("--annotation-path", type=Path, default=DEFAULT_ANNOTATION_PATH)
    parser.add_argument("--trim-manifest", type=Path, default=DEFAULT_TRIM_MANIFEST)
    parser.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    has_ground_truth = visualize_target(
        video_id=args.video_id,
        target_video=args.target_video.expanduser().absolute(),
        predicted_boundaries_path=args.predicted_boundaries.expanduser().absolute(),
        output_path=args.output.expanduser().absolute(),
        reference_video=args.reference_video.expanduser().absolute(),
        reference_boundaries_path=args.reference_boundaries.expanduser().absolute(),
        annotation_path=args.annotation_path.expanduser().absolute(),
        trim_manifest=args.trim_manifest.expanduser().absolute(),
        reference_video_id=args.reference_video_id,
        sample_fps=args.sample_fps,
    )
    column_count = 3 if has_ground_truth else 2
    print(
        f"[{args.video_id}] saved {column_count}-column visualization: {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
