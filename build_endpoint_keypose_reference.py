#!/usr/bin/env python
"""Build the BV1WE411W7JB endpoint-KeyPose reference bundle."""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw

from aqa3d.reference_manifest import (
    EndpointKeypose,
    attach_tracking_availability,
    build_endpoint_keyposes,
    build_reference_manifest,
    endpoint_keypose_rows,
    load_final_technique_steps,
)
from aqa3d.smpl_dtw import inspect_video_sampling
from aqa3d.smpl_pose_segmentation import load_gold_boundaries
from aqa3d.tracking import load_stitched_primary_track
from visualize_tas_boundary_frames import fit_frame, load_font, read_video_frames


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE_VIDEO_ID = "BV1WE411W7JB"
DEFAULT_REFERENCE_SAMPLE_SEQUENCE_ID = "BV1WE411W7JB_5FPS"
DEFAULT_REFERENCE_VIDEO = Path(
    "/home/sqw/VisualSearch/aqa/teach_trimmed/BV1WE411W7JB.mp4"
)
DEFAULT_REFERENCE_TRACKING = Path(
    "/home/sqw/VisualSearch/aqa/Tracking/teach_trimmed/"
    "BV1WE411W7JB/results/demo_BV1WE411W7JB.pkl"
)
DEFAULT_GOLD_BOUNDARIES = Path(
    "/home/sqw/VisualSearch/aqa/teach_trimmed/"
    "BV1WE411W7JB_5FPS_Boundary_ft.txt"
)
DEFAULT_TECHNIQUE_FILE = PROJECT_ROOT / "TechPoint.md"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reference_data" / DEFAULT_REFERENCE_VIDEO_ID


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_endpoint_overview(
    output_path: Path,
    video_path: Path,
    keyposes: list[EndpointKeypose],
) -> None:
    columns = 4
    rows = 6
    margin = 18
    gap = 14
    card_width = 440
    card_height = 325
    frame_width = 416
    frame_height = 234
    header_height = 38
    canvas_width = margin * 2 + columns * card_width + (columns - 1) * gap
    canvas_height = 70 + margin + rows * card_height + (rows - 1) * gap
    canvas = Image.new("RGB", (canvas_width, canvas_height), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(25)
    header_font = load_font(18)
    detail_font = load_font(13)
    status_font = load_font(12)

    draw.text(
        (margin, 18),
        "BV1WE411W7JB | 24-form Gold endpoint KeyPoses",
        font=title_font,
        fill="#111827",
    )
    frame_indices = [item.source_frame_index_0based for item in keyposes]
    frames = read_video_frames(video_path, frame_indices)

    for item_index, item in enumerate(keyposes):
        row, column = divmod(item_index, columns)
        left = margin + column * (card_width + gap)
        top = 70 + margin + row * (card_height + gap)
        right = left + card_width
        bottom = top + card_height
        draw.rounded_rectangle(
            (left, top, right, bottom),
            radius=6,
            fill="#ffffff",
            outline="#cbd5e1",
            width=1,
        )
        draw.text(
            (left + 12, top + 8),
            f"{item.move_id:02d}  {item.move_name_zh}",
            font=header_font,
            fill="#111827",
        )
        name_width = draw.textbbox((0, 0), item.move_name_pinyin, font=detail_font)[2]
        draw.text(
            (right - 12 - name_width, top + 12),
            item.move_name_pinyin,
            font=detail_font,
            fill="#475569",
        )
        frame_left = left + (card_width - frame_width) // 2
        frame_top = top + header_height
        fitted = fit_frame(
            frames[item.source_frame_index_0based],
            frame_width,
            frame_height,
        )
        canvas.paste(fitted, (frame_left, frame_top))
        draw.rectangle(
            (frame_left, frame_top, frame_left + frame_width, frame_top + frame_height),
            outline="#94a3b8",
            width=1,
        )
        details = (
            f"Gold {item.boundary_time_seconds:.1f}s | 5FPS index {item.sample_index_0based} | "
            f"source frame {item.source_frame_1based}"
        )
        draw.text(
            (left + 12, frame_top + frame_height + 8),
            details,
            font=detail_font,
            fill="#334155",
        )
        status = (
            f"{item.pose_id} | tracking exact: "
            f"{'yes' if item.tracking_pose_available else 'NO'}"
        )
        draw.text(
            (left + 12, frame_top + frame_height + 31),
            status,
            font=status_font,
            fill="#15803d" if item.tracking_pose_available else "#b91c1c",
        )
        if not item.final_technique_step:
            warning = "Missing final technique step"
            draw.text(
                (left + 12, bottom - 18),
                textwrap.shorten(warning, width=55),
                font=status_font,
                fill="#b91c1c",
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=93, subsampling=0, optimize=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-video-id", default=DEFAULT_REFERENCE_VIDEO_ID)
    parser.add_argument(
        "--reference-sample-sequence-id",
        default=DEFAULT_REFERENCE_SAMPLE_SEQUENCE_ID,
    )
    parser.add_argument("--reference-video", type=Path, default=DEFAULT_REFERENCE_VIDEO)
    parser.add_argument("--reference-tracking", type=Path, default=DEFAULT_REFERENCE_TRACKING)
    parser.add_argument("--gold-boundaries", type=Path, default=DEFAULT_GOLD_BOUNDARIES)
    parser.add_argument("--technique-file", type=Path, default=DEFAULT_TECHNIQUE_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--metric-window-seconds", type=float, default=0.2)
    parser.add_argument(
        "--allow-missing-tracking-keyposes",
        action="store_true",
        help="Write the bundle even if an exact endpoint frame is absent from tracking.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reference_video = args.reference_video.expanduser().absolute()
    reference_tracking = args.reference_tracking.expanduser().absolute()
    gold_boundaries_path = args.gold_boundaries.expanduser().absolute()
    technique_path = args.technique_file.expanduser().absolute()
    output_dir = args.output_dir.expanduser().absolute()
    for path in (reference_video, reference_tracking, gold_boundaries_path, technique_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    sampling = inspect_video_sampling(reference_video, args.sample_fps)
    boundaries = load_gold_boundaries(
        gold_boundaries_path,
        args.sample_fps,
        expected_url_id=args.reference_sample_sequence_id,
    )
    final_steps = load_final_technique_steps(technique_path)
    keyposes = build_endpoint_keyposes(
        boundaries,
        sampling,
        final_steps,
        reference_video_id=args.reference_video_id,
        reference_sample_sequence_id=args.reference_sample_sequence_id,
    )
    track = load_stitched_primary_track(reference_tracking)
    keyposes = attach_tracking_availability(keyposes, track)
    missing = [item.pose_id for item in keyposes if not item.tracking_pose_available]
    if missing and not args.allow_missing_tracking_keyposes:
        raise ValueError(
            "Exact Gold endpoint frames are absent from tracking: " + ", ".join(missing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = endpoint_keypose_rows(keyposes)
    write_csv(output_dir / "endpoint_keyposes.csv", rows)
    (output_dir / "endpoint_keyposes.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = build_reference_manifest(
        keyposes=keyposes,
        sampling=sampling,
        reference_video_id=args.reference_video_id,
        reference_sample_sequence_id=args.reference_sample_sequence_id,
        reference_video_path=reference_video,
        reference_tracking_path=reference_tracking,
        gold_boundary_path=gold_boundaries_path,
        technique_path=technique_path,
        output_dir=output_dir,
        metric_window_seconds=args.metric_window_seconds,
        tracking_used_track_ids=track.used_track_ids,
    )
    (output_dir / "reference_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    render_endpoint_overview(
        output_dir / "endpoint_keypose_overview.jpg",
        reference_video,
        keyposes,
    )
    print(
        f"[{args.reference_video_id}] wrote 24 endpoint KeyPoses to {output_dir}; "
        f"tracking exact={24 - len(missing)}/24",
        flush=True,
    )


if __name__ == "__main__":
    main()
