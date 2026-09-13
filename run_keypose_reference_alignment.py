#!/usr/bin/env python
"""Align sparse KeyPose images to a tracked teaching video with per-move SMPL DTW."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from aqa3d.keypose_alignment import (
    KeyPoseImage,
    MoveBoundary,
    align_keyposes_to_reference_segment,
    discover_keypose_images,
    format_timestamp,
    load_move_boundaries,
    time_interval_to_inclusive_frames,
)
from aqa3d.tracking import load_stitched_primary_track
from export_tas_ground_truth_comparison import build_ground_truth_segments


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO_ID = "i8kMrJmAfjU"
DEFAULT_KEYPOSE_ROOT = Path("/home/sqw/VisualSearch/aqa/FrameData/KeyPose")
DEFAULT_VIDEO_ROOT = Path("/home/sqw/VisualSearch/aqa/teach_trimmed")
DEFAULT_ORIGINAL_VIDEO_ROOT = Path("/home/sqw/VisualSearch/aqa/teach")
DEFAULT_TRACKING_ROOT = Path("/home/sqw/VisualSearch/aqa/Tracking/teach_trimmed")
DEFAULT_GROUND_TRUTH = PROJECT_ROOT / "tas_ground_truth" / "ground_truth_segments.csv"
DEFAULT_TRIM_MANIFEST = Path("/home/sqw/VisualSearch/aqa/teach_trimmed/trim_manifest.csv")
DEFAULT_KEYPOSE_ARTIFACT_ROOT = Path(
    "/home/sqw/VisualSearch/aqa/keypose_alignment_results/i8kMrJmAfjU"
)
DEFAULT_OUTPUT_BASE = Path("/home/sqw/VisualSearch/aqa/keypose_alignment_results")
DEFAULT_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def _video_metadata(path: Path) -> tuple[float, int, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0 or frame_count <= 0:
        raise ValueError(f"Invalid video metadata for {path}.")
    return fps, frame_count, width, height


def _load_trim_offset(path: Path, video_id: str) -> float:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["video_id"] == video_id]
    if len(rows) != 1:
        raise ValueError(f"Expected one trim manifest row for {video_id}, got {len(rows)}.")
    return float(rows[0]["start_time"])


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_video_frames(path: Path, frame_indices: list[int]) -> dict[int, np.ndarray]:
    requested = sorted(set(int(index) for index in frame_indices))
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    frames: dict[int, np.ndarray] = {}
    for frame_index in requested:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Cannot decode frame {frame_index} from {path}.")
        frames[frame_index] = frame
    capture.release()
    return frames


def _letterbox(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_width, target_height = size
    result = Image.new("RGB", size, "white")
    source = image.convert("RGB")
    scale = min(target_width / source.width, target_height / source.height)
    resized = source.resize(
        (
            max(1, int(round(source.width * scale))),
            max(1, int(round(source.height * scale))),
        ),
        Image.Resampling.LANCZOS,
    )
    result.paste(
        resized,
        ((target_width - resized.width) // 2, (target_height - resized.height) // 2),
    )
    return result


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if DEFAULT_FONT.is_file():
        return ImageFont.truetype(str(DEFAULT_FONT), size=size)
    return ImageFont.load_default()


def _build_match_grid(
    output_path: Path,
    video_id: str,
    move_items: list[KeyPoseImage],
    rows: list[dict],
    matched_frames: dict[int, np.ndarray],
    overlay_root: Path,
) -> None:
    image_size = (300, 168)
    header_height = 44
    label_height = 72
    row_height = label_height + image_size[1] + 12
    margin = 14
    gap = 12
    width = margin * 2 + image_size[0] * 3 + gap * 2
    height = header_height + margin + row_height * len(rows)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(19)
    label_font = _font(15)
    small_font = _font(13)
    headers = ("KeyPose input", "KeyPose 3D reconstruction", f"{video_id} match")
    for column, label in enumerate(headers):
        x = margin + column * (image_size[0] + gap)
        draw.text((x, 8), label, fill="black", font=title_font)

    for row_index, (item, row) in enumerate(zip(move_items, rows)):
        top = header_height + margin + row_index * row_height
        image_top = top + label_height
        source = Image.open(item.image_path)
        overlay_path = overlay_root / item.relative_path
        overlay = Image.open(overlay_path) if overlay_path.is_file() else source
        matched_bgr = matched_frames[int(row["matched_trimmed_frame_0based"])]
        matched = Image.fromarray(cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2RGB))
        images = (source, overlay, matched)
        for column, image in enumerate(images):
            x = margin + column * (image_size[0] + gap)
            canvas.paste(_letterbox(image, image_size), (x, image_top))

        left_x = margin
        middle_x = margin + image_size[0] + gap
        right_x = margin + 2 * (image_size[0] + gap)
        draw.text(
            (left_x, top),
            (
                f"Move {item.move_id:02d} | KeyPose {item.keypose_order}\n"
                f"{item.relative_path.name} | source frame {item.source_frame_number}"
            ),
            fill="black",
            font=label_font,
            spacing=2,
        )
        draw.text(
            (middle_x, top),
            (
                f"Detection score {float(row['detection_score']):.3f}\n"
                "SMPL local body_pose (23 joints)"
            ),
            fill="black",
            font=small_font,
            spacing=2,
        )
        draw.text(
            (right_x, top),
            (
                f"trim {row['trimmed_timestamp']} | frame "
                f"{row['matched_trimmed_frame_0based']}\n"
                f"original {row['original_timestamp']} | "
                f"geodesic {float(row['local_geodesic_degrees']):.2f} deg"
            ),
            fill="black",
            font=small_font,
            spacing=2,
        )
        separator_y = top + row_height - 2
        draw.line((margin, separator_y, width - margin, separator_y), fill=(215, 215, 215))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=94, subsampling=0)


def _plot_dtw_diagnostic(
    output_path: Path,
    local_costs: np.ndarray,
    path: np.ndarray,
    selected_targets: np.ndarray,
    move_name: str,
    fps: float,
) -> None:
    costs_degrees = np.degrees(local_costs)
    path_target = path[:, 0]
    path_keypose = path[:, 1]
    figure, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    image = axis.imshow(
        costs_degrees.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(0.0, (len(local_costs) - 1) / fps, -0.5, local_costs.shape[1] - 0.5),
        cmap="viridis",
        vmax=float(np.percentile(costs_degrees, 95.0)),
    )
    axis.plot(path_target / fps, path_keypose, color="white", linewidth=1.3, label="DTW path")
    axis.scatter(
        selected_targets / fps,
        np.arange(len(selected_targets)),
        color="#e53935",
        edgecolors="white",
        linewidths=0.7,
        s=42,
        label="selected minimum local cost",
    )
    axis.set(
        title=f"Per-move SMPL geodesic DTW: {move_name}",
        xlabel="i8k trimmed move-relative time (seconds)",
        ylabel="KeyPose order (zero-based)",
    )
    axis.legend(loc="upper left")
    figure.colorbar(image, ax=axis, label="Mean local geodesic distance (degrees)")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _load_keypose_archive(
    path: Path,
    items: list[KeyPoseImage],
) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"KeyPose SMPL archive does not exist: {path}. "
            "Run export_keypose_smpl.py first."
        )
    with np.load(path) as archive:
        relative_paths = archive["relative_paths"].astype(str)
        expected = np.asarray([str(item.relative_path) for item in items])
        if not np.array_equal(relative_paths, expected):
            raise ValueError(
                "KeyPose archive image order does not match the current KeyPose directory."
            )
        body_poses = archive["body_poses"].astype(np.float64)
        detection_scores = archive["detection_scores"].astype(np.float64)
    if body_poses.shape != (len(items), 23, 3, 3):
        raise ValueError(f"Expected KeyPose poses shaped (K, 23, 3, 3), got {body_poses.shape}.")
    return body_poses, detection_scores


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", default=DEFAULT_VIDEO_ID)
    parser.add_argument("--keypose-root", type=Path, default=DEFAULT_KEYPOSE_ROOT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument(
        "--original-video-root",
        type=Path,
        default=DEFAULT_ORIGINAL_VIDEO_ROOT,
    )
    parser.add_argument("--tracking-root", type=Path, default=DEFAULT_TRACKING_ROOT)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument(
        "--annotation-path",
        type=Path,
        help="annotation-tool file used instead of --ground-truth",
    )
    parser.add_argument("--trim-manifest", type=Path, default=DEFAULT_TRIM_MANIFEST)
    parser.add_argument(
        "--keypose-smpl-archive",
        type=Path,
        default=DEFAULT_KEYPOSE_ARTIFACT_ROOT / "keypose_smpl.npz",
    )
    parser.add_argument(
        "--keypose-overlay-root",
        type=Path,
        default=DEFAULT_KEYPOSE_ARTIFACT_ROOT / "keypose_reconstruction_overlays",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Defaults to keypose_alignment_results/<video-id>",
    )
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--dtw-coefficient", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output_root is None:
        args.output_root = DEFAULT_OUTPUT_BASE / args.video_id
    args.output_root.mkdir(parents=True, exist_ok=True)
    video_path = args.video_root / f"{args.video_id}.mp4"
    original_video_path = args.original_video_root / f"{args.video_id}.mp4"
    tracking_path = (
        args.tracking_root
        / args.video_id
        / "results"
        / f"demo_{args.video_id}.pkl"
    )
    archive_path = args.keypose_smpl_archive

    items = discover_keypose_images(args.keypose_root)
    keypose_poses, detection_scores = _load_keypose_archive(archive_path, items)
    fps, frame_count, _, _ = _video_metadata(video_path)
    original_fps, original_frame_count, _, _ = _video_metadata(original_video_path)
    trim_offset = _load_trim_offset(args.trim_manifest, args.video_id)
    if args.annotation_path is None:
        boundaries = load_move_boundaries(args.ground_truth, args.video_id)
        ground_truth_source = args.ground_truth
    else:
        ground_truth_segments = build_ground_truth_segments(
            args.annotation_path,
            args.video_id,
            trim_offset,
            5.0,
        )
        _write_csv(
            args.output_root / "ground_truth_segments.csv",
            [asdict(segment) for segment in ground_truth_segments],
        )
        boundaries = [
            MoveBoundary(
                move_id=segment.move_id,
                move_name=segment.move_name,
                trimmed_start_time=segment.ground_truth_start_time,
                trimmed_end_time=segment.ground_truth_end_time,
                original_start_time=segment.source_annotated_first_active_time,
                original_end_time=segment.source_boundary_end_time,
            )
            for segment in ground_truth_segments
        ]
        ground_truth_source = args.annotation_path
    track = load_stitched_primary_track(tracking_path)
    if track.frame_numbers[0] != 1 or track.frame_numbers[-1] != frame_count:
        raise ValueError(
            "Reference tracking does not cover every trimmed-video frame: "
            f"{track.frame_numbers[0]}..{track.frame_numbers[-1]} vs 1..{frame_count}."
        )

    all_rows: list[dict] = []
    move_summaries: list[dict] = []
    for boundary in boundaries:
        move_indices = [
            index for index, item in enumerate(items) if item.move_id == boundary.move_id
        ]
        move_items = [items[index] for index in move_indices]
        move_keyposes = keypose_poses[move_indices]
        start_frame, end_frame = time_interval_to_inclusive_frames(
            boundary.trimmed_start_time,
            boundary.trimmed_end_time,
            fps,
            frame_count,
        )
        source_frames = np.arange(start_frame, end_frame + 1, dtype=np.int64)
        reference_poses = track.at_source_frames(source_frames)
        alignment = align_keyposes_to_reference_segment(
            reference_poses,
            move_keyposes,
            chunk_size=args.chunk_size,
            coefficient=args.dtw_coefficient,
        )

        move_dir = (
            args.output_root
            / "moves"
            / f"{boundary.move_id:02d}_{move_items[0].move_name}"
        )
        move_dir.mkdir(parents=True, exist_ok=True)
        path_target = alignment.path[:, 0]
        path_keypose = alignment.path[:, 1]
        path_costs = alignment.local_costs[path_target, path_keypose]
        np.savez_compressed(
            move_dir / "dtw_alignment.npz",
            local_cost_matrix_radians=alignment.local_costs.astype(np.float32),
            accumulated_cost_matrix=alignment.accumulated_costs.astype(np.float32),
            path_reference_segment_indices_0based=path_target.astype(np.int32),
            path_keypose_indices_0based=path_keypose.astype(np.int16),
            path_reference_source_frames_0based=source_frames[path_target].astype(np.int32),
            path_local_geodesic_radians=path_costs.astype(np.float32),
            path_local_geodesic_degrees=np.degrees(path_costs).astype(np.float32),
            reference_segment_source_frames_0based=source_frames.astype(np.int32),
            keypose_source_frame_numbers=np.asarray(
                [item.source_frame_number for item in move_items],
                dtype=np.int32,
            ),
        )

        move_rows: list[dict] = []
        selected_targets = np.asarray(
            [match.target_index for match in alignment.matches],
            dtype=np.int64,
        )
        for local_index, (item, match) in enumerate(
            zip(move_items, alignment.matches)
        ):
            matched_trimmed_frame = int(source_frames[match.target_index])
            trimmed_time = matched_trimmed_frame / fps
            original_time = trim_offset + trimmed_time
            original_frame = int(round(original_time * original_fps))
            original_frame = int(np.clip(original_frame, 0, original_frame_count - 1))
            row = {
                "video_id": args.video_id,
                "move_id": boundary.move_id,
                "move_name": boundary.move_name,
                "keypose_order": item.keypose_order,
                "keypose_source_frame_number": item.source_frame_number,
                "keypose_image": str(item.image_path),
                "keypose_relative_path": str(item.relative_path),
                "detection_score": float(detection_scores[move_indices[local_index]]),
                "matched_segment_frame_index_0based": match.target_index,
                "matched_trimmed_frame_0based": matched_trimmed_frame,
                "matched_trimmed_frame_1based": matched_trimmed_frame + 1,
                "trimmed_time_seconds": trimmed_time,
                "trimmed_timestamp": format_timestamp(trimmed_time),
                "matched_original_frame_0based": original_frame,
                "matched_original_frame_1based": original_frame + 1,
                "original_time_seconds": original_time,
                "original_timestamp": format_timestamp(original_time),
                "dtw_candidate_count": match.candidate_count,
                "local_geodesic_radians": match.local_cost,
                "local_geodesic_degrees": float(np.degrees(match.local_cost)),
                "move_trimmed_start_time": boundary.trimmed_start_time,
                "move_trimmed_end_time": boundary.trimmed_end_time,
                "move_original_start_time": boundary.original_start_time,
                "move_original_end_time": boundary.original_end_time,
                "selection_policy": "minimum_local_geodesic_on_per_move_dtw_path",
            }
            move_rows.append(row)
            all_rows.append(row)

        _write_csv(move_dir / "matches.csv", move_rows)
        matched_frames = _read_video_frames(
            video_path,
            [int(row["matched_trimmed_frame_0based"]) for row in move_rows],
        )
        _build_match_grid(
            move_dir / "match_grid.jpg",
            args.video_id,
            move_items,
            move_rows,
            matched_frames,
            args.keypose_overlay_root,
        )
        _plot_dtw_diagnostic(
            move_dir / "dtw_diagnostic.png",
            alignment.local_costs,
            alignment.path,
            selected_targets,
            boundary.move_name,
            fps,
        )
        move_summaries.append(
            {
                "move_id": boundary.move_id,
                "move_name": boundary.move_name,
                "keypose_count": len(move_items),
                "reference_start_frame_0based": start_frame,
                "reference_end_frame_0based": end_frame,
                "reference_duration_seconds": (end_frame - start_frame) / fps,
                "dtw_path_length": len(alignment.path),
                "dtw_path_mean_geodesic_degrees": float(
                    np.degrees(np.mean(path_costs))
                ),
                "selected_mean_geodesic_degrees": float(
                    np.mean([row["local_geodesic_degrees"] for row in move_rows])
                ),
                "selected_frames_nondecreasing": bool(
                    np.all(np.diff(selected_targets) >= 0)
                ),
            }
        )
        print(
            f"[{boundary.move_id:02d}/24] {boundary.move_name}: "
            f"{len(move_items)} KeyPoses, selected mean "
            f"{move_summaries[-1]['selected_mean_geodesic_degrees']:.2f} deg",
            flush=True,
        )

    _write_csv(args.output_root / "keypose_matches.csv", all_rows)
    _write_csv(args.output_root / "move_summary.csv", move_summaries)
    review_rows = [
        {"manual_review_priority": priority, **row}
        for priority, row in enumerate(
            sorted(
                all_rows,
                key=lambda item: float(item["local_geodesic_degrees"]),
                reverse=True,
            ),
            start=1,
        )
    ]
    _write_csv(args.output_root / "manual_review_priority.csv", review_rows)
    (args.output_root / "keypose_matches.json").write_text(
        json.dumps(all_rows, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "video_id": args.video_id,
        "keypose_root": str(args.keypose_root),
        "keypose_smpl_archive": str(archive_path),
        "trimmed_video": str(video_path),
        "original_video": str(original_video_path),
        "tracking_path": str(tracking_path),
        "ground_truth_path": str(ground_truth_source),
        "keypose_overlay_root": str(args.keypose_overlay_root),
        "trim_offset_seconds": trim_offset,
        "trimmed_video_fps": fps,
        "trimmed_video_frame_count": frame_count,
        "original_video_fps": original_fps,
        "original_video_frame_count": original_frame_count,
        "move_count": len(boundaries),
        "keypose_count": len(items),
        "dtw_mode": "independent_per_move_full_30fps",
        "dtw_local_cost": "mean_geodesic_radians_over_23_local_smpl_joints",
        "keypose_selection": "minimum_local_geodesic_among_dtw_path_candidates",
        "move_boundaries": [asdict(boundary) for boundary in boundaries],
        "move_summaries": move_summaries,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Matches: {args.output_root / 'keypose_matches.csv'}")
    print(f"Per-move visualizations: {args.output_root / 'moves'}")


if __name__ == "__main__":
    main()
