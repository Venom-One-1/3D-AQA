#!/usr/bin/env python
"""Compare 2D-DTW and SMPL-geodesic-DTW teacher-anchor matches."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OLD_ROOT = PROJECT_ROOT / "teacher_keyframe_results"
DEFAULT_NEW_ROOT = PROJECT_ROOT / "teacher_keyframe_smpl_dtw_results"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "smpl_dtw_diagnostics"


def read_keyframe_csv(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {int(row["anchor_order"]): row for row in csv.DictReader(handle)}


def read_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_frame(video_path: Path, frame_index: int) -> tuple[np.ndarray, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    candidates = [int(frame_index)]
    fallback_start = min(int(frame_index), frame_count - 1) if frame_count > 0 else int(frame_index)
    candidates.extend(range(fallback_start, max(-1, fallback_start - 90), -1))
    seen: set[int] = set()
    for candidate in candidates:
        if candidate < 0 or candidate in seen:
            continue
        seen.add(candidate)
        capture.set(cv2.CAP_PROP_POS_FRAMES, candidate)
        ok, frame = capture.read()
        if ok:
            capture.release()
            return frame, candidate
    capture.release()
    raise ValueError(f"Cannot read frame {frame_index} or a nearby fallback from {video_path}")


def letterbox(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_width, target_height = size
    height, width = frame.shape[:2]
    scale = min(target_width / width, target_height / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_height, target_width, 3), 245, dtype=np.uint8)
    x0 = (target_width - new_width) // 2
    y0 = (target_height - new_height) // 2
    canvas[y0 : y0 + new_height, x0 : x0 + new_width] = resized
    return canvas


def labeled_tile(frame: np.ndarray, lines: list[str], size: tuple[int, int], label_height: int = 48) -> np.ndarray:
    image = letterbox(frame, size)
    label = np.full((label_height, size[0], 3), 255, dtype=np.uint8)
    y = 15
    for line in lines[:3]:
        cv2.putText(label, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20, 20, 20), 1, cv2.LINE_AA)
        y += 15
    return np.concatenate((label, image), axis=0)


def make_grid(
    rows: list[dict[str, object]],
    *,
    teacher_video: Path,
    student_video: Path,
    output_dir: Path,
    tile_size: tuple[int, int],
    rows_per_page: int,
) -> list[Path]:
    if rows_per_page <= 0:
        raise ValueError(f"rows_per_page must be positive, got {rows_per_page}.")
    rendered_rows: list[np.ndarray] = []
    for row in rows:
        rendered_rows.append(render_row(row, teacher_video=teacher_video, student_video=student_video, tile_size=tile_size))

    if not rendered_rows:
        raise ValueError("No rows to render.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_page in output_dir.glob("frame_grid_page_*.jpg"):
        old_page.unlink()
    written: list[Path] = []
    for page_index, start in enumerate(range(0, len(rendered_rows), rows_per_page), start=1):
        page = np.concatenate(rendered_rows[start : start + rows_per_page], axis=0)
        page_path = output_dir / f"frame_grid_page_{page_index:02d}.jpg"
        cv2.imwrite(str(page_path), page)
        written.append(page_path)
    if written:
        cv2.imwrite(str(output_dir / "frame_grid.jpg"), cv2.imread(str(written[0])))
    return written


def write_anchor_images(
    rows: list[dict[str, object]],
    *,
    teacher_video: Path,
    student_video: Path,
    output_dir: Path,
    tile_size: tuple[int, int],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for row in rows:
        anchor = int(row["anchor_order"])
        path = output_dir / f"anchor_{anchor:02d}.jpg"
        image = render_row(row, teacher_video=teacher_video, student_video=student_video, tile_size=tile_size)
        cv2.imwrite(str(path), image)
        written.append(path)
    return written


def render_row(
    row: dict[str, object],
    *,
    teacher_video: Path,
    student_video: Path,
    tile_size: tuple[int, int],
) -> np.ndarray:
    teacher_requested = int(row["teacher_source_frame_0based"])
    old_requested = int(row["old_student_source_frame_0based"])
    new_requested = int(row["new_student_source_frame_0based"])
    teacher_frame, teacher_shown = read_frame(teacher_video, teacher_requested)
    old_frame, old_shown = read_frame(student_video, old_requested)
    new_frame, new_shown = read_frame(student_video, new_requested)

    anchor = int(row["anchor_order"])
    delta_frame = int(row["student_source_delta_new_minus_2d"])
    old_geo = float(row["old_mean_geodesic_error_degrees"])
    new_geo = float(row["new_mean_geodesic_error_degrees"])
    delta_geo = float(row["geodesic_delta_new_minus_2d_degrees"])

    teacher_tile = labeled_tile(
        teacher_frame,
        [f"Anchor {anchor} | Teacher", _frame_label(teacher_requested, teacher_shown), ""],
        tile_size,
    )
    old_tile = labeled_tile(
        old_frame,
        ["2D-DTW student", _frame_label(old_requested, old_shown), f"geo {old_geo:.2f} deg"],
        tile_size,
    )
    new_tile = labeled_tile(
        new_frame,
        [
            "SMPL-DTW student",
            f"{_frame_label(new_requested, new_shown)} | dF {delta_frame:+d}",
            f"geo {new_geo:.2f} ({delta_geo:+.2f})",
        ],
        tile_size,
    )
    return np.concatenate((teacher_tile, old_tile, new_tile), axis=1)


def _frame_label(requested: int, shown: int) -> str:
    if requested == shown:
        return f"src {requested}"
    return f"src {requested} shown {shown}"


def compare_clip(
    old_dir: Path,
    new_dir: Path,
    output_root: Path,
    tile_size: tuple[int, int],
    rows_per_page: int,
) -> list[dict[str, object]]:
    old_summary = read_summary(old_dir / "summary.json")
    new_summary = read_summary(new_dir / "summary.json")
    old_rows = read_keyframe_csv(old_dir / "teacher_anchored_keyframes.csv")
    new_rows = read_keyframe_csv(new_dir / "teacher_anchored_keyframes.csv")
    common_anchors = sorted(set(old_rows) & set(new_rows))
    if not common_anchors:
        raise ValueError(f"No shared anchors between {old_dir} and {new_dir}")

    clip = old_dir.name
    move = clip.rsplit("_", 1)[-1]
    student_id = clip.split("_", 1)[0]
    teacher_video = Path(old_summary["teacher_video"])
    student_video = Path(old_summary["student_video"])

    comparisons: list[dict[str, object]] = []
    for anchor in common_anchors:
        old = old_rows[anchor]
        new = new_rows[anchor]
        old_student_source = int(old["student_source_frame_0based"])
        new_student_source = int(new["student_source_frame_0based"])
        old_student_sample = int(old["student_sample_index"])
        new_student_sample = int(new["student_sample_index"])
        old_geo = float(old["mean_geodesic_error_degrees"])
        new_geo = float(new["mean_geodesic_error_degrees"])
        comparisons.append(
            {
                "clip": clip,
                "move": move,
                "student_id": student_id,
                "anchor_order": anchor,
                "teacher_sample_index": int(old["teacher_sample_index"]),
                "teacher_source_frame_0based": int(old["teacher_source_frame_0based"]),
                "old_student_sample_index": old_student_sample,
                "new_student_sample_index": new_student_sample,
                "old_student_source_frame_0based": old_student_source,
                "new_student_source_frame_0based": new_student_source,
                "student_sample_delta_new_minus_2d": new_student_sample - old_student_sample,
                "student_source_delta_new_minus_2d": new_student_source - old_student_source,
                "old_mean_geodesic_error_degrees": old_geo,
                "new_mean_geodesic_error_degrees": new_geo,
                "geodesic_delta_new_minus_2d_degrees": new_geo - old_geo,
                "old_selected_local_2d_distance": float(old["selected_local_2d_distance"]),
                "new_selected_local_smpl_geodesic_distance_degrees": float(
                    new["selected_local_smpl_geodesic_distance_degrees"]
                ),
                "same_student_frame": old_student_source == new_student_source,
            }
        )

    clip_output = output_root / clip
    clip_output.mkdir(parents=True, exist_ok=True)
    write_csv(clip_output / "anchor_comparison.csv", comparisons)
    page_paths = make_grid(
        comparisons,
        teacher_video=teacher_video,
        student_video=student_video,
        output_dir=clip_output,
        tile_size=tile_size,
        rows_per_page=rows_per_page,
    )
    anchor_paths = write_anchor_images(
        comparisons,
        teacher_video=teacher_video,
        student_video=student_video,
        output_dir=clip_output / "anchors",
        tile_size=tile_size,
    )

    summary = {
        "clip": clip,
        "move": move,
        "student_id": student_id,
        "teacher_video": str(teacher_video),
        "student_video": str(student_video),
        "old_result_dir": str(old_dir),
        "new_result_dir": str(new_dir),
        "anchor_count": len(comparisons),
        "same_student_frame_count": int(sum(row["same_student_frame"] for row in comparisons)),
        "mean_abs_source_frame_delta": float(np.mean([abs(int(row["student_source_delta_new_minus_2d"])) for row in comparisons])),
        "max_abs_source_frame_delta": int(max(abs(int(row["student_source_delta_new_minus_2d"])) for row in comparisons)),
        "old_mean_geodesic_distance_degrees": float(old_summary["mean_geodesic_distance_degrees"]),
        "new_mean_geodesic_distance_degrees": float(new_summary["mean_geodesic_distance_degrees"]),
        "rows_per_page": rows_per_page,
        "page_images": [path.name for path in page_paths],
        "anchor_image_count": len(anchor_paths),
    }
    (clip_output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return comparisons


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_aggregate_reports(output_root: Path, rows: list[dict[str, object]], top_k: int = 30) -> None:
    by_clip: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_clip[str(row["clip"])].append(row)

    summaries: list[dict[str, object]] = []
    for clip, clip_rows in sorted(by_clip.items()):
        source_deltas = np.asarray([int(row["student_source_delta_new_minus_2d"]) for row in clip_rows], dtype=np.float64)
        geo_deltas = np.asarray([float(row["geodesic_delta_new_minus_2d_degrees"]) for row in clip_rows], dtype=np.float64)
        summaries.append(
            {
                "clip": clip,
                "move": clip_rows[0]["move"],
                "student_id": clip_rows[0]["student_id"],
                "anchor_count": len(clip_rows),
                "same_student_frame_count": int(sum(bool(row["same_student_frame"]) for row in clip_rows)),
                "mean_abs_source_frame_delta": float(np.mean(np.abs(source_deltas))),
                "median_abs_source_frame_delta": float(np.median(np.abs(source_deltas))),
                "max_abs_source_frame_delta": int(np.max(np.abs(source_deltas))),
                "new_lower_geodesic_count": int(np.sum(geo_deltas < 0)),
                "mean_geodesic_delta_new_minus_2d_degrees": float(np.mean(geo_deltas)),
            }
        )
    write_csv(output_root / "clip_summary.csv", summaries)

    shifted = sorted(rows, key=lambda row: abs(int(row["student_source_delta_new_minus_2d"])), reverse=True)
    write_csv(output_root / "largest_frame_shifts.csv", shifted[:top_k])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT, help="2D-DTW teacher-keyframe result root.")
    parser.add_argument("--new-root", type=Path, default=DEFAULT_NEW_ROOT, help="SMPL-DTW teacher-keyframe result root.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--clip", action="append", help="Clip stem to diagnose; repeatable. Defaults to all shared clips.")
    parser.add_argument("--tile-width", type=int, default=220)
    parser.add_argument("--tile-height", type=int, default=124)
    parser.add_argument("--rows-per-page", type=int, default=999, help="Use a large value to keep each clip in one image.")
    parser.add_argument("--top-shifts", type=int, default=30)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    old_dirs = {path.name: path for path in args.old_root.iterdir() if (path / "summary.json").is_file()}
    new_dirs = {path.name: path for path in args.new_root.iterdir() if (path / "summary.json").is_file()}
    clips = sorted(set(old_dirs) & set(new_dirs))
    if args.clip:
        requested = set(args.clip)
        clips = [clip for clip in clips if clip in requested]
    if not clips:
        raise ValueError("No shared result clips matched the request.")

    all_rows: list[dict[str, object]] = []
    for clip in clips:
        rows = compare_clip(
            old_dirs[clip],
            new_dirs[clip],
            args.output_root,
            tile_size=(args.tile_width, args.tile_height),
            rows_per_page=args.rows_per_page,
        )
        all_rows.extend(rows)
        print(f"{clip}: wrote {args.output_root / clip / 'anchor_comparison.csv'} and paged frame grids")

    write_csv(args.output_root / "all_anchor_comparison.csv", all_rows)
    write_aggregate_reports(args.output_root, all_rows, top_k=args.top_shifts)
    print(f"Processed {len(clips)} clips and {len(all_rows)} anchors.")


if __name__ == "__main__":
    main()
