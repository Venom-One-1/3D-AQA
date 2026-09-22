#!/usr/bin/env python
"""Transfer the 24 manifest endpoint events to a tracked student video."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from aqa3d.endpoint_transfer import (
    manifest_boundaries, resolve_tracking_samples, transferred_keypose_rows,
)
from aqa3d.smpl_dtw import (
    inspect_video_sampling, pairwise_geodesic_costs, dtw_from_cost_matrix,
    backtrack_dtw_path, select_reference_frame_matches,
)
from aqa3d.smpl_pose_segmentation import build_mapped_boundaries, build_mapped_segments
from aqa3d.tracking import load_stitched_primary_track
from run_smpl_pose_dtw_segmentation import write_dict_csv, save_dtw_path, plot_diagnostics
from visualize_tas_boundary_frames import fit_frame, load_font, read_video_frames


PROJECT_ROOT = Path(__file__).resolve().parent


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def render_keyposes(output_dir: Path, rows: list[dict], reference_video: Path,
                    student_video: Path) -> None:
    """Keep full source images below separate labels, including portrait video."""
    frames = {
        "reference": read_video_frames(reference_video, [r["reference_source_frame_0based"] for r in rows]),
        "student": read_video_frames(student_video, [r["student_source_frame_0based"] for r in rows]),
    }
    width, row_height, header_height = 1160, 420, 64
    canvas = Image.new("RGB", (width, header_height + row_height * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, detail_font = load_font(23), load_font(17)
    draw.text((16, 14), "Reference Gold endpoint / Student DTW endpoint", font=title_font, fill="black")
    frame_dir = output_dir / "keypose_frames"
    frame_dir.mkdir(exist_ok=True)
    for i, row in enumerate(rows):
        top = header_height + i * row_height
        draw.line((0, top, width, top), fill="#bbbbbb", width=1)
        draw.text((16, top + 7), f'{row["move_id"]:02d} {row["move_name_zh"]} | {row["pose_id"]}',
                  font=title_font, fill="black")
        for column, prefix in enumerate(("reference", "student")):
            left = 12 + 580 * column
            source = row[f"{prefix}_source_frame_0based"]
            frame = frames[prefix][source]
            image_path = frame_dir / f'{row["move_id"]:02d}_{prefix}.jpg'
            frame.save(image_path, quality=95)
            row[f"{prefix}_image"] = str(image_path.relative_to(output_dir))
            label = (f'{prefix} {row[f"{prefix}_boundary_time_seconds"]:.3f}s | '
                     f'frame0 {source} | PHALP {row[f"{prefix}_tracking_actual_phalp_frame_1based"]}')
            draw.text((left, top + 44), label, font=detail_font, fill="#333333")
            canvas.paste(fit_frame(frame, 556, 278), (left, top + 72))
        detail = f'Geodesic {row["local_geodesic_degrees"]:.2f} deg | candidates {row["candidate_count"]}'
        if row["review_required"]:
            detail += " | REVIEW: " + row["diagnostic_flags"]
        lines = [""]
        for character in detail:
            if draw.textlength(lines[-1] + character, font=detail_font) > width - 32:
                lines.append("")
            lines[-1] += character
        if len(lines) > 3:
            raise ValueError("Diagnostic label exceeds reserved space.")
        for line_index, line in enumerate(lines):
            draw.text((16, top + 355 + 20 * line_index), line, font=detail_font,
                      fill="#a12525" if row["review_required"] else "#333333")
    canvas.save(output_dir / "boundary_frames.jpg", quality=95)


def run(args: argparse.Namespace, output_dir: Path) -> dict:
    started = time.perf_counter()
    manifest_path = args.reference_manifest.expanduser().absolute()
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    reference = manifest["reference"]
    reference_video = Path(reference["video_path"]).expanduser()
    reference_tracking = Path(reference["tracking_path"]).expanduser()
    # Relative paths in a portable manifest are relative to that manifest.
    if not reference_video.is_absolute():
        reference_video = manifest_path.parent / reference_video
    if not reference_tracking.is_absolute():
        reference_tracking = manifest_path.parent / reference_tracking
    sample_fps = float(reference["sample_fps"])
    if sample_fps != 5.0:
        raise ValueError("This endpoint workflow requires a 5 FPS manifest.")
    ref_sampling = inspect_video_sampling(reference_video, sample_fps)
    gold = manifest_boundaries(manifest, ref_sampling)
    target_sampling = inspect_video_sampling(args.input_video, sample_fps)
    ref_track = load_stitched_primary_track(reference_tracking)
    target_track = load_stitched_primary_track(args.input_tracking)
    ref_positions = resolve_tracking_samples(ref_track, ref_sampling, args.max_tracking_gap_seconds)
    target_positions = resolve_tracking_samples(target_track, target_sampling, args.max_tracking_gap_seconds)
    print(f'[{output_dir.name}] reference={ref_sampling.sample_count}, '
          f'student={target_sampling.sample_count}; computing SMPL costs', flush=True)
    costs = pairwise_geodesic_costs(target_track.body_poses[target_positions],
                                   ref_track.body_poses[ref_positions],
                                   chunk_size=args.pairwise_chunk_size)
    distance, _, accumulated = dtw_from_cost_matrix(costs)
    path = backtrack_dtw_path(accumulated)
    save_dtw_path(output_dir, path, costs, ref_sampling, target_sampling)
    matches = select_reference_frame_matches(costs, path, [b.sample_index_0based for b in gold])
    # Preserve selected candidates even when duplicate boundaries prevent segmentation.
    write_json(output_dir / "selected_matches.json", [asdict(m) for m in matches])
    boundaries = build_mapped_boundaries(output_dir.name, gold, matches, ref_sampling, target_sampling)
    segments = build_mapped_segments(output_dir.name, boundaries, target_sampling)
    rows = transferred_keypose_rows(manifest, boundaries, path, ref_sampling, target_sampling,
                                    ref_track, target_track, ref_positions, target_positions,
                                    args.review_geodesic_degrees)
    plot_diagnostics(output_dir / "dtw_diagnostics.png", output_dir.name, costs, path,
                     gold, boundaries, ref_sampling, target_sampling)
    print(f'[{output_dir.name}] 24 ordered endpoints; rendering full-frame comparison', flush=True)
    render_keyposes(output_dir, rows, reference_video, args.input_video)
    write_dict_csv(output_dir / "endpoint_keyposes.csv", rows)
    write_json(output_dir / "endpoint_keyposes.json", rows)
    write_dict_csv(output_dir / "boundaries.csv", [asdict(b) for b in boundaries])
    write_dict_csv(output_dir / "segments.csv", [asdict(s) for s in segments])
    # Also retain actual tracking-frame provenance for every 5 FPS sample.
    np.savez_compressed(
        output_dir / "sample_tracking_provenance.npz",
        reference_source_indices_0based=ref_sampling.source_indices,
        student_source_indices_0based=target_sampling.source_indices,
        reference_actual_phalp_frames_1based=ref_track.frame_numbers[ref_positions],
        student_actual_phalp_frames_1based=target_track.frame_numbers[target_positions],
    )
    write_json(output_dir / "reference_manifest_snapshot.json", manifest)
    summary = {
        "schema_version": "1.0", "status": "ok", "video_id": output_dir.name,
        "reference_manifest": str(manifest_path),
        "reference_manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "reference_video_id": reference["video_id"], "reference_video": str(reference_video),
        "reference_tracking": str(reference_tracking),
        "input_video": str(args.input_video.absolute()),
        "input_tracking": str(args.input_tracking.absolute()),
        "input_assumed_trimmed_to_24_form": True,
        "sample_fps": sample_fps, "input_source_fps": target_sampling.source_fps,
        "input_source_frame_count": target_sampling.source_frame_count,
        "reference_source_fps": ref_sampling.source_fps,
        "input_used_track_ids": list(target_track.used_track_ids),
        "reference_used_track_ids": list(ref_track.used_track_ids),
        "max_tracking_gap_seconds": args.max_tracking_gap_seconds,
        "reference_substituted_sample_count": int(np.count_nonzero(
            ref_track.frame_numbers[ref_positions] != ref_sampling.source_indices + 1)),
        "student_substituted_sample_count": int(np.count_nonzero(
            target_track.frame_numbers[target_positions] != target_sampling.source_indices + 1)),
        "endpoint_count": len(rows), "strictly_increasing": True,
        "review_required_count": sum(r["review_required"] for r in rows),
        "manual_review_status": "pending",
        "review_geodesic_degrees": args.review_geodesic_degrees,
        "review_threshold_policy": "Optional user diagnostic threshold, not a quality decision.",
        "local_cost": "mean geodesic over 23 local body_pose joints; excludes global_orient",
        "dtw_distance_degrees": float(np.degrees(distance)),
        "dtw_path_mean_geodesic_degrees": float(np.degrees(costs[path[:, 0], path[:, 1]].mean())),
        "dtw_path_length": len(path),
        "unmapped_tail_source_frame_count": target_sampling.source_frame_count - boundaries[-1].target_source_frame_1based,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output_dir / "segmentation.json", {"summary": summary,
               "boundaries": [asdict(b) for b in boundaries], "segments": [asdict(s) for s in segments]})
    write_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", type=Path,
                        default=PROJECT_ROOT / "reference_data/BV1WE411W7JB/reference_manifest.json")
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--input-tracking", type=Path, required=True)
    parser.add_argument("--video-id")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "student_keypose_results")
    parser.add_argument("--max-tracking-gap-seconds", type=float, default=0.0)
    parser.add_argument("--pairwise-chunk-size", type=int, default=32)
    parser.add_argument("--review-geodesic-degrees", type=float,
                        help="Optional manual-review flag threshold, not a calibrated quality threshold.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    video_id = args.video_id or args.input_video.stem
    if not video_id.strip() or Path(video_id).name != video_id or video_id in (".", ".."):
        raise ValueError("Invalid output video ID.")
    if args.pairwise_chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")
    if args.review_geodesic_degrees is not None and (
        not np.isfinite(args.review_geodesic_degrees) or args.review_geodesic_degrees <= 0
    ):
        raise ValueError("Review threshold must be finite and positive.")
    output_dir = args.output_root.expanduser().absolute() / video_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use a fresh output directory to retain previous results: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        summary = run(args, output_dir)
    except Exception as exc:
        write_json(output_dir / "summary.json", {"status": "failed", "error": str(exc)})
        raise
    print(f'[{video_id}] complete in {summary["elapsed_seconds"]:.1f}s -> {output_dir}', flush=True)


if __name__ == "__main__":
    main()
