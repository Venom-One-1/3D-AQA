#!/usr/bin/env python
"""Segment every tracked teaching video against the BV1WE411W7JB gold reference."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from run_smpl_pose_dtw_segmentation import (
    DEFAULT_GOLD_BOUNDARIES,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REFERENCE_TRACKING,
    DEFAULT_REFERENCE_VIDEO,
)
from export_tas_reference_annotations import load_point_labels
from visualize_smpl_pose_dtw_teach_boundaries import (
    DEFAULT_ANNOTATION_PATH,
    DEFAULT_TRIM_MANIFEST,
)


DEFAULT_VIDEO_ROOT = Path("/home/sqw/VisualSearch/aqa/teach_trimmed")
DEFAULT_TRACKING_ROOT = Path("/home/sqw/VisualSearch/aqa/Tracking/teach_trimmed")
DEFAULT_REFERENCE_ID = "BV1WE411W7JB"


@dataclass(frozen=True)
class TargetFiles:
    video_id: str
    video_path: Path
    tracking_path: Path


@dataclass(frozen=True)
class BatchResult:
    video_id: str
    status: str
    segmentation_status: str
    visualization_status: str
    has_complete_ground_truth: bool | None
    elapsed_seconds: float
    video_path: str
    tracking_path: str
    output_dir: str
    visualization_path: str
    error: str


def discover_targets(
    video_root: Path,
    tracking_root: Path,
    reference_video_id: str,
    selected_ids: set[str] | None = None,
) -> tuple[list[TargetFiles], list[str]]:
    targets: list[TargetFiles] = []
    problems: list[str] = []
    if not tracking_root.is_dir():
        raise FileNotFoundError(tracking_root)
    for directory in sorted(
        path
        for path in tracking_root.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    ):
        video_id = directory.name
        if video_id == reference_video_id:
            continue
        if selected_ids is not None and video_id not in selected_ids:
            continue
        video_path = video_root / f"{video_id}.mp4"
        exact_tracking = directory / "results" / f"demo_{video_id}.pkl"
        if exact_tracking.is_file():
            tracking_path = exact_tracking
        else:
            candidates = sorted((directory / "results").glob("demo_*.pkl"))
            if len(candidates) != 1:
                problems.append(
                    f"{video_id}: expected one tracking PKL under {directory / 'results'}, "
                    f"found {len(candidates)}"
                )
                continue
            tracking_path = candidates[0]
        if not video_path.is_file():
            problems.append(f"{video_id}: missing video {video_path}")
            continue
        targets.append(TargetFiles(video_id, video_path, tracking_path))

    found_ids = {target.video_id for target in targets}
    if selected_ids is not None:
        for missing_id in sorted(selected_ids - found_ids):
            if not any(problem.startswith(f"{missing_id}:") for problem in problems):
                problems.append(f"{missing_id}: no tracking directory found")
    return targets, problems


def result_is_current(
    summary_path: Path,
    target: TargetFiles,
    reference_video: Path,
    reference_tracking: Path,
    gold_boundaries: Path,
    max_tracking_gap_seconds: float,
) -> bool:
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "status": "ok",
        "video_id": target.video_id,
        "input_video": str(target.video_path.absolute()),
        "input_tracking": str(target.tracking_path.absolute()),
        "reference_video": str(reference_video.absolute()),
        "reference_tracking": str(reference_tracking.absolute()),
        "gold_boundaries": str(gold_boundaries.absolute()),
        "sample_fps": 5.0,
    }
    if not all(summary.get(key) == value for key, value in expected.items()):
        return False
    # A result produced with a stricter gap policy remains valid when the caller
    # permits a larger gap, but not the other way around.
    return float(summary.get("max_tracking_gap_seconds", 0.0)) <= max_tracking_gap_seconds


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def write_batch_summary(
    output_root: Path,
    *,
    reference_video_id: str,
    results: list[BatchResult],
    discovery_problems: list[str],
) -> None:
    payload = {
        "reference_video_id": reference_video_id,
        "target_count": len(results),
        "completed_count": sum(item.status == "ok" for item in results),
        "failed_count": sum(item.status == "failed" for item in results),
        "discovery_problems": discovery_problems,
        "results": [asdict(item) for item in results],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "teach_batch_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--tracking-root", type=Path, default=DEFAULT_TRACKING_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reference-video-id", default=DEFAULT_REFERENCE_ID)
    parser.add_argument("--reference-video", type=Path, default=DEFAULT_REFERENCE_VIDEO)
    parser.add_argument("--reference-tracking", type=Path, default=DEFAULT_REFERENCE_TRACKING)
    parser.add_argument("--gold-boundaries", type=Path, default=DEFAULT_GOLD_BOUNDARIES)
    parser.add_argument("--annotation-path", type=Path, default=DEFAULT_ANNOTATION_PATH)
    parser.add_argument("--trim-manifest", type=Path, default=DEFAULT_TRIM_MANIFEST)
    parser.add_argument("--video-id", action="append", help="Target ID; repeatable.")
    parser.add_argument("--force", action="store_true", help="Recompute existing segmentations.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--max-tracking-gap-seconds",
        type=float,
        default=0.0,
        help="Nearest-pose tolerance passed to the single-video segmenter.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = {
        "video_root": args.video_root.expanduser().absolute(),
        "tracking_root": args.tracking_root.expanduser().absolute(),
        "output_root": args.output_root.expanduser().absolute(),
        "reference_video": args.reference_video.expanduser().absolute(),
        "reference_tracking": args.reference_tracking.expanduser().absolute(),
        "gold_boundaries": args.gold_boundaries.expanduser().absolute(),
        "annotation_path": args.annotation_path.expanduser().absolute(),
        "trim_manifest": args.trim_manifest.expanduser().absolute(),
    }
    if args.max_tracking_gap_seconds < 0:
        raise ValueError("--max-tracking-gap-seconds must be non-negative.")
    for key in (
        "video_root",
        "tracking_root",
        "reference_video",
        "reference_tracking",
        "gold_boundaries",
        "annotation_path",
        "trim_manifest",
    ):
        if not paths[key].exists():
            raise FileNotFoundError(paths[key])

    selected_ids = set(args.video_id) if args.video_id else None
    targets, discovery_problems = discover_targets(
        paths["video_root"],
        paths["tracking_root"],
        args.reference_video_id,
        selected_ids,
    )
    if not targets:
        raise ValueError("No complete target video/tracking pairs were found.")
    for problem in discovery_problems:
        print(f"[discovery warning] {problem}", flush=True)
    print(f"Discovered {len(targets)} target teaching videos.", flush=True)

    script_root = Path(__file__).resolve().parent
    segmentation_script = script_root / "run_smpl_pose_dtw_segmentation.py"
    visualization_script = script_root / "visualize_smpl_pose_dtw_teach_boundaries.py"
    results: list[BatchResult] = []
    for index, target in enumerate(targets, start=1):
        started = time.perf_counter()
        output_dir = paths["output_root"] / target.video_id
        summary_path = output_dir / "summary.json"
        visualization_path = output_dir / "boundary_frames.jpg"
        segmentation_status = "pending"
        visualization_status = "pending"
        has_gt: bool | None = None
        error = ""
        print(f"[{index}/{len(targets)}] {target.video_id}", flush=True)
        try:
            if not args.force and result_is_current(
                summary_path,
                target,
                paths["reference_video"],
                paths["reference_tracking"],
                paths["gold_boundaries"],
                args.max_tracking_gap_seconds,
            ):
                segmentation_status = "reused"
                print(f"[{target.video_id}] reusing current segmentation", flush=True)
            else:
                run_command(
                    [
                        sys.executable,
                        str(segmentation_script),
                        "--input-video",
                        str(target.video_path),
                        "--input-tracking",
                        str(target.tracking_path),
                        "--video-id",
                        target.video_id,
                        "--output-root",
                        str(paths["output_root"]),
                        "--reference-video",
                        str(paths["reference_video"]),
                        "--reference-tracking",
                        str(paths["reference_tracking"]),
                        "--gold-boundaries",
                        str(paths["gold_boundaries"]),
                        "--max-tracking-gap-seconds",
                        str(args.max_tracking_gap_seconds),
                    ]
                )
                segmentation_status = "computed"

            run_command(
                [
                    sys.executable,
                    str(visualization_script),
                    "--video-id",
                    target.video_id,
                    "--target-video",
                    str(target.video_path),
                    "--predicted-boundaries",
                    str(output_dir / "boundaries.csv"),
                    "--output",
                    str(visualization_path),
                    "--reference-video-id",
                    args.reference_video_id,
                    "--reference-video",
                    str(paths["reference_video"]),
                    "--reference-boundaries",
                    str(paths["gold_boundaries"]),
                    "--annotation-path",
                    str(paths["annotation_path"]),
                    "--trim-manifest",
                    str(paths["trim_manifest"]),
                ]
            )
            visualization_status = "computed"
            # The third column is present exactly when all 24 GT move IDs exist.
            try:
                labels = load_point_labels(paths["annotation_path"], target.video_id)
            except ValueError:
                has_gt = False
            else:
                move_ids = {
                    item.tag_id
                    for item in labels
                    if item.state == 1 and 1 <= item.tag_id <= 24
                }
                has_gt = move_ids == set(range(1, 25))
            status = "ok"
        except Exception as exception:
            status = "failed"
            error = f"{type(exception).__name__}: {exception}"
            print(f"[{target.video_id}] FAILED: {error}", flush=True)

        elapsed = time.perf_counter() - started
        results.append(
            BatchResult(
                video_id=target.video_id,
                status=status,
                segmentation_status=segmentation_status,
                visualization_status=visualization_status,
                has_complete_ground_truth=has_gt,
                elapsed_seconds=round(elapsed, 3),
                video_path=str(target.video_path),
                tracking_path=str(target.tracking_path),
                output_dir=str(output_dir),
                visualization_path=str(visualization_path),
                error=error,
            )
        )
        write_batch_summary(
            paths["output_root"],
            reference_video_id=args.reference_video_id,
            results=results,
            discovery_problems=discovery_problems,
        )
        print(f"[{target.video_id}] batch step finished in {elapsed:.1f}s", flush=True)
        if status == "failed" and args.fail_fast:
            break

    failed = [item.video_id for item in results if item.status == "failed"]
    print(
        f"Batch complete: {len(results) - len(failed)}/{len(results)} succeeded; "
        f"summary={paths['output_root'] / 'teach_batch_summary.json'}",
        flush=True,
    )
    if failed:
        raise SystemExit(f"Failed targets: {', '.join(failed)}")


if __name__ == "__main__":
    main()
