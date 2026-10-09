#!/usr/bin/env python
"""Locate semantic temporal key-pose events in a tracked Tai Chi move."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from aqa3d.temporal_keyposes import (
    compute_raw_metrics,
    locate_events,
    prepare_metric_signals,
    validate_temporal_plan,
)
from aqa3d.tracking import load_stitched_primary_track, load_stitched_root_track


PROJECT = Path(__file__).resolve().parent
DEFAULT_PLAN = PROJECT / "temporal_query_specs" / "qishi_v1.json"
DEFAULT_MANIFEST = PROJECT / "reference_data" / "BV1WE411W7JB" / "reference_manifest.json"
DEFAULT_VIDEO = Path("/home/sqw/VisualSearch/aqa/teach_trimmed/BV1WE411W7JB.mp4")
DEFAULT_TRACKING = Path(
    "/home/sqw/VisualSearch/aqa/Tracking/teach_trimmed/BV1WE411W7JB/results/demo_BV1WE411W7JB.pkl"
)
DEFAULT_OUTPUT = PROJECT / "temporal_keypose_results" / "qishi_v1" / "BV1WE411W7JB"
DEFAULT_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, include_hash: bool) -> dict[str, Any]:
    stat = path.stat()
    record = {"path": str(path.resolve()), "size_bytes": stat.st_size,
              "mtime_ns": stat.st_mtime_ns}
    if include_hash:
        record["sha256"] = sha256(path)
    return record


def video_metadata(path: Path) -> tuple[float, int, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0 or count <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid video metadata for {path}")
    return fps, count, width, height


def move_event(manifest: dict[str, Any], move_id: int) -> dict[str, Any]:
    selected = [event for event in manifest["endpoint_keyposes"] if event["move_id"] == move_id]
    if len(selected) != 1:
        raise ValueError(f"Expected one manifest endpoint for move {move_id}, got {len(selected)}")
    return selected[0]


def load_segment_joints(
    tracking_path: Path,
    start_frame: int,
    end_frame: int,
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    primary = load_stitched_primary_track(tracking_path)
    root = load_stitched_root_track(tracking_path)
    frames = np.arange(start_frame, end_frame + 1, dtype=np.int64)
    joints = np.full((len(frames), 24, 3), np.nan, dtype=np.float64)
    statuses = np.full(len(frames), "missing_tracking", dtype="<U32")
    source_track_ids = np.full(len(frames), -1, dtype=np.int64)
    primary_index = {int(frame): index for index, frame in enumerate(primary.frame_numbers)}
    root_index = {int(frame): index for index, frame in enumerate(root.frame_numbers)}
    switches = primary.frame_numbers[
        np.flatnonzero(np.diff(primary.source_track_ids) != 0) + 1
    ]
    available: list[int] = []
    output_indices: list[int] = []
    for output_index, source_frame in enumerate(frames):
        phalp_frame = int(source_frame) + 1
        primary_position = primary_index.get(phalp_frame)
        root_position = root_index.get(phalp_frame)
        if primary_position is None or root_position is None:
            continue
        source_track_ids[output_index] = int(primary.source_track_ids[primary_position])
        if (root.source_track_ids[root_position] != primary.source_track_ids[primary_position]
                or not np.allclose(
                    root.body_poses[root_position], primary.body_poses[primary_position],
                    atol=1e-7, rtol=0,
                )):
            statuses[output_index] = "tracking_selection_mismatch"
        elif np.any(np.abs(switches - phalp_frame) <= 1):
            statuses[output_index] = "near_track_switch"
        else:
            statuses[output_index] = "valid"
            available.append(int(source_frame))
            output_indices.append(output_index)
    if available:
        selected = root.at_source_frames(available)
        joints[output_indices] = selected.to_smpl24_joints(
            device=device, batch_size=batch_size,
        )
    quality = {
        "primary_track_id": int(primary.track_id),
        "used_track_ids": list(primary.used_track_ids),
        "valid_frame_count": int(np.count_nonzero(statuses == "valid")),
        "requested_frame_count": len(frames),
        "excluded_frame_count": int(np.count_nonzero(statuses != "valid")),
        "status_counts": {
            str(status): int(count)
            for status, count in zip(*np.unique(statuses, return_counts=True))
        },
        "track_switch_phalp_frames_1based": switches.astype(int).tolist(),
    }
    return frames, joints, statuses, source_track_ids, quality


def sensitivity_analysis(
    plan: dict[str, Any],
    raw_metrics: dict[str, np.ndarray],
    frames: np.ndarray,
    times: np.ndarray,
    fps: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    execution = plan["execution"]
    rows = []
    for smoothing in execution["sensitivity_smoothing_seconds"]:
        signals = prepare_metric_signals(
            raw_metrics,
            times,
            smoothing_seconds=float(smoothing),
            derivative_window_seconds=float(execution["derivative_window_seconds"]),
        )
        for multiplier in execution["sensitivity_threshold_multipliers"]:
            detections = locate_events(
                plan, signals, frames, times, fps,
                threshold_multiplier=float(multiplier),
            )
            for event_id, detection in detections.items():
                result = detection["result"]
                rows.append({
                    "event_id": event_id,
                    "smoothing_seconds": float(smoothing),
                    "threshold_multiplier": float(multiplier),
                    "status": result["status"],
                    "start_time_seconds": result["start_time_seconds"],
                    "representative_time_seconds": result["representative_time_seconds"],
                    "end_time_seconds": result["end_time_seconds"],
                })
    summary: dict[str, Any] = {}
    for event in plan["events"]:
        selected = [row for row in rows if row["event_id"] == event["event_id"]]
        located = [row for row in selected if row["status"] == "located"]
        item: dict[str, Any] = {
            "run_count": len(selected),
            "located_run_count": len(located),
            "all_variants_located": len(located) == len(selected),
        }
        for field in ("start_time_seconds", "representative_time_seconds", "end_time_seconds"):
            values = [float(row[field]) for row in located]
            item[field] = ({"minimum": min(values), "maximum": max(values),
                            "span": max(values) - min(values)} if values else None)
        summary[event["event_id"]] = item
    return rows, summary


def result_rows(
    plan: dict[str, Any], detections: dict[str, dict[str, Any]], fps: float,
    sensitivity: dict[str, Any], boundary_time: float,
) -> list[dict[str, Any]]:
    rows = []
    for event in plan["events"]:
        result = dict(detections[event["event_id"]]["result"])
        result["interval_start_source_frame_1based"] = (
            None if result["interval_start_source_frame_0based"] is None
            else result["interval_start_source_frame_0based"] + 1
        )
        result["interval_end_source_frame_1based"] = (
            None if result["interval_end_source_frame_0based"] is None
            else result["interval_end_source_frame_0based"] + 1
        )
        representative = result["representative_source_frame_0based"]
        result["representative_source_frame_1based"] = (
            None if representative is None else representative + 1
        )
        result["representative_sample_index_5fps_0based"] = (
            None if result["representative_time_seconds"] is None
            else int(np.floor(result["representative_time_seconds"] * 5 + 0.5))
        )
        result["representative_sample_frame_5fps_1based"] = (
            None if result["representative_sample_index_5fps_0based"] is None
            else result["representative_sample_index_5fps_0based"] + 1
        )
        result["representative_grid_time_5fps_seconds"] = (
            None if representative is None
            else result["representative_sample_index_5fps_0based"] / 5
        )
        result["representative_grid_source_frame_0based"] = (
            None if representative is None
            else int(np.floor(result["representative_grid_time_5fps_seconds"] * fps + .5))
        )
        result["boundary_delta_seconds"] = (
            None if result["representative_time_seconds"] is None
            else result["representative_time_seconds"] - boundary_time
        )
        result["sensitivity"] = sensitivity[event["event_id"]]
        result["auxiliary_reference_image"] = event["auxiliary_reference_image"]
        result["auxiliary_reference_used_for_detection"] = False
        rows.append(result)
    return rows


def save_signal_archive(
    path: Path,
    frames: np.ndarray,
    times: np.ndarray,
    statuses: np.ndarray,
    source_track_ids: np.ndarray,
    signals: dict[str, Any],
    detections: dict[str, dict[str, Any]],
) -> None:
    values: dict[str, np.ndarray] = {
        "source_frames_0based": frames,
        "time_seconds": times,
        "tracking_status": statuses,
        "source_track_ids": source_track_ids,
        "valid_frame_mask": statuses == "valid",
    }
    for metric_id, series in signals.items():
        values[f"metric__{metric_id}__raw"] = series.raw
        values[f"metric__{metric_id}__smoothed"] = series.smoothed
        values[f"metric__{metric_id}__velocity"] = series.velocity
        values[f"metric__{metric_id}__acceleration"] = series.acceleration
    for event_id, detection in detections.items():
        values[f"event__{event_id}__score"] = detection["score"]
        values[f"event__{event_id}__search_mask"] = detection["search_mask"]
        values[f"event__{event_id}__candidate_mask"] = detection["candidate_mask"]
        for condition_id, score in detection["condition_scores"].items():
            values[f"condition__{event_id}__{condition_id}__score"] = score
            values[f"condition__{event_id}__{condition_id}__mask"] = detection["condition_masks"][condition_id]
    np.savez_compressed(path, **values)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = (Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
            if bold else DEFAULT_FONT)
    return ImageFont.truetype(str(path), size=size)


def read_video_frames(path: Path, frame_ids: list[int]) -> dict[int, Image.Image]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    result = {}
    for frame_id in sorted(set(frame_ids)):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Cannot decode frame {frame_id} from {path}")
        result[frame_id] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return result


def letterbox(image: Image.Image, width: int, height: int) -> Image.Image:
    source = image.convert("RGB")
    source.thumbnail((width, height), Image.Resampling.LANCZOS)
    result = Image.new("RGB", (width, height), "#eef1f4")
    result.paste(source, ((width - source.width) // 2, (height - source.height) // 2))
    return result


def save_overview(
    path: Path,
    video: Path,
    plan: dict[str, Any],
    rows: list[dict[str, Any]],
    fps: float,
    auxiliary_correspondences: dict[str, str] | None = None,
) -> None:
    frame_ids = []
    for row in rows:
        for key in ("interval_start_source_frame_0based", "representative_source_frame_0based",
                    "interval_end_source_frame_0based"):
            if row[key] is not None:
                frame_ids.append(int(row[key]))
        if row["status"] == "located":
            frame_ids.append((row["interval_start_source_frame_0based"]
                              + row["interval_end_source_frame_0based"]) // 2)
    frames = read_video_frames(video, frame_ids)
    width, cell_width, cell_height = 1760, 410, 250
    header_height, row_height = 125, 390
    canvas = Image.new("RGB", (width, header_height + len(rows) * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 18), plan["move"]["move_name"]+"：时序关键事件定位", fill="#111827", font=_font(34, True))
    draw.text(
        (30, 65),
        plan['video_id']+" | 已知招式边界限定搜索范围；frame0 表示原视频 0-based 帧号",
        fill="#52606d", font=_font(20),
    )
    for row_index, (event, row) in enumerate(zip(plan["events"], rows)):
        top = header_height + row_index * row_height
        draw.rectangle((0, top, width, top + row_height), fill="#f8fafc" if row_index % 2 == 0 else "white")
        status_text = (f"{row['start_time_seconds']:.3f}s–{row['end_time_seconds']:.3f}s | "
                       f"代表 {row['representative_time_seconds']:.3f}s"
                       if row["status"] == "located" else f"未定位：{row['reason']}")
        draw.text((30, top + 15), f"{row_index + 1}. {event['label']}", fill="#0f172a", font=_font(25, True))
        draw.text((30, top + 50), status_text, fill="#216e5a", font=_font(18))
        images: list[Image.Image | None] = []
        auxiliary = Path(event["auxiliary_reference_image"])
        start, rep, end = [row[k] for k in (
            "interval_start_source_frame_0based", "representative_source_frame_0based",
            "interval_end_source_frame_0based")]
        if auxiliary.is_file():
            labels = ["辅助 KeyPose（非金标准）", "事件区间起点", "代表帧", "事件区间终点"]
            columns = [None, start, rep, end]
            with Image.open(auxiliary) as reference_image:
                images.append(reference_image.convert("RGB"))
            images.extend(None if f is None else frames[int(f)] for f in columns[1:])
        else:
            labels = ["事件区间起点", "区间中点", "代表帧", "事件区间终点"]
            columns = [start, None if start is None else (start+end)//2, rep, end]
            images = [None if f is None else frames[int(f)] for f in columns]
        if row.get("representative_source") == "known_move_end":
            labels[2] = "代表帧（已知结束边界）"
            labels[3] = "已知招式结束边界"
        for column, (label, image) in enumerate(zip(labels, images)):
            left = 30 + column * (cell_width + 20)
            draw.text((left, top + 84), label, fill="#334155", font=_font(17, column == 2))
            if image is None:
                draw.rectangle((left, top + 116, left + cell_width, top + 116 + cell_height), fill="#e5e7eb")
            else:
                canvas.paste(letterbox(image, cell_width, cell_height - 38), (left, top + 116))
            if columns[column] is not None:
                frame_id = int(columns[column])
                draw.rectangle((left, top + 328, left + cell_width, top + 366), fill=(255, 255, 255))
                draw.text((left + 8, top + 334), f"{frame_id / fps:.3f}s | frame0 {frame_id}",
                          fill="#111827", font=_font(16))
            elif column == 0 and auxiliary.is_file():
                correspondence = (auxiliary_correspondences or {}).get(event['event_id'])
                caption = auxiliary.name + (f" | {correspondence}" if correspondence else '')
                draw.rectangle((left, top + 328, left + cell_width, top + 366), fill='white')
                draw.text((left + 8, top + 334), caption, fill='#111827', font=_font(16))
    canvas.save(path, quality=94)


def save_diagnostics(
    path: Path,
    plan: dict[str, Any],
    signals: dict[str, Any],
    detections: dict[str, dict[str, Any]],
    times: np.ndarray,
) -> None:
    font = font_manager.FontProperties(fname=str(DEFAULT_FONT))
    groups = [[], [], []]
    for metric_id in signals:
        group_id = 2 if ('angle' in metric_id or 'knee' in metric_id) else int('distance' in metric_id)
        groups[group_id].append(metric_id)
    groups = [group for group in groups if group]
    count = len(groups) + len(plan['events'])
    figure, axes = plt.subplots(count, 1, figsize=(15, 2.5*count), sharex=True)
    for axis, group in zip(axes, groups):
        for metric_id in group:
            axis.plot(times, signals[metric_id].smoothed, label=metric_id)
        axis.set_ylabel('degree' if all('angle' in m or 'knee' in m for m in group) else 'ratio')
        axis.legend(loc='upper left', fontsize=8, ncol=2)
    for index, event in enumerate(plan["events"]):
        color = plt.get_cmap('tab10')(index % 10)
        detection = detections[event["event_id"]]
        axis = axes[len(groups)+index]
        scores = np.where(detection['search_mask'], detection['score'], np.nan)
        axis.plot(times, scores, label=event["label"], color=color)
        result = detection["result"]
        if result["status"] == "located":
            axis.axvspan(result['start_time_seconds'], result['end_time_seconds'], color=color, alpha=.10)
            axis.axvspan(result['detected_match_start_time_seconds'],
                         result['detected_match_end_time_seconds'], color=color, alpha=.2)
            axis.axvline(result['representative_time_seconds'], color=color, linestyle='--')
        axis.set_ylabel("查询得分", fontproperties=font)
        axis.set_ylim(-.03, 1.05)
        axis.legend(loc="upper left", prop=font)
    axes[-1].set_xlabel("裁剪后视频时间（s）", fontproperties=font)
    for axis in axes:
        axis.grid(True, alpha=.22)
        axis.set_xlim(plan["move"]["start_seconds"], plan["move"]["end_seconds"])
    figure.suptitle(plan['move']['move_name']+"：逐帧信号与时序查询结果", fontproperties=font, fontsize=18)
    figure.tight_layout(rect=(0, 0, 1, .97))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_report(path, plan, rows):
    lines = [f"# {plan['move']['move_name']}：时序查询定位", '',
             f"参考视频：{plan['video_id']}；已知招式范围："
             f"{plan['move']['start_seconds']:.3f}–{plan['move']['end_seconds']:.3f}s。", '',
             '时间均为裁剪后完整视频时间。查询得分用于选取候选，不是动作质量分或正确概率。', '',
             '| 事件 | 输出区间 | 信号匹配区间 | 代表帧时间 | 代表帧来源 | 9组扰动定位次数 |',
             '|---|---|---|---|---|---|']
    for row in rows:
        if row['status'] != 'located':
            lines.append(f"| {row['label']} | 未定位：{row['reason']} | | | | |")
            continue
        source = '已知招式结束边界' if row['representative_source']=='known_move_end' else '信号查询'
        lines.append(
            f"| {row['label']} | {row['start_time_seconds']:.3f}–{row['end_time_seconds']:.3f}s | "
            f"{row['detected_match_start_time_seconds']:.3f}–{row['detected_match_end_time_seconds']:.3f}s | "
            f"{row['representative_time_seconds']:.3f}s | {source} | "
            f"{row['sensitivity']['located_run_count']}/{row['sensitivity']['run_count']} |"
        )
    lines += ['', '![区间与代表帧](keypose_overview.jpg)', '',
              '![逐帧信号](temporal_diagnostics.png)', '',
              '诊断图深色阴影为信号匹配区间，浅色阴影为输出区间，虚线为代表帧。', '',
              '## 查询依据', '']
    for event in plan['events']:
        lines += [f"### {event['label']}", '', event['technique'], '',
                  event['query']['rationale'], '']
    lines += ['## 解释范围', '',
              '本次为同一参考视频上的查询开发与可视化检查，内部事件尚无人工时间金标准。'
              '参数扰动是稳定性诊断，不能解释为独立准确率。'
              '身体坐标 up 跟随躯干，不是地面竖直方向；腕、踝只用于几何定位。', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--tracking", type=Path, default=DEFAULT_TRACKING)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smpl-batch-size", type=int, default=256)
    args = parser.parse_args()
    for path in (args.plan, args.manifest, args.video, args.tracking):
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    if args.smpl_batch_size <= 0:
        parser.error("--smpl-batch-size must be positive")

    plan = load_json(args.plan)
    validate_temporal_plan(plan)
    if args.output_dir is None:
        args.output_dir = PROJECT/'temporal_keypose_results'/args.plan.stem/plan['video_id']
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Refusing to overwrite nonempty output directory: {args.output_dir}")
    manifest = load_json(args.manifest)
    if plan["video_id"] != manifest["reference"]["video_id"]:
        raise ValueError("Plan and reference manifest video IDs differ")
    event = move_event(manifest, plan["move"]["move_id"])
    fps, frame_count, _, _ = video_metadata(args.video)
    if not np.isclose(fps, manifest["reference"]["source_fps"], atol=1e-6):
        raise ValueError("Video FPS differs from the reference manifest")
    if frame_count != manifest["reference"]["source_frame_count"]:
        raise ValueError("Video frame count differs from the reference manifest")
    start_frame = int(event["segment_source_start_index_0based"])
    end_frame = int(event["segment_source_end_index_0based"])
    if (not np.isclose(start_frame / fps, plan["move"]["start_seconds"], atol=1 / fps)
            or not np.isclose(end_frame / fps, plan["move"]["end_seconds"], atol=1 / fps)):
        raise ValueError("Plan move interval differs from the Gold reference boundary")

    frames, joints, statuses, source_track_ids, tracking_quality = load_segment_joints(
        args.tracking, start_frame, end_frame,
        device=args.device, batch_size=args.smpl_batch_size,
    )
    times = frames.astype(np.float64) / fps
    raw_metrics = compute_raw_metrics(plan, joints, frames, statuses)
    execution = plan["execution"]
    signals = prepare_metric_signals(
        raw_metrics, times,
        smoothing_seconds=float(execution["smoothing_seconds"]),
        derivative_window_seconds=float(execution["derivative_window_seconds"]),
    )
    detections = locate_events(plan, signals, frames, times, fps)
    sensitivity_rows, sensitivity_summary = sensitivity_analysis(
        plan, raw_metrics, frames, times, fps,
    )
    rows = result_rows(
        plan, detections, fps, sensitivity_summary,
        boundary_time=float(event["boundary_time_seconds"]),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "temporal_queries.json", plan)
    write_json(args.output_dir / "keypose_events.json", {
        "schema_version": "1.0",
        "video_id": plan["video_id"],
        "move": plan["move"],
        "frame_convention": {
            "source_frame_0based": "canonical OpenCV frame index",
            "source_frame_1based": "source_frame_0based + 1",
            "time_seconds": "source_frame_0based / source_fps",
            "sample_index_5fps_0based": "floor(time_seconds * 5 + 0.5); nearest grid, not exact source frame",
        },
        "gold_move_end_time_seconds": float(event["boundary_time_seconds"]),
        "events": rows,
    })
    flat_rows = []
    for row in rows:
        flat_rows.append({
            key: (json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value)
            for key, value in row.items()
        })
    write_csv(args.output_dir / "keypose_events.csv", flat_rows)
    write_csv(args.output_dir / "sensitivity_runs.csv", sensitivity_rows)
    save_signal_archive(
        args.output_dir / "motion_signals.npz", frames, times, statuses,
        source_track_ids, signals, detections,
    )
    save_diagnostics(
        args.output_dir / "temporal_diagnostics.png", plan, signals, detections, times,
    )
    save_overview(
        args.output_dir / "keypose_overview.jpg", args.video, plan, rows, fps,
    )
    save_report(args.output_dir/'report.md', plan, rows)
    write_json(args.output_dir / "summary.json", {
        "schema_version": "1.0",
        "status": "completed" if all(row["status"] == "located" for row in rows) else "needs_review",
        "scope": f"{plan['video_id']} move {plan['move']['move_id']}; authored temporal queries",
        "video": {"video_id": plan["video_id"], "source_fps": fps,
                  "source_frame_count": frame_count},
        "move": {**plan["move"], "segment_start_source_frame_0based": start_frame,
                 "segment_end_source_frame_0based": end_frame,
                 "gold_end_time_seconds": event["boundary_time_seconds"]},
        "tracking_quality": tracking_quality,
        "event_count": len(rows),
        "located_event_count": sum(row["status"] == "located" for row in rows),
        "sensitivity": sensitivity_summary,
        "supplemental_keypose_policy": {
            "root": "/home/sqw/VisualSearch/aqa/FrameData/KeyPose",
            "used_for_detection": False,
            "purpose": "visual reference only; no explicit Gold correspondence assumed",
        },
        "limitations": [
            "Internal events do not have human Gold timestamps.",
            "Event intervals are semantic candidates, not statistical confidence intervals.",
            "torso_v1 up follows the torso and is not gravity vertical.",
            "A successful result on one teacher does not establish cross-person generalization.",
            "The terminal transition uses the known Gold move end only as its explicit end bound.",
        ],
        "inputs": {
            "plan": file_record(args.plan, include_hash=True),
            "manifest": file_record(args.manifest, include_hash=True),
            "video": file_record(args.video, include_hash=False),
            "tracking": file_record(args.tracking, include_hash=False),
            "executor": file_record(PROJECT/'aqa3d/temporal_keyposes.py', include_hash=True),
            "runner": file_record(Path(__file__), include_hash=True),
        },
        "outputs": [
            "temporal_queries.json", "keypose_events.json", "keypose_events.csv",
            "sensitivity_runs.csv", "motion_signals.npz", "temporal_diagnostics.png",
            "keypose_overview.jpg", "report.md", "summary.json",
        ],
    })
    print(f"Saved {len(rows)} temporal events to {args.output_dir}")
    for row in rows:
        if row["status"] == "located":
            print(f"{row['event_id']}: {row['start_time_seconds']:.3f}s–{row['end_time_seconds']:.3f}s "
                  f"(representative {row['representative_time_seconds']:.3f}s)")
        else:
            print(f"{row['event_id']}: not located ({row['reason']})")


if __name__ == "__main__":
    main()
