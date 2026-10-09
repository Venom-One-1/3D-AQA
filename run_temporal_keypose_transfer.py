#!/usr/bin/env python
"""Transfer the reviewed first-three-form events using frozen SMPL-DTW paths."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw

from aqa3d.endpoint_transfer import manifest_boundaries, resolve_tracking_samples
from aqa3d.smpl_dtw import inspect_video_sampling
from aqa3d.temporal_transfer import (
    POINT_FIELDS, apply_correspondence_reviews, apply_reference_corrections,
    flag_event_order, flatten_event, map_event, path_pose_costs, track_frame_quality,
    verify_path_archive,
)
from aqa3d.tracking import load_stitched_primary_track
from run_temporal_keypose_localization import file_record, sha256, write_csv, write_json
from visualize_tas_boundary_frames import fit_frame, load_font, read_video_frames


PROJECT = Path(__file__).resolve().parent
FOLDERS = {1: "01_qishi", 2: "02_yemafenzong", 3: "03_baiheliangchi"}


def load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def verify_plan_snapshot(snapshot: Path, original_record: dict) -> dict:
    original = Path(original_record["path"])
    if sha256(original) != original_record["sha256"]:
        raise ValueError("Reference localization used a different plan.")
    plan = load_object(snapshot)
    if plan != load_object(original):
        raise ValueError("Reference query snapshot differs from its original plan.")
    return plan


def freeze_reference(args, manifest: dict, sampling, output: Path) -> list[dict]:
    records = []
    for mid in args.moves:
        folder = args.reference_events_root / FOLDERS[mid]
        summary_path, events_path, plan_path = [folder / name for name in (
            "summary.json", "keypose_events.json", "temporal_queries.json")]
        summary, result = [load_object(p) for p in (summary_path, events_path)]
        plan = verify_plan_snapshot(plan_path, summary["inputs"]["plan"])
        gold = manifest["endpoint_keyposes"][mid - 1]
        if (summary["status"] != "completed" or result["video_id"] != manifest["reference"]["video_id"]
                or result["move"]["move_id"] != mid
                or [e["event_id"] for e in plan["events"]] != [e["event_id"] for e in result["events"]]):
            raise ValueError(f"Reference event identity/order mismatch: {folder}")
        if summary["inputs"]["manifest"]["sha256"] != sha256(args.reference_manifest):
            raise ValueError("Reference localization used a different manifest.")
        for name, actual in (("video", Path(manifest["reference"]["video_path"])),
                             ("tracking", Path(manifest["reference"]["tracking_path"]))):
            saved = summary["inputs"][name]
            stat = actual.stat()
            if (Path(saved["path"]).resolve() != actual.resolve()
                    or saved["size_bytes"] != stat.st_size or saved["mtime_ns"] != stat.st_mtime_ns):
                raise ValueError(f"Reference localization input changed: {name}.")
        if (summary["move"]["segment_start_source_frame_0based"] != gold["segment_source_start_index_0based"]
                or summary["move"]["segment_end_source_frame_0based"] != gold["segment_source_end_index_0based"]
                or summary["video"]["source_fps"] != sampling.source_fps):
            raise ValueError("Reference events and current form boundaries disagree.")
        auxiliary = {}
        mapping_path = folder / "auxiliary_keypose_mapping.json"
        if mapping_path.is_file():
            mapping = load_object(mapping_path)
            if mapping["source_results"]["keypose_events.json"]["sha256"] != sha256(events_path):
                raise ValueError("Auxiliary correspondence belongs to different event results.")
            auxiliary = {row["event_id"]: row for row in mapping["auxiliary_keyposes"]}
        snapshot = output / "reference" / FOLDERS[mid]
        snapshot.mkdir(parents=True)
        for source in (summary_path, events_path, plan_path):
            name = source.name if source == plan_path else f"source_{source.name}"
            shutil.copy2(source, snapshot / name)
        for event in result["events"]:
            if event["status"] != "located":
                raise ValueError(f"Reference event was not located: {event['event_id']}")
            row = dict(event)
            row.update(
                pose_id=f"{mid}.{event['event_id']}", move_id=mid,
                move_name=result["move"]["move_name"],
                reference_segment_start_source_frame_0based=gold["segment_source_start_index_0based"],
                reference_segment_end_source_frame_0based=gold["segment_source_end_index_0based"],
                annotation_status="user_reviewed_candidate_not_internal_timestamp_gold",
                source_event_file=file_record(events_path, include_hash=True),
                auxiliary_correspondence="visual_reference_only",
            )
            for field in POINT_FIELDS.values():
                source = row[field]
                if (not isinstance(source, int) or not gold["segment_source_start_index_0based"]
                        <= source <= gold["segment_source_end_index_0based"]):
                    raise ValueError(f"Reference event out of form: {row['pose_id']}")
            if event["event_id"] in auxiliary:
                selected = auxiliary[event["event_id"]]
                image = Path(selected["image"]["path"])
                if sha256(image) != selected["image"]["sha256"]:
                    raise ValueError(f"Auxiliary image changed: {image}")
                row["auxiliary_reference_image"] = str(image)
                row["auxiliary_correspondence"] = selected["correspondence"]
            records.append(row)
    if len({r["pose_id"] for r in records}) != len(records):
        raise ValueError("Duplicate reference event IDs.")
    review_records = load_object(args.review_records)
    records = apply_reference_corrections(records, review_records, sampling,
                                          manifest["reference"]["video_id"])
    write_json(output / "manual_review_records_snapshot.json", review_records)
    for mid in args.moves:
        selected = [event for event in records if event["move_id"] == mid]
        write_json(output / "reference" / FOLDERS[mid] / "keypose_events.json", {
            "schema_version": "1.0", "video_id": manifest["reference"]["video_id"],
            "move": {"move_id": mid, "move_name": selected[0]["move_name"]},
            "events": selected,
        })
    write_json(output / "reference_event_manifest.json", {
        "schema_version": "1.0", "reference_video_id": manifest["reference"]["video_id"],
        "reference_manifest_sha256": sha256(args.reference_manifest),
        "internal_events_are_temporal_gold": False,
        "manual_review_records": file_record(args.review_records, include_hash=True),
        "auxiliary_keyposes_used_for_alignment": False, "events": records,
    })
    return records


def verified_alignment(folder: Path, manifest: dict, manifest_hash: str, reference,
                       target, ref_track, target_track, ref_positions):
    metadata = load_object(folder / "alignment.json")
    if metadata["manifest_sha256"] != manifest_hash:
        raise ValueError("Cached alignment used a different reference manifest.")
    if Path(metadata["input_video"]).resolve() != target.video_path.resolve():
        raise ValueError("Cached alignment video identity mismatch.")
    if not np.isclose(metadata["source_fps"], target.source_fps, rtol=0, atol=1e-8):
        raise ValueError("Cached alignment FPS mismatch.")
    positions = resolve_tracking_samples(
        target_track, target, float(metadata["alignment_tracking_gap_seconds"]),
    )
    with np.load(folder / "dtw_path.npz", allow_pickle=False) as data:
        path = np.column_stack((data["target_sample_indices_0based"],
                                data["reference_sample_indices_0based"]))
        costs = data["local_geodesic_radians"].astype(np.float64)
        verify_path_archive(data, path, costs, reference, target)
    actual_costs = path_pose_costs(target_track.body_poses[positions[path[:, 0]]],
                                  ref_track.body_poses[ref_positions[path[:, 1]]])
    if not np.allclose(actual_costs, costs, rtol=1e-5, atol=1e-5):
        raise ValueError("Cached path costs differ from current tracking.")
    endpoints = metadata["endpoints"]
    if [e["move_id"] for e in endpoints] != [1, 2, 3]:
        raise ValueError("Cached alignment must have ordered first-three endpoints.")
    previous = -1
    for endpoint in endpoints:
        mid = endpoint["move_id"]
        gold = manifest["endpoint_keyposes"][mid - 1]
        ref_index = gold["sample_index_0based"]
        candidates = np.flatnonzero(path[:, 1] == ref_index)
        best = candidates[np.argmin(costs[candidates])]
        chosen = int(path[best, 0])
        source = int(target.source_indices[chosen])
        if (endpoint["sample_index_0based"] != chosen
                or endpoint["reference_source_frame_0based"] != int(reference.source_indices[ref_index])
                or endpoint["segment_start_source_frame_0based"] != previous + 1
                or endpoint["segment_end_source_frame_0based"] != source
                or source <= previous):
            raise ValueError(f"Cached form {mid} boundary/index is inconsistent with its path.")
        previous = source
    return metadata, path, costs, positions


def draw_cell(canvas, draw, left, top, image, title, footer, *, width=330, height=205):
    font = load_font(16)
    draw.text((left, top), title, font=font, fill="#253139")
    canvas.paste(fit_frame(image, width - 12, height - 54), (left, top + 26))
    draw.text((left, top + height - 23), footer, font=load_font(14), fill="#475963")


def render_subject(directory: Path, role: str, subject: str, events: list[dict],
                   target_video: Path, reference_frames: dict) -> None:
    indices = [point["target_source_frame_0based"] for e in events for point in e["points"].values()
               if point["target_source_frame_0based"] is not None]
    frames = read_video_frames(target_video, indices)
    frame_dir = directory / "frames"
    frame_dir.mkdir()
    width, row_height, header = 1344, 295, 62
    canvas = Image.new("RGB", (width, header + row_height * len(events)), "#f8fafb")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 14), f"前三式事件迁移 | {role} {subject} | DTW 5 FPS", font=load_font(24), fill="#172b35")
    for i, event in enumerate(events):
        top = header + i * row_height
        draw.line((12, top, width - 12, top), fill="#cbd5db")
        draw.text((12, top + 8), f"{event['move_id']:02d} {event['move_name']} | {event['label']}",
                  font=load_font(20), fill="#172b35")
        ref_point = event["points"]["representative"]
        ref_source = ref_point["reference_source_frame_0based"]
        draw_cell(canvas, draw, 12, top + 40, reference_frames[ref_source], "标准参考代表帧 BV1WE411W7JB",
                  f"{ref_point['reference_source_time_seconds']:.3f}s | frame0 {ref_source}")
        for col, name in enumerate(POINT_FIELDS, 1):
            point = event["points"][name]
            source = point["target_source_frame_0based"]
            left = 12 + col * 330
            title = {"start": "事件起点", "representative": "迁移代表帧", "end": "事件终点"}[name]
            if source is None:
                draw.text((left, top + 50), title + "：无候选", font=load_font(16), fill="#b02f3a")
                continue
            relative = f"frames/{event['pose_id']}_{name}.jpg"
            frames[source].save(directory / relative, quality=92)
            point["image"] = relative
            draw_cell(canvas, draw, left, top + 40, frames[source], title,
                      f"{point['target_source_time_seconds']:.3f}s | frame0 {source}")
        rep = event["points"]["representative"]
        note = f"代表帧候选 {rep['candidate_count']} | 跨度 {rep['candidate_span_seconds']:.3f}s"
        confirmed = event["manual_review_status"] == "correct"
        if confirmed:
            note += " | 已人工确认事件对应（保留自动诊断）"
        elif event["review_required"]:
            note += f" | 需检查：{len(event['diagnostic_flags'])}项（详见 CSV）"
        else:
            note += " | 无自动诊断警告（不表示已经人工确认）"
        draw.text((12, top + 255), note, font=load_font(16),
                  fill="#23785b" if confirmed else "#b02f3a" if event["review_required"] else "#475963")
    canvas.save(directory / "keypose_overview.jpg", quality=93)


def render_group(output: Path, reference_events: list[dict], reference_frames: dict,
                 subjects: list[dict], move_id: int, filename: str) -> None:
    events = [e for e in reference_events if e["move_id"] == move_id]
    width, cell, height, header = (len(subjects) + 1) * 270 + 24, 270, 250, 70
    canvas = Image.new("RGB", (width, header + height * len(events)), "#f8fafb")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 15), f"{events[0]['move_name']} | 参考与迁移代表帧", font=load_font(24), fill="#172b35")
    for i, ref_event in enumerate(events):
        top = header + i * height
        draw.text((12, top), ref_event["label"], font=load_font(20), fill="#172b35")
        source = ref_event["representative_source_frame_0based"]
        draw_cell(canvas, draw, 12, top + 30, reference_frames[source], "BV1WE411W7JB",
                  f"{ref_event['representative_time_seconds']:.3f}s | frame0 {source}", width=cell)
        for col, subject in enumerate(subjects, 1):
            event = next(e for e in subject["events"] if e["pose_id"] == ref_event["pose_id"])
            point = event["points"]["representative"]
            left = 12 + col * cell
            if point["target_source_frame_0based"] is None:
                draw.text((left, top + 35), subject["subject_id"] + " 无候选", font=load_font(16), fill="#b02f3a")
                continue
            with Image.open(subject["directory"] / point["image"]) as image:
                draw_cell(canvas, draw, left, top + 30, image, subject["subject_id"],
                          f"{point['target_source_time_seconds']:.3f}s | frame0 {point['target_source_frame_0based']}",
                          width=cell)
            if event["review_required"]:
                color = "#23785b" if event["manual_review_status"] == "correct" else "#c34445"
                draw.rectangle((left, top + 30, left + cell - 12, top + 30 + 204), outline=color, width=2)
    canvas.save(output / "figures" / filename, quality=93)


def write_report(output: Path, subjects: list[dict], reference_events: list[dict],
                 failures: list[dict], summary: dict) -> None:
    lines = ["# 前三式关键事件 DTW 迁移", "",
        "参考：BV1WE411W7JB。复用既有完整视频 5 FPS、23 个局部 SMPL 关节的 Geodesic DTW 路径，"
        "逐条核对路径的身份、索引、时间和实际 tracking 姿态距离。未重新运行 DTW、人体重建或 LLM。", "",
        f"共 {len(reference_events)} 个参考事件，{len(subjects)} 个目标视频，{summary['mapped_event_count']}/"
        f"{summary['event_count']} 个事件获得映射；{summary['review_required_count']} 个事件带自动诊断警告，"
        f"其中 {summary['confirmed_diagnostic_count']} 个已人工确认事件对应。", "",
        "## 解读边界", "",
        "内部代表帧取该参考 5 FPS 帧在 DTW 路径上、当前招式内候选的较早中位帧，不按学生姿态误差挑最好帧。"
        "结束事件复用已有最小局部距离的招式边界。参考原始帧、量化后的参考帧和目标实际帧分别保存。"
        "区间起止取候选包络，并限定在当前式内；越界候选全部留在 candidates.csv，且标记复核。", "",
        "红框仅表示尚未确认的匹配或 tracking 诊断，不是动作错误；绿框表示该诊断条目的事件对应已人工确认。无框不代表匹配已人工确认。"
        "内部事件没有人工时间戳金标准，因此不能把映射数量、单调性或候选跨度称为定位准确率。"
        "事件允许重叠；重复代表帧不静默改帧。白鹤亮翅抱手已按人工更正改用区间起点作代表帧；"
        "原查询中点仅留作历史溯源，不再参与当前迁移。", "",
        f"候选跨度警告阈值为 {summary['review_candidate_span_seconds']:.2f}s，仅用于优先检查，未经标定。", "",
        "## 人工更正参考帧", ""]
    for event in reference_events:
        correction = event.get("manual_reference_correction")
        if correction:
            lines += [f"- {event['move_name']}／{event['label']}：frame0 "
                      f"{correction['expected_original_source_frame_0based']} → "
                      f"{event['representative_source_frame_0based']}（{event['representative_time_seconds']:.3f}s）。"
                      "事件区间和招式边界不变；目标代表帧按更正后的参考帧重新迁移。", ""]
    lines += ["## 集中对照", ""]
    for mid in summary["move_ids"]:
        name = next(e["move_name"] for e in reference_events if e["move_id"] == mid)
        lines += [f"### {mid}. {name}", ""]
        if (output / "figures" / f"{mid:02d}_students.jpg").is_file():
            lines += [f"![学生对照](figures/{mid:02d}_students.jpg)", ""]
        for page in range(1, 3):
            filename = f"{mid:02d}_teachers_{page}.jpg"
            if (output / "figures" / filename).is_file():
                lines += [f"[教师对照：第{page}组](figures/{filename})", ""]
    lines += ["## 逐视频区间与代表帧", "", "| 类型 | 视频 | 映射 | 尚待确认的诊断 | 对照 |", "|---|---|---:|---:|---|"]
    for subject in subjects:
        relative = subject["directory"].relative_to(output).as_posix()
        events = subject["events"]
        lines.append(f"| {subject['role']} | {subject['subject_id']} | "
                     f"{sum(e['status']=='mapped' for e in events)}/{len(events)} | "
                     f"{sum(e['review_required'] and e['manual_review_status'] != 'correct' for e in events)} "
                     f"| [查看]({relative}/keypose_overview.jpg) |")
    lines += ["", "## 诊断与人工确认", "", "| 类型 | 视频 | 招式／事件 | 代表帧时间 | 警告 | 人工复核 |", "|---|---|---|---:|---|---|"]
    for subject in subjects:
        for event in subject["events"]:
            if event["review_required"]:
                point = event["points"]["representative"]
                timestamp = point["target_source_time_seconds"]
                text = "无候选" if timestamp is None else f"{timestamp:.3f}s"
                lines.append(f"| {subject['role']} | {subject['subject_id']} | {event['move_name']}／{event['label']} | "
                             f"{text} | {'; '.join(event['diagnostic_flags'])} | {event['manual_review_status']}："
                             f"{event['manual_review_note']} |")
    lines += ["", "警告释义：wide_candidate_span 为某一参考帧匹配到较宽的目标时间跨度；"
              "candidates_outside_form 为部分路径候选落在当前式以外；"
              "missing_tracking / nearest_substitution 为原始精确帧缺失或旧路径用了近邻姿态；"
              "near_track_switch 为 ID 切换附近；duplicate_event_representative 为相邻事件选到同帧；"
              "interval_collapsed_to_single_sample 为区间被压到一个 5 FPS 帧。", ""]
    if failures:
        lines += ["", "## 未完成视频", ""] + [f"- {r['role']}/{r['subject_id']}: {r['error']}" for r in failures]
    lines += ["", "## 数据文件", "",
              "- reference_event_manifest.json：冻结参考事件、动作要领、原始区间及来源哈希。",
              "- all_events.csv：逐视频逐事件的时间、索引、候选跨度、选帧策略及 tracking 状态。",
              "- candidates.csv：起点、代表帧、终点的全部 DTW 候选及是否越界／被选中。",
              "- review_queue.csv：自动诊断警告的事件及已有人工复核，不重复要求已确认条目复核。",
              "- manual_review_records_snapshot.json：人工参考帧更正和事件对应确认的冻结记录。",
              "- 各视频 keypose_events.json/csv、dtw_path.npz、summary.json 和 keypose_overview.jpg。", "",
              "本轮只验证跨视频事件对应，不生成质量分、教师指标范围或训练反馈。下一阶段再在确认可用的事件上测量姿态。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", type=Path,
                        default=PROJECT / "reference_data/BV1WE411W7JB/reference_manifest.json")
    parser.add_argument("--reference-events-root", type=Path,
                        default=PROJECT / "temporal_keypose_results/first3_v2/BV1WE411W7JB")
    parser.add_argument("--alignment-root", type=Path,
                        default=PROJECT / "endpoint_metric_results/first3_five_students")
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT / "temporal_keypose_transfer_results/first3_v2")
    parser.add_argument("--review-records", type=Path,
                        default=PROJECT / "reference_data/BV1WE411W7JB/temporal_event_review.json",
                        help="Explicit manual reference corrections and correspondence reviews.")
    parser.add_argument("--moves", nargs="+", type=int, choices=(1, 2, 3), default=[1, 2, 3])
    parser.add_argument("--students", nargs="*", default=["01", "02", "03", "04", "10"])
    parser.add_argument("--teachers", nargs="*", help="Defaults to the frozen alignment cohort.")
    parser.add_argument("--review-candidate-span-seconds", type=float, default=0.8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (not np.isfinite(args.review_candidate_span_seconds) or args.review_candidate_span_seconds < 0
            or len(set(args.moves)) != len(args.moves) or args.moves != sorted(args.moves)):
        raise ValueError("Invalid move order or candidate-span threshold.")
    cohort = load_object(args.alignment_root / "summary.json")
    teachers = cohort["teacher_ids"] if args.teachers is None else args.teachers
    for role, ids, allowed in (("teacher", teachers, cohort["teacher_ids"]),
                               ("student", args.students, cohort["student_ids"])):
        if len(set(ids)) != len(ids) or any(sid not in allowed for sid in ids):
            raise ValueError(f"Unknown/duplicate {role} ID.")
    if not teachers and not args.students:
        raise ValueError("Select at least one target video.")
    output = args.output_root.absolute()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {output}")
    manifest = load_object(args.reference_manifest)
    manifest_hash = sha256(args.reference_manifest)
    if manifest_hash != cohort["reference_manifest_sha256"]:
        raise ValueError("Alignment cohort uses a different reference manifest.")
    reference = inspect_video_sampling(manifest["reference"]["video_path"], 5.)
    manifest_boundaries(manifest, reference)
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir()
    shutil.copy2(args.reference_manifest, output / "reference_manifest_snapshot.json")
    events = freeze_reference(args, manifest, reference, output)
    review_records = load_object(output / "manual_review_records_snapshot.json")
    ref_track = load_stitched_primary_track(manifest["reference"]["tracking_path"])
    ref_positions = resolve_tracking_samples(ref_track, reference)
    ref_quality = track_frame_quality(ref_track, reference)
    for event in events:
        if any(ref_quality["status"][event[field]] != "valid" for field in POINT_FIELDS.values()):
            raise ValueError(f"Reference event tracking invalid: {event['pose_id']}")
    reference_frames = read_video_frames(reference.video_path,
        [event["representative_source_frame_0based"] for event in events])
    subjects, failures, candidate_rows, all_rows = [], [], [], []
    started = time.perf_counter()
    for role, ids in (("teacher", teachers), ("student", args.students)):
        for sid in ids:
            begin = time.perf_counter()
            folder = args.alignment_root / role / sid
            directory = output / role / sid
            directory.mkdir(parents=True)
            print(f"[{role}/{sid}] verify cached full DTW and transfer {len(events)} events", flush=True)
            try:
                metadata = load_object(folder / "alignment.json")
                target = inspect_video_sampling(metadata["input_video"], 5.)
                target_track = load_stitched_primary_track(metadata["input_tracking"])
                metadata, path, costs, positions = verified_alignment(
                    folder, manifest, manifest_hash, reference, target, ref_track, target_track, ref_positions,
                )
                if metadata["subject_id"] != sid or metadata["role"] != role:
                    raise ValueError("Cached alignment subject identity mismatch.")
                quality = track_frame_quality(target_track, target)
                endpoints = {e["move_id"]: e for e in metadata["endpoints"]}
                transferred, candidates = [], []
                for event in events:
                    result, rows = map_event(
                        event, path, costs, reference, target, endpoints[event["move_id"]],
                        positions, target_track, ref_quality, quality,
                        review_span_seconds=args.review_candidate_span_seconds,
                    )
                    result.update(role=role, subject_id=sid)
                    transferred.append(result)
                    candidates.extend({**row, "role": role, "subject_id": sid} for row in rows)
                flag_event_order(transferred)
                apply_correspondence_reviews(transferred, review_records, role, sid)
                render_subject(directory, role, sid, transferred, target.video_path, reference_frames)
                write_json(directory / "keypose_events.json", {"schema_version": "1.0", "role": role,
                    "subject_id": sid, "reference_video_id": manifest["reference"]["video_id"], "events": transferred})
                flat = [flatten_event(e) for e in transferred]
                write_csv(directory / "keypose_events.csv", flat)
                shutil.copy2(folder / "dtw_path.npz", directory / "dtw_path.npz")
                substituted = int(np.count_nonzero(target_track.frame_numbers[positions] != target.source_indices + 1))
                write_json(directory / "summary.json", {
                    "status": "completed", "role": role, "subject_id": sid,
                    "source_fps": target.source_fps, "source_frame_count": target.source_frame_count,
                    "sample_fps": target.sample_fps, "dtw_path_length": len(path),
                    "input_video": file_record(target.video_path, include_hash=False),
                    "input_tracking": file_record(Path(metadata["input_tracking"]), include_hash=False),
                    "alignment_metadata": file_record(folder / "alignment.json", include_hash=True),
                    "alignment_path": file_record(folder / "dtw_path.npz", include_hash=True),
                    "path_costs_verified_against_current_tracking": True,
                    "alignment_tracking_gap_seconds": metadata["alignment_tracking_gap_seconds"],
                    "alignment_substituted_sample_count": substituted,
                    "used_track_ids": list(target_track.used_track_ids),
                    "track_switch_source_frames_0based": quality["switch_source_frames"].tolist(),
                    "event_count": len(transferred), "review_required_count": sum(e["review_required"] for e in transferred),
                    "confirmed_diagnostic_count": sum(e["review_required"] and e["manual_review_status"] == "correct"
                                                      for e in transferred),
                    "elapsed_seconds": time.perf_counter() - begin,
                })
                candidate_rows.extend(candidates)
                all_rows.extend(flat)
                subjects.append({"role": role, "subject_id": sid, "directory": directory, "events": transferred})
                print(f"[{role}/{sid}] completed, flagged {sum(e['review_required'] for e in transferred)} events", flush=True)
            except Exception as exc:
                failure = {"role": role, "subject_id": sid, "error": str(exc)}
                failures.append(failure)
                write_json(directory / "summary.json", {"status": "failed", **failure})
                print(f"[{role}/{sid}] FAILED: {exc}", flush=True)
    if all_rows:
        write_csv(output / "all_events.csv", all_rows)
        write_csv(output / "candidates.csv", candidate_rows)
        queue = [{**row, "manual_verdict": row["manual_review_status"],
                  "reviewer": "user" if row["manual_review_date"] else "", "notes": row["manual_review_note"]}
                 for row in all_rows if row["review_required"]]
        if queue:
            write_csv(output / "review_queue.csv", queue)
    for mid in args.moves:
        students = [s for s in subjects if s["role"] == "student"]
        if students:
            render_group(output, events, reference_frames, students, mid, f"{mid:02d}_students.jpg")
        successful_teachers = [s for s in subjects if s["role"] == "teacher"]
        for i in range(0, len(successful_teachers), 5):
            render_group(output, events, reference_frames, successful_teachers[i:i + 5], mid,
                         f"{mid:02d}_teachers_{i // 5 + 1}.jpg")
    flag_counts = Counter(flag for s in subjects for e in s["events"] for flag in e["diagnostic_flags"])
    summary = {
        "schema_version": "1.0", "status": "completed" if not failures else "partial_failure",
        "reference_video_id": manifest["reference"]["video_id"], "teacher_ids": teachers,
        "student_ids": args.students, "move_ids": args.moves, "reference_event_count": len(events),
        "completed_subject_count": len(subjects), "event_count": len(all_rows),
        "mapped_event_count": sum(r["status"] == "mapped" for r in all_rows),
        "review_required_count": sum(r["review_required"] for r in all_rows),
        "confirmed_diagnostic_count": sum(r["review_required"] and r["manual_review_status"] == "correct"
                                          for r in all_rows),
        "unconfirmed_diagnostic_count": sum(r["review_required"] and r["manual_review_status"] != "correct"
                                            for r in all_rows),
        "diagnostic_flag_counts": dict(flag_counts), "failures": failures,
        "review_candidate_span_seconds": args.review_candidate_span_seconds,
        "reference_event_manifest": file_record(output / "reference_event_manifest.json", include_hash=True),
        "reference_manifest": file_record(args.reference_manifest, include_hash=True),
        "manual_review_records": file_record(args.review_records, include_hash=True),
        "transfer_source": file_record(PROJECT / "aqa3d/temporal_transfer.py", include_hash=True),
        "runner_source": file_record(Path(__file__), include_hash=True),
        "internal_event_accuracy_measured": False, "llm_called": False,
        "new_dtw_computed": False, "new_3d_reconstruction_run": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_report(output, subjects, events, failures, summary)
    write_json(output / "summary.json", summary)
    print(f"Completed -> {output / 'report.md'}", flush=True)
    if failures:
        raise SystemExit("Some videos failed; see summary.json and report.md.")


if __name__ == "__main__":
    main()
