#!/usr/bin/env python
"""Compare frozen temporal event geometry without rerunning DTW or an LLM."""

from __future__ import annotations

import argparse
from collections import Counter
import io
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from aqa3d.endpoint_metrics import MOVE_METRICS, endpoint_window
from aqa3d.pose_measurements import measurement_contract, reference_statistics
from aqa3d.temporal_event_metrics import (
    LIMITATIONS, compare_measurement, event_metric_plan, metric_catalog, process_change,
    scalar_series, stability_warnings, summarize_scalar,
)
from run_pose_measurements import FrozenInputs, load_window
from run_temporal_keypose_localization import (
    _font, letterbox, load_segment_joints, read_video_frames, sha256, write_csv, write_json,
)


PROJECT = Path(__file__).resolve().parent


def verify_identity(record: dict[str, Any]) -> None:
    path = Path(record["path"])
    stat = path.stat()
    if stat.st_size != record["size_bytes"] or stat.st_mtime_ns != record["mtime_ns"]:
        raise ValueError(f"Input identity changed: {path}")
    if record.get("sha256") and sha256(path) != record["sha256"]:
        raise ValueError(f"Input hash changed: {path}")


def source_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def reference_subject(events: list[dict], manifest: dict) -> tuple[dict, list[dict]]:
    reference = manifest["reference"]
    meta = {"role": "reference", "subject_id": reference["video_id"],
            "source_fps": reference["source_fps"], "source_frame_count": reference["source_frame_count"],
            "input_tracking": source_identity(Path(reference["tracking_path"])),
            "input_video": source_identity(Path(reference["video_path"]))}
    translated = []
    for event in events:
        points = {}
        for name, field in (("start", "interval_start_source_frame_0based"),
                            ("representative", "representative_source_frame_0based"),
                            ("end", "interval_end_source_frame_0based")):
            frame = event[field]
            points[name] = {"target_source_frame_0based": frame,
                            "target_phalp_frame_1based": frame + 1,
                            "target_source_time_seconds": frame / reference["source_fps"]}
        translated.append({**event, "status": "mapped", "points": points,
            "target_segment_start_source_frame_0based": event["reference_segment_start_source_frame_0based"],
            "target_segment_end_source_frame_0based": event["reference_segment_end_source_frame_0based"],
            "review_required": False, "manual_review_status": event["annotation_status"]})
    return meta, translated


def validate_events(events: list[dict], reference: list[dict], fps: float, frame_count: int) -> None:
    if len(events) != len(reference) or {e["pose_id"] for e in events} != {e["pose_id"] for e in reference}:
        raise ValueError("Subject does not contain exactly the frozen reference events")
    lookup = {e["pose_id"]: e for e in reference}
    for event in events:
        source = lookup[event["pose_id"]]
        if event["status"] != "mapped" or event["technique"] != source["technique"]:
            raise ValueError("Unmapped event or changed technique")
        start, stop = event["target_segment_start_source_frame_0based"], event["target_segment_end_source_frame_0based"]
        selected = [event["points"][p]["target_source_frame_0based"] for p in ("start", "representative", "end")]
        if any(type(f) is not int for f in [start, stop, *selected]) or not 0 <= start <= min(selected) <= max(selected) <= stop < frame_count:
            raise ValueError("Invalid source frame bounds")
        if not selected[0] <= selected[1] <= selected[2]:
            raise ValueError("Nonmonotone event points")
        for name, point in event["points"].items():
            f = point["target_source_frame_0based"]
            if point["target_phalp_frame_1based"] != f + 1 or not np.isclose(point["target_source_time_seconds"], f / fps, atol=1e-8, rtol=0):
                raise ValueError("Time / frame convention mismatch")
            if "reference_source_frame_0based" in point:
                field = {"start": "interval_start_source_frame_0based", "representative": "representative_source_frame_0based", "end": "interval_end_source_frame_0based"}[name]
                if point["reference_source_frame_0based"] != source[field]:
                    raise ValueError("Subject transfer uses a stale reference event")


def endpoint_compatibility(rows: list[dict], previous: list[dict], event: dict) -> list[dict]:
    expected = set(MOVE_METRICS[event["move_id"]])
    old = {r["metric_id"]: r for r in previous if int(r["move_id"]) == event["move_id"]}
    new = {r["metric_id"]: r for r in rows if r["kind"] == "pose" and r["metric_id"] in expected}
    if set(old) != expected or set(new) != expected:
        raise ValueError("F-group endpoint coverage changed")
    records = []
    for mid in MOVE_METRICS[event["move_id"]]:
        errors = []
        if int(old[mid]["center_source_frame_0based"]) != new[mid]["representative_source_frame_0based"]:
            raise ValueError("Frozen endpoint frame changed")
        if old[mid]["status"] != new[mid]["status"]:
            raise ValueError("Frozen endpoint validity changed")
        for key in ("value", "window_median", "center_value", "window_min", "window_max",
                    "valid_frame_count", "requested_frame_count"):
            a = float(old[mid][key]) if old[mid][key] else None
            b = new[mid][key]
            if a is None or b is None:
                if a != b:
                    raise ValueError(f"Frozen endpoint null value changed: {mid}:{key}")
            else:
                errors.append(abs(a - b))
        error = max(errors, default=0)
        if error > 1e-6:
            raise ValueError(f"Frozen endpoint geometry changed: {mid}, error={error}")
        records.append({"role": new[mid]["role"], "subject_id": new[mid]["subject_id"],
                        "pose_id": event["pose_id"], "metric_id": mid, "status": "pass", "max_abs_error": error})
    return records


def measure_subject(meta: dict, events: list[dict], plans: dict, output: Path,
                    transfer: FrozenInputs, baseline: FrozenInputs, device: str,
                    batch_size: int) -> tuple[list[dict], list[dict]]:
    role, sid, fps = meta["role"], meta["subject_id"], meta["source_fps"]
    directory = output / role / sid
    (directory / "frames").mkdir(parents=True)
    for name in ("input_video", "input_tracking"):
        verify_identity(meta[name])
    start = min(e["target_segment_start_source_frame_0based"] for e in events)
    end = max(e["target_segment_end_source_frame_0based"] for e in events)
    frames, joints, statuses, track_ids, quality = load_segment_joints(
        Path(meta["input_tracking"]["path"]), start, end, device=device, batch_size=batch_size)
    np.savez_compressed(directory / "native_joints.npz", source_frames_0based=frames,
                        native_smpl24_joints=joints, tracking_status=statuses, source_track_ids=track_ids)
    catalog = metric_catalog()
    ids = sorted({m for e in events for k in ("pose_metric_ids", "process_metric_ids") for m in plans[e["pose_id"]][k]})
    series = {mid: scalar_series(joints, frames, statuses, catalog[mid]) for mid in ids}
    archive = {"source_frames_0based": frames}
    for mid, (values, validity) in series.items():
        archive[mid + "__value"] = values
        archive[mid + "__status"] = validity
    np.savez_compressed(directory / "frame_metrics.npz", **archive)
    old_meta = baseline.json(f"{role}/{sid}/alignment.json") if role != "reference" else None
    previous = baseline.csv(f"{role}/{sid}/endpoint_metrics.csv") if role != "reference" else []
    if old_meta is not None:
        if (old_meta["role"], old_meta["subject_id"]) != (role, sid):
            raise ValueError("Frozen endpoint subject identity changed")
        for key in ("input_video", "input_tracking"):
            if Path(old_meta[key]).resolve() != Path(meta[key]["path"]).resolve():
                raise ValueError("Frozen endpoint cache refers to different video / tracking")
    reference_images = read_video_frames(Path(meta["input_video"]["path"]),
        [p["target_source_frame_0based"] for e in events for p in e["points"].values()]) if role == "reference" else {}
    rows, compatible, contexts = [], [], []
    for event in events:
        plan = plans[event["pose_id"]]
        center = event["points"]["representative"]["target_source_frame_0based"]
        segment_start = event["target_segment_start_source_frame_0based"]
        segment_end = event["target_segment_end_source_frame_0based"]
        expected = endpoint_window(center, segment_start, segment_end, fps)
        cached = None
        if plan["contains_f_group_endpoint_metrics"] and old_meta is not None:
            old_event = next(e for e in old_meta["endpoints"] if e["move_id"] == event["move_id"])
            if (old_event["center_source_frame_0based"] != center or
                    old_event["segment_start_source_frame_0based"] != segment_start or
                    old_event["segment_end_source_frame_0based"] != segment_end or
                    not np.isclose(old_meta["source_fps"], fps, atol=1e-8, rtol=0)):
                raise ValueError("New endpoint is not identical to the F-group endpoint")
            cached = load_window(baseline, f"{role}/{sid}", old_event, fps, 0.2)
        image_paths = {}
        for name, point in event["points"].items():
            relative = f"frames/{event['pose_id']}_{name}.jpg"
            destination = directory / relative
            if role == "reference":
                reference_images[point["target_source_frame_0based"]].save(destination, quality=93)
            else:
                frozen = transfer.read(f"{role}/{sid}/{point['image']}")
                destination.write_bytes(frozen)
                with Image.open(io.BytesIO(frozen)) as image:
                    image.verify()
            image_paths[name] = destination.relative_to(output).as_posix()
        context = {**event, "role": role, "subject_id": sid, "images": image_paths,
                   "endpoint_technique": plan.get("endpoint_technique"),
                   "pose_window_geometry_source": "frozen_f_group_window" if cached is not None else "tracking_native_smpl24"}
        contexts.append(context)
        common = {"role": role, "subject_id": sid, "move_id": event["move_id"],
                  "pose_id": event["pose_id"], "event_label": event["label"],
                  "representative_source_frame_0based": center, "representative_phalp_frame_1based": center + 1,
                  "representative_time_seconds": center / fps,
                  "source_fps": fps, "representative_image": image_paths["representative"]}
        event_rows = []
        for mid in plan["pose_metric_ids"]:
            spec = catalog[mid]
            if cached is not None:
                cj, cf, cs = cached
                values, validity = scalar_series(cj, cf, cs, spec)
                if not np.array_equal(cf, expected):
                    raise ValueError("Frozen F-group window frames changed")
            else:
                values, validity = series[mid]
                selected = np.isin(frames, expected)
                values, validity = values[selected], validity[selected]
            stats = summarize_scalar(values, validity, expected, center)
            warnings = stability_warnings(stats, spec.unit)
            event_rows.append({**common, "kind": "pose", "metric_id": mid, "label": spec.label,
                               "unit": spec.unit, "aspect": spec.aspect, **stats,
                               "warnings": warnings, "raw_endpoint_change": None,
                               "geometry_source": context["pose_window_geometry_source"]})
        if plan["contains_f_group_endpoint_metrics"] and old_meta is not None:
            compatible += endpoint_compatibility(event_rows, previous, event)
        for mid in plan["process_metric_ids"]:
            spec = catalog[mid]
            values, validity = series[mid]
            interval_start = event["points"]["start"]["target_source_frame_0based"]
            interval_end = event["points"]["end"]["target_source_frame_0based"]
            stats = process_change(values, validity, frames, interval_start, interval_end, fps, spec.unit)
            event_rows.append({**common, "kind": "process", "metric_id": mid + "__change",
                "label": spec.label + "：末窗减首窗", "unit": spec.unit, "aspect": spec.aspect,
                **stats, "center_value": None, "window_median": None, "window_min": None,
                "window_max": None, "valid_frame_count": None, "requested_frame_count": None,
                "geometry_source": "tracking_native_smpl24", "interval_start_source_frame_0based": interval_start,
                "interval_end_source_frame_0based": interval_end})
        rows += event_rows
    write_json(directory / "measurements.json", {"subject": meta, "tracking_quality": quality,
                                               "events": contexts, "measurements": rows})
    return rows, compatible


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


RELATIONS = {"below": "低于", "above": "高于", "within": "区间内",
             "unavailable": "无效", "insufficient_teachers": "教师不足", "degenerate_reference": "区间退化"}
CATEGORIES = {"difference_evidence": "一致差异证据", "review_measurement": "谨慎解读", "no_interval_difference": "未见区间差异"}


def render_gallery(output: Path, event: dict, metas: list[dict]) -> None:
    width, cell_h, margin = 310, 235, 12
    canvas = Image.new("RGB", (6 * width + 7 * margin, 3 * cell_h + 70), "#f7f9fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 12), event["label"], font=_font(22, True), fill="#202a32")
    draw.text((margin, 40), "参考锚点与五位学生 / 十位教师；显示冻结的代表帧", font=_font(15), fill="#52606a")
    order = [m for m in metas if m["role"] == "reference"] + [m for m in metas if m["role"] == "student"]
    order += [m for m in metas if m["role"] == "teacher"]
    for index, meta in enumerate(order):
        row, column = (0, index) if index < 6 else (1 + (index - 6) // 5, (index - 6) % 5)
        left, top = margin + column * (width + margin), 70 + row * cell_h
        data = json.loads((output / meta["role"] / meta["subject_id"] / "measurements.json").read_text())
        context = next(e for e in data["events"] if e["pose_id"] == event["pose_id"])
        label = {"reference": "参考", "student": "学生", "teacher": "教师"}[meta["role"]] + " " + meta["subject_id"]
        draw.text((left, top), label, font=_font(15), fill="#202a32")
        with Image.open(output / context["images"]["representative"]) as image:
            canvas.paste(letterbox(image, width, 175), (left, top + 25))
        point = context["points"]["representative"]
        draw.text((left, top + 206), f"{point['target_source_time_seconds']:.3f}s | frame0 {point['target_source_frame_0based']}",
                  font=_font(14), fill="#52606a")
    canvas.save(output / "figures" / f"{event['pose_id']}.jpg", quality=93)


def write_reports(output: Path, events: list[dict], references: list[dict],
                  comparisons: list[dict], summary: dict, metas: list[dict]) -> None:
    (output / "events").mkdir()
    (output / "cases").mkdir()
    ref_lookup = {(r["pose_id"], r["metric_id"]): r for r in references}
    header = ["# 前三式：多事件姿态与过程指标", "",
        "本轮只做几何测量与教师样本比较，不调用 LLM，不生成动作错误结论、训练指导、总分或排名。", "",
        f"10 个事件、10 位教师、5 位学生；保留全部 26 项 F 组结束定势指标。"
        f"共 {summary['student_comparison_count']} 项学生指标比较，{summary['endpoint_compatibility_count']} 项历史结束定势数值校验通过。", "",
        "姿势单元格采用 **窗口中位数（中心帧值）**；过程单元格采用 **末窗中位数减首窗中位数（原始末帧减首帧）**。"
        "两类指标分别统计教师区间，不能相互混用。默认区间是未缩放 MAD 的 median ± 2×MAD，P10–P90 为辅助对照。", "",
        "所有姿势窗口为代表帧前后各 0.2s，截断于当前招式。过程采用区间内首尾各不超过 0.2s 的单侧、互不重叠窗口。"
        "不插值、不补帧、不重新挑选代表帧；失效教师值保留为 null，不按异常值删教师。", "",
        "白鹤亮翅转体抱手固定使用参考 frame0 **1328**。两项已人工确认的 DTW 诊断保留为溯源，不视为动作错误。", "",
        "## 阅读入口", "", "| 事件 | 完整指标与图片 |", "| --- | --- |"]
    for event in events:
        render_gallery(output, event, metas)
        eid = event["pose_id"]
        header.append(f"| {event['move_id']}. {event['label']} | [对比表](events/{eid}.md) |")
        lines = [f"# {event['label']}", "", event.get("endpoint_technique", event["technique"]), "",
                 f"参考区间 {event['start_time_seconds']:.3f}–{event['end_time_seconds']:.3f}s；"
                 f"代表 {event['representative_time_seconds']:.3f}s / frame0 {event['representative_source_frame_0based']}。", "",
                 f"![参考、学生与教师](../figures/{eid}.jpg)", "",
                 "数值差异不是动作错误判定；全部指标保留，未按显著性筛除。谨慎解读的原因详见 JSON/CSV。", ""]
        for kind, title in (("pose", "代表帧姿势"), ("process", "区间变化量（末减首）")):
            selected = [r for r in references if r["pose_id"] == eid and r["kind"] == kind]
            if not selected:
                continue
            lines += [f"## {title}", "", "| 指标 | 教师 median ± 2MAD | P10–P90 | 有效教师 | 01 | 02 | 03 | 04 | 10 |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            for ref in selected:
                stats = ref["reference"]
                cells = [f"{ref['label']} ({ref['unit']})", f"{fmt(stats['median_minus_2mad'])}–{fmt(stats['median_plus_2mad'])}",
                         f"{fmt(stats['p10'])}–{fmt(stats['p90'])}", f"{stats['valid_teacher_count']}/10"]
                for sid in summary["student_ids"]:
                    row = next(r for r in comparisons if (r["pose_id"], r["metric_id"], r["subject_id"]) == (eid, ref["metric_id"], sid))
                    raw = row["center_value"] if kind == "pose" else row["raw_endpoint_change"]
                    cells.append(f"{fmt(row['value'])} ({fmt(raw)})；{RELATIONS[row['median_2mad_relation']]}；{CATEGORIES[row['category']]}")
                lines.append("| " + " | ".join(cells) + " |")
            lines += ["", "### 教师原始值（窗口汇总，不截尾）", "",
                      "| 指标 | " + " | ".join(summary["teacher_ids"]) + " |", "| --- | " + " | ".join(["---"] * 10) + " |"]
            for ref in selected:
                cells = [ref["label"]] + [fmt(s["value"]) + (" *" if s["warnings"] else "") for s in ref["reference"]["teacher_values"]]
                lines.append("| " + " | ".join(cells) + " |")
            lines += ["", "* 表示短窗变化、中心值差异或缺帧提醒；仍保留该教师原始值并参与有效样本统计。", ""]
        (output / "events" / f"{eid}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    header += ["", "## 集中导航", "", "| 学生 | 两种区间一致的差异（含谨慎项） | 稳定的一致差异证据 | 谨慎解读 | 案例文件 |",
               "| --- | --- | --- | --- | --- |"]
    for sid in summary["student_ids"]:
        rows = [r for r in comparisons if r["subject_id"] == sid]
        header.append(f"| {sid} | {sum(r['consistent_outside'] for r in rows)} | "
                      f"{sum(r['category'] == 'difference_evidence' for r in rows)} | "
                      f"{sum(r['category'] == 'review_measurement' for r in rows)} | [逐事件证据](cases/{sid}.md) |")
        lines = [f"# 学生 {sid}：几何差异导航", "", "不表示动作错误数量或学生水平。先看一致差异，再看谨慎项；未丢弃其余指标。", ""]
        for event in events:
            eid = event["pose_id"]
            selected = [r for r in rows if r["pose_id"] == eid]
            context = json.loads((output / "student" / sid / "measurements.json").read_text())
            context = next(e for e in context["events"] if e["pose_id"] == eid)
            lines += [f"## {event['label']}", "", event.get("endpoint_technique", event["technique"]), "",
                      f"![学生代表帧](../{context['images']['representative']})", "",
                      f"[参考锚点与十位教师对照、完整数值](../events/{eid}.md)", "",
                      "| 指标 | 数值 | 主区间判断 | 辅区间判断 | 证据状态 | 提醒 |", "| --- | --- | --- | --- | --- | --- |"]
            ordered = sorted(selected, key=lambda r: (not r["consistent_outside"], r["category"] != "difference_evidence"))
            for r in ordered:
                lines.append(f"| {r['label']} | {fmt(r['value'])} | {RELATIONS[r['median_2mad_relation']]} | "
                             f"{RELATIONS[r['p10_p90_relation']]} | {CATEGORIES[r['category']]} | "
                             f"{', '.join(r['review_reasons']) or '无自动提醒'} |")
            write_json(output / "cases" / f"{sid}_{eid}.json", {
                "event": context, "limitations": LIMITATIONS,
                "measurements": [{**r, "teacher_reference": ref_lookup[(eid, r["metric_id"])]["reference"]} for r in selected]})
            lines.append("")
        (output / "cases" / f"{sid}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    header += ["", "## 局限", ""] + [f"- {s}" for s in LIMITATIONS]
    header += ["", "以上数量只用于导航，不是准确率。没有内部事件质量人工金标准；本轮不估计动作诊断性能。", "",
               "原始关节与逐帧信号在各视频目录的 native_joints.npz / frame_metrics.npz；教师十个值、中心值、窗口范围及状态在 teacher_reference.json。", "",
               "重跑方式：`python run_temporal_event_metrics.py --output-root <新的输出目录>`。"]
    (output / "comparison_report.md").write_text("\n".join(header) + "\n", encoding="utf-8")


def compact_csv(rows: list[dict]) -> list[dict]:
    keys = ("role", "subject_id", "move_id", "pose_id", "kind", "metric_id", "label", "unit",
            "representative_source_frame_0based", "representative_phalp_frame_1based", "representative_time_seconds",
            "value", "center_value", "window_median", "window_min", "window_max", "raw_endpoint_change",
            "status", "geometry_source", "representative_image")
    result = []
    for row in rows:
        record = {k: row[k] for k in keys}
        for key in ("warnings", "median_2mad_relation", "p10_p90_relation", "category", "review_reasons"):
            value = row.get(key, "")
            record[key] = ";".join(value) if isinstance(value, list) else value
        result.append(record)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    transfer, baseline = FrozenInputs(args.transfer_root), FrozenInputs(args.endpoint_root)
    source_summary = transfer.json("summary.json")
    old_summary = baseline.json("summary.json")
    if source_summary["status"] != "completed" or old_summary["status"] != "ok":
        raise ValueError("Source experiments must be complete")
    teachers, students = source_summary["teacher_ids"], source_summary["student_ids"]
    if len(teachers) != 10 or students != ["01", "02", "03", "04", "10"]:
        raise ValueError("This version requires the frozen ten-teacher / five-student cohort")
    if teachers != old_summary["teacher_ids"] or students != old_summary["student_ids"]:
        raise ValueError("Cohort changed between transfer and endpoint experiments")
    manifest = transfer.json("reference_manifest_snapshot.json")
    reference = transfer.json("reference_event_manifest.json")
    events = reference["events"]
    if (len(events) != 10 or len({e["pose_id"] for e in events}) != 10 or
            old_summary["reference_manifest_sha256"] != reference["reference_manifest_sha256"] or
            transfer.hashes["reference_manifest_snapshot.json"] != reference["reference_manifest_sha256"]):
        raise ValueError("Reference snapshot / endpoint provenance mismatch")
    gather = next(e for e in events if e["pose_id"] == "3.gather_hands")
    if gather["representative_source_frame_0based"] != 1328:
        raise ValueError("Expected corrected Baiheliangchi gather representative frame0 1328")
    if reference["reference_video_id"] in teachers:
        raise ValueError("The reference anchor must not be included as an eleventh teacher")
    for event in events:
        if event_metric_plan(event)["contains_f_group_endpoint_metrics"]:
            endpoint = next(e for e in manifest["endpoint_keyposes"] if e["move_id"] == event["move_id"])
            event["endpoint_technique"] = endpoint["final_technique_step"]
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory; refusing to overwrite {output}")
    output.mkdir(parents=True)
    (output / "figures").mkdir()
    (output / "snapshots").mkdir()
    for name in ("summary.json", "reference_event_manifest.json", "reference_manifest_snapshot.json", "manual_review_records_snapshot.json"):
        (output / "snapshots" / name).write_bytes(transfer.read(name))
    plans = {e["pose_id"]: event_metric_plan(e) for e in events}
    for event in events:
        if "endpoint_technique" in event:
            plans[event["pose_id"]]["endpoint_technique"] = event["endpoint_technique"]
    catalog = metric_catalog()
    write_json(output / "metric_definitions.json", {
        "schema_version": "1.0", "measurement_contract": measurement_contract(),
        "plans": list(plans.values()), "metrics": {m: s.definition() for m, s in catalog.items()},
        "limitations": LIMITATIONS, "window_seconds_each_side": 0.2,
        "window_clipping": "within_current_move", "process_endpoint_windows": "inside_interval_disjoint",
        "minimum_valid_frames": 3, "minimum_valid_ratio": 0.7, "minimum_teachers": 7,
        "stability_warning_thresholds": {"degree": {"center_delta": 5, "window_span": 10},
                                          "ratio": {"center_delta": 0.05, "window_span": 0.1}},
        "thresholds_are_uncalibrated_diagnostics": True})
    metas, subjects = [], []
    meta, translated = reference_subject(events, manifest)
    metas.append(meta)
    subjects.append(translated)
    for role, ids in (("teacher", teachers), ("student", students)):
        for sid in ids:
            prefix = f"{role}/{sid}"
            meta = transfer.json(prefix + "/summary.json")
            data = transfer.json(prefix + "/keypose_events.json")
            if (meta["role"], meta["subject_id"], data["role"], data["subject_id"], data["reference_video_id"]) != (
                    role, sid, role, sid, reference["reference_video_id"]):
                raise ValueError("Subject transfer identity mismatch")
            for key in ("alignment_metadata", "alignment_path"):
                verify_identity(meta[key])
            metas.append(meta)
            subjects.append(data["events"])
    rows, compatibility, unconfirmed = [], [], {}
    for index, (meta, subject_events) in enumerate(zip(metas, subjects), 1):
        validate_events(subject_events, events, meta["source_fps"], meta["source_frame_count"])
        for event in subject_events:
            unconfirmed[(meta["role"], meta["subject_id"], event["pose_id"])] = bool(
                event.get("review_required") and event.get("manual_review_status") != "correct")
        print(f"[{index}/{len(metas)}] {meta['role']} {meta['subject_id']}: native joints + frozen event metrics", flush=True)
        measured, checks = measure_subject(meta, subject_events, plans, output, transfer, baseline, args.device, args.batch_size)
        rows += measured
        compatibility += checks
    refs = []
    for event in events:
        for row in [r for r in rows if r["role"] == "reference" and r["pose_id"] == event["pose_id"]]:
            samples = []
            for sid in teachers:
                measured = next(r for r in rows if (r["role"], r["subject_id"], r["pose_id"], r["metric_id"]) == ("teacher", sid, row["pose_id"], row["metric_id"]))
                samples.append({"subject_id": sid, "value": measured["value"], "status": measured["status"],
                                "center_value": measured["center_value"], "raw_endpoint_change": measured["raw_endpoint_change"],
                                "window_min": measured["window_min"], "window_max": measured["window_max"],
                                "warnings": measured["warnings"]})
            refs.append({"pose_id": row["pose_id"], "kind": row["kind"], "metric_id": row["metric_id"],
                         "label": row["label"], "unit": row["unit"], "reference": reference_statistics(samples)})
    lookup = {(r["pose_id"], r["metric_id"]): r["reference"] for r in refs}
    comparisons = [{**r, **compare_measurement(r, lookup[(r["pose_id"], r["metric_id"])],
        unconfirmed[(r["role"], r["subject_id"], r["pose_id"])])} for r in rows if r["role"] == "student"]
    write_json(output / "measurements.json", rows)
    write_json(output / "teacher_reference.json", refs)
    write_json(output / "student_comparison.json", comparisons)
    write_json(output / "endpoint_compatibility.json", compatibility)
    write_csv(output / "measurements.csv", compact_csv(rows))
    write_csv(output / "student_comparison.csv", compact_csv(comparisons))
    summary = {"schema_version": "1.0", "status": "completed", "reference_video_id": reference["reference_video_id"],
        "teacher_ids": teachers, "student_ids": students, "reference_event_count": len(events),
        "mapped_subject_event_count": len(events) * (len(teachers) + len(students)),
        "metric_definition_count": len(refs), "pose_metric_slot_count": sum(r["kind"] == "pose" for r in refs),
        "process_metric_slot_count": sum(r["kind"] == "process" for r in refs),
        "student_comparison_count": len(comparisons), "endpoint_compatibility_count": len(compatibility),
        "endpoint_compatibility_passed": all(r["status"] == "pass" for r in compatibility),
        "measurement_status_counts": dict(Counter(r["status"] for r in rows)),
        "comparison_categories": dict(Counter(r["category"] for r in comparisons)),
        "reference_validity_counts": dict(Counter(r["reference"]["status"] for r in refs)),
        "corrected_gather_reference_source_frame_0based": 1328, "reference_anchor_in_teacher_statistics": False,
        "llm_called": False, "new_dtw_computed": False, "new_video_reconstruction_run": False,
        "teacher_mad_scaled": False, "action_quality_accuracy_estimated": False,
        "device": args.device, "batch_size": args.batch_size}
    write_reports(output, events, refs, comparisons, summary, metas)
    for meta in metas:
        for key in ("input_video", "input_tracking"):
            verify_identity(meta[key])
    transfer.verify()
    baseline.verify()
    for relative in ("run_temporal_event_metrics.py", "aqa3d/temporal_event_metrics.py", "aqa3d/pose_measurements.py",
                     "run_temporal_keypose_localization.py", "run_pose_measurements.py"):
        dest = output / "snapshots" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / relative, dest)
    write_json(output / "input_provenance.json", {"transfer_root": str(transfer.root),
        "endpoint_root": str(baseline.root), "transfer_input_sha256": transfer.hashes,
        "endpoint_input_sha256": baseline.hashes, "subjects": metas})
    summary["elapsed_seconds"] = time.monotonic() - started
    write_json(output / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-root", type=Path, default=PROJECT / "temporal_keypose_transfer_results" / "first3_v2")
    parser.add_argument("--endpoint-root", type=Path, default=PROJECT / "endpoint_metric_results" / "first3_five_students")
    parser.add_argument("--output-root", type=Path, default=PROJECT / "temporal_event_metric_results" / "first3_five_students_v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
