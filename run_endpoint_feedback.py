"""Generate auditable first-three-form feedback from saved endpoint metrics, without rerunning DTW."""

import argparse
from collections import Counter
from dataclasses import asdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

from PIL import Image

from aqa3d.endpoint_feedback import (
    FeedbackConfig, STATUS_LABELS, coach_summary, evaluate_feedback, make_rules, validate_rules,
)
from aqa3d.endpoint_metrics import MOVE_METRICS


PROJECT = Path(__file__).resolve().parent
NUMERIC_FIELDS = {
    "move_id", "center_source_frame_0based", "center_phalp_frame_1based", "boundary_time_seconds",
    "source_frame_time_seconds", "window_start_source_frame_0based", "window_end_source_frame_0based",
    "local_geodesic_degrees", "center_value", "window_median", "value", "center_minus_window_median",
    "window_min", "window_max", "valid_frame_count", "requested_frame_count",
}
INTEGER_FIELDS = {"move_id", "center_source_frame_0based", "center_phalp_frame_1based",
    "window_start_source_frame_0based", "window_end_source_frame_0based",
    "valid_frame_count", "requested_frame_count"}


def read_metrics(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in NUMERIC_FIELDS:
            raw = row.get(field, "")
            value = float(raw) if raw else None
            row[field] = value if value is not None and math.isfinite(value) else None
        for field in INTEGER_FIELDS:
            if row[field] is not None:
                if not row[field].is_integer():
                    raise ValueError(f"Expected integer field: {field}")
                row[field] = int(row[field])
        if row["move_id"] is None:
            raise ValueError("Missing or invalid move_id")
    return rows


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty result table")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value) -> str:
    return "NA" if value is None else f"{value:.3f}"


def interval(ref: dict, lo: str, hi: str) -> str:
    return f"{number(ref[lo])} ~ {number(ref[hi])}"


def flat_record(record: dict) -> dict:
    evidence, ref = record["student_evidence"], record["teacher_reference"]
    return {"record_id": record["record_id"], "student_id": record["subject_id"],
        "move_id": record["move_id"], "move_name": record["move_name_pinyin"],
        "metric_id": record["metric_id"], "metric_label_zh": record["metric_label_zh"],
        "technique_aspect": record["technique_aspect"], "value": evidence["value"],
        "center_value": evidence["center_value"], "unit": record["unit"],
        "teacher_median": ref["median"], "teacher_mad": ref["mad"],
        "median_minus_2mad": ref["median_minus_2mad"], "median_plus_2mad": ref["median_plus_2mad"],
        "p10": ref["p10"], "p90": ref["p90"], "teacher_count": ref["valid_teacher_count"],
        "direction": record["direction"], "p10_p90_direction": record["interval_directions"]["p10_p90"],
        "decision": record["decision"], "review_reasons": ";".join(record["review_reasons"]),
        "teacher_warning_ids": ";".join(w["subject_id"] for w in record["teacher_warnings"]),
        "feedback": record["feedback"], "boundary_time_seconds": evidence["boundary_time_seconds"],
        "source_frame_0based": evidence["center_source_frame_0based"],
        "phalp_frame_1based": evidence["center_phalp_frame_1based"], "image": evidence["image"]}


def report_images(records: list[dict], input_root: Path, directory: Path) -> dict:
    """Bundle JPEGs beside reports so previews never depend on remote absolute paths."""
    images = {}

    def copy_image(source, destination, relative):
        with Image.open(source) as image:
            image.load()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
        return relative

    for mid in (1, 2, 3):
        row = next(r for r in records if r["move_id"] == mid)
        sources = {"student": input_root/"student"/row["subject_id"]/f"{mid:02d}_endpoint.jpg",
            "students": input_root/"figures"/f"{mid:02d}_student_keyposes.jpg",
            "teachers": input_root/"figures"/f"{mid:02d}_teacher_keyposes.jpg"}
        if not sources["student"].is_file() and row["student_evidence"].get("image"):
            sources["student"] = Path(row["student_evidence"]["image"])
        for kind, source in sources.items():
            if kind == "student":
                relative = f"assets/{mid:02d}_endpoint.jpg"
            else:
                suffix = "student" if kind == "students" else "teacher"
                relative = f"../assets/{mid:02d}_{suffix}_keyposes.jpg"
            destination = directory/relative
            if source.is_file():
                images[(mid,kind)] = copy_image(source,destination,relative)
            elif destination.is_file():
                # An exported report remains refreshable after its input tree is moved.
                with Image.open(destination) as image:
                    image.load()
                images[(mid,kind)] = relative
            elif kind == "student" and row["student_evidence"].get("image"):
                raise FileNotFoundError(f"Missing endpoint image: {source}")
    return images


def markdown_report(records: list[dict], input_root: Path, subject_id: str,
                    directory: Path) -> str:
    images = report_images(records, input_root, directory)
    lines = [f"# Student {subject_id}：前三式结束定势反馈", "",
        "本报告是规则基线的待核验结果，不是专家定论、整体合格判断或质量排名。",
        "使用括号外的窗口中位数，括号内为中心帧值；角度单位为度，距离均已归一化。",
        "两种区间、中心帧判断或窗口稳定性检查不一致时，停止输出纠正建议。",
        "教师原始值保持不变：教师稳定性警告不会触发静默剔除或重新拟合参考范围。",
        "稳定性门槛是未标定的实验参数；稳定的三维估计仍可能有系统误差。",
        "人工复核在根目录 manual_review.csv 中填写 correct / false_positive / uncertain；不是自动标注。", ""]
    for mid in (1, 2, 3):
        selected = [r for r in records if r["move_id"] == mid]
        row = selected[0]; evidence = row["student_evidence"]
        lines += [f"## {mid}. {row['move_name_zh']}", "", row["final_technique_step"], "",
            f"边界：{number(evidence['boundary_time_seconds'])}s；原视频 0-based 帧：{evidence['center_source_frame_0based']}。", ""]
        for kind, label in (("student", "学生结束定势"), ("students", "参考与五位学生对照"),
                            ("teachers", "十位教师对照")):
            lines += [f"### {label}", ""]
            if (mid,kind) in images:
                lines += [f"![{label}]({images[(mid,kind)]})", ""]
            else:
                lines += ["未提供对应图片。", ""]
        lines += ["### 简短训练提示", "", coach_summary(selected), "",
            "### 逐项依据", "",
            "| 动作要领 / 指标 | 学生值（中心帧） | 教师 median ± 2MAD / P10–P90 | 偏差 | 状态 / 复核原因 | 训练建议 |",
            "|---|---|---|---|---|---|"]
        for item in selected:
            r, s = item["teacher_reference"], item["student_evidence"]
            reasons = "；".join(item["review_reasons_zh"])
            if item["teacher_warnings"]:
                reasons += "；教师警告：" + ", ".join(w["subject_id"] for w in item["teacher_warnings"])
            direction = {"above": "偏高", "below": "偏低", "within": "区间内"}.get(item["direction"], item["direction"])
            cells = [item["technique_aspect"]+" / "+item["metric_label_zh"],
                f"{number(s['value'])} ({number(s['center_value'])}) {item['unit']}",
                interval(r,"median_minus_2mad","median_plus_2mad")+" / "+interval(r,"p10","p90"),
                direction, item["decision_zh"]+("；"+reasons if reasons else ""), item["feedback"] or "不生成纠正建议"]
            lines.append("| "+" | ".join(c.replace("|", " / ").replace("\n", " ") for c in cells)+" |")
        lines.append("")
    return "\n".join(lines)


def refresh_reports(input_root: Path, output_root: Path, students: list[str]) -> None:
    """Refresh presentation only; never overwrite feedback data or manual review labels."""
    for sid in students:
        if not sid.isdigit():
            raise ValueError("Student IDs must be numeric filenames")
        directory = output_root/sid
        records = json.loads((directory/"feedback.json").read_text(encoding="utf-8"))
        if not records or any(r["subject_id"] != sid for r in records):
            raise ValueError(f"Feedback subject mismatch: {sid}")
        text = markdown_report(records,input_root,sid,directory)
        (directory/"feedback_report.md").write_text(text,encoding="utf-8")


def run(input_root: Path, output_root: Path, students: list[str], config: FeedbackConfig,
        rules_path: Path | None = None) -> dict:
    input_root, output_root = input_root.absolute(), output_root.absolute()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("Use a fresh output directory; prior results and manual reviews are never overwritten")
    if not students or len(set(students)) != len(students) or any(not s.isdigit() for s in students):
        raise ValueError("Student IDs must be unique numeric filenames")
    sources = [input_root / name for name in ("summary.json", "all_endpoint_metrics.csv",
        "endpoint_metric_rules_first3.json", "reference_manifest_snapshot.json")]
    summary, metric_rules, manifest = (json.loads(sources[i].read_text()) for i in (0, 2, 3))
    # The experiment hashes the original manifest bytes; its JSON snapshot may be reformatted.
    events = {e["move_id"]: e for e in manifest["endpoint_keyposes"]}
    for pose in metric_rules["poses"]:
        event = events[pose["move_id"]]
        if any(pose[k] != event[k] for k in ("pose_id", "final_technique_step")):
            raise ValueError("Metric rules and manifest snapshot disagree")
    rules = json.loads(rules_path.read_text()) if rules_path else make_rules(metric_rules)
    rule_lookup = validate_rules(rules, metric_rules)
    rows = read_metrics(sources[1]); lookup = {}
    for row in rows:
        key = (row["role"], row["subject_id"], row["move_id"], row["metric_id"])
        if key in lookup:
            raise ValueError(f"Duplicate metric row: {key}")
        lookup[key] = row
    teacher_ids = summary["teacher_ids"]
    if len(teacher_ids) != 10 or len(set(teacher_ids)) != 10:
        raise ValueError("This experiment expects ten unique teacher IDs")
    records = []
    for sid in students:
        for (mid, metric), rule in rule_lookup.items():
            student = lookup[("student", sid, mid, metric)]
            teachers = [lookup[("teacher", tid, mid, metric)] for tid in teacher_ids]
            for row in [student]+teachers:
                if row["pose_id"] != rule["pose_id"] or row["final_technique_step"] != rule["final_technique_step"]:
                    raise ValueError("Metric row and endpoint rule disagree")
                if row["unit"] != metric_rules["metric_definitions"][metric]["unit"]:
                    raise ValueError("Metric units disagree")
                if row["status"] == "valid" and (row["value"] != row["window_median"] or
                        row["center_phalp_frame_1based"] != row["center_source_frame_0based"]+1):
                    raise ValueError("Inconsistent aggregation or source-frame convention")
            records.append(evaluate_feedback(student, teachers, rule, config))
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root/"feedback_rules.json", rules)
    write_json(output_root/"feedback.json", records)
    flat = [flat_record(r) for r in records]
    write_csv(output_root/"feedback.csv", flat)
    write_csv(output_root/"manual_review.csv", [{**r, "manual_verdict": "pending",
        "suspected_cause": "", "reviewer": "", "notes": ""} for r in flat])
    index = ["# 前三式规则反馈实验", "", "## 人工核验入口", "",
        "这是基于冻结分割、冻结指标的反馈层实验。不重跑 DTW，不修改参考范围，不生成质量总分或排名。",
        "反馈候选仅表示通过本轮门槛，不能当作已确认错误；未生成反馈也不表示动作合格。", "",
        "| 学生 | 候选建议 | 需复核 | 仅记录差异 | 未发现区间偏离 | 报告 |",
        "|---|---|---|---|---|---|"]
    counts = []
    for sid in students:
        selected = [r for r in records if r["subject_id"] == sid]
        directory = output_root/sid; directory.mkdir()
        write_json(directory/"feedback.json", selected)
        write_csv(directory/"feedback.csv", [flat_record(r) for r in selected])
        (directory/"feedback_report.md").write_text(markdown_report(selected,input_root,sid,directory),encoding="utf-8")
        c = Counter(r["decision"] for r in selected)
        index.append(f"| {sid} | {c['feedback_candidate']} | {c['needs_review']} | {c['observed_difference']} | {c['within_reference']} | [查看]({sid}/feedback_report.md) |")
        for mid in (1, 2, 3):
            subset = [r for r in selected if r["move_id"] == mid]
            values = Counter(r["decision"] for r in subset)
            counts.append({"student_id":sid,"move_id":mid,**{k:values[k] for k in STATUS_LABELS}})
    write_csv(output_root/"decision_counts.csv", counts)
    index += ["", "数量仅用于监督反馈筛选流程，不代表学生水平，也不能用来排名。", "",
        "## 可靠性门槛", "", "以下是预先设定、尚未人工标定的门槛，不来自学生质量排名拟合：", "",
        f"- 中心帧与窗口中位数差异：角度 > {config.center_delta_degree}°；归一化距离 > {config.center_delta_ratio}。",
        f"- 窗口最大值减最小值：角度 > {config.window_spread_degree}°；归一化距离 > {config.window_spread_ratio}。",
        "- 学生两种参考区间判断不同，或中心帧与中位数判断不同，转人工复核。",
        "- 任一参与范围统计的教师存在上述稳定性警告，该指标转人工复核；保留其原始值，不剔除教师。",
        "- 不把自然运动变化直接认作重建噪声；门槛仅提示当前窗口可能不适合作确定判断。", "",
        "## 人工检查", "", "填写 manual_review.csv 的 manual_verdict：correct / false_positive / uncertain，默认 pending。",
        "correct 表示该行的建议或暂缓决定恰当；false_positive 表示不恰当，不是整体动作是否合格。",
        "suspected_cause 可填 reconstruction / alignment / reference_range / rule / temporal_window / other；在 notes 说明漏报或补充意见。",
        "下一步统计候选建议的人工确认率，单独分析漏报；不要把暂缓判断当作正确反馈。", ""]
    (output_root/"feedback_report.md").write_text("\n".join(index),encoding="utf-8")
    if rules_path:
        sources.append(rules_path)
    provenance = {str(p.absolute()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    result = {"status":"ok", "student_ids":students, "teacher_ids":teacher_ids, "move_ids":[1,2,3],
        "record_count":len(records), "decision_counts":dict(Counter(r["decision"] for r in records)),
        "config":asdict(config), "threshold_status":"experimental_not_calibrated",
        "reference_manifest_sha256":summary["reference_manifest_sha256"], "input_sha256":provenance,
        "code_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__).absolute(),PROJECT/"aqa3d/endpoint_feedback.py",PROJECT/"aqa3d/endpoint_metrics.py")},
        "alignment_review":"user_reported_teachers_accurate_students_acceptable_not_per_record_labels",
        "metric_review":"user_reported_mostly_accurate_with_reconstruction_limitations",
        "feedback_manual_review":"pending", "input_root":str(input_root), "output_root":str(output_root)}
    write_json(output_root/"summary.json",result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root",type=Path,default=PROJECT/"endpoint_metric_results/first3_five_students")
    parser.add_argument("--output-root",type=Path,default=PROJECT/"endpoint_feedback_results/first3_five_students")
    parser.add_argument("--students",nargs="+",default=["01","02","03","04","10"])
    parser.add_argument("--rules-json",type=Path)
    parser.add_argument("--refresh-reports-only",action="store_true",
        help="Refresh Markdown and bundled images without changing existing data or reviews")
    for name, value in asdict(FeedbackConfig()).items():
        parser.add_argument("--"+name.replace("_","-"),type=type(value),default=value)
    args = parser.parse_args()
    if args.refresh_reports_only:
        refresh_reports(args.input_root,args.output_root,args.students)
        print(f"Refreshed reports and images: {args.output_root}")
        return
    config = FeedbackConfig(**{k:getattr(args,k) for k in asdict(FeedbackConfig())})
    result = run(args.input_root,args.output_root,args.students,config,args.rules_json)
    print(json.dumps({"output":result["output_root"],"counts":result["decision_counts"]},ensure_ascii=False))


if __name__ == "__main__":
    main()
