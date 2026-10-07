"""Freeze, run and audit endpoint coaching from reviewed LLM measurement plans."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import json
from pathlib import Path

from aqa3d.check_planning import strict_loads
from aqa3d.endpoint_feedback import coach_summary
from aqa3d.plan_feedback import POLICY, compile_packet, extract_response, select_checks, validate_response
from run_check_plan_experiment import load_key
from run_f_group_api import run_case, validate_base_url


PROJECT = Path(__file__).resolve().parent
STUDENTS = ("01", "02", "03", "04", "10")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return strict_loads(path.read_text(encoding="utf-8"))


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Sources:
    def __init__(self):
        self.hashes = {}

    def read(self, path):
        raw = path.read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        key = str(path.resolve())
        if key in self.hashes and self.hashes[key] != checksum:
            raise ValueError(f"Source changed during preparation: {path}")
        self.hashes[key] = checksum
        return raw

    def json(self, path):
        return strict_loads(self.read(path).decode("utf-8"))

    def verify(self):
        for path, checksum in self.hashes.items():
            if sha(Path(path)) != checksum:
                raise ValueError(f"Source changed: {path}")


def prepare(args):
    output = args.output_root
    if any(p.name != ".run.lock" for p in output.iterdir()):
        raise ValueError("Preparation needs a new empty directory; existing runs are immutable")
    sources = Sources()
    original_manifest = sources.json(args.plans_root/"source_manifest_snapshot.json")
    config = original_manifest["api_config"]
    validate_base_url(config["base_url"])
    if config["model"] != "qwen3.8-max-0902":
        raise ValueError("Unexpected model; do not silently substitute models")
    summary = sources.json(args.plans_root/"summary.json")
    selected = summary["selected_first_valid_repaired"]
    review_raw = sources.read(args.review_csv)
    reviews = list(csv.DictReader(review_raw.decode("utf-8-sig").splitlines()))
    review_map = {r["record_id"]: r for r in reviews}
    if len(review_map) != len(reviews):
        raise ValueError("Duplicate review record IDs")
    (output/"review_snapshot.csv").write_bytes(review_raw)
    prompt = sources.read(PROJECT/"prompt/plan_feedback.md")
    (output/"prompt.md").write_bytes(prompt)
    write_json(output/"policy.json", POLICY)
    f_manifest = sources.json(args.f_root/"experiment_manifest.json")
    f_config = sources.json(args.f_root/"api_config.json")
    baseline = sources.json(args.f_root/"evaluation/baseline_snapshot.json")
    entries, frozen_plans, coverage, comparisons = [], {}, [], {}
    for mid in (1, 2, 3):
        pid = selected[str(mid)]
        plan = sources.json(args.plans_root/pid/"plan.json")
        source = sources.json(args.plans_root/pid/"input.json")
        frozen = select_checks(plan, source, review_map, pid)
        frozen_plans[str(mid)] = frozen
        write_json(output/"plan_snapshots"/f"{pid}.json", plan)
        for c in frozen["checks"]:
            coverage.append({"move_id": mid, "plan_id": pid, "clause_id": c["clause_id"],
                "text": c["text"], "included": True, "reason": c["acceptance"]})
        for c in frozen["excluded_clauses"]:
            coverage.append({"move_id": mid, "plan_id": pid, "clause_id": c["clause_id"],
                "text": c["text"], "included": False, "reason": c["reason"]})
    # Freeze plan selection before reading any student values.
    write_json(output/"frozen_plans.json", frozen_plans)
    for sid in STUDENTS:
        for mid in (1, 2, 3):
            frozen = frozen_plans[str(mid)]
            cid = f"PF_{sid}_{mid:02d}"
            measure = sources.json(args.plans_root/frozen["plan_id"]/"measurements/cases"/f"{sid}_{mid:02d}.json")
            packet = compile_packet(cid, frozen, measure)
            if packet["student_id"] != sid:
                raise ValueError("Student identity mismatch")
            path = output/"inputs"/f"{cid}.json"
            write_json(path, packet)
            entries.append({"case_id": cid, "student_id": sid, "move_id": mid,
                "input_file": str(path.relative_to(output)), "input_sha256": sha(path),
                "response_file": f"responses/{cid}.json", "prompt_file": "prompt.md"})
            f_entry = next(e for e in f_manifest["cases"] if e["student_id"] == sid and e["move_id"] == mid)
            rows = [r for r in baseline if r["subject_id"] == sid and r["move_id"] == mid]
            if not rows or any(r["final_technique_step"] != packet["technique"] for r in rows):
                raise ValueError("Baseline technique mismatch")
            old_answer = sources.json(args.f_root/f_entry["response_file"])
            comparisons[cid] = {"old_f_case_id": f_entry["case_id"], "old_f_response": old_answer,
                "rule_baseline_summary": coach_summary(rows), "rule_baseline": rows}
            for role, image_path in {
                "reference": args.metric_root/"figures"/f"{mid:02d}_reference.jpg",
                "student": args.metric_root/"student"/sid/f"{mid:02d}_endpoint.jpg",
            }.items():
                dest = output/"images"/f"{cid}_{role}.jpg"
                dest.parent.mkdir(exist_ok=True)
                dest.write_bytes(sources.read(image_path))
    (output/"responses").mkdir()
    write_json(output/"comparison_snapshot.json", comparisons)
    write_csv(output/"plan_coverage.csv", coverage)
    manifest = {"version": "1.0", "status": "prepared", "api_config": config,
        "old_f_model": f_config["model"], "cases": entries,
        "scope": "development_set_not_held_out; endpoint_only; no_tracking_or_dtw_rerun",
        "comparison_caveat": "Historical F differs in model label, input queries and prompt; not a controlled ablation.",
        "source_sha256": sources.hashes,
        "implementation_sha256": {str(p.relative_to(PROJECT)): sha(p) for p in
            (Path(__file__), PROJECT/"aqa3d/plan_feedback.py", PROJECT/"prompt/plan_feedback.md",
             PROJECT/"run_f_group_api.py")}}
    sources.verify()
    manifest["frozen_sha256"] = {str(p.relative_to(output)): sha(p)
        for p in output.rglob("*") if p.is_file() and p.name != ".run.lock"}
    write_json(output/"experiment_manifest.json", manifest)
    print(f"Prepared {len(entries)} cases; included clauses per move: " +
          str({m: len(p['checks']) for m, p in frozen_plans.items()}), flush=True)
    return manifest


def verify_frozen(output):
    manifest = load(output/"experiment_manifest.json")
    for relative, checksum in manifest["frozen_sha256"].items():
        path = (output/relative).resolve()
        if not path.is_relative_to(output.resolve()) or sha(path) != checksum:
            raise ValueError(f"Frozen artifact changed: {relative}")
    validate_base_url(manifest["api_config"]["base_url"])
    return manifest


def call(output, manifest, key, workers, retry_failed=False):
    entries, config = manifest["cases"], manifest["api_config"]
    first = run_case(entries[0], output, config, key, retries=0, retry_failed=retry_failed)
    print(f"{first['case_id']}: {first['status']}", flush=True)
    if first["status"] != "completed":
        return
    # Format failure is an outcome, not permission to regenerate until accepted.
    for result in first["attempts"]:
        if result.get("model") and result["model"] != config["model"]:
            raise ValueError("Server returned a different model; remaining calls stopped")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_case, e, output, config, key, 0, retry_failed) for e in entries[1:]]
        for future in as_completed(futures):
            result = future.result()
            print(f"{result['case_id']}: {result['status']}", flush=True)


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def analyze(output, manifest):
    comparisons = load(output/"comparison_snapshot.json")
    syntax_path = output/"syntax_repairs.json"
    syntax_policies = load(syntax_path)["cases"] if syntax_path.exists() else {}
    validation, new_reviews, cases, all_findings = [], [], [], []
    def add_review(cid, method, kind, item_id, text):
        new_reviews.append({"record_id": f"{cid}:{method}:{kind}:{item_id}", "case_id": cid,
            "method": method, "kind": kind, "item_id": item_id, "text": text,
            "manual_verdict": "pending", "notes": ""})
    for entry in manifest["cases"]:
        cid = entry["case_id"]
        packet = load(output/entry["input_file"])
        answer, issues, normalization = None, [], []
        status_path = output/"api_calls"/cid/"status.json"
        status = load(status_path) if status_path.exists() else {"status": "not_called", "attempts": []}
        path = output/entry["response_file"]
        if path.exists():
            if status["status"] != "completed" or sha(path) != status.get("response_sha256"):
                issues.append("response provenance mismatch")
            if not status["attempts"] or status["attempts"][-1].get("model") != manifest["api_config"]["model"]:
                issues.append("response model mismatch")
            try:
                answer, normalization = extract_response(path.read_text(encoding="utf-8"), syntax_policies.get(cid))
                issues.extend(validate_response(answer, packet))
            except (ValueError, TypeError, KeyError) as exc:
                issues.append(f"invalid JSON/schema: {type(exc).__name__}")
        else:
            issues.append("missing response")
        valid = not issues
        if isinstance(answer, dict):
            write_json(output/"parsed_responses"/f"{cid}.json", answer)
        write_json(output/"normalization"/f"{cid}.json", {
            "raw_sha256": sha(path) if path.exists() else None,
            "actions": normalization, "valid_after_extraction": valid})
        validation.append({"case_id": cid, "transport_status": status["status"], "valid": valid,
                           "raw_json_only": answer is not None and not normalization,
                           "extracted_json": any(e["action"] == "extract_json" for e in normalization),
                           "syntax_repaired": any(e["action"] == "escape_reviewed_literal" for e in normalization),
                           "errors": "; ".join(issues)})
        old = comparisons[cid]
        sid, mid = entry["student_id"], entry["move_id"]
        lines = [f"# 学生 {sid} · {packet['move_name']}\n", packet["technique"],
            "\n本页为定势几何证据实验，不是完整动作质量判定。新方法自动校验只检查结构、关联和数值引用，不证明语义正确。\n",
            f"计划：{packet['plan_id']}；可用要领 {len(packet['checks'])} 条；未用于诊断 {len(packet['excluded_clauses'])} 条。\n",
            "| 参考锚点 BV1WE411W7JB | 学生结束定势 |", "|---|---|",
            f"| ![参考](../images/{cid}_reference.jpg) | ![学生](../images/{cid}_student.jpg) |",
            f"| {packet['reference_anchor']['boundary_time_seconds']:.3f}s | {packet['student_endpoint']['boundary_time_seconds']:.3f}s |",
            "\n## 三种方法的训练提示\n", "| 方法 | 训练提示 |", "|---|---|"]
        summary_text = answer["coach_summary"] if valid else "回答未通过校验，见原始回答与 validation.csv；不展示为有效训练反馈。"
        lines.extend([f"| 新方法：计划驱动、选择性反馈 | {cell(summary_text)} |",
            f"| 历史 F 组 | {cell(old['old_f_response']['coach_summary'])} |",
            f"| Rule Baseline | {cell(old['rule_baseline_summary'])} |"])
        if not valid:
            lines.extend(["\n## 未通过自动校验\n", "；".join(issues),
                          "\n保留原始结果，不自动改写诊断或措辞等级。"])
            if isinstance(answer, dict):
                lines.append(f"\n[提取后的原始判断，仅供核验](../parsed_responses/{cid}.json)")
                flagged_summary = answer.get("coach_summary")
                if isinstance(flagged_summary, str):
                    lines.extend(["\n### 待核验的原始总结\n", flagged_summary])
                    add_review(cid, "plan_feedback", "flagged_summary", "summary", flagged_summary)
                flagged_findings = answer.get("findings", [])
                if isinstance(flagged_findings, list):
                    for index, finding in enumerate(flagged_findings):
                        if isinstance(finding, dict):
                            add_review(cid, "plan_feedback", "flagged_finding", str(index+1),
                                       json.dumps(finding, ensure_ascii=False))
        if valid:
            add_review(cid, "plan_feedback", "summary", "summary", summary_text)
            for finding in answer["findings"]:
                fid = finding["finding_id"]
                all_findings.append({"case_id": cid, **finding})
                lines.extend([f"\n## {fid}：{finding['observation']}\n",
                    f"**训练建议：** {finding['training_guidance']}\n",
                    f"综合依据：{finding['synthesis']}\n", f"优先理由：{finding['why_priority']}\n",
                    f"局限（{finding['certainty']}）：{finding['caveat']}\n",
                    "| 证据 | 学生窗口中位数 | 教师中位数 | P10–P90 | 方向 |",
                    "|---|---|---|---|---|"])
                for e in finding["evidence"]:
                    lines.append(f"| {e['evidence_id']} | {e['student_value']:.3f} | {e['teacher_median']:.3f} | {e['p10']:.3f}–{e['p90']:.3f} | {e['direction']} |")
                add_review(cid, "plan_feedback", "finding", fid, json.dumps(finding, ensure_ascii=False))
            lines.append("\n## 未选择的候选\n")
            for r in answer["not_selected"]:
                lines.append(f"- {', '.join(r['clause_ids'])}：{r['reason']}")
        add_review(cid, "old_F", "summary", "summary", old["old_f_response"]["coach_summary"])
        add_review(cid, "rule_baseline", "summary", "summary", old["rule_baseline_summary"])
        lines.append("\n## 历史 F 组逐项反馈\n")
        for f in old["old_f_response"]["findings"]:
            lines.extend([f"### {f['finding_id']}\n", "```json", json.dumps(f, ensure_ascii=False, indent=2), "```"])
        lines.append("\n## Rule Baseline 候选建议与需复核项\n")
        for r in old["rule_baseline"]:
            if r["decision"] in ("feedback_candidate", "needs_review"):
                lines.append(f"- {r['metric_label_zh']}：{r['decision_zh']}。{r['feedback'] or '暂不纠正'}。{'；'.join(r['review_reasons_zh'])}")
        lines.append("\n## 本轮检查空缺\n")
        lines.extend(f"- {r['clause_id']} {r['text']}：{r['reason']}" for r in packet["excluded_clauses"])
        lines.append(f"\n[完整输入](../{entry['input_file']}) · [原始回答](../{entry['response_file']})\n")
        report = output/"reports"/f"{cid}.md"
        report.parent.mkdir(exist_ok=True)
        report.write_text("\n".join(lines)+"\n", encoding="utf-8")
        cases.append({"case_id": cid, "student_id": sid, "move_id": mid, "valid": valid,
            "findings": len(answer["findings"]) if valid else None,
            "tentative": sum(f["certainty"] == "tentative" for f in answer["findings"]) if valid else None,
            "multimetric_findings": sum(len(f["evidence"]) > 1 for f in answer["findings"]) if valid else None,
            "old_F_findings": len(old["old_f_response"]["findings"]),
            "rule_candidates": sum(r["decision"] == "feedback_candidate" for r in old["rule_baseline"])})
    write_csv(output/"validation.csv", validation)
    write_csv(output/"case_summary.csv", cases)
    write_json(output/"findings.json", all_findings)
    review_path = output/"manual_feedback_review.csv"
    if review_path.exists():
        with review_path.open(encoding="utf-8-sig", newline="") as handle:
            previous = list(csv.DictReader(handle))
        by_id = {r["record_id"]: r for r in previous}
        for row in new_reviews:
            old = by_id.get(row["record_id"])
            if old and old["text"] != row["text"]:
                raise ValueError("Review text changed; preserve review and use a new output directory")
        previous.extend(r for r in new_reviews if r["record_id"] not in by_id)
        if previous:
            write_csv(review_path, previous)
    elif new_reviews:
        write_csv(review_path, new_reviews)
    frozen = load(output/"frozen_plans.json")
    summary = {"case_count": len(cases), "valid_responses": sum(r["valid"] for r in cases),
        "raw_json_only": sum(r["raw_json_only"] for r in validation),
        "extracted_json": sum(r["extracted_json"] for r in validation),
        "syntax_repaired": sum(r["syntax_repaired"] for r in validation),
        "syntax_policy_sha256": sha(syntax_path) if syntax_path.exists() else None,
        "finding_count": len(all_findings),
        "multimetric_findings": sum(len(f["evidence"]) > 1 for f in all_findings),
        "tentative_findings": sum(f["certainty"] == "tentative" for f in all_findings),
        "coverage_by_move": {m: {"included": len(p["checks"]), "excluded": len(p["excluded_clauses"])} for m, p in frozen.items()},
        "model": manifest["api_config"]["model"], "old_F_model": manifest["old_f_model"],
        "semantic_quality": "not_established; requires final_feedback_review",
        "source_files_unchanged": all(sha(Path(p)) == h for p, h in manifest["source_sha256"].items()),
        "analysis_implementation_sha256": {str(p.relative_to(PROJECT)): sha(p) for p in
            (Path(__file__), PROJECT/"aqa3d/plan_feedback.py")}}
    write_json(output/"summary.json", summary)
    report = ["# 计划驱动的选择性训练反馈实验\n",
        "冻结三式首个有效轮次，复用 15 个结束定势和十位教师数据。只复核最终提示，不重新复核全部检查计划。\n",
        f"新模型：{summary['model']}；历史 F：{summary['old_F_model']}。输入指标、提示词和筛选策略也不同，因此本轮是开发性对照，不是证明 LLM 优越性或因果效果的受控消融。\n",
        f"格式/引用校验通过：{summary['valid_responses']}/15；选中建议 {summary['finding_count']} 条；其中多项证据 {summary['multimetric_findings']} 条、谨慎措辞 {summary['tentative_findings']} 条。条数不等于正确率。\n",
        f"原始回答直接为 JSON：{summary['raw_json_only']}；移除外围文字后提取 JSON：{summary['extracted_json']}；精确转义已核对的字符串引号：{summary['syntax_repaired']}。提取/转义未改写诊断内容、数值或原始回答；修复规则绑定原始文件哈希。日志在 normalization/，解析结果在 parsed_responses/。\n",
        "## 检查覆盖\n", "| 招式 | 可用于诊断的要领 | 空缺/不支持 |", "|---|---:|---:|"]
    for m, p in frozen.items():
        report.append(f"| {m} {p['move_name']} | {len(p['checks'])} | {len(p['excluded_clauses'])} |")
    report.extend(["\n人工否定条目整条排除；不按学生结果换轮次。保留 Head/Wrist 代理及局限，但不把旧计划的 expected_relation 当作质量标准。未发现问题也可能是检查覆盖不足，不能解释为动作合格。\n",
        "## 个案对照\n", "| 学生 | 招式 | 新建议 | 多证据建议 | 历史 F | Rule 候选 | 对照报告 |", "|---|---|---:|---:|---:|---:|---|"])
    for r in cases:
        report.append(f"| {r['student_id']} | {r['move_id']} | {r['findings']} | {r['multimetric_findings']} | {r['old_F_findings']} | {r['rule_candidates']} | [查看](reports/{r['case_id']}.md) |")
    report.extend(["\n## 下一步复核\n",
        "先检查新方法选中的建议与总结：correct 表示建议有证据且合理，false_positive 表示误判或越界，uncertain 表示无法判断。填写 manual_feedback_review.csv 的 manual_verdict 与 notes，保留英文枚举。没有建议的案例也要检查总结是否恰当；未标注条目保留 pending。\n",
        "该 CSV 不把此前 check-plan 对应性标注当作反馈正确性标签。最终建议的 precision、每例有效建议数、重复/矛盾或越界情况应在复核后统计；缺少穷尽式错误标注，暂不能估计召回率。\n",
        "kind=flagged_summary / flagged_finding 的记录来自未通过自动校验的回答，单独保留供复核，不计入上述合规建议数量。自动校验失败也不等于人工判定误判。\n",
        "本轮自动校验范围：JSON 结构、要领与查询关联、数字引用、P10-P90 方向、稳定性警告下的措辞等级。自由文本含义、训练价值和总结是否新增判断仍需复核。\n",
        "## 复现\n", "```bash", "python run_plan_feedback_experiment.py prepare --output-root <新目录>",
        "python run_plan_feedback_experiment.py run --output-root <目录> --api-key-file <权限600的密钥文件>",
        "python run_plan_feedback_experiment.py analyze --output-root <目录>", "```",
        "run 默认不重试失败调用、不覆盖原始回答；显式 --retry-failed 才会重试。原始 API 回答、请求、耗时与用量在 responses/、api_calls/。输入和审查来源哈希冻结在 experiment_manifest.json。"])
    (output/"comparison_report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "analyze"))
    parser.add_argument("--output-root", type=Path, default=PROJECT/"plan_feedback_results/first3_five_students_v1")
    parser.add_argument("--plans-root", type=Path, default=PROJECT/"check_plan_experiment/first3_v1_repaired_v1")
    parser.add_argument("--review-csv", type=Path, default=PROJECT/"check_plan_judge_results/first3_v1_astra/manual_agreement_v1/input_review.csv")
    parser.add_argument("--f-root", type=Path, default=PROJECT/"f_group_experiment/first3_five_students_api")
    parser.add_argument("--metric-root", type=Path, default=PROJECT/"endpoint_metric_results/first3_five_students")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root/".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("This experiment is already running; keep its lock file")
        if args.action == "prepare":
            prepare(args)
        else:
            manifest = verify_frozen(args.output_root)
            if args.action == "run":
                call(args.output_root, manifest, load_key(args.api_key_file), args.workers, args.retry_failed)
            summary = analyze(args.output_root, manifest)
            if summary["valid_responses"] != len(manifest["cases"]):
                raise SystemExit(1)


if __name__ == "__main__":
    main()
