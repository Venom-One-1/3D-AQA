"""Repair saved check plans offline and execute copies without replacing first-pass results."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from aqa3d.check_plan_repair import repair_response
from aqa3d.check_planning import queries_payload, stability, strict_loads
from aqa3d.pose_measurements import MeasurementQuery
from run_check_plan_probes import probe_query
from run_pose_measurements import export_experiment, write_csv, write_json


PROJECT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return strict_loads(path.read_bytes().decode("utf-8"))


def source_file(root: Path, relative: str) -> Path:
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Source path escapes experiment root")
    return path


def snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*")) if p.is_file()}


def cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def run_repair(source: Path, output: Path, policy_path: Path, *, execute: bool = True) -> dict:
    source, output = source.resolve(), output.resolve()
    manifest = read_json(source/"experiment_manifest.json")
    cache_root = Path(manifest["input_root"]).resolve()
    for protected in (source, cache_root):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("Repair output must be separate from original experiment and frozen caches")
    if output.exists():
        raise FileExistsError("Use a new output directory; repairs and reviews are never overwritten")
    policy_raw = policy_path.read_bytes()
    policy = strict_loads(policy_raw.decode("utf-8"))
    if policy["version"] != "1.0":
        raise ValueError("Unsupported repair policy")
    before = snapshot(source)
    for relative, checksum in manifest["frozen_file_sha256"].items():
        if digest(source_file(source, relative)) != checksum:
            raise ValueError(f"Frozen input changed: {relative}")
    entries = sorted(manifest["cases"], key=lambda e: (e["move_id"], e["repeat"]))
    if (len({e["case_id"] for e in entries}) != len(entries)
            or any(not re.fullmatch(r"M\d{2}_R\d{2}", e["case_id"]) for e in entries)):
        raise ValueError("Invalid or duplicate case IDs")
    unknown = set(policy["cases"]) - {e["case_id"] for e in entries}
    if unknown:
        raise ValueError(f"Repair policy contains unknown cases: {sorted(unknown)}")
    output.mkdir(parents=True)
    write_json(output/"summary.json", {"status": "in_progress", "api_calls": 0})
    (output/"repair_policy.json").write_bytes(policy_raw)
    write_json(output/"source_manifest_snapshot.json", manifest)
    provenance = {
        "source_root": str(source), "frozen_cache_root": str(cache_root),
        "source_files_sha256_before": before,
        "policy_sha256": hashlib.sha256(policy_raw).hexdigest(),
        "implementation_sha256": {str(p.relative_to(PROJECT)): digest(p) for p in (
            Path(__file__), PROJECT/"aqa3d/check_plan_repair.py", PROJECT/"aqa3d/check_planning.py",
            PROJECT/"aqa3d/pose_measurements.py", PROJECT/"run_pose_measurements.py",
            PROJECT/"run_check_plan_probes.py")},
        "repair_type": "offline deterministic normalization plus hash-bound assistant-reviewed corrections",
        "api_calls": 0, "selection": "first_valid_repaired_plan_in_repeat_order_not_student_outcomes",
        "semantic_correctness": "not_reviewed", "diagnosis_generated": False,
    }
    write_json(output/"repair_manifest.json", provenance)
    rows, action_rows, review_rows, executions, probes, cache_hashes = [], [], [], [], [], {}
    selected, groups = {}, {}
    plan_report = ["# 修复后检查计划对照", "", "本表中的动作解释仍是模型输出，不是经过专家确认的规则。"
                   "修复只恢复序列化、接口引用和明确坐标含义；不按这些文字执行学生纠错。", ""]
    for entry in entries:
        cid, mid = entry["case_id"], entry["move_id"]
        folder = output/cid
        folder.mkdir()
        raw_path = source_file(source, entry["response_file"])
        raw = raw_path.read_bytes()
        status = read_json(source/"api_calls"/cid/"status.json")
        if status["status"] != "completed" or digest(raw_path) != status["response_sha256"]:
            raise ValueError(f"Source response incomplete or changed: {cid}")
        returned_model = status["attempts"][-1].get("model")
        if returned_model != manifest["api_config"]["model"]:
            raise ValueError(f"Source model does not match frozen configuration: {cid}")
        request = read_json(source_file(source, entry["input_file"]))
        write_json(folder/"input.json", request)
        (folder/"original_answer.txt").write_bytes(raw)
        result = repair_response(raw.decode("utf-8"), request, policy["cases"].get(cid))
        row = {"case_id": cid, "move_id": mid, "move_name": request["move_name"],
               "original_valid": result["original_validation"]["schema_valid"],
               "repaired_valid": result["repaired_validation"]["schema_valid"],
               "repair_actions": len(result["changes"]),
               "assistant_reviewed_changes": sum(e.get("approval", "").startswith("assistant_reviewed")
                                                 for e in result["changes"]),
               "query_count": result["repaired_validation"].get("query_count", 0),
               "execution_status": "not_run", "semantic_correctness": "not_reviewed"}
        rows.append(row)
        logs = {"source_sha256": digest(raw_path), "original_validation": result["original_validation"],
                "repaired_validation": result["repaired_validation"], "changes": result["changes"],
                "semantic_warnings": result["semantic_warnings"], "api_calls": 0}
        for action in result["changes"]:
            action_rows.append({"case_id": cid, "kind": action["kind"], "path": action["path"],
                                "reason": action["reason"],
                                "before": json.dumps(action["before"], ensure_ascii=False),
                                "after": json.dumps(action["after"], ensure_ascii=False)})
        plan = result["plan"]
        if plan is None:
            write_json(folder/"repair_log.json", logs)
            write_json(folder/"candidate.json", result["candidate"])
            continue
        write_json(folder/"plan.json", plan)
        logs["repaired_sha256"] = digest(folder/"plan.json")
        write_json(folder/"repair_log.json", logs)
        write_json(folder/"queries.json", queries_payload(plan))
        selected.setdefault(str(mid), cid)
        groups.setdefault(mid, []).append((cid, plan))
        clauses = {c["clause_id"]: c["text"] for c in request["clauses"]}
        definitions = {q["query_id"]: q for q in plan["queries"]}
        plan_report += [f"## {cid} · {request['move_name']}", "",
                        f"[修复后 JSON]({cid}/plan.json) | [逐项修改日志]({cid}/repair_log.json)", ""]
        plan_report += [f"- 待语义复核：{note}" for note in result["semantic_warnings"]]
        plan_report += ["", "| 要领原文 | 支持类型 | 查询与选用分量 | 理由及预期关系 | 局限 |", "|---|---|---|---|---|"]
        for check in plan["checks"]:
            query_text, explanations = [], []
            for evidence in check["evidence"]:
                q = definitions[evidence["query_id"]]
                query_text.append(f"{q['operation']}({', '.join(q['points'])}); "
                                  f"{q.get('normalizer')}; {q.get('coordinate_frame')}; {evidence['components']}")
                explanations.append(evidence["purpose"]+"；"+evidence["expected_relation"])
            review = {"case_id": cid, "move_id": mid, "clause_id": check["clause_id"],
                      "source_text": clauses[check["clause_id"]], "support": check["support"],
                      "queries": "；".join(query_text), "explanation": "；".join(explanations) or check["endpoint_interpretation"],
                      "limitations": "；".join(check["limitations"]), "manual_verdict": "pending", "notes": ""}
            review_rows.append(review)
            plan_report.append("| " + " | ".join(cell(review[k]) for k in
                               ("source_text", "support", "queries", "explanation", "limitations")) + " |")
        plan_report.append("")
        if not execute or not plan["queries"]:
            continue
        measured = export_experiment(cache_root, folder/"measurements", queries_path=folder/"queries.json", moves=[mid])
        executions.append({"case_id": cid, **measured})
        row["execution_status"] = measured["status"]
        input_provenance = read_json(folder/"measurements/provenance.json")
        for relative, checksum in input_provenance["source_sha256"].items():
            path = str(source_file(cache_root, relative))
            if path in cache_hashes and cache_hashes[path] != checksum:
                raise ValueError("Frozen input changed between repeated plan executions")
            cache_hashes[path] = checksum
        cache = cache_root/"student/01"/f"{mid:02d}_window_joints.npz"
        if cache_hashes.get(str(cache)) != digest(cache):
            raise ValueError("Probe cache differs from executed measurement input")
        with np.load(cache, allow_pickle=False) as data:
            joints = data["native_smpl24_joints"]
        for query in plan["queries"]:
            probes.extend({"case_id": cid, **p} for p in probe_query(joints, MeasurementQuery.from_dict(query)))
    after = snapshot(source)
    if before != after:
        raise ValueError("Original experiment changed during repair; do not treat this run as verified")
    for path, checksum in cache_hashes.items():
        if digest(Path(path)) != checksum:
            raise ValueError("Probe cache changed during repair")
    provenance["source_files_unchanged"] = True
    provenance["frozen_cache_sha256"] = cache_hashes
    write_json(output/"repair_manifest.json", provenance)
    pairs = [{"move_id": mid, **p} for mid, plans in groups.items() for p in stability(plans)]
    write_json(output/"stability.json", pairs)
    write_json(output/"executions.json", executions)
    write_json(output/"geometry_probes.json", {"meaning": "Numerical tests only; not semantic accuracy", "records": probes})
    write_csv(output/"validation.csv", rows)
    write_csv(output/"repair_actions.csv", action_rows)
    write_csv(output/"plan_review.csv", review_rows)
    summary = {
        "status": "complete" if execute and len(executions) == len(entries) else "repair_only_or_partial",
        "source_answers": len(entries), "original_valid": sum(r["original_valid"] for r in rows),
        "repaired_valid": sum(r["repaired_valid"] for r in rows),
        "changed_answers": sum(r["repair_actions"] > 0 for r in rows),
        "assistant_reviewed_correction_answers": sum(r["assistant_reviewed_changes"] > 0 for r in rows),
        "repair_action_counts": dict(Counter(r["kind"] for r in action_rows)),
        "selected_first_valid_repaired": selected, "executed_plans": len(executions),
        "successful_executions": sum(e["status"] == "ok" for e in executions),
        "unique_student_move_cases": len({(sid, mid) for e in executions for sid in e["student_ids"] for mid in e["move_ids"]}),
        "probe_checks": len(probes), "probe_passed": sum(p["status"] == "pass" for p in probes),
        "probe_failed": sum(p["status"] == "fail" for p in probes),
        "probe_not_testable": sum(p["status"] == "not_testable" for p in probes),
        "api_calls": 0, "original_experiment_unchanged": True, "semantic_accuracy": None,
        "diagnosis_generated": False,
    }
    if summary["successful_executions"] != len(executions) or summary["probe_failed"]:
        summary["status"] = "execution_or_probe_failed"
    write_json(output/"summary.json", summary)
    (output/"plan_report.md").write_text("\n".join(plan_report), encoding="utf-8")
    report = ["# 检查计划离线修复报告", "", "首次结果、原始回答、API 记录及冻结缓存均保持不变。"
              "本轮没有新增 API 调用。修复通过率不是模型首次成功率，也不是动作诊断准确率。", "",
              "| 回答 | 首次严格校验 | 修复后 | 查询数 | 修改记录数 | 缓存执行 |", "|---|---|---|---:|---:|---|"]
    for r in rows:
        target = "plan.json" if r["repaired_valid"] else "repair_log.json"
        report.append(f"| [{r['case_id']}]({r['case_id']}/{target}) | {r['original_valid']} | {r['repaired_valid']} | "
                      f"{r['query_count']} | [{r['repair_actions']}]({r['case_id']}/repair_log.json) | {r['execution_status']} |")
    report += ["", f"首次通过 {summary['original_valid']}/{len(entries)}；修复后通过 {summary['repaired_valid']}/{len(entries)}。",
               f"改动 {summary['changed_answers']} 份；其中 {summary['assistant_reviewed_correction_answers']} 份涉及助手核对后锁定的转义、引用或坐标修正，不能宣称为模型自行修好。",
               f"执行 {summary['successful_executions']}/{len(executions)} 份计划成功；覆盖 {summary['unique_student_move_cases']} 个不同学生定势案例。",
               f"数值核验 {summary['probe_passed']}/{summary['probe_checks']} 通过，{summary['probe_not_testable']} 项不可测。", "",
               "## 修复边界", "",
               "- 仅提取唯一完整 JSON；不拼接多个回答、不补写缺失字段。",
               "- 未转义字符串由原文哈希和逐条文字锁定，保持解码后的中文内容。",
               "- 合并相同测量时同步引用；同一要领重复引用则合并分量并保留解释。闲置查询删除，不删要领。",
               "- M02_R01 的 left>0 转为 right<0，查询本身不变；不是仅把 left 改名。",
               "- M03_R02 的引用通过关节顺序、坐标和尺度核对后修正；不进行模糊匹配。",
               "- M03_R01 右侧却写 right<0 的矛盾改为 right>0，属于显式坐标修正，不算纯格式修复。", "",
               "## 尚未证明的部分", "",
               "几何可计算不等于要领理解正确。脊柱共线与上体正直、膝角与承重、手腕高度与肘部下坠等代理关系仍需语义复核。"
               "模型自行给出的角度范围未获得专家验证，不作为学生质量阈值。本轮不生成训练反馈或质量排名。", "",
               "## 查看结果", "",
               "[逐句计划对照](plan_report.md) · [语义复核表](plan_review.csv) · [所有修改前后对照](repair_actions.csv) · [汇总 JSON](summary.json)", "",
               "按重复顺序选择首份可执行计划，未根据学生结果挑选：", ""]
    for mid, cid in selected.items():
        line = f"- 第 {mid} 式：[{cid}]({cid}/plan.json)"
        if (output/cid/"measurements/cases"/f"01_{int(mid):02d}.json").exists():
            line += f"；[学生 01 测量证据]({cid}/measurements/cases/01_{int(mid):02d}.json)"
        report.append(line)
    (output/"repair_report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=PROJECT/"check_plan_experiment/first3_v1")
    parser.add_argument("--output-root", type=Path, default=PROJECT/"check_plan_experiment/first3_v1_repaired_v1")
    parser.add_argument("--policy", type=Path, default=PROJECT/"prompt/check_plan_repair_policy_v1.json")
    parser.add_argument("--skip-execution", action="store_true")
    args = parser.parse_args()
    result = run_repair(args.source_root, args.output_root, args.policy, execute=not args.skip_execution)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["repaired_valid"] != result["source_answers"] or result["status"] == "execution_or_probe_failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
