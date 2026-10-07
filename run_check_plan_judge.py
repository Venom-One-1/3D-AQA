"""Prepare blinded review packets, import independent answers, and triage clauses."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil

from aqa3d.check_plan_judge import model_verdict, probe_plan, route_check, signed_axis_probes, validate_judgment
from aqa3d.check_planning import parse_plan, strict_loads

PROJECT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return strict_loads(path.read_text(encoding="utf-8-sig"))


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def verify(root):
    manifest = read(root / "manifest.json")
    hashes = dict(manifest["source_sha256"])
    hashes[str(root / manifest["prompt_file"])] = manifest["prompt_sha256"]
    for entry in manifest["cases"]:
        hashes[str(root / entry["input_file"])] = entry["input_sha256"]
    for path, digest in hashes.items():
        if sha(Path(path)) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    return manifest


def prepare(source, root, prompt, seed):
    if root.exists():
        raise ValueError("Output exists; use another directory to preserve frozen inputs")
    entries = read(source / "source_manifest_snapshot.json")["cases"]
    random.Random(seed).shuffle(entries)
    cases, hashes = [], {}
    for i, entry in enumerate(entries, 1):
        cid, jid = entry["case_id"], f"J{i:03d}"
        ip, pp = source / cid / "input.json", source / cid / "plan.json"
        packet = {"case_id": jid, "request": read(ip), "candidate_plan": read(pp)}
        target = root / "inputs" / f"{jid}.json"
        dump(target, packet)
        hashes.update({str(ip): sha(ip), str(pp): sha(pp)})
        cases.append({"judge_case_id": jid, "case_id": cid, "move_id": entry["move_id"],
                      "input_file": f"inputs/{jid}.json", "input_sha256": sha(target)})
    shutil.copyfile(prompt, root / "judge_prompt.md")
    dump(root / "manifest.json", {"schema_version": 1, "requested_model": "gpt-6-astra",
         "transport": "codex_subagent", "reasoning_effort": "high", "fork_context": False,
         "source_root": str(source), "source_sha256": hashes, "cases": cases,
         "prompt_file": "judge_prompt.md", "prompt_sha256": sha(prompt),
         "blinding": {"human_labels": False, "generator_model": False, "repair_notes": False,
                      "images": False, "student_measurements": False}, "calibration_validated": False})


def import_answers(root, answers):
    manifest = verify(root)
    results = {}
    for entry in manifest["cases"]:
        jid = entry["judge_case_id"]
        source, target = answers / f"{jid}.json", root / "responses" / f"{jid}.json"
        if not source.exists():
            results[jid] = "missing"
            continue
        try:
            validate_judgment(source.read_text(encoding="utf-8"), read(root / entry["input_file"]))
        except ValueError as error:
            results[jid] = str(error)
            continue
        if target.exists() and sha(source) != sha(target):
            raise ValueError(f"Refusing to overwrite imported answer: {target}")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        results[jid] = "valid"
    return results


def audit_sample(rows, rate, seed):
    if not 0 <= rate <= 1:
        raise ValueError("Audit rate must be between 0 and 1")
    rng = random.Random(seed)
    chosen = set()
    for move in sorted({r["move_id"] for r in rows}):
        candidates = sorted(r["record_id"] for r in rows if r["move_id"] == move
                            and r["triage"] == "auto_pass_provisional")
        count = min(len(candidates), max(1, round(len(candidates)*rate))) if candidates and rate else 0
        chosen.update(rng.sample(candidates, count))
    return chosen


def write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["record_id"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("'"+v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v)
                             for k, v in row.items()})


def analyze(root, output, rate=0.1, seed=20261002, human_csv=None):
    manifest = verify(root)
    if output.exists():
        raise ValueError("Analysis output exists; use another directory to preserve results")
    rows, probes, errors = [], [], {}
    hashes = dict(manifest["source_sha256"])
    hashes[str(root / "manifest.json")] = sha(root / "manifest.json")
    hashes[str(root / manifest["prompt_file"])] = manifest["prompt_sha256"]
    axis = signed_axis_probes()
    for entry in manifest["cases"]:
        cid, jid = entry["case_id"], entry["judge_case_id"]
        packet = read(root / entry["input_file"])
        hashes[str(root / entry["input_file"])] = entry["input_sha256"]
        judgments, judge_error, schema_error = {}, None, None
        try:
            plan = parse_plan(json.dumps(packet["candidate_plan"], ensure_ascii=False), packet["request"])
            numeric = probe_plan(plan)
        except (ValueError, KeyError) as error:
            schema_error, numeric = str(error), []
        answer = root / "responses" / f"{jid}.json"
        try:
            result = validate_judgment(answer.read_text(encoding="utf-8"), packet)
            judgments = {c["clause_id"]: c for c in result["checks"]}
            hashes[str(answer)] = sha(answer)
        except (OSError, ValueError) as error:
            judge_error = str(error)
        if judge_error or schema_error:
            errors[cid] = {"judge": judge_error, "schema": schema_error}
        probes.extend({"case_id": cid, **p} for p in numeric)
        checks = {c["clause_id"]: c for c in packet["candidate_plan"]["checks"]}
        for clause in packet["request"]["clauses"]:
            clid = clause["clause_id"]
            check, judgment = checks.get(clid, {}), judgments.get(clid)
            ids = {e["query_id"] for e in check.get("evidence", [])}
            relevant = [p for p in numeric if p["query_id"] in ids] + axis
            route, reason = route_check(schema_error is None, relevant, judgment, judge_error)
            rows.append({"record_id": cid+":"+clid, "case_id": cid, "move_id": entry["move_id"],
                         "clause_id": clid, "source_text": clause["text"], "triage": route,
                         "triage_reason": reason, "model_verdict": model_verdict(judgment) if judgment else "unavailable",
                         "judgment": judgment, "probe_counts": dict(Counter(p["status"] for p in relevant))})
    sampled = audit_sample(rows, rate, seed)
    # Human labels are consulted only after all independent judgments and routing are fixed.
    human = {}
    if human_csv:
        with human_csv.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                rid = row.get("record_id") or row["case_id"]+":"+row["clause_id"]
                if rid in human:
                    raise ValueError("Duplicate human review record")
                human[rid] = row
        hashes[str(human_csv)] = sha(human_csv)
    for row in rows:
        prior = human.get(row["record_id"], {})
        row["audit_sample"] = row["record_id"] in sampled
        row["prior_manual_verdict"] = prior.get("manual_verdict", "pending")
        row["prior_manual_notes"] = prior.get("notes", "")
        expected = {"correct": "pass", "false_positive": "fail"}.get(row["prior_manual_verdict"])
        row["human_model_disagreement"] = bool(expected and row["model_verdict"] != "unavailable" and expected != row["model_verdict"])
        row["human_queue"] = (row["triage"] == "needs_review" or row["audit_sample"] or row["human_model_disagreement"])
    summary = {"schema_version": 1, "requested_model": manifest["requested_model"],
               "transport": manifest["transport"], "calibration_validated": False,
               "total": len(rows), "triage": dict(Counter(r["triage"] for r in rows)),
               "model_verdicts": dict(Counter(r["model_verdict"] for r in rows)),
               "probe_counts": dict(Counter(p["status"] for p in probes+axis)),
               "audit_sample_count": len(sampled), "audit_rate": rate, "audit_seed": seed,
               "human_queue_count": sum(r["human_queue"] for r in rows),
               "human_model_disagreement_count": sum(r["human_model_disagreement"] for r in rows),
               "prior_reviewed_count": sum(r["prior_manual_verdict"] != "pending" for r in rows),
               "dimension_counts": {d:dict(Counter(r["judgment"]["dimensions"][d]["verdict"] for r in rows if r["judgment"]))
                                    for d in ("relevance", "direction", "coverage", "scope")}, "errors": errors}
    output.mkdir(parents=True)
    dump(output / "summary.json", summary)
    dump(output / "review_overlay.json", {"schema_version": 1, "summary": summary,
         "source_hashes": hashes, "records": {r["record_id"]: r for r in rows}})
    dump(output / "geometric_probes.json", {"signed_axes": axis, "plans": probes})
    table = [{k:v for k,v in r.items() if k not in ("judgment", "probe_counts")} | {
             "issues": json.dumps(r["judgment"]["issues"] if r["judgment"] else [], ensure_ascii=False)} for r in rows]
    write_csv(output / "clause_triage.csv", table)
    write_csv(output / "human_queue.csv", [r for r in table if r["human_queue"]])
    groups = {}
    for r in rows:
        if r["human_queue"]:
            groups.setdefault((r["move_id"], r["clause_id"], r["source_text"]), []).append(r["record_id"])
    dump(output / "human_queue_groups.json", [{"move_id":m,"clause_id":c,"source_text":t,"records":ids}
                                               for (m,c,t),ids in groups.items()])
    n = len(rows)
    lines = ["# 检查计划独立预审", "", "本轮为未校准的预审试运行，不是专家正确率评测。", "",
             f"评审模型请求：{manifest['requested_model']}；通道：{manifest['transport']}。未提供人工标签、修复提示或学生指标。", "",
             f"共 {n} 条；人工队列 {summary['human_queue_count']} 条（含 {len(sampled)} 条分层随机抽检及既有标注分歧）。",
             f"相对逐条复核的待看条目缩减：{(1-summary['human_queue_count']/n):.1%}。这不是质量正确率或已证明的安全放行率。", "",
             "| 路由 | 条数 |", "|---|---:|", *[f"| {k} | {v} |" for k,v in summary['triage'].items()], "",
             "数值探针验证接口行为，不验证自由文本预期关系，也不证明动作要领理解正确。",
             "旧人工标签存在左右坐标解释争议，只保留对照；不据此计算准确率或调整原标注。", "",
             "## 人工队列按要领合并", "", "| 招式 | 要领 | 记录 |", "|---|---|---|"]
    lines += [f"| {m} | {t} | {', '.join(ids)} |" for (m,c,t),ids in groups.items()]
    (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "import", "analyze"))
    parser.add_argument("--root", type=Path, default=PROJECT / "check_plan_judge_results/first3_v1_astra")
    parser.add_argument("--source", type=Path, default=PROJECT / "check_plan_experiment/first3_v1_repaired_v1")
    parser.add_argument("--prompt", type=Path, default=PROJECT / "prompt/check_plan_judge.md")
    parser.add_argument("--answers", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--human-csv", type=Path)
    parser.add_argument("--audit-rate", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.source.resolve(), args.root.resolve(), args.prompt, args.seed)
    elif args.action == "import":
        if not args.answers:
            parser.error("--answers required")
        print(json.dumps(import_answers(args.root.resolve(), args.answers), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(analyze(args.root.resolve(), args.output or args.root / "analysis_v1",
                                args.audit_rate, args.seed, args.human_csv), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
