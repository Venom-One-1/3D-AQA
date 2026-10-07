"""Generate independent endpoint check plans, then execute on frozen pose caches."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import uuid

from aqa3d.check_planning import (
    build_input, queries_payload, stability, strict_loads, validate_response,
)
from run_f_group_api import run_case, validate_base_url
from run_pose_measurements import export_experiment, write_csv, write_json


PROJECT = Path(__file__).resolve().parent
MODEL = "qwen3.8-max-0902"
BASE_URL = "https://ws-05bnmtv66yx74w8r.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return strict_loads(path.read_text(encoding="utf-8"))


def prepare(input_root: Path, output: Path, config: dict) -> dict:
    input_root, output = input_root.resolve(), output.resolve()
    if output.is_relative_to(input_root) or input_root.is_relative_to(output):
        raise ValueError("Output must be separate from frozen inputs")
    manifest_path = input_root / "reference_manifest_snapshot.json"
    manifest = read_json(manifest_path)
    cache_config = read_json(input_root / "summary.json")
    if cache_config["reference_manifest_sha256"] != digest(manifest_path):
        raise ValueError("Frozen reference manifest checksum mismatch")
    prompt = PROJECT / "prompt/check_planning_prompt.md"
    sources = [Path(__file__), PROJECT/"aqa3d/check_planning.py", PROJECT/"aqa3d/pose_measurements.py",
               PROJECT/"run_pose_measurements.py", PROJECT/"run_f_group_api.py", prompt]
    inputs = {f"inputs/M{mid:02d}.json": build_input(manifest, mid) for mid in (1, 2, 3)}
    entries = [{"case_id": f"M{mid:02d}_R{repeat:02d}", "move_id": mid, "repeat": repeat,
                "input_file": f"inputs/M{mid:02d}.json", "prompt_file": "planner_prompt.md",
                "response_file": f"responses/M{mid:02d}_R{repeat:02d}.json"}
               for mid in (1, 2, 3) for repeat in (1, 2, 3)]
    frozen = {
        "experiment": "technique_to_endpoint_check_plan_v1", "api_config": config,
        "input_root": str(input_root), "reference_manifest_sha256": digest(manifest_path),
        "implementation_sha256": {str(p.relative_to(PROJECT)): digest(p) for p in sources},
        "cases": entries, "selection": "first_schema_valid_response_in_repeat_order",
        "semantic_accuracy": None, "held_out_evaluation": False,
        "development_moves": [1, 2, 3],
        "inputs_exclude": ["student_values", "teacher_ranges", "old_queries", "old_rules",
                           "old_feedback", "human_reviews", "videos", "images"],
        "sampling_note": "Independent identical requests; low randomness does not imply deterministic outputs",
    }
    saved = output / "experiment_manifest.json"
    if saved.exists():
        previous = read_json(saved)
        hashes = previous.pop("frozen_file_sha256")
        if previous != frozen:
            raise ValueError("Run source or configuration changed; use a fresh output directory")
        for relative, checksum in hashes.items():
            if digest(output/relative) != checksum:
                raise ValueError(f"Frozen request changed: {relative}")
        return dict(previous, frozen_file_sha256=hashes)
    output.mkdir(parents=True, exist_ok=True)
    if any(p.name != ".run.lock" for p in output.iterdir()):
        raise ValueError("New experiment output must be empty")
    (output/"inputs").mkdir()
    (output/"responses").mkdir()
    (output/"planner_prompt.md").write_bytes(prompt.read_bytes())
    for relative, payload in inputs.items():
        write_json(output/relative, payload)
    frozen["frozen_file_sha256"] = {relative: digest(output/relative)
                                    for relative in ["planner_prompt.md", *inputs]}
    write_json(saved, frozen)
    return frozen


def load_key(path: Path | None) -> str:
    if path is None:
        key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    else:
        with path.open(encoding="utf-8") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise ValueError("API key file must be an owner-only regular file (mode 600)")
            key = handle.read().strip()
    if not key or any(c.isspace() for c in key):
        raise ValueError("Set DASHSCOPE_API_KEY or provide --api-key-file; key is never saved")
    return key


def call_plans(manifest: dict, output: Path, key: str, *, workers: int = 2,
               retry_failed: bool = False) -> list[dict]:
    entries, config = manifest["cases"], manifest["api_config"]
    first = run_case(entries[0], output, config, key, retries=0, retry_failed=retry_failed)
    print(f"{first['case_id']}: {first['status']}", flush=True)
    results = [first]
    if first["status"] != "completed":
        return results
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_case, e, output, config, key, 0, retry_failed) for e in entries[1:]]
        for future in as_completed(futures):
            result = future.result()
            print(f"{result['case_id']}: {result['status']}", flush=True)
            results.append(result)
    return results


def _cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def analyze(manifest: dict, output: Path, *, execute: bool = True) -> tuple[Path, dict]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:6]
    analysis = output / ("analysis_" + stamp)
    analysis.mkdir()
    validations, valid_plans, selections, execution = [], {}, {}, []
    review_rows, report = [], ["# 动作要领到检查计划：开发实验", "",
        "前三式各三次独立调用。输入不含学生值、教师区间或旧规则。", "",
        "程序只验证格式、可执行性和逐条引用；语义正确性仍未人工确认。"
        "重复一致不代表正确，区间不是动作合格阈值。本轮不输出诊断或训练反馈。", ""]
    for entry in manifest["cases"]:
        cid, mid = entry["case_id"], entry["move_id"]
        status_path = output/"api_calls"/cid/"status.json"
        request = read_json(output/entry["input_file"])
        record = {"case_id": cid, "move_id": mid, "repeat": entry["repeat"],
                  "schema_valid": False, "semantic_correctness": "not_reviewed"}
        plan = None
        if not status_path.exists():
            record.update(api_status="not_run", errors=["API response not available"])
        else:
            status = read_json(status_path)
            record["api_status"] = status["status"]
            if status["status"] == "completed":
                answer = output/entry["response_file"]
                if digest(answer) != status["response_sha256"]:
                    raise ValueError(f"Saved response changed: {cid}")
                validation, plan = validate_response(answer.read_text(encoding="utf-8"), request)
                record.update(validation)
                record["returned_model"] = status["attempts"][-1].get("model")
                record["model_matches_requested"] = record["returned_model"] == manifest["api_config"]["model"]
                if not record["model_matches_requested"]:
                    record["errors"].append("Returned model differs from frozen requested model; excluded")
                    plan = None
            else:
                record["errors"] = ["API request incomplete; inspect saved attempt metadata"]
        record["eligible"] = plan is not None
        validations.append(record)
        report += [f"## {cid} · {request['move_name']}", "",
                   f"API: {record['api_status']}；格式: {record['schema_valid']}；语义: 未复核", ""]
        if plan is None:
            report += ["；".join(record["errors"]), ""]
            continue
        valid_plans.setdefault(mid, []).append((cid, plan))
        selections.setdefault(str(mid), cid)
        folder = analysis/cid
        folder.mkdir()
        write_json(folder/"plan.json", plan)
        write_json(folder/"queries.json", queries_payload(plan))
        clauses = {c["clause_id"]: c["text"] for c in request["clauses"]}
        definitions = {q["query_id"]: q for q in plan["queries"]}
        report += ["| 原文 | 支持类型 | 几何查询 / 依据 | 局限 |", "|---|---|---|---|"]
        for check in plan["checks"]:
            details = []
            for evidence in check["evidence"]:
                q = definitions[evidence["query_id"]]
                details.append(f"{q['operation']}({', '.join(q['points'])}) "
                               f"[{', '.join(evidence['components'])}]; "
                               f"{evidence['purpose']}; {evidence['expected_relation']}")
            limitations = "；".join(check["limitations"])
            row = {"case_id": cid, "move_id": mid, "clause_id": check["clause_id"],
                   "source_text": clauses[check["clause_id"]], "support": check["support"],
                   "endpoint_interpretation": check["endpoint_interpretation"],
                   "evidence": "；".join(details), "limitations": limitations,
                   "manual_verdict": "pending", "notes": ""}
            review_rows.append(row)
            report.append("| " + " | ".join(_cell(v) for v in
                (row["source_text"], row["support"], row["evidence"] or row["endpoint_interpretation"], limitations)) + " |")
        report += ["", f"[完整计划与坐标/归一化定义]({cid}/plan.json)", ""]
        if execute and plan["queries"]:
            result = export_experiment(Path(manifest["input_root"]), folder/"measurements",
                                       queries_path=folder/"queries.json", moves=[mid])
            execution.append({"case_id": cid, **result})
    pairs = [{"move_id": mid, **pair} for mid, plans in valid_plans.items() for pair in stability(plans)]
    write_json(analysis/"validation.json", validations)
    write_json(analysis/"stability.json", pairs)
    write_json(analysis/"executions.json", execution)
    if review_rows:
        write_csv(analysis/"plan_review.csv", review_rows)
    if pairs:
        write_csv(analysis/"stability.csv", pairs)
    summary = {
        "expected_calls": len(manifest["cases"]),
        "completed_calls": sum(v["api_status"] == "completed" for v in validations),
        "schema_valid_plans": sum(v["schema_valid"] for v in validations),
        "eligible_plans": sum(v["eligible"] for v in validations),
        "selected_by_first_valid_repeat": selections, "executed_plans": len(execution),
        "successful_executions": sum(e["status"] == "ok" for e in execution),
        "semantic_accuracy": None, "diagnosis_accuracy": None, "feedback_generated": False,
        "held_out_evaluation": False, "source_files_verified_by_each_execution": bool(execution),
    }
    write_json(analysis/"summary.json", summary)
    (analysis/"plan_report.md").write_text("\n".join(report), encoding="utf-8")
    return analysis, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=PROJECT/"endpoint_metric_results/first3_five_students")
    parser.add_argument("--output-root", type=Path, default=PROJECT/"check_plan_experiment/first3_v1")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retry-failed", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--skip-execution", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        parser.error("workers must be 1..3")
    config = dict(model=MODEL, base_url=validate_base_url(args.base_url), temperature=0.2,
                  top_p=0.8, max_tokens=16384, enable_thinking=True, thinking_budget=4000,
                  timeout_seconds=300, enable_search=False)
    output = args.output_root.resolve()
    source = args.input_root.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        parser.error("Output must be separate from frozen input data")
    output.mkdir(parents=True, exist_ok=True)
    with (output/".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another process is using this experiment directory")
        manifest = prepare(source, output, config)
        if args.prepare_only:
            print(f"Prepared {len(manifest['cases'])} requests. No API calls sent. {output}")
            return
        if not args.analyze_only:
            key = load_key(args.api_key_file)
            call_plans(manifest, output, key, workers=args.workers, retry_failed=args.retry_failed)
        path, summary = analyze(manifest, output, execute=not args.skip_execution)
        print(json.dumps({"analysis": str(path), **summary}, ensure_ascii=False, indent=2))
        if summary["eligible_plans"] != len(manifest["cases"]):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
