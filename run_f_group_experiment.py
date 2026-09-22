"""Prepare F-group inputs and validate Qwen responses collected through the web UI."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np

from aqa3d.endpoint_metrics import MOVE_METRICS, teacher_interval
from run_endpoint_feedback import read_metrics

PROJECT = Path(__file__).resolve().parent
STUDENTS = ("01", "02", "03", "04", "10")
QUALITY_FIELDS = ("center_value", "window_min", "window_max", "valid_frame_count", "requested_frame_count")


def strict_json(path):
    def invalid(token):
        raise ValueError(f"Non-finite JSON number: {token}")
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=invalid)


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fresh_directory(path):
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def quality(row):
    return {key: row.get(key) for key in QUALITY_FIELDS}


def make_case(case_id, event, definitions, normalizers, coordinate_frame, rows, teacher_ids, summary):
    metrics = []
    for metric_id in MOVE_METRICS[event["move_id"]]:
        student = rows[("student", metric_id)]
        teachers = [rows[(tid, metric_id)] for tid in teacher_ids]
        for r in [student]+teachers:
            if r["pose_id"] != event["pose_id"] or r["final_technique_step"] != event["final_technique_step"]:
                raise ValueError("Metric and manifest technique mismatch")
            if r["unit"] != definitions[metric_id]["unit"]:
                raise ValueError("Metric units do not match")
            if r["status"] == "valid" and r["value"] != r["window_median"]:
                raise ValueError("Expected window median as scoring value")
        values = [r["value"] if r["status"] == "valid" else None for r in teachers]
        stats = teacher_interval(values, summary["minimum_teachers"])
        metrics.append({"metric_id": metric_id, **definitions[metric_id],
            "student": {"value": student["value"] if student["status"] == "valid" else None,
                "computation_status": student["status"], **quality(student)},
            "teacher": {"sample_count": len(teachers), "valid_count": stats["valid_teacher_count"],
                "median": stats["median"], "mad_unscaled": stats["mad"],
                "median_2mad": {"lower": stats["median_minus_2mad"], "upper": stats["median_plus_2mad"]},
                "p10_p90": {"lower": stats["p10"], "upper": stats["p90"]},
                "samples": [{"teacher_code": f"T{i+1:02d}", "value": value,
                    "computation_status": r["status"], **quality(r)}
                    for i, (r, value) in enumerate(zip(teachers, values))]}})
    return {"case_id": case_id, "input_mode": "text_and_numeric_only",
        "move": {k: event[k] for k in ("move_id", "move_name_pinyin", "move_name_zh")},
        "pose_stage": "结束定势", "technique": event["final_technique_step"],
        "measurement_protocol": {"skeleton": "native SMPL 24 joints",
            "primary_student_value": "window_median", "angle_convention": "180 degrees = straight",
            "window_half_seconds": summary["window_seconds_each_side"],
            "window_clipping": "within current move; endpoint uses preceding frames and endpoint only",
            "coordinate_frame": coordinate_frame, "normalizers": normalizers,
            "minimum_valid_frames": summary["minimum_valid_frames"],
            "minimum_window_valid_ratio": summary["minimum_window_valid_ratio"],
            "minimum_valid_teachers": summary["minimum_teachers"],
            "teacher_range_interpretation": "descriptive sample ranges, not expert pass/fail thresholds",
            "mad_definition": "median(abs(x-median(x))); no 1.4826 scaling",
            "limitations": ["Torso up is not gravity.", "Head/Wrist are not forehead, eye or palm center.",
                "Cannot measure actual weight bearing, muscle relaxation or palm orientation.",
                "An endpoint alone cannot assess whole-move rhythm, continuity or motion amplitude.",
                "Window stability cannot exclude systematic reconstruction error."]},
        "metrics": metrics}


def prepare(metric_root, baseline_root, output):
    source_paths = [metric_root/name for name in ("summary.json", "reference_manifest_snapshot.json",
        "endpoint_metric_rules_first3.json", "all_endpoint_metrics.csv", "teacher_reference_values.json")]
    source_paths.append(baseline_root/"feedback.json")
    summary, manifest, definitions = [strict_json(p) for p in source_paths[:3]]
    baseline = strict_json(source_paths[-1])
    lookup = {}
    for row in read_metrics(source_paths[3]):
        key = (row["role"], row["subject_id"], row["move_id"], row["metric_id"])
        if key in lookup:
            raise ValueError(f"Duplicate metric row: {key}")
        lookup[key] = row
    tids = summary["teacher_ids"]
    if len(tids) != 10 or len(set(tids)) != 10:
        raise ValueError("Expected ten distinct teachers")
    references = {(r["move_id"], r["metric_id"]):r for r in strict_json(source_paths[4])}
    events = {e["move_id"]:e for e in manifest["endpoint_keyposes"]}
    cases, mapping = [], []
    for sid in STUDENTS:
        for mid in (1,2,3):
            event = events[mid]
            case_id = f"F{len(cases)+1:03d}"
            selected = {("student",m):lookup[("student",sid,mid,m)] for m in MOVE_METRICS[mid]}
            selected.update({(tid,m):lookup[("teacher",tid,mid,m)] for tid in tids for m in MOVE_METRICS[mid]})
            case = make_case(case_id,event,definitions["metric_definitions"],definitions["normalizers"],
                             definitions["coordinate_frame"],selected,tids,summary)
            for metric in case["metrics"]:
                ref = references[(mid,metric["metric_id"])]; t = metric["teacher"]
                if not np.allclose([t["median_2mad"]["lower"],t["median_2mad"]["upper"],
                                    t["p10_p90"]["lower"],t["p10_p90"]["upper"]],
                                   [ref["median_minus_2mad"],ref["median_plus_2mad"],ref["p10"],ref["p90"]],atol=1e-9,rtol=0):
                    raise ValueError("Teacher ranges differ from frozen experiment")
            cases.append(case)
            mapping.append({"case_id":case_id,"student_id":sid,"move_id":mid,
                "move_name":event["move_name_pinyin"],"metric_count":len(case["metrics"]),
                "input_file":f"inputs/{case_id}.json","response_file":f"responses/{case_id}.json"})
    fresh_directory(output)
    for folder in ("inputs", "responses", "evaluation"):
        (output/folder).mkdir()
    shutil.copyfile(PROJECT/"prompt/f_group_prompt.md",output/"F_prompt.md")
    for case in cases:
        write_json(output/"inputs"/(case["case_id"]+".json"),case)
    for item in mapping:
        item["input_sha256"] = digest(output/item["input_file"])
    write_csv(output/"case_index.csv",mapping,list(mapping[0]))
    write_json(output/"evaluation/baseline_snapshot.json",baseline)
    write_json(output/"experiment_manifest.json", {"status":"prepared_awaiting_model_responses",
        "group":"F","expected_model_label":"Qwen3.8-max","cases":mapping,
        "case_count":len(cases),"metric_count":sum(len(c["metrics"]) for c in cases),
        "source_sha256":{str(p.absolute()):digest(p) for p in source_paths},
        "prompt_sha256":digest(output/"F_prompt.md"),
        "baseline_sha256":digest(output/"evaluation/baseline_snapshot.json"),
        "reference_manifest_sha256":summary["reference_manifest_sha256"],
        "teacher_code_mapping":{f"T{i+1:02d}":tid for i,tid in enumerate(tids)}})
    fields=["case_id","model_display_name","model_version","run_time","temperature","top_p","thinking_mode",
            "thinking_budget","search_enabled","conversation_id","notes"]
    write_csv(output/"run_log.csv",[{**dict.fromkeys(fields,""),"case_id":m["case_id"],
        "model_display_name":"Qwen3.8-max","search_enabled":"false"} for m in mapping],fields)
    write_readme(output,mapping)
    return {"cases":len(cases),"metrics":sum(len(c["metrics"]) for c in cases),"status":"prepared"}


def write_readme(output,mapping):
    lines = ["# F 组：结束定势数值证据实验", "",
        "已准备 15 份真实定势输入，覆盖 5 位学生的前三式、共 130 项指标。目前等待 Qwen 回答，尚不能报告模型准确率。", "",
        "## 在百炼网页运行", "",
        "1. 每案例新建独立会话，选择相同 Qwen3.8-max 模型及参数，关闭搜索。",
        "2. 粘贴 [统一提示词](F_prompt.md)，只上传 inputs 内对应的一个 JSON。不上传报告、baseline、评价文件、图片或视频。",
        "3. 保持所有案例参数一致，在 run_log.csv 记录实际模型名称、参数、时间及会话信息。使用界面支持的低随机性设置，不在看到个别结果后单独调参。",
        "4. 把最终 JSON 回答按编号保存至 responses，保留第一次完整回答。若格式失败，记录后另建目录重试，不静默替换。",
        "5. 全部完成后运行校验命令，程序保留原始回答。", "",
        "| 编号 | 学生 | 招式 | 指标数 | 上传文件 | 回答保存位置 |", "|---|---|---|---|---|---|"]
    for r in mapping:
        lines.append(f"| {r['case_id']} | {r['student_id']} | {r['move_name']} | {r['metric_count']} | [{r['case_id']}.json]({r['input_file']}) | {r['response_file']} |")
    lines += ["", "## 校验与人工比较", "", "```bash", "conda activate 4d-humans",
        "cd /home/sqw/Projects/3D-AQA", "python run_f_group_experiment.py validate --experiment-root "+str(output.absolute()), "```", "",
        "自动核验案例、指标覆盖、数值引用、区间比较及证据 ID；动作诊断与指导正确性仍需人工判断。",
        "输出 validation/ 内的问题清单、数值检查及模型/baseline 人工对照表。复查时用 --output-root 指定新目录，保留已有人工记录。",
        "manual_comparison.csv 中 diagnosis_verdict / guidance_verdict 填 correct / false_positive / uncertain；multi_metric_supported 填 yes / no / not_applicable。",
        "case_review.csv 中独立列出真实问题及双方漏报，记录多指标解释是否成立。不能仅按引用了多个指标就认定有效综合。",
        "规则 baseline 的暂缓判断与模型的诊断错误分别记录，不强行一对一匹配不同数量的建议。",
        "主要观察有依据建议比例、无依据判断、真实问题覆盖和人工确认的多指标综合。15 例仅用于预实验。",
        "已有复核 UI 的标签评价规则建议或暂缓决定，不能直接作为模型诊断的完整真值，尤其不能用于计算漏报。", "",
        "## 输入约定", "",
        "输入包含完整动作要领、对应的所有指标、定义/单位/坐标系/归一化方式、学生中位数和中心值、教师双区间及十个教师原始值和窗口信息。",
        "模型输入不含学生真实编号、人工排名、baseline 判断或建议、图片路径。case_index.csv 仅供研究者追溯。",
        "沿用现有数值及区间；不附带规则系统的稳定性阈值和判定结果。",
        "相同 JSON 加视频可用于后续配对实验；F 组单独不能证明模型是否使用视觉。", ""]
    (output/"README.md").write_text("\n".join(lines),encoding="utf-8")


def relation(value, interval, teacher_count, minimum_count=7):
    lo,hi = interval["lower"],interval["upper"]
    if teacher_count < minimum_count or any(not isinstance(v,(int,float)) or isinstance(v,bool)
           or not math.isfinite(v) for v in (value,lo,hi)) or hi-lo <= 1e-8:
        return "not_evaluable"
    return "below" if value < lo else "above" if value > hi else "within"


def numbers_match(left,right):
    if left is None or right is None:
        return left is right
    if isinstance(right,list):
        return isinstance(left,list) and len(left)==len(right) and all(numbers_match(a,b) for a,b in zip(left,right))
    return isinstance(left,(float,int)) and not isinstance(left,bool) and math.isfinite(left) and abs(left-right)<=1e-6


def validate_response(case, response):
    errors, checks = [], []
    if not isinstance(response,dict):
        return ["Response must be a JSON object"],[]
    if set(response) != {"case_id","metric_comparisons","findings","coach_summary"}:
        errors.append("Response fields do not match requested schema")
    if response.get("case_id") != case["case_id"]:
        errors.append("case_id mismatch")
    metrics = {m["metric_id"]:m for m in case["metrics"]}
    comparisons = response.get("metric_comparisons")
    if not isinstance(comparisons,list):
        comparisons=[];errors.append("metric_comparisons must be a list")
    seen = set()
    for result in comparisons:
        if not isinstance(result,dict) or not isinstance(result.get("metric_id"),str):
            errors.append("Malformed metric comparison");continue
        mid=result["metric_id"]
        if mid not in metrics or mid in seen:
            errors.append(f"Unknown/duplicate metric: {mid}");continue
        seen.add(mid);metric=metrics[mid]
        for method in ("median_2mad","p10_p90"):
            interval=metric["teacher"][method];expected=relation(metric["student"]["value"],interval,
                metric["teacher"]["valid_count"],case["measurement_protocol"]["minimum_valid_teachers"])
            quote_ok=numbers_match(result.get("student_value"),metric["student"]["value"]) and numbers_match(
                result.get(method+"_interval"),[interval["lower"],interval["upper"]])
            actual=result.get(method+"_relation")
            checks.append({"case_id":case["case_id"],"metric_id":mid,"method":method,
                "expected_relation":expected,"model_relation":actual,"relation_correct":actual==expected,
                "numbers_correct":quote_ok})
            if actual!=expected or not quote_ok:
                errors.append(f"Incorrect relation or numerical quote: {mid}/{method}")
    if seen != set(metrics):errors.append("Missing metric comparisons: "+", ".join(sorted(set(metrics)-seen)))
    findings=response.get("findings")
    if not isinstance(findings,list):findings=[];errors.append("findings must be a list")
    ids=set();possible=0
    for finding in findings:
        if not isinstance(finding,dict):errors.append("Malformed finding");continue
        fid=finding.get("finding_id")
        if not isinstance(fid,str) or not fid or fid in ids:
            errors.append("Missing/duplicate finding ID")
        else:ids.add(fid)
        for key in ("problem","explanation","uncertainty"):
            if not isinstance(finding.get(key),str) or not finding[key].strip():errors.append(f"Missing finding text: {key}")
        evidence=finding.get("evidence_metric_ids")
        if not isinstance(evidence,list) or not evidence or any(not isinstance(x,str) or x not in metrics for x in evidence):
            errors.append(f"Invalid evidence IDs: {fid}")
        elif len(set(evidence))!=len(evidence):errors.append(f"Duplicate evidence IDs: {fid}")
        assessment=finding.get("assessment")
        if assessment=="possible_issue":
            possible+=1
            if not isinstance(finding.get("guidance"),str) or not finding["guidance"].strip():errors.append(f"Missing guidance: {fid}")
        elif assessment=="insufficient_evidence":
            if finding.get("guidance") is not None:errors.append(f"Abstention must not give guidance: {fid}")
        else:errors.append(f"Unknown assessment: {fid}")
    if possible>3:errors.append("More than three possible issues")
    if not isinstance(response.get("coach_summary"),str) or not response["coach_summary"].strip():errors.append("Missing coach_summary")
    return errors,checks


def validate(experiment, output):
    manifest=strict_json(experiment/"experiment_manifest.json")
    if digest(experiment/"F_prompt.md")!=manifest["prompt_sha256"]:
        raise ValueError("Prompt changed; create a separate experiment")
    baseline_path=experiment/"evaluation/baseline_snapshot.json"
    if "baseline_sha256" in manifest and digest(baseline_path)!=manifest["baseline_sha256"]:
        raise ValueError("Frozen baseline changed")
    baseline=strict_json(baseline_path)
    errors,checks,reviews,case_reviews=[],[],[],[]
    received=[];missing=[];response_hashes={}

    def review_row(case_id,source,fid,assessment,metrics,problem,explanation,guidance,needs_review):
        return {"case_id":case_id,"source":source,"finding_id":fid,"assessment":assessment,
            "evidence_metric_ids":";".join(metrics),"problem":problem,"explanation":explanation,
            "guidance":guidance or "","automatic_errors_present":needs_review,
            "diagnosis_verdict":"pending","guidance_verdict":"pending","multi_metric_supported":"pending",
            "unsupported_claims":"","reviewer":"","notes":""}

    for entry in manifest["cases"]:
        inp=experiment/entry["input_file"]
        if digest(inp)!=entry["input_sha256"]:raise ValueError(f"Input changed: {entry['case_id']}")
        case=strict_json(inp);path=experiment/entry["response_file"]
        if not path.exists():missing.append(entry["case_id"]);continue
        received.append(entry["case_id"]);response_hashes[entry["case_id"]]=digest(path)
        try:
            response=strict_json(path);issues,numeric=validate_response(case,response)
        except (ValueError,TypeError) as exc:
            response={};issues=[str(exc)];numeric=[]
        checks.extend(numeric)
        errors.extend({"case_id":entry["case_id"],"error":issue} for issue in issues)
        if isinstance(response,dict) and isinstance(response.get("findings"),list):
            for f in response["findings"]:
                if not isinstance(f,dict):continue
                ids=f.get("evidence_metric_ids",[])
                ids=[str(x) for x in ids] if isinstance(ids,list) else []
                reviews.append(review_row(entry["case_id"],"F",f.get("finding_id",""),f.get("assessment",""),
                    ids,f.get("problem",""),f.get("explanation",""),f.get("guidance",""),bool(issues)))
        selected=[r for r in baseline if r["subject_id"]==entry["student_id"] and r["move_id"]==entry["move_id"]]
        for r in selected:
            if r["decision"]=="feedback_candidate":
                reviews.append(review_row(entry["case_id"],"rule_baseline",r["record_id"],"possible_issue",
                    [r["metric_id"]],r["metric_label_zh"],r["technique_aspect"],r["feedback"],False))
        case_reviews.append({"case_id":entry["case_id"],"student_id":entry["student_id"],"move":entry["move_name"],
            "F_coach_summary":response.get("coach_summary","") if isinstance(response,dict) else "",
            "expert_problem_list":"","F_missed_problems":"","baseline_missed_problems":"",
            "F_summary_adds_unsupported_claims":"pending","F_useful_metric_integration":"pending",
            "baseline_abstention_count":sum(r["decision"]=="needs_review" for r in selected),
            "reviewer":"","notes":""})
    fresh_directory(output)
    write_csv(output/"numeric_checks.csv",checks,["case_id","metric_id","method","expected_relation","model_relation","relation_correct","numbers_correct"])
    write_csv(output/"manual_comparison.csv",reviews,list(review_row("","","","",[],"","","",False)))
    write_csv(output/"case_review.csv",case_reviews,["case_id","student_id","move","F_coach_summary","expert_problem_list",
        "F_missed_problems","baseline_missed_problems","F_summary_adds_unsupported_claims","F_useful_metric_integration",
        "baseline_abstention_count","reviewer","notes"])
    result={"status":"awaiting_responses" if not received else "partial" if missing else "responses_received_pending_manual_review",
        "submitted_count":len(received),"expected_case_count":len(manifest["cases"]),"missing_cases":missing,
        "automatic_errors":errors,"checked_relations":len(checks),
        "expected_relations_for_submitted_cases":sum(2*c["metric_count"] for c in manifest["cases"] if c["case_id"] in received),
        "correct_relations":sum(c["relation_correct"] for c in checks),
        "correct_numerical_quotes":sum(c["numbers_correct"] for c in checks),
        "response_sha256":response_hashes,"diagnosis_accuracy":None,"manual_review_status":"pending"}
    write_json(output/"validation_summary.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare")
    p.add_argument("--metric-root",type=Path,default=PROJECT/"endpoint_metric_results/first3_five_students")
    p.add_argument("--baseline-root",type=Path,default=PROJECT/"endpoint_feedback_results/first3_five_students")
    p.add_argument("--output-root",type=Path,default=PROJECT/"f_group_experiment/first3_five_students")
    p=sub.add_parser("validate")
    p.add_argument("--experiment-root",type=Path,default=PROJECT/"f_group_experiment/first3_five_students")
    p.add_argument("--output-root",type=Path)
    args=parser.parse_args()
    result=prepare(args.metric_root,args.baseline_root,args.output_root) if args.command=="prepare" else validate(
        args.experiment_root,args.output_root or args.experiment_root/"validation")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
