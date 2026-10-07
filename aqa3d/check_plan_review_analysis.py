"""Compare human plan verdicts with independent judge verdicts, without relabeling."""

from collections import Counter
import csv
from pathlib import Path

HUMAN_VERDICTS = {"correct": "pass", "false_positive": "fail", "uncertain": "uncertain", "pending": "pending"}
GROUPS = {
    "astra_reject": "Astra 判不正确",
    "random_audit": "Astra 通过项随机抽检",
    "astra_uncertain": "Astra 不确定",
    "other_pass": "其他通过项（非本轮随机抽检）",
    "astra_unavailable": "缺少 Astra 结论",
}
DERIVED_FIELDS = ["human_plan_verdict", "analysis_group", "judge_agreement",
                  "agreement_exclusion", "review_history"]


def read_review(path: Path, require_judge: bool = True) -> tuple[list[str], list[dict]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        required = {"record_id", "case_id", "move_id", "clause_id", "source_text", "manual_verdict", "notes"}
        if require_judge:
            required |= {"model_verdict", "audit_sample", "human_queue"}
        if not required <= set(fields) or len(fields) != len(set(fields)):
            raise ValueError("Missing required columns or duplicate CSV headers")
        if set(fields) & set(DERIVED_FIELDS):
            raise ValueError("Expected original review CSV, not an already analyzed export")
        rows, seen = [], set()
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV row at line {reader.line_num}")
            rid = row["record_id"]
            if not rid or rid in seen or rid != row["case_id"]+":"+row["clause_id"]:
                raise ValueError(f"Invalid or duplicate record_id: {rid}")
            seen.add(rid)
            if not row["move_id"].isdigit() or int(row["move_id"]) < 1:
                raise ValueError(f"Invalid move_id: {rid}")
            if row["manual_verdict"] not in HUMAN_VERDICTS:
                raise ValueError(f"Unknown manual verdict: {rid}")
            if require_judge:
                if row["model_verdict"] not in ("pass", "fail", "uncertain", "unavailable"):
                    raise ValueError(f"Unknown model verdict: {rid}")
                for key in ("audit_sample", "human_queue"):
                    if row[key].lower() not in ("true", "false"):
                        raise ValueError(f"Invalid boolean {key}: {rid}")
                if row["audit_sample"].lower() == "true" and row["model_verdict"] != "pass":
                    raise ValueError(f"Random audit must come from passed plans: {rid}")
            rows.append(row)
    if not rows:
        raise ValueError("Empty review CSV")
    return fields, rows


def classify(row: dict, prior: dict | None = None) -> dict:
    human, model = HUMAN_VERDICTS[row["manual_verdict"]], row["model_verdict"]
    audit = row["audit_sample"].lower() == "true"
    group = {"fail": "astra_reject", "uncertain": "astra_uncertain", "unavailable": "astra_unavailable"}.get(model)
    group = group or ("random_audit" if audit else "other_pass")
    exclusions = []
    if human in ("pending", "uncertain"):
        exclusions.append("human_"+human)
    if model in ("uncertain", "unavailable"):
        exclusions.append("astra_"+model)
    agreement = "excluded" if exclusions else "agree" if human == model else "disagree"
    history = "not_reviewed" if human == "pending" else "unknown"
    if prior is not None and human != "pending":
        old = prior.get(row["record_id"])
        if old is None or old["manual_verdict"] == "pending":
            history = "newly_reviewed"
        elif all(old[key] == row[key] for key in ("manual_verdict", "notes")):
            history = "previously_reviewed_unchanged"
        else:
            history = "previously_reviewed_updated"
    return dict(row, human_plan_verdict=human, analysis_group=group, judge_agreement=agreement,
                agreement_exclusion=";".join(exclusions), review_history=history)


def group_statistics(rows: list[dict]) -> dict:
    counts = Counter(r["judge_agreement"] for r in rows)
    comparable = counts["agree"] + counts["disagree"]
    return {"total": len(rows), "human_verdicts": dict(Counter(r["manual_verdict"] for r in rows)),
            "agree": counts["agree"], "disagree": counts["disagree"], "excluded": counts["excluded"],
            "comparable": comparable, "agreement_rate": counts["agree"]/comparable if comparable else None,
            "exclusion_reasons": dict(Counter(reason for r in rows for reason in r["agreement_exclusion"].split(";") if reason))}


def summarize(rows: list[dict]) -> dict:
    groups = {group: group_statistics([r for r in rows if r["analysis_group"] == group]) for group in GROUPS}
    return {"schema_version": 1, "total": len(rows),
            "semantics": {"manual_verdict": HUMAN_VERDICTS,
                          "agreement": "Both verdicts are binary and equal",
                          "denominator": "agree + disagree; uncertain, pending and unavailable are excluded",
                          "unit": "One clause-plan record, not one independent source technique"},
            "human_verdicts": dict(Counter(r["manual_verdict"] for r in rows)),
            "review_history": dict(Counter(r["review_history"] for r in rows)),
            "groups": groups,
            "human_queue": group_statistics([r for r in rows if r["human_queue"].lower() == "true"]),
            "all_records_descriptive_only": group_statistics(rows),
            "by_move": {move: {g: group_statistics([r for r in rows if r["move_id"] == move and r["analysis_group"] == g])
                               for g in GROUPS} for move in sorted({r["move_id"] for r in rows}, key=int)},
            "by_case": {case: group_statistics([r for r in rows if r["case_id"] == case])
                        for case in sorted({r["case_id"] for r in rows})}}


def report_markdown(summary: dict, rows: list[dict]) -> str:
    def table_row(label, s):
        rate = f"{s['agree']}/{s['comparable']} = {s['agreement_rate']:.1%}" if s["comparable"] else "不计算"
        return f"| {label} | {s['total']} | {s['agree']} | {s['disagree']} | {s['excluded']} | {rate} |"

    def escape(text):
        return str(text).replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")

    lines = ["# Astra 与人工检查计划判定的一致性", "",
             "人工 correct（对应）=计划合理；false_positive（不对应）=计划不合理。",
             "Astra fail + 人工 false_positive 为一致；Astra pass + 人工 correct 为一致。",
             "Astra 不确定不是 fail；任一方不确定或人工未复核，不进入二元一致率分母。", "",
             "## 分组统计", "", "| 分组 | 全部条目 | 一致 | 不一致 | 排除 | 一致率 |",
             "|---|---:|---:|---:|---:|---|",
             *[table_row(GROUPS[g], s) for g, s in summary["groups"].items() if s["total"]], "",
             "## 统计口径", "",
             "- 原始标注原样保留；不因几何解释争议、备注缺失或模型理由而擅自改标。",
             "- 一致仅指结论标签一致，不意味着模型指出的每个原因都得到人工认可。",
             "- 原 human_model_disagreement 是复核前的历史字段，不用于本轮计算；使用新字段 judge_agreement。",
             "- 两组经过不同抽样，且同一要领有多轮生成，不合并报告为总体准确率，也不推算总体漏检率。",
             "- 模型评审理由在复核界面可见，因此这是当前复核流程中的描述性一致性，不是盲测准确率。", "",
             f"人工判定计数：`{summary['human_verdicts']}`。",
             f"与上轮 CSV 的关系：`{summary['review_history']}`。", "",
             "## 按招式分组", "", "| 分组 | 全部条目 | 一致 | 不一致 | 排除 | 一致率 |",
             "|---|---:|---:|---:|---:|---|" ]
    for move, groups in summary["by_move"].items():
        for group in ("astra_reject", "random_audit", "astra_uncertain"):
            if groups[group]["total"]:
                lines.append(table_row(f"第 {move} 式 / {GROUPS[group]}", groups[group]))
    lines += ["", "## 不一致条目", "", "| 记录 | 要领 | Astra | 人工 | 人工备注 |",
              "|---|---|---|---|---|"]
    for r in rows:
        if r["judge_agreement"] == "disagree":
            lines.append("| " + " | ".join(escape(r[k]) or "无备注" for k in
                         ("record_id", "source_text", "model_verdict", "manual_verdict", "notes")) + " |")
    lines += ["", "## 已复核但不计入二元一致率", "", "| 记录 | 要领 | Astra | 人工 | 排除原因 |", "|---|---|---|---|---|"]
    for r in rows:
        if r["judge_agreement"] == "excluded" and r["manual_verdict"] != "pending":
            lines.append("| " + " | ".join(escape(r[k]) for k in
                         ("record_id", "source_text", "model_verdict", "manual_verdict", "agreement_exclusion")) + " |")
    lines += ["", f"完整 {len(rows)} 条见 agreement_details.csv；disagreements.csv 仅保存明确分歧；excluded_cases.csv 保留排除项。",
              "输入快照及哈希记录在 input_review.csv、summary.json 中。原实验结果和人工进度均不修改。"]
    return "\n".join(lines)+"\n"
