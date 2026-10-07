"""Compile reviewed plans into neutral evidence and validate cited coaching output."""

from dataclasses import asdict
import hashlib
import json
import math

from .check_planning import strict_loads
from .endpoint_feedback import FeedbackConfig, stability_reasons


POLICY = {
    "version": "1.0",
    "plan_selection": "first_valid_repaired_repeat_before_student_values",
    "human_false_positive": "exclude_whole_clause",
    "human_uncertain": "exclude_whole_clause",
    "human_correct": "allow_measurement_mapping_not_unbounded_interpretations",
    "unreviewed": "allow_only_astra_pass_as_provisional",
    "unsupported": "context_only_no_diagnosis",
    "expected_relation": "omit_all_unvalidated_prose_and_numeric_targets",
    "permission_clauses": {"3:C12": "permission_not_mandatory; fingers_not_measured"},
    "maximum_findings": 3,
    "stability": asdict(FeedbackConfig()),
    "stability_is": "uncalibrated_caution_not_reconstruction_error_or_quality_label",
}


def extract_response(text: str, syntax_policy: dict | None = None) -> tuple[dict, list[dict]]:
    """Remove only unambiguous surrounding prose; never repair diagnosis content."""
    events = []
    if syntax_policy:
        if hashlib.sha256(text.encode('utf-8')).hexdigest() != syntax_policy['source_sha256']:
            raise ValueError('Syntax repair policy hash mismatch')
        for fragment in syntax_policy['escape_fragments']:
            literal = fragment if isinstance(fragment, str) else fragment['text']
            count = 1 if isinstance(fragment, str) else fragment['occurrences']
            before, after = literal, literal.replace('"', '\\"')
            if (not literal.startswith('"') or not literal.endswith('"')
                    or literal.count('"') != 2 or type(count) is not int or count < 1
                    or text.count(before) != count):
                raise ValueError('Reviewed malformed literal occurrence count mismatch')
            text = text.replace(before, after, count)
            events.append({'action': 'escape_reviewed_literal', 'before': before, 'after': after,
                           'occurrences': count, 'decoded_text_changed': False})
    try:
        return strict_loads(text), events
    except ValueError:
        pass
    decoder, candidates = json.JSONDecoder(), []
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            candidate, end = decoder.raw_decode(text, start)
        except ValueError:
            continue
        if isinstance(candidate, dict) and {"case_id", "findings", "coach_summary"} <= candidate.keys():
            candidates.append((start, end, strict_loads(text[start:end])))
    if len(candidates) != 1:
        raise ValueError(f"Expected one unambiguous feedback JSON; found {len(candidates)}")
    start, end, answer = candidates[0]
    return answer, events + [{"action": "extract_json", "start": start, "end": end,
                     "raw_characters": len(text), "json_content_changed": False}]


def select_checks(plan: dict, source: dict, reviews: dict, plan_id: str) -> dict:
    clauses = {c["clause_id"]: c["text"] for c in source["clauses"]}
    included, excluded = [], []
    for check in plan["checks"]:
        cid = check["clause_id"]
        row = reviews[f"{plan_id}:{cid}"]
        if row["source_text"] != clauses[cid] or int(row["move_id"]) != source["move_id"]:
            raise ValueError("Review and plan clause mismatch")
        human, judge = row["manual_verdict"], row["model_verdict"]
        if human not in {"correct", "false_positive", "uncertain", "pending"}:
            raise ValueError(f"Unknown manual verdict: {human}")
        reason = None
        if human in {"false_positive", "uncertain"}:
            reason = "human_" + human
        elif check["support"] == "unsupported" or not check["evidence"]:
            reason = "unsupported_at_endpoint"
        elif f"{source['move_id']}:{cid}" in POLICY["permission_clauses"]:
            reason = POLICY["permission_clauses"][f"{source['move_id']}:{cid}"]
        elif human != "correct" and judge != "pass":
            reason = "no_review_acceptance"
        if reason:
            excluded.append({"clause_id": cid, "text": clauses[cid], "reason": reason})
            continue
        included.append({
            "clause_id": cid, "text": clauses[cid], "support": check["support"],
            "acceptance": "human_reviewed_mapping" if human == "correct" else "astra_pass_provisional",
            "limitations": check["limitations"],
            "evidence_ids": list(dict.fromkeys(
                f"{e['query_id']}:{c}" for e in check["evidence"] for c in e["components"])),
        })
    return {"plan_id": plan_id, "move_id": source["move_id"],
            "technique": source["technique"], "move_name": source["move_name"],
            "checks": included, "excluded_clauses": excluded}


def direction(value, lo, hi):
    if any(not finite(x) for x in (value, lo, hi)) or lo >= hi:
        return "unavailable"
    return "below" if value < lo else "above" if value > hi else "within"


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def compile_packet(case_id: str, frozen: dict, measurement: dict) -> dict:
    if measurement["move_id"] != frozen["move_id"] or measurement["technique"] != frozen["technique"]:
        raise ValueError("Measurement/plan technique mismatch")
    requested = {}
    for check in frozen["checks"]:
        for eid in check["evidence_ids"]:
            requested.setdefault(eid, []).append(check["clause_id"])
    evidence = []
    for measure in measurement["measurements"]:
        qid, student = measure["query_id"], measure["student"]
        definition = student["definition"]
        for comp, values in student["components"].items():
            eid = f"{qid}:{comp}"
            if eid not in requested:
                continue
            reference = measure["teachers"]["components"][comp]
            unit = definition["unit"]
            student_row = {**values, "status": student["status"], "unit": unit,
                           "valid_frame_count": student["valid_frame_count"],
                           "requested_frame_count": student["requested_frame_count"]}
            warnings = stability_reasons(student_row, FeedbackConfig())
            teacher_values = []
            for original in reference["teacher_values"]:
                row = {k: v for k, v in original.items() if k != "endpoint"}
                row["warnings"] = stability_reasons({**row, "unit": unit}, FeedbackConfig())
                teacher_values.append(row)
            intervals = {"median_2mad": [reference["median_minus_2mad"], reference["median_plus_2mad"]],
                         "p10_p90": [reference["p10"], reference["p90"]]}
            positions = {k: direction(values["value"], *v) for k, v in intervals.items()}
            centers = {k: direction(values["center_value"], *v) for k, v in intervals.items()}
            if len(set(positions.values())) != 1:
                warnings.append("interval_disagreement")
            if centers != positions:
                warnings.append("center_direction_disagreement")
            if any(t["warnings"] for t in teacher_values):
                warnings.append("unstable_teacher_reference")
            valid = (student["status"] == "valid" and reference["status"] == "valid"
                     and "invalid_window" not in warnings
                     and all(x != "unavailable" for x in positions.values()))
            if not valid:
                warnings.append("unusable_numeric_evidence")
            evidence.append({"evidence_id": eid, "query_id": qid, "component": comp,
                "clause_ids": requested[eid], "definition": definition,
                "student": student_row,
                "reference": {k: v for k, v in reference.items() if k != "teacher_values"},
                "teacher_values": teacher_values,
                "interval_directions": positions, "center_directions": centers,
                "warnings": warnings, "usable": valid,
                "supported_claim_allowed": valid and not warnings})
    if set(requested) != {e["evidence_id"] for e in evidence}:
        raise ValueError("A selected query/component is absent from frozen measurements")
    return {"schema_version": "1.0", "case_id": case_id,
        "student_id": measurement["student_id"], "move_id": measurement["move_id"],
        "move_name": frozen["move_name"], "scope": "endpoint_only_structured_evidence_no_video",
        "technique": frozen["technique"], "plan_id": frozen["plan_id"],
        "student_endpoint": measurement["student_endpoint"],
        "reference_anchor": measurement["reference_anchor"],
        "coordinate_contract": {
            "right": "signed positive toward performer's anatomical right; negative toward left",
            "up": "Neck-Pelvis torso direction, NOT gravity",
            "forward": "body forward, NOT camera forward",
            "left_outward": "negative right component; no separate left component",
            "proxies": "Head != forehead/eyes/top of head; Wrist != palm; no contact or load",
        },
        "checks": frozen["checks"], "excluded_clauses": frozen["excluded_clauses"],
        "evidence": evidence, "policy": POLICY}


def validate_response(answer: dict, packet: dict) -> list[str]:
    """Check structure and literal citations, not the truth of free-text diagnoses."""
    errors = []
    if not isinstance(answer, dict):
        return ["response must be an object"]
    if set(answer) != {"case_id", "findings", "not_selected", "coach_summary"}:
        errors.append("unexpected/missing response fields")
    if answer.get("case_id") != packet["case_id"]:
        errors.append("case_id mismatch")
    findings = answer.get("findings")
    if not isinstance(findings, list) or len(findings) > POLICY["maximum_findings"]:
        return errors + ["findings must contain 0-3 items"]
    checks = {c["clause_id"]: c for c in packet["checks"]}
    evidence = {e["evidence_id"]: e for e in packet["evidence"]}
    ids, used = set(), set()
    for i, finding in enumerate(findings):
        prefix = f"findings[{i}]"
        fields = {"finding_id", "priority", "clause_ids", "observation", "evidence",
                  "synthesis", "why_priority", "training_guidance", "caveat", "certainty"}
        if not isinstance(finding, dict) or set(finding) != fields:
            errors.append(prefix + ": invalid fields")
            continue
        fid = finding["finding_id"]
        if not isinstance(fid, str) or fid != f"P{i+1}" or fid in ids:
            errors.append(prefix + ": invalid finding_id")
        else:
            ids.add(fid)
        if type(finding["priority"]) is not int or finding["priority"] != i+1:
            errors.append(prefix + ": priority must follow order")
        for key in ("observation", "synthesis", "why_priority", "training_guidance", "caveat"):
            if not isinstance(finding[key], str) or not finding[key].strip():
                errors.append(prefix + f": empty {key}")
        certainty = finding["certainty"]
        if certainty not in ("supported", "tentative"):
            errors.append(prefix + ": invalid certainty")
        clauses = finding["clause_ids"]
        if (not isinstance(clauses, list) or not clauses or
                any(not isinstance(c, str) or c not in checks for c in clauses)):
            errors.append(prefix + ": unknown/excluded clause")
            continue
        citations = finding["evidence"]
        if not isinstance(citations, list) or not citations:
            errors.append(prefix + ": no evidence")
            continue
        covered, local = set(), set()
        for citation in citations:
            expected = {"evidence_id", "student_value", "teacher_median", "p10", "p90", "direction"}
            if not isinstance(citation, dict) or set(citation) != expected:
                errors.append(prefix + ": invalid citation fields")
                continue
            eid = citation["evidence_id"]
            if not isinstance(eid, str) or eid not in evidence:
                errors.append(prefix + ": unknown evidence")
                continue
            e = evidence[eid]
            if eid in local:
                errors.append(prefix + ": duplicate evidence")
            local.add(eid)
            related = set(clauses).intersection(e["clause_ids"])
            if not related or not e["usable"]:
                errors.append(prefix + ": unusable/unrelated evidence")
            covered.update(related)
            if certainty == "supported" and not e["supported_claim_allowed"]:
                errors.append(prefix + ": supported claim with reliability warnings")
            numbers = {"student_value": e["student"]["value"],
                       "teacher_median": e["reference"]["median"],
                       "p10": e["reference"]["p10"], "p90": e["reference"]["p90"]}
            for key, actual in numbers.items():
                if not finite(citation[key]) or not finite(actual) or abs(citation[key]-actual) > 0.001:
                    errors.append(prefix + f": incorrect numeric citation {eid}/{key}")
            if citation["direction"] != e["interval_directions"]["p10_p90"]:
                errors.append(prefix + ": incorrect P10-P90 direction")
        if set(clauses) != covered:
            errors.append(prefix + ": a claimed clause has no linked evidence")
        signature = frozenset(local)
        if signature in used:
            errors.append(prefix + ": duplicate finding evidence set")
        used.add(signature)
    omitted = answer.get("not_selected")
    if not isinstance(omitted, list):
        errors.append("not_selected must be a list")
    else:
        all_clauses = set(checks) | {c["clause_id"] for c in packet["excluded_clauses"]}
        for row in omitted:
            if (not isinstance(row, dict) or set(row) != {"clause_ids", "reason"}
                    or not isinstance(row["clause_ids"], list) or not row["clause_ids"]
                    or any(not isinstance(c, str) or c not in all_clauses for c in row["clause_ids"])
                    or not isinstance(row["reason"], str) or not row["reason"].strip()):
                errors.append("invalid not_selected entry")
    if not isinstance(answer.get("coach_summary"), str) or not answer["coach_summary"].strip():
        errors.append("coach_summary must be nonempty text")
    return errors
