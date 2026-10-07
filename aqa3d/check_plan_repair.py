"""Auditable offline repairs; source answers and geometric definitions stay intact."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

from aqa3d.check_planning import strict_loads, validate_response
from aqa3d.pose_measurements import MeasurementQuery


def _event(kind: str, path: str, before: Any, after: Any, reason: str, **extra) -> dict:
    return {"kind": kind, "path": path, "before": before, "after": after,
            "reason": reason, **extra}


def extract_plan(text: str) -> tuple[dict, list[dict]]:
    """Only extract one complete JSON plan; never stitch fragments or pick between plans."""
    try:
        return strict_loads(text), []
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    candidates = []
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            candidate, end = decoder.raw_decode(text, start)
        except ValueError:
            continue
        if isinstance(candidate, dict) and {"schema_version", "move_id", "queries", "checks"} <= candidate.keys():
            # The second parser rejects duplicate keys and non-finite numbers.
            candidates.append((start, end, strict_loads(text[start:end])))
    if len(candidates) != 1:
        raise ValueError(f"Expected one unambiguous complete plan JSON; found {len(candidates)}")
    start, end, plan = candidates[0]
    return plan, [_event("extract_json", "response", {"characters": len(text)},
                        {"start": start, "end": end, "characters": end-start},
                        "Removed surrounding prose/fences; JSON content unchanged.")]


def _parent(document: dict, path: list[str | int]):
    if not isinstance(path, list) or not path:
        raise ValueError("Correction path must be nonempty")
    current = document
    for key in path[:-1]:
        current = current[key]
    return current, path[-1]


def _same_query(a: dict, b: dict) -> bool:
    def definition(q):
        value = MeasurementQuery.from_dict(q)
        return value.operation, value.points, value.normalizer, value.coordinate_frame
    return definition(a) == definition(b)


def normalize_queries(plan: dict) -> list[dict]:
    """Merge exact definitions, relink evidence, then remove genuinely unreferenced queries."""
    logs, unique, aliases, ids = [], [], {}, set()
    for query in plan["queries"]:
        MeasurementQuery.from_dict(query)
        qid = query["query_id"]
        if qid in ids:
            raise ValueError("Repeated query IDs are ambiguous; no automatic repair")
        ids.add(qid)
        canonical = next((q for q in unique if _same_query(q, query)), None)
        if canonical is None:
            unique.append(query)
        else:
            aliases[qid] = canonical["query_id"]
            logs.append(_event("merge_identical_query", f"queries/{qid}", query, canonical,
                               "Identical ordered points, operation, scale and coordinate frame."))
    plan["queries"] = unique
    used = set()
    for ci, check in enumerate(plan["checks"]):
        merged = {}
        evidence = []
        for ei, item in enumerate(check["evidence"]):
            previous = deepcopy(item)
            item["query_id"] = aliases.get(item["query_id"], item["query_id"])
            if previous != item:
                logs.append(_event("relink_duplicate", f"checks/{ci}/evidence/{ei}", previous,
                                   deepcopy(item), "Use retained identical measurement ID."))
            qid = item["query_id"]
            if qid not in ids:
                raise ValueError(f"Unknown query reference remains: {qid}; no fuzzy matching")
            if qid in merged:
                target = merged[qid]
                before = deepcopy(target)
                target["components"] = list(dict.fromkeys(target["components"]+item["components"]))
                for key in ("purpose", "expected_relation"):
                    if target[key] != item[key]:
                        target[key] += "\n" + item[key]
                logs.append(_event("merge_evidence", f"checks/{ci}/evidence/{qid}",
                                   [before, deepcopy(item)], deepcopy(target),
                                   "Union components; retain both explanations without choosing between them."))
            else:
                merged[qid] = item
                evidence.append(item)
            used.add(qid)
        check["evidence"] = evidence
    kept = []
    for query in plan["queries"]:
        if query["query_id"] in used:
            kept.append(query)
        else:
            logs.append(_event("remove_unused_query", f"queries/{query['query_id']}", query, None,
                               "No clause references this measurement; no clause/evidence was removed."))
    plan["queries"] = kept
    return logs


def repair_response(text: str, request: dict, policy: dict | None = None) -> dict:
    """Apply only hash-bound reviewed edits plus meaning-preserving normalization."""
    original_validation, _ = validate_response(text, request)
    source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    logs = []
    policy = policy or {}
    try:
        if policy and policy.get("source_sha256") != source_hash:
            raise ValueError("Reviewed repair policy source hash mismatch")
        for value in policy.get("escape_literals", []):
            before, after = '"'+value+'"', json.dumps(value, ensure_ascii=False)
            if before == after or text.count(before) != 1:
                raise ValueError("Expected one exact malformed string literal")
            start = text.index(before)
            text = text[:start] + after + text[start+len(before):]
            logs.append(_event("escape_reviewed_literal", f"response/offset/{start}", before, after,
                               "Escape embedded quotes without changing decoded prose.",
                               approval="assistant_reviewed_exact_literal"))
        plan, extracted = extract_plan(text)
        logs.extend(extracted)
        for edit in policy.get("edits", []):
            path = edit["path"]
            if (len(path) != 5 or path[0] != "checks" or path[2] != "evidence"
                    or any(type(path[i]) is not int or path[i] < 0 for i in (1, 3))
                    or path[4] not in {"query_id", "components", "purpose", "expected_relation"}):
                raise ValueError("Reviewed edits may only correct evidence references/components/text")
            parent, key = _parent(plan, edit["path"])
            if parent[key] != edit["before"]:
                raise ValueError(f"Reviewed field does not match: {edit['path']}")
            if "target_query" in edit:
                targets = [q for q in plan["queries"] if q["query_id"] == edit["after"]]
                if len(targets) != 1 or targets[0] != edit["target_query"]:
                    raise ValueError("Reference repair target definition mismatch")
            parent[key] = deepcopy(edit["after"])
            logs.append(_event(edit["kind"], "/".join(map(str, edit["path"])),
                               edit["before"], edit["after"], edit["reason"],
                               approval="assistant_reviewed_contract_correction"))
        logs.extend(normalize_queries(plan))
        repaired_validation, valid = validate_response(json.dumps(plan, ensure_ascii=False), request)
        return {"original_validation": original_validation, "repaired_validation": repaired_validation,
                "changes": logs, "plan": valid, "candidate": plan,
                "semantic_warnings": policy.get("semantic_warnings", [])}
    except (ValueError, TypeError, KeyError, IndexError, RecursionError) as exc:
        return {"original_validation": original_validation,
                "repaired_validation": {"schema_valid": False, "errors": [str(exc)],
                                        "semantic_correctness": "not_reviewed"},
                "changes": logs, "plan": None, "candidate": None,
                "semantic_warnings": policy.get("semantic_warnings", [])}
