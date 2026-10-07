"""Constrained requirement-to-measurement planning, without subject observations."""

from __future__ import annotations

from dataclasses import asdict
import itertools
import json
import re
from typing import Any

from aqa3d.pose_measurements import (
    JOINT_NAMES, OPERATIONS, VERSION, MeasurementQuery, measurement_contract,
)


PLAN_VERSION = "1.0"
MAX_QUERIES = 24
SUPPORT_LEVELS = {"direct", "proxy", "unsupported"}


def strict_loads(text: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Non-finite JSON literal: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def planner_contract() -> dict[str, Any]:
    contract = measurement_contract()
    for operation, count in OPERATIONS.items():
        angle = operation.endswith("angle")
        definition = MeasurementQuery(
            "contract_only", operation, JOINT_NAMES[:count],
            None if angle else "torso_length",
            "torso_v1" if operation == "relative_position" else None,
        ).definition()
        contract["operations"][operation].update({
            k: definition[k] for k in ("formula", "unit", "components", "limitations")
        })
        contract["operations"][operation]["constraints"] = (
            "normalizer=null, coordinate_frame=null" if angle else
            "registered normalizer required; coordinate_frame="
            + ("torso_v1" if operation == "relative_position" else "null")
        )
    contract["input_representation"] = "Native SMPL-24 joint positions, not images or contact forces"
    contract["later_execution"] = (
        "Same query for every subject. Frozen endpoint window, clipped to current move. "
        "No trajectory, transition direction, ground plane or global orientation is supplied."
    )
    return contract


def build_input(manifest: dict[str, Any], move_id: int) -> dict[str, Any]:
    matches = [e for e in manifest["endpoint_keyposes"] if e["move_id"] == move_id]
    if len(matches) != 1:
        raise ValueError("Move must have exactly one frozen endpoint")
    event = matches[0]
    technique = event["final_technique_step"]
    clauses = [s.strip() for s in re.split("[\u3002\uff1b\uff0c;]", technique) if s.strip()]
    return {
        "schema_version": PLAN_VERSION,
        "move_id": move_id,
        "move_name": event["move_name_zh"],
        "technique": technique,
        "clauses": [{"clause_id": f"C{i:02d}", "text": text}
                    for i, text in enumerate(clauses, 1)],
        "scope": "endpoint_geometry_only",
        "measurement_tools": planner_contract(),
    }


def query_signature(query: MeasurementQuery) -> str:
    points = list(query.points)
    if query.operation == "distance":
        points.sort()
    elif query.operation == "joint_angle":
        ends = sorted((points[0], points[2]))
        points = [ends[0], points[1], ends[1]]
    elif query.operation == "vector_angle":
        a, b = tuple(points[:2]), tuple(points[2:])
        points = list(min(a+b, b+a, a[::-1]+b[::-1], b[::-1]+a[::-1]))
    return json.dumps([query.operation, points, query.normalizer, query.coordinate_frame])


def _fields(value: Any, names: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != names:
        raise ValueError(f"{label}: expected fields {sorted(names)}")


def _text(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 3000:
        raise ValueError(f"{label}: expected nonempty text, max 3000 characters")


def parse_plan(text: str, request: dict[str, Any]) -> dict[str, Any]:
    plan = strict_loads(text)
    _fields(plan, {"schema_version", "move_id", "scope", "queries", "checks"}, "plan")
    if (plan["schema_version"] != PLAN_VERSION or type(plan["move_id"]) is not int
            or plan["move_id"] != request["move_id"] or plan["scope"] != request["scope"]):
        raise ValueError("Plan version, move or scope mismatch")
    if not isinstance(plan["queries"], list) or len(plan["queries"]) > MAX_QUERIES:
        raise ValueError(f"At most {MAX_QUERIES} queries are allowed")
    queries = [MeasurementQuery.from_dict(q) for q in plan["queries"]]
    by_id = {q.query_id: q for q in queries}
    if len(by_id) != len(queries) or len({query_signature(q) for q in queries}) != len(queries):
        raise ValueError("Duplicate query ID or equivalent query")
    if not isinstance(plan["checks"], list):
        raise ValueError("checks must be a list")
    seen, used = set(), set()
    for check in plan["checks"]:
        _fields(check, {"clause_id", "support", "endpoint_interpretation", "evidence", "limitations"}, "check")
        cid = check["clause_id"]
        if not isinstance(cid, str) or cid in seen:
            raise ValueError("Invalid or duplicate clause ID")
        seen.add(cid)
        if not isinstance(check["support"], str) or check["support"] not in SUPPORT_LEVELS:
            raise ValueError("Unknown support level")
        _text(check["endpoint_interpretation"], "endpoint_interpretation")
        limitations = check["limitations"]
        if not isinstance(limitations, list) or not limitations:
            raise ValueError("Every check must state its limitations")
        for limitation in limitations:
            _text(limitation, "limitation")
        evidence = check["evidence"]
        if not isinstance(evidence, list):
            raise ValueError("evidence must be a list")
        if (check["support"] == "unsupported") != (len(evidence) == 0):
            raise ValueError("Unsupported checks need no evidence; other checks need evidence")
        referenced = set()
        for item in evidence:
            _fields(item, {"query_id", "components", "purpose", "expected_relation"}, "evidence")
            qid = item["query_id"]
            if not isinstance(qid, str) or qid not in by_id or qid in referenced:
                raise ValueError("Unknown or duplicate evidence query")
            referenced.add(qid)
            used.add(qid)
            components = item["components"]
            allowed = by_id[qid].definition()["components"]
            if (not isinstance(components, list) or not components
                    or any(not isinstance(c, str) or c not in allowed for c in components)
                    or len(components) != len(set(components))):
                raise ValueError("Evidence must name legal query components")
            _text(item["purpose"], "purpose")
            _text(item["expected_relation"], "expected_relation")
    if seen != {c["clause_id"] for c in request["clauses"]}:
        raise ValueError("Every source clause must occur exactly once")
    if used != set(by_id):
        raise ValueError("Unused queries are not allowed")
    return plan


def validate_response(text: str, request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        plan = parse_plan(text, request)
    except (ValueError, TypeError, KeyError, RecursionError) as exc:
        return {"schema_valid": False, "errors": [str(exc)], "semantic_correctness": "not_reviewed"}, None
    return {
        "schema_valid": True, "errors": [], "query_count": len(plan["queries"]),
        "source_clause_count": len(request["clauses"]),
        "support_counts": {s: sum(c["support"] == s for c in plan["checks"]) for s in sorted(SUPPORT_LEVELS)},
        "semantic_correctness": "not_reviewed",
        "coverage_meaning": "Clause IDs accounted for, not proof of correct or complete geometric interpretation",
        "can_diagnose": False,
    }, plan


def queries_payload(plan: dict[str, Any]) -> dict[str, Any]:
    return {"version": VERSION, "queries": [asdict(MeasurementQuery.from_dict(q)) for q in plan["queries"]]}


def stability(plans: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    results = []
    for (aid, a), (bid, b) in itertools.combinations(plans, 2):
        def index(plan):
            return {q["query_id"]: query_signature(MeasurementQuery.from_dict(q)) for q in plan["queries"]}
        ai, bi = index(a), index(b)
        aset, bset = set(ai.values()), set(bi.values())
        ac = {c["clause_id"]: c for c in a["checks"]}
        bc = {c["clause_id"]: c for c in b["checks"]}
        def evidence(check, idx):
            return {(idx[e["query_id"]], tuple(sorted(e["components"]))) for e in check["evidence"]}
        results.append({
            "first": aid, "second": bid,
            "query_jaccard": len(aset & bset)/len(aset | bset) if aset | bset else 1.0,
            "support_agreement": sum(ac[c]["support"] == bc[c]["support"] for c in ac)/len(ac),
            "clause_evidence_exact_agreement": sum(evidence(ac[c], ai) == evidence(bc[c], bi) for c in ac)/len(ac),
            "meaning": "Reproducibility only; not semantic accuracy",
        })
    return results
