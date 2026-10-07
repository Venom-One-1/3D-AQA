"""Strict semantic-judge validation and independent geometric contract probes."""

from dataclasses import replace
import json
import numpy as np

from .check_planning import strict_loads
from .pose_measurements import JOINTS, JOINT_NAMES, MeasurementQuery, measure_pose
from run_check_plan_probes import probe_query

DIMENSIONS = ("relevance", "direction", "coverage", "scope")
VERDICTS = ("pass", "fail", "uncertain")
ISSUE_CODES = ("wrong_body_part", "coordinate_sign", "unsupported_inference", "support_level",
               "missing_measurement", "extra_constraint", "domain_ambiguity", "other")


def _fields(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"Expected fields: {keys}")


def _text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 4000:
        raise ValueError("Expected nonempty text of at most 4000 characters")


def validate_judgment(text: str, packet: dict) -> dict:
    result = strict_loads(text)
    _fields(result, ("schema_version", "case_id", "checks"))
    if result["schema_version"] != "1.0" or result["case_id"] != packet["case_id"]:
        raise ValueError("Judge version or case mismatch")
    if not isinstance(result["checks"], list):
        raise ValueError("Expected checks array")
    clauses = {c["clause_id"]: c["text"] for c in packet["request"]["clauses"]}
    planned = {c["clause_id"]: c for c in packet["candidate_plan"]["checks"]}
    seen = set()
    for check in result["checks"]:
        _fields(check, ("clause_id", "dimensions", "issues", "summary"))
        cid = check["clause_id"]
        if not isinstance(cid, str) or cid not in clauses or cid in seen:
            raise ValueError("Unknown or duplicate judge clause")
        seen.add(cid)
        _text(check["summary"])
        _fields(check["dimensions"], DIMENSIONS)
        allowed = {e["query_id"] for e in planned[cid]["evidence"]}

        def query_ids(value):
            if (not isinstance(value, list) or any(not isinstance(q, str) or q not in allowed for q in value)
                    or len(value) != len(set(value))):
                raise ValueError("Judge query IDs must belong to this clause")

        for d in check["dimensions"].values():
            _fields(d, ("verdict", "reason", "query_ids"))
            if d["verdict"] not in VERDICTS:
                raise ValueError("Invalid dimension verdict")
            _text(d["reason"])
            query_ids(d["query_ids"])
        if not isinstance(check["issues"], list):
            raise ValueError("Expected issues array")
        texts = [planned[cid]["endpoint_interpretation"], *planned[cid]["limitations"]]
        texts += [e[k] for e in planned[cid]["evidence"] for k in ("purpose", "expected_relation")]
        covered = set()
        for issue in check["issues"]:
            _fields(issue, ("dimension", "code", "query_ids", "source_quote", "plan_quote", "reason"))
            if issue["dimension"] not in DIMENSIONS or issue["code"] not in ISSUE_CODES:
                raise ValueError("Unknown issue dimension/code")
            if check["dimensions"][issue["dimension"]]["verdict"] == "pass":
                raise ValueError("An issue cannot contradict a pass dimension")
            query_ids(issue["query_ids"])
            for field in ("source_quote", "plan_quote", "reason"):
                _text(issue[field])
            if issue["source_quote"] not in clauses[cid] or not any(issue["plan_quote"] in t for t in texts):
                raise ValueError("Judge quoted text absent from the source clause or planned evidence")
            covered.add(issue["dimension"])
        required = {d for d in DIMENSIONS if check["dimensions"][d]["verdict"] != "pass"}
        if covered != required:
            raise ValueError("Every fail/uncertain dimension needs a grounded issue")
    if seen != set(clauses):
        raise ValueError("Judge must account for every source clause exactly once")
    return result


def model_verdict(check: dict) -> str:
    values = {v["verdict"] for v in check["dimensions"].values()}
    return "fail" if "fail" in values else "uncertain" if "uncertain" in values else "pass"


def fixture() -> np.ndarray:
    """Algebraic fixture, NOT a model of a correct or anatomically valid Tai Chi pose."""
    p = np.zeros((24, 3))
    for name, xyz in (("Spine1", [0,.2,0]), ("Spine2", [0,.4,0]), ("Spine3", [0,.65,0]),
                      ("Neck", [0,1,0]), ("Head", [0,1.25,0])):
        p[JOINTS[name]] = xyz
    for side, sign in (("L", 1), ("R", -1)):
        for name, xyz in (("Hip", [sign*.2,0,0]), ("Knee", [sign*.2,-.5,.1]),
                          ("Ankle", [sign*.2,-1,0]), ("Foot", [sign*.2,-1,.2]),
                          ("Collar", [sign*.1,.75,0]), ("Shoulder", [sign*.3,.8,0]),
                          ("Elbow", [sign*.5,.5,.1]), ("Wrist", [sign*.6,.3,.2]),
                          ("Hand", [sign*.65,.25,.25])):
            p[JOINTS[side+"_"+name]] = xyz
    return p


def swapped(name):
    return "R_"+name[2:] if name.startswith("L_") else "L_"+name[2:] if name.startswith("R_") else name


def probe_plan(plan: dict) -> list[dict]:
    p = fixture()
    mirrored = p[[JOINTS[swapped(n)] for n in JOINT_NAMES]] * [-1, 1, 1]
    rows = []
    for data in plan["queries"]:
        q = MeasurementQuery.from_dict(data)
        rows.extend(probe_query(p, q))
        a = measure_pose(p, q)
        b = measure_pose(mirrored, replace(q, points=tuple(swapped(n) for n in q.points)))
        expected = a.values * ([-1,1,1] if q.operation == "relative_position" else [1])
        valid = bool(np.all(a.statuses == "valid") and np.all(b.statuses == "valid"))
        error = float(np.max(np.abs(b.values-expected))) if valid else None
        rows.append({"query_id": q.query_id, "probe": "mirror_and_swap_anatomical_sides",
                     "max_abs_error": error, "status": "pass" if valid and error < 1e-7 else "fail" if valid else "not_testable"})
    return rows


def signed_axis_probes() -> list[dict]:
    """Known directions in a fixture with anatomical left=+x, up=+y, front=+z."""
    rows = []
    for side in ("L", "R"):
        p = fixture()
        q = MeasurementQuery("elbow_"+side, "relative_position", (side+"_Elbow", side+"_Shoulder"), "torso_length", "torso_v1")
        base = measure_pose(p, q).values
        for label, delta, expected in (("left", [.1,0,0], [-.1,0,0]),
                                       ("right", [-.1,0,0], [.1,0,0]),
                                       ("up", [0,.1,0], [0,.1,0])):
            changed = p.copy()
            changed[JOINTS[side+"_Elbow"]] += delta
            error = float(np.max(np.abs(measure_pose(changed, q).values-base-expected)))
            rows.append({"query_id": q.query_id, "probe": "signed_displacement_"+label,
                         "max_abs_error": error, "status": "pass" if error < 1e-7 else "fail"})
        changed = p.copy()
        changed[JOINTS[side+"_Elbow"]] = p[JOINTS[side+"_Shoulder"]] + [0,.1,0]
        value = measure_pose(changed, q).values[0,1]
        rows.append({"query_id": q.query_id, "probe": "elbow_crosses_shoulder_up_sign",
                     "status": "pass" if base[0,1] < 0 < value else "fail", "max_abs_error": None})
    return rows


def route_check(schema_valid: bool, probes: list[dict], judgment: dict | None, judge_error=None):
    if not schema_valid or any(p["status"] == "fail" for p in probes):
        return "auto_return", "程序发现接口或几何行为错误"
    if judge_error or judgment is None:
        return "pending_judge", "缺少合法完整的模型预审，不能自动通过"
    if any(p["status"] == "not_testable" for p in probes):
        return "needs_review", "相关数值测试无法执行"
    verdict = model_verdict(judgment)
    if verdict != "pass":
        return "needs_review", "模型提出语义问题或不确定项，需要裁决"
    return "auto_pass_provisional", "数值检查与模型预审通过；尚未经独立校准，不是专家确认"
