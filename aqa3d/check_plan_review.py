"""Read-only measurement plans and independently persisted semantic reviews."""

import copy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
from threading import RLock

from .check_planning import parse_plan, strict_loads
from .feedback_review import VERDICTS, _atomic_json, _clean_text, _sha256
from .pose_measurements import MeasurementQuery


CSV_FIELDS = [
    "case_id", "move_id", "clause_id", "source_text", "support", "queries",
    "explanation", "limitations", "manual_verdict", "notes", "record_id",
    "repeat", "endpoint_interpretation", "evidence_json", "query_definitions_json",
    "reviewer", "updated_at", "save_status",
]
JUDGE_FIELDS = ["model_verdict", "triage", "audit_sample", "human_queue", "human_model_disagreement",
                "judge_summary", "judge_issues_json"]


class PlanReviewStore:
    def __init__(self, experiment_root: Path, metric_root: Path, state_path: Path, judge_report: Path | None = None):
        self.root = Path(experiment_root).resolve()
        self.metric_root = Path(metric_root).resolve()
        self.state_path = Path(state_path).resolve()
        # Never overwrite frozen plans, inputs, or metric evidence with annotations.
        if any(self.state_path.is_relative_to(root) for root in (self.root, self.metric_root)):
            raise ValueError("Review state must be outside the experiment and metric directories")
        self.lock = RLock()
        self.revision = 0
        self.source_hashes = {}
        self.cases, self.records, self.image_paths = [], {}, {}

        def load(path):
            self.source_hashes[str(path)] = _sha256(path)
            return strict_loads(path.read_text(encoding="utf-8-sig"))

        manifest = load(self.root / "source_manifest_snapshot.json")
        reference = load(self.metric_root / "reference_manifest_snapshot.json")
        if reference["reference"]["video_id"] != "BV1WE411W7JB":
            raise ValueError("Expected BV1WE411W7JB reference anchor")
        endpoints = {r["move_id"]: r for r in reference["endpoint_keyposes"]}
        self.model = manifest["api_config"]["model"]
        seen = set()
        for entry in manifest["cases"]:
            cid, mid, repeat = entry["case_id"], entry["move_id"], entry["repeat"]
            if not re.fullmatch(r"M\d{2}_R\d{2}", cid) or cid in seen:
                raise ValueError("Invalid or duplicate case ID")
            seen.add(cid)
            request = load(self.root / cid / "input.json")
            raw_plan = load(self.root / cid / "plan.json")
            plan = parse_plan(json.dumps(raw_plan, ensure_ascii=False), request)
            repair = load(self.root / cid / "repair_log.json")
            endpoint = endpoints[mid]
            if request["move_id"] != mid or request["technique"] != endpoint["final_technique_step"]:
                raise ValueError(f"Technique/identity mismatch: {cid}")
            image_path = self.metric_root / "figures" / f"{mid:02d}_reference.jpg"
            self.source_hashes[str(image_path)] = _sha256(image_path)
            self.image_paths[f"reference-{mid}"] = image_path
            case = {"case_id": cid, "move_id": mid, "repeat": repeat,
                    "move_name": request["move_name"], "technique": request["technique"],
                    "reference_time": endpoint["boundary_time_seconds"],
                    "image": f"/api/image/reference-{mid}", "record_ids": [],
                    "repair_changes": repair.get("changes", []),
                    "semantic_warnings": repair.get("semantic_warnings", [])}
            queries = {q["query_id"]: MeasurementQuery.from_dict(q).definition() for q in plan["queries"]}
            checks = {check["clause_id"]: check for check in plan["checks"]}
            for clause in request["clauses"]:
                check = checks[clause["clause_id"]]
                rid = f"{cid}:{clause['clause_id']}"
                definitions = {e["query_id"]: queries[e["query_id"]] for e in check["evidence"]}
                self.records[rid] = {"record_id": rid, "case_id": cid, "move_id": mid,
                                     "repeat": repeat, "source_text": clause["text"],
                                     **check, "query_definitions": definitions}
                case["record_ids"].append(rid)
            self.cases.append(case)
        if not self.records:
            raise ValueError("No checks to review")
        self.dataset_id = hashlib.sha256(json.dumps(self.source_hashes, sort_keys=True).encode()).hexdigest()
        self.annotations = {rid: {"manual_verdict": "pending", "notes": "", "reviewer": "", "updated_at": ""}
                            for rid in self.records}
        if self.state_path.exists():
            saved = strict_loads(self.state_path.read_text(encoding="utf-8"))
            if saved["source_hashes"] != self.source_hashes or set(saved["annotations"]) != set(self.records):
                raise ValueError("Review sources changed; preserve old state and use a new --state-path")
            if type(saved["revision"]) is not int or saved["revision"] < 0:
                raise ValueError("Invalid saved revision")
            for rid, annotation in saved["annotations"].items():
                self.validate_annotation(rid, annotation)
            self.annotations, self.revision = saved["annotations"], saved["revision"]
        self.judge, self.judge_hashes = None, {}
        if judge_report is not None:
            path = Path(judge_report).resolve()
            overlay = strict_loads(path.read_text(encoding="utf-8"))
            if overlay.get("schema_version") != 1 or set(overlay["records"]) != set(self.records):
                raise ValueError("Judge report does not match review records")
            self.judge_hashes = dict(overlay["source_hashes"], **{str(path): _sha256(path)})
            for file, digest in self.judge_hashes.items():
                if _sha256(Path(file)) != digest:
                    raise ValueError(f"Judge source changed: {file}")
            for cid in seen:
                for filename in ("input.json", "plan.json"):
                    source = self.root / cid / filename
                    matches = [digest for file, digest in overlay["source_hashes"].items() if Path(file).resolve() == source]
                    if matches != [self.source_hashes[str(source)]]:
                        raise ValueError("Judge source does not match the displayed plan")
            for rid, row in overlay["records"].items():
                if row["record_id"] != rid or row["source_text"] != self.records[rid]["source_text"]:
                    raise ValueError("Judge clause mismatch")
                self.records[rid]["judge"] = row
            self.judge = overlay["summary"]
        self.export_fields = CSV_FIELDS + (JUDGE_FIELDS if self.judge else [])

    def validate_annotation(self, rid, annotation):
        if not isinstance(rid, str) or rid not in self.records or not isinstance(annotation, dict):
            raise ValueError("Unknown review record")
        if annotation.get("manual_verdict") not in VERDICTS:
            raise ValueError("Invalid manual_verdict")
        _clean_text(annotation.get("notes", ""), "notes", 4000)
        _clean_text(annotation.get("reviewer", ""), "reviewer", 100)

    def update_many(self, updates, revision):
        if not isinstance(updates, list) or not 1 <= len(updates) <= 200:
            raise ValueError("Expected 1-200 review updates")
        with self.lock:
            if type(revision) is not int or revision != self.revision:
                raise ValueError("Review changed in another browser; reload before saving")
            if any(_sha256(Path(p)) != sha for p, sha in self.source_hashes.items()):
                raise ValueError("Source files changed; restart with a new review state")
            if any(_sha256(Path(p)) != sha for p, sha in self.judge_hashes.items()):
                raise ValueError("Judge files changed; restart before saving")
            new, seen = copy.deepcopy(self.annotations), set()
            for item in updates:
                if not isinstance(item, dict):
                    raise ValueError("Expected review object")
                rid = item.get("record_id")
                self.validate_annotation(rid, item)
                if rid in seen:
                    raise ValueError("Duplicate review update")
                seen.add(rid)
                new[rid] = {"manual_verdict": item["manual_verdict"],
                            "notes": _clean_text(item.get("notes", ""), "notes", 4000),
                            "reviewer": _clean_text(item.get("reviewer", ""), "reviewer", 100),
                            "updated_at": datetime.now(timezone.utc).isoformat()}
            _atomic_json(self.state_path, {"schema_version": 1, "source_hashes": self.source_hashes,
                                          "revision": self.revision + 1, "annotations": new})
            self.annotations, self.revision = new, self.revision + 1
            return {"annotations": copy.deepcopy(new), "revision": self.revision}

    def export_row(self, record):
        definitions = record["query_definitions"]
        row = {k: record[k] for k in ("record_id", "case_id", "move_id", "repeat", "clause_id",
                                     "source_text", "support", "endpoint_interpretation")} | {
            "queries": "; ".join(f"{e['query_id']}: {definitions[e['query_id']]['operation']}"
                                   f"({', '.join(definitions[e['query_id']]['points'])}); {','.join(e['components'])}"
                                   for e in record["evidence"]),
            "explanation": "; ".join(e["purpose"] + "; " + e["expected_relation"] for e in record["evidence"])
                           or record["endpoint_interpretation"],
            "limitations": "; ".join(record["limitations"]),
            "evidence_json": json.dumps(record["evidence"], ensure_ascii=False),
            "query_definitions_json": json.dumps(definitions, ensure_ascii=False),
        }
        if self.judge:
            judge = record["judge"]
            row.update({key: judge[key] for key in JUDGE_FIELDS[:5]})
            result = judge["judgment"] or {}
            row["judge_summary"] = result.get("summary", "")
            row["judge_issues_json"] = json.dumps(result.get("issues", []), ensure_ascii=False)
        return row

    def client_payload(self):
        with self.lock:
            return {"cases": self.cases, "records": self.records, "model": self.model,
                    "dataset_id": self.dataset_id, "state_path": str(self.state_path),
                    "annotations": copy.deepcopy(self.annotations), "revision": self.revision,
                    "judge": self.judge, "export_fields": self.export_fields,
                    "export_rows": {rid: self.export_row(r) for rid, r in self.records.items()}}

    def export_csv(self):
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=self.export_fields)
        writer.writeheader()
        with self.lock:
            for rid, record in self.records.items():
                row = self.export_row(record) | self.annotations[rid] | {"save_status": "saved"}
                for key, value in row.items():
                    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                        row[key] = "'" + value
                writer.writerow(row)
        return ("\ufeff" + output.getvalue()).encode("utf-8")
