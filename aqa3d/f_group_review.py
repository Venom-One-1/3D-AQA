"""Paired F-group and frozen rule-baseline review data, without model calls."""

import copy
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from threading import RLock

from .endpoint_feedback import coach_summary
from .feedback_review import VERDICTS, _atomic_json, _clean_text, _sha256


def read_json(path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=invalid)


class ComparisonStore:
    def __init__(self, experiment_root, metric_root, state_path=None):
        self.root = Path(experiment_root).resolve()
        self.metric_root = Path(metric_root).resolve()
        self.state_path = Path(state_path) if state_path else self.root / "ui_review/review_progress.json"
        self.lock = RLock()
        self.revision = 0
        self.source_hashes = {}
        self.records = {}
        self.cases = []
        self.image_paths = {}

        def load(path):
            self.source_hashes[str(path)] = _sha256(path)
            return read_json(path)

        manifest = load(self.root / "experiment_manifest.json")
        baseline = load(self.root / "evaluation/baseline_snapshot.json")
        reference = load(self.metric_root / "reference_manifest_snapshot.json")
        if reference["reference"]["video_id"] != "BV1WE411W7JB":
            raise ValueError("Expected BV1WE411W7JB reference anchor")
        endpoints = {r["move_id"]: r for r in reference["endpoint_keyposes"]}
        seen = set()
        for entry in manifest["cases"]:
            cid, sid, mid = entry["case_id"], entry["student_id"], entry["move_id"]
            if cid in seen:
                raise ValueError(f"Duplicate case: {cid}")
            seen.add(cid)
            source = load(self.root / entry["input_file"])
            response = load(self.root / entry["response_file"])
            if response.get("case_id") != cid or not isinstance(response.get("findings"), list):
                raise ValueError(f"Invalid response case/findings: {cid}")
            if not isinstance(response.get("coach_summary"), str):
                raise ValueError(f"Invalid coach_summary: {cid}")
            rows = [r for r in baseline if r["subject_id"] == sid and r["move_id"] == mid]
            if not rows:
                raise ValueError(f"No frozen baseline: {cid}")
            if any(r["final_technique_step"] != source["technique"] for r in rows):
                raise ValueError(f"Technique mismatch: {cid}")
            baseline_summary = coach_summary(rows)
            case = {"case_id": cid, "student_id": sid, "move_id": mid,
                    "move_name": source["move"]["move_name_zh"], "technique": source["technique"],
                    "metrics": source["metrics"], "response": response, "baseline": rows,
                    "baseline_summary": baseline_summary,
                    "reference_time": endpoints[mid]["boundary_time_seconds"],
                    "student_time": rows[0]["student_evidence"]["boundary_time_seconds"],
                    "images": {}, "record_ids": []}
            for role, path in {
                "reference": self.metric_root / "figures" / f"{mid:02d}_reference.jpg",
                "student": self.metric_root / "student" / sid / f"{mid:02d}_endpoint.jpg",
            }.items():
                if not path.is_file():
                    raise FileNotFoundError(path)
                token = f"{cid}-{role}"
                self.image_paths[token] = path
                case["images"][role] = "/api/image/" + token
            self.add_record(case, "F", "coach_summary", "summary", response["coach_summary"])
            self.add_record(case, "rule_baseline", "coach_summary", "summary", baseline_summary)
            finding_ids = set()
            for finding in response["findings"]:
                fid = finding.get("finding_id")
                if not isinstance(fid, str) or not fid or fid in finding_ids:
                    raise ValueError(f"Invalid or duplicate finding_id: {cid}")
                finding_ids.add(fid)
                self.add_record(case, "F", "finding", fid, finding)
            for row in rows:
                if row["decision"] in ("feedback_candidate", "needs_review"):
                    self.add_record(case, "rule_baseline", "metric", row["record_id"], {
                        k: row[k] for k in ("metric_id", "metric_label_zh", "decision", "decision_zh",
                                            "feedback", "review_reasons_zh", "technique_aspect")})
            self.cases.append(case)
        self.annotations = {rid: {"manual_verdict": "pending", "notes": "", "reviewer": "", "updated_at": ""}
                            for rid in self.records}
        if self.state_path.exists():
            saved = read_json(self.state_path)
            if saved["source_hashes"] != self.source_hashes or set(saved["annotations"]) != set(self.records):
                raise ValueError("Review sources changed; keep old state and use a new --state-path")
            for rid, annotation in saved["annotations"].items():
                self.check_annotation(rid, annotation)
            self.annotations = saved["annotations"]
            self.revision = saved["revision"]

    def add_record(self, case, source, scope, item_id, content):
        rid = f"{case['case_id']}:{source}:{scope}:{item_id}"
        if rid in self.records:
            raise ValueError(f"Duplicate review ID: {rid}")
        self.records[rid] = {"record_id": rid, "case_id": case["case_id"],
                             "student_id": case["student_id"], "move_id": case["move_id"],
                             "move_name": case["move_name"], "source": source,
                             "review_scope": scope, "item_id": item_id, "content": content}
        case["record_ids"].append(rid)

    def check_annotation(self, rid, annotation):
        if rid not in self.records or not isinstance(annotation, dict):
            raise ValueError("Unknown review record")
        if annotation.get("manual_verdict") not in VERDICTS:
            raise ValueError("Invalid manual_verdict")
        _clean_text(annotation.get("notes", ""), "notes", 4000)
        _clean_text(annotation.get("reviewer", ""), "reviewer", 100)

    def update_many(self, updates, revision):
        if not isinstance(updates, list) or not updates or len(updates) > 200:
            raise ValueError("Expected 1-200 review updates")
        with self.lock:
            if type(revision) is not int or revision != self.revision:
                raise ValueError("Review changed in another browser; reload before saving")
            if any(_sha256(Path(p)) != checksum for p, checksum in self.source_hashes.items()):
                raise ValueError("Source files changed; restart using a new review state")
            new = copy.deepcopy(self.annotations)
            seen = set()
            for item in updates:
                rid = item.get("record_id")
                if rid in seen:
                    raise ValueError("Duplicate review update")
                seen.add(rid)
                self.check_annotation(rid, item)
                new[rid] = {"manual_verdict": item["manual_verdict"],
                            "notes": _clean_text(item.get("notes", ""), "notes", 4000),
                            "reviewer": _clean_text(item.get("reviewer", ""), "reviewer", 100),
                            "updated_at": datetime.now(timezone.utc).isoformat()}
            _atomic_json(self.state_path, {"schema_version": 1, "source_hashes": self.source_hashes,
                                          "revision": self.revision + 1, "annotations": new})
            self.annotations = new
            self.revision += 1
            return {"annotations": copy.deepcopy(new), "revision": self.revision}

    def client_payload(self):
        with self.lock:
            return {"cases": self.cases, "records": self.records,
                    "annotations": copy.deepcopy(self.annotations), "revision": self.revision}

    def export_csv(self):
        fields = ["record_id", "case_id", "student_id", "move_id", "move_name", "source",
                  "review_scope", "item_id", "content", "manual_verdict", "notes", "reviewer", "updated_at"]
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        with self.lock:
            for rid, record in self.records.items():
                row = dict(record, **self.annotations[rid])
                row["content"] = json.dumps(row["content"], ensure_ascii=False) if not isinstance(row["content"], str) else row["content"]
                # Keep spreadsheet software from executing reviewer-entered formulas.
                for key, value in row.items():
                    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                        row[key] = "'" + value
                writer.writerow(row)
        return ("\ufeff" + output.getvalue()).encode("utf-8")
