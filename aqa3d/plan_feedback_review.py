"""Read-only experiment adapter for new-method/F paired human review."""

import copy
import hashlib
import json
from pathlib import Path
from threading import RLock

from .f_group_review import ComparisonStore, read_json
from .feedback_review import _sha256
from .plan_feedback import extract_response, validate_response


class PlanFeedbackReviewStore(ComparisonStore):
    """Reuse the existing atomic, revision-checked annotation store and CSV schema."""

    def __init__(self, experiment_root, state_path=None):
        self.root = Path(experiment_root).resolve()
        self.state_path = Path(state_path).resolve() if state_path else self.root/"ui_comparison/review_progress.json"
        if self.state_path.is_relative_to(self.root) and not self.state_path.is_relative_to(self.root/"ui_comparison"):
            raise ValueError("Inside the experiment, review state must be under ui_comparison/")
        self.lock, self.revision = RLock(), 0
        self.source_hashes, self.records, self.image_paths = {}, {}, {}
        self.cases, self.mode = [], "plan_feedback"

        def path(relative):
            target = (self.root/relative).resolve()
            if not target.is_relative_to(self.root):
                raise ValueError("Source path escapes experiment")
            return target

        def track(target):
            self.source_hashes[str(target)] = _sha256(target)
            return target

        def load(relative):
            return read_json(track(path(relative)))

        manifest = load("experiment_manifest.json")
        for relative, expected in manifest["frozen_sha256"].items():
            if _sha256(track(path(relative))) != expected:
                raise ValueError(f"Frozen experiment file changed: {relative}")
        comparisons = load("comparison_snapshot.json")
        policies = load("syntax_repairs.json")["cases"] if (self.root/"syntax_repairs.json").exists() else {}
        self.models = {"plan_feedback": manifest["api_config"]["model"], "F": manifest["old_f_model"]}
        seen = set()
        for entry in manifest["cases"]:
            cid, sid, mid = entry["case_id"], entry["student_id"], entry["move_id"]
            if cid in seen:
                raise ValueError("Duplicate case ID")
            seen.add(cid)
            packet = load(entry["input_file"])
            if (packet["case_id"], packet["student_id"], packet["move_id"]) != (cid, sid, mid):
                raise ValueError("Case identity mismatch")
            if packet["reference_anchor"]["reference_video_id"] != "BV1WE411W7JB":
                raise ValueError("Unexpected reference anchor")
            status = load(f"api_calls/{cid}/status.json")
            raw_path = track(path(entry["response_file"]))
            if status["status"] != "completed" or _sha256(raw_path) != status["response_sha256"]:
                raise ValueError("Original response provenance mismatch")
            new, _ = extract_response(raw_path.read_text(encoding="utf-8"), policies.get(cid))
            errors = validate_response(new, packet)
            other = comparisons[cid]
            old = other["old_f_response"]
            if old["case_id"] != other["old_f_case_id"]:
                raise ValueError("F response identity mismatch")
            rows = other["rule_baseline"]
            if not rows or any((r["subject_id"], r["move_id"], r["final_technique_step"]) !=
                               (sid, mid, packet["technique"]) for r in rows):
                raise ValueError("F evidence identity mismatch")
            case = {"case_id": cid, "student_id": sid, "move_id": mid, "move_name": packet["move_name"],
                "technique": packet["technique"], "plan_id": packet["plan_id"],
                "reference_time": packet["reference_anchor"]["boundary_time_seconds"],
                "student_time": packet["student_endpoint"]["boundary_time_seconds"],
                "responses": {"plan_feedback": new, "F": old}, "validation_errors": errors,
                "new_evidence": packet["evidence"], "checks": packet["checks"],
                "excluded_clauses": packet["excluded_clauses"],
                "f_metrics": [{"metric_id": r["metric_id"], "label": r["metric_label_zh"], "unit": r["unit"],
                               "student": r["student_evidence"], "reference": r["teacher_reference"]} for r in rows],
                "images": {}, "record_ids": []}
            for role in ("reference", "student"):
                token = f"{cid}-{role}"
                self.image_paths[token] = track(path(f"images/{cid}_{role}.jpg"))
                case["images"][role] = "/api/image/"+token
            for method, response in case["responses"].items():
                if not isinstance(response.get("coach_summary"), str) or not isinstance(response.get("findings"), list):
                    raise ValueError(f"Response cannot be rendered: {cid}/{method}")
                self.add_record(case, method, "coach_summary", "summary", response["coach_summary"])
                for f in response["findings"]:
                    if not isinstance(f, dict) or not isinstance(f.get("finding_id"), str) or not f["finding_id"]:
                        raise ValueError("Invalid finding ID")
                    self.add_record(case, method, "finding", f["finding_id"], f)
            self.cases.append(case)
        self.annotations = {rid: {"manual_verdict": "pending", "notes": "", "reviewer": "", "updated_at": ""}
                            for rid in self.records}
        if self.state_path.exists():
            saved = read_json(self.state_path)
            if saved["source_hashes"] != self.source_hashes or set(saved["annotations"]) != set(self.records):
                raise ValueError("Review sources changed; keep old state and use a new --state-path")
            for rid, annotation in saved["annotations"].items():
                self.check_annotation(rid, annotation)
            self.annotations, self.revision = saved["annotations"], saved["revision"]
        self.dataset_id = hashlib.sha256(json.dumps(self.source_hashes, sort_keys=True).encode()).hexdigest()
        self.state_id = hashlib.sha256(str(self.state_path).encode()).hexdigest()

    def client_payload(self):
        with self.lock:
            return {"mode": self.mode, "models": self.models, "cases": self.cases, "records": self.records,
                    "annotations": copy.deepcopy(self.annotations), "revision": self.revision,
                    "dataset_id": self.dataset_id, "state_id": self.state_id,
                    "export_filename": "plan_feedback_f_review.csv"}
