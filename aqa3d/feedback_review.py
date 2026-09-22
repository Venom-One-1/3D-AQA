"""Data model and persistence for the endpoint-feedback review UI."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone
from threading import RLock

from .endpoint_feedback import coach_summary


REVIEW_DECISIONS = ("feedback_candidate", "needs_review")
VERDICTS = ("pending", "correct", "false_positive", "uncertain")
CAUSES = ("", "reconstruction", "alignment", "reference_range", "rule",
          "temporal_window", "other")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _clean_text(value, field: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    value = value.replace("\x00", "").strip()
    if len(value) > maximum:
        raise ValueError(f"{field} is longer than {maximum} characters")
    return value


class FeedbackReviewStore:
    """Validated review records with atomic, resumable annotations."""

    def __init__(self, result_root: Path):
        self.result_root = Path(result_root).resolve()
        self.feedback_path = self.result_root / "feedback.json"
        self.review_csv_path = self.result_root / "manual_review.csv"
        self.state_path = self.result_root / "review_progress.json"
        if not self.feedback_path.is_file() or not self.review_csv_path.is_file():
            raise FileNotFoundError("feedback.json and manual_review.csv are required")
        self.feedback_sha256 = _sha256(self.feedback_path)
        self._lock = RLock()
        self.all_records = json.loads(self.feedback_path.read_text(encoding="utf-8"))
        if not isinstance(self.all_records, list) or not self.all_records:
            raise ValueError("feedback.json must contain a non-empty list")
        self.records_by_id = {}
        for record in self.all_records:
            record_id = record.get("record_id")
            if not isinstance(record_id, str) or not record_id or record_id in self.records_by_id:
                raise ValueError("Feedback record IDs must be non-empty and unique")
            self.records_by_id[record_id] = record
        self.review_records = [r for r in self.all_records if r.get("decision") in REVIEW_DECISIONS]
        if not self.review_records:
            raise ValueError("No candidate or needs-review records found")
        with self.review_csv_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            self.csv_fields = reader.fieldnames
            self.csv_rows = list(reader)
        if not self.csv_fields or "record_id" not in self.csv_fields:
            raise ValueError("manual_review.csv is missing record_id")
        csv_ids = [r["record_id"] for r in self.csv_rows]
        if len(csv_ids) != len(set(csv_ids)) or set(csv_ids) != set(self.records_by_id):
            raise ValueError("manual_review.csv record IDs do not match feedback.json")
        required = {"manual_verdict", "suspected_cause", "reviewer", "notes"}
        if not required.issubset(self.csv_fields):
            raise ValueError("manual_review.csv is missing annotation columns")
        self.annotations = {
            row["record_id"]: {
                "manual_verdict": row.get("manual_verdict") or "pending",
                "suspected_cause": row.get("suspected_cause") or "",
                "reviewer": row.get("reviewer") or "",
                "notes": row.get("notes") or "",
            }
            for row in self.csv_rows
        }
        self._validate_annotations(self.annotations)
        if self.state_path.is_file():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("feedback_sha256") != self.feedback_sha256:
                raise ValueError("feedback.json changed after review began; archive or reconcile review_progress.json")
            saved = state.get("annotations", {})
            if not isinstance(saved, dict) or not set(saved).issubset(self.records_by_id):
                raise ValueError("review_progress.json contains unknown records")
            self._validate_annotations(saved)
            self.annotations.update(saved)
        self.image_paths = self._build_image_paths()

    @staticmethod
    def _validate_annotations(annotations: dict) -> None:
        for record_id, annotation in annotations.items():
            if not isinstance(annotation, dict):
                raise ValueError(f"Invalid annotation for {record_id}")
            if annotation.get("manual_verdict", "pending") not in VERDICTS:
                raise ValueError(f"Invalid verdict for {record_id}")
            if annotation.get("suspected_cause", "") not in CAUSES:
                raise ValueError(f"Invalid cause for {record_id}")
            _clean_text(annotation.get("reviewer", ""), "reviewer", 100)
            _clean_text(annotation.get("notes", ""), "notes", 2000)

    def _build_image_paths(self) -> dict[str, Path]:
        metric_root = self.result_root.parent.parent / "endpoint_metric_results" / self.result_root.name
        paths = {}
        for record in self.review_records:
            sid, mid = record["subject_id"], int(record["move_id"])
            expected = {
                f"student-{sid}-{mid}": metric_root / "student" / sid / f"{mid:02d}_endpoint.jpg",
                f"reference-{mid}": metric_root / "figures" / f"{mid:02d}_reference.jpg",
                f"teachers-{mid}": metric_root / "figures" / f"{mid:02d}_teacher_keyposes.jpg",
            }
            for token, path in expected.items():
                if token in paths and paths[token] != path:
                    raise ValueError(f"Conflicting image token: {token}")
                paths[token] = path.resolve()
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing review images: "+", ".join(missing[:5]))
        return paths

    def update(self, record_id: str, verdict: str, cause: str, reviewer: str, notes: str) -> dict:
        if record_id not in {r["record_id"] for r in self.review_records}:
            raise ValueError("Record is outside the 93-item review scope")
        annotation = {"manual_verdict": verdict, "suspected_cause": cause,
                      "reviewer": _clean_text(reviewer, "reviewer", 100),
                      "notes": _clean_text(notes, "notes", 2000)}
        self._validate_annotations({record_id: annotation})
        with self._lock:
            self.annotations[record_id] = annotation
            state = {"schema_version": "1.0", "feedback_sha256": self.feedback_sha256,
                     "updated_at": datetime.now(timezone.utc).isoformat(),
                     "annotations": self.annotations}
            _atomic_json(self.state_path, state)
        return annotation

    def progress(self) -> dict:
        with self._lock:
            scoped = [self.annotations[r["record_id"]]["manual_verdict"] for r in self.review_records]
        counts = {verdict: scoped.count(verdict) for verdict in VERDICTS}
        return {"total": len(scoped), "completed": len(scoped)-counts["pending"], "counts": counts}

    def export_csv(self) -> bytes:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=self.csv_fields)
        writer.writeheader()
        with self._lock:
            for source in self.csv_rows:
                row = dict(source)
                row.update(self.annotations[source["record_id"]])
                writer.writerow(row)
        return ("\ufeff"+output.getvalue()).encode("utf-8")

    def client_payload(self) -> dict:
        grouped = {}
        for record in self.all_records:
            key = (record["subject_id"], int(record["move_id"]))
            grouped.setdefault(key, []).append(record)
        records = []
        with self._lock:
            review_records = [(record, dict(self.annotations[record["record_id"]]))
                              for record in self.review_records]
        for record, annotation in review_records:
            sid, mid = record["subject_id"], int(record["move_id"])
            evidence, reference = record["student_evidence"], record["teacher_reference"]
            records.append({
                "record_id": record["record_id"], "student_id": sid, "move_id": mid,
                "move_name_zh": record["move_name_zh"], "move_name_pinyin": record["move_name_pinyin"],
                "final_technique_step": record["final_technique_step"],
                "coach_summary": coach_summary(grouped[(sid,mid)]),
                "metric_id": record["metric_id"], "metric_label_zh": record["metric_label_zh"],
                "technique_aspect": record["technique_aspect"], "unit": record["unit"],
                "value": evidence["value"], "center_value": evidence["center_value"],
                "teacher_reference": reference, "direction": record["direction"],
                "decision": record["decision"], "decision_zh": record["decision_zh"],
                "review_reasons_zh": record["review_reasons_zh"], "feedback": record["feedback"],
                "limitation": record["limitation"], "boundary_time_seconds": evidence["boundary_time_seconds"],
                "source_frame_0based": evidence["center_source_frame_0based"],
                "images": {"student": f"/api/image/student-{sid}-{mid}",
                           "reference": f"/api/image/reference-{mid}",
                           "teachers": f"/api/image/teachers-{mid}"},
                "annotation": annotation,
            })
        return {"records": records, "progress": self.progress(),
                "verdicts": list(VERDICTS[1:]), "causes": list(CAUSES)}
