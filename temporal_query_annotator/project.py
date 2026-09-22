from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .domain import (
    ProjectPaths,
    ValidationError,
    atomic_write_json,
    default_annotation,
    deep_copy_json,
    load_json,
    resolve_project_paths,
    seconds_to_frame,
    validate_annotation,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AnnotationProject:
    def __init__(self, config_path: Path):
        self.config = load_json(config_path)
        self.paths: ProjectPaths = resolve_project_paths(config_path, self.config)
        self._lock = threading.RLock()
        self._validate_sources()

        self.rules = load_json(self.paths.rules)
        self.sample_fps = float(self.config["sample_fps"])
        self.metrics = self.rules.get("metric_definitions", {})
        if not isinstance(self.metrics, dict) or not self.metrics:
            raise ValidationError("Rules file does not contain metric_definitions")
        self.metric_ids = set(self.metrics)
        self.tasks = self._build_tasks()
        self.task_by_id = {task["technique_id"]: task for task in self.tasks}
        self.frame_names = sorted(path.name for path in self.paths.frames.glob("*.jpg"))
        if not self.frame_names:
            raise ValidationError(f"No JPG frames found in {self.paths.frames}")
        self.frame_name_set = set(self.frame_names)
        self.state = self._load_or_initialize_state()

    def _validate_sources(self) -> None:
        for label, path in (
            ("video", self.paths.video),
            ("frames directory", self.paths.frames),
            ("rules", self.paths.rules),
        ):
            if not path.exists():
                raise FileNotFoundError(f"Missing {label}: {path}")
        if self.config.get("schema_version") != "0.1":
            raise ValidationError("Project schema_version must be '0.1'")

    def _build_tasks(self) -> list[dict[str, Any]]:
        move_ids = {int(value) for value in self.config.get("move_ids", [])}
        boundaries = {
            int(item["move_id"]): item
            for item in self.config.get("move_boundaries_seconds", [])
        }
        rules_moves = self.rules.get("moves")
        if not isinstance(rules_moves, list):
            raise ValidationError("Rules file does not contain a moves list")

        tasks: list[dict[str, Any]] = []
        for move in rules_moves:
            move_id = int(move.get("move_id", -1))
            if move_id not in move_ids:
                continue
            if move_id not in boundaries:
                raise ValidationError(f"Missing boundary for move {move_id}")
            boundary = boundaries[move_id]
            move_start = seconds_to_frame(float(boundary["start"]), self.sample_fps)
            move_end = seconds_to_frame(float(boundary["end"]), self.sample_fps)
            for keypose in move.get("keyposes", []):
                technique_id = str(keypose.get("pose_id", "")).strip()
                if not technique_id:
                    raise ValidationError(f"Move {move_id} contains a keypose without pose_id")
                tasks.append(
                    {
                        "technique_id": technique_id,
                        "move_id": move_id,
                        "move_name": move.get("move_name", ""),
                        "move_display_name": move.get("display_name", ""),
                        "stage_name": keypose.get("stage_name", ""),
                        "technique": keypose.get("technique", ""),
                        "existing_checks": keypose.get("checks", []),
                        "unsupported_observations": keypose.get(
                            "unsupported_observations", []
                        ),
                        "legacy_keypose": {
                            "frame": keypose.get("frame"),
                            "image": keypose.get("image"),
                        },
                        "move_start_seconds": float(boundary["start"]),
                        "move_end_seconds": float(boundary["end"]),
                        "move_start_frame_5fps": move_start,
                        "move_end_frame_5fps": move_end,
                    }
                )
        if not tasks:
            raise ValidationError("No annotation tasks were selected")
        return tasks

    def _load_or_initialize_state(self) -> dict[str, Any]:
        if self.paths.annotations.exists():
            state = load_json(self.paths.annotations)
        else:
            now = utc_now()
            state = {
                "schema_version": "0.1",
                "project_id": self.config["project_id"],
                "video_id": self.config["video_id"],
                "sample_fps": self.sample_fps,
                "created_at": now,
                "updated_at": now,
                "annotations": {
                    task["technique_id"]: default_annotation(task)
                    for task in self.tasks
                },
            }
            atomic_write_json(self.paths.annotations, state)

        if state.get("project_id") != self.config["project_id"]:
            raise ValidationError("Annotation file belongs to a different project")
        annotations = state.get("annotations")
        if not isinstance(annotations, dict):
            raise ValidationError("Annotation file does not contain annotations")
        changed = False
        for task in self.tasks:
            task_id = task["technique_id"]
            if task_id not in annotations:
                annotations[task_id] = default_annotation(task)
                changed = True
            validate_annotation(annotations[task_id], task, self.metric_ids)
        if changed:
            state["updated_at"] = utc_now()
            atomic_write_json(self.paths.annotations, state)
        return state

    def public_project(self, llm_status: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": "0.1",
            "project_id": self.config["project_id"],
            "title": self.config["title"],
            "video_id": self.config["video_id"],
            "sample_fps": self.sample_fps,
            "frame_count": len(self.frame_names),
            "first_frame": self.frame_names[0],
            "last_frame": self.frame_names[-1],
            "tasks": deep_copy_json(self.tasks),
            "metrics": deep_copy_json(self.metrics),
            "llm": llm_status,
        }

    def annotations(self) -> dict[str, Any]:
        with self._lock:
            return deep_copy_json(self.state)

    def get_annotation(self, technique_id: str) -> dict[str, Any]:
        with self._lock:
            if technique_id not in self.task_by_id:
                raise KeyError(technique_id)
            return deep_copy_json(self.state["annotations"][technique_id])

    def save_annotation(
        self, technique_id: str, annotation: dict[str, Any]
    ) -> dict[str, Any]:
        with self._lock:
            task = self.task_by_id.get(technique_id)
            if task is None:
                raise KeyError(technique_id)
            validate_annotation(annotation, task, self.metric_ids)
            previous = self.state["annotations"].get(technique_id)
            now = utc_now()
            audit = annotation.setdefault("audit", {})
            previous_audit = previous.get("audit", {}) if isinstance(previous, dict) else {}
            audit["llm_generated_at"] = audit.get("llm_generated_at") or previous_audit.get(
                "llm_generated_at"
            )
            audit["first_saved_at"] = previous_audit.get("first_saved_at") or now
            audit["last_saved_at"] = now
            candidate = annotation.get("llm_candidate")
            if isinstance(candidate, dict) and isinstance(candidate.get("query"), dict):
                audit["llm_candidate_accepted_without_edit"] = (
                    annotation.get("query") == candidate.get("query")
                )
            self.state["annotations"][technique_id] = deep_copy_json(annotation)
            self.state["updated_at"] = now
            atomic_write_json(self.paths.annotations, self.state)
            return deep_copy_json(annotation)

    def save_candidate(
        self, technique_id: str, candidate: dict[str, Any]
    ) -> dict[str, Any]:
        with self._lock:
            annotation = self.get_annotation(technique_id)
            annotation["llm_candidate"] = deep_copy_json(candidate)
            annotation.setdefault("audit", {})["llm_generated_at"] = utc_now()
            if annotation["review_status"] == "unannotated":
                annotation["review_status"] = "draft"
            return self.save_annotation(technique_id, annotation)

    def frame_path(self, frame_name: str) -> Path:
        if frame_name not in self.frame_name_set:
            raise FileNotFoundError(frame_name)
        return self.paths.frames / frame_name

    def export_payload(self) -> dict[str, Any]:
        with self._lock:
            payload = deep_copy_json(self.state)
            payload["project"] = {
                "title": self.config["title"],
                "video_id": self.config["video_id"],
                "sample_fps": self.sample_fps,
                "task_count": len(self.tasks),
            }
            payload["tasks"] = deep_copy_json(self.tasks)
            return payload


def compact_metric_catalog(
    metrics: dict[str, dict[str, Any]], preferred_ids: set[str]
) -> list[dict[str, Any]]:
    ordered_ids = sorted(metrics, key=lambda item: (item not in preferred_ids, item))
    catalog = []
    for metric_id in ordered_ids:
        definition = metrics[metric_id]
        catalog.append(
            {
                "metric_id": metric_id,
                "type": definition.get("type"),
                "unit": definition.get("unit"),
                "calculation": definition.get("calculation"),
            }
        )
    return catalog


def serialize_pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)
