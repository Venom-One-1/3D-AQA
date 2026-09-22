from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from temporal_query_annotator.domain import (
    ValidationError,
    blank_query,
    extract_json_object,
    frame_to_seconds,
    seconds_to_frame,
    validate_query,
)
from temporal_query_annotator.llm import build_user_prompt, validate_candidate
from temporal_query_annotator.project import AnnotationProject


class FrameConversionTests(unittest.TestCase):
    def test_five_fps_grid(self) -> None:
        self.assertEqual(seconds_to_frame(16.0, 5.0), 80)
        self.assertEqual(seconds_to_frame(41.4, 5.0), 207)
        self.assertAlmostEqual(frame_to_seconds(239, 5.0), 47.8)


class QueryValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metric_ids = {"right_elbow_angle", "right_wrist_height"}

    def test_blank_query_is_valid(self) -> None:
        validate_query(blank_query(), self.metric_ids)

    def test_unknown_metric_is_rejected(self) -> None:
        query = blank_query()
        query["conditions"].append(
            {
                "metric_id": "invented_metric",
                "signal": "value",
                "aggregation": "mean",
                "operator": "within_teacher_range",
                "role": "required",
                "min_duration_seconds": 0.4,
            }
        )
        with self.assertRaisesRegex(ValidationError, "unknown metric_id"):
            validate_query(query, self.metric_ids)

    def test_candidate_scope_must_match_query(self) -> None:
        candidate = {
            "observability": "observable",
            "observability_reason": "test",
            "temporal_scope": "instant_event",
            "query": blank_query("stable_window"),
        }
        with self.assertRaisesRegex(ValidationError, "differ"):
            validate_candidate(candidate, self.metric_ids)

    def test_extracts_json_from_code_fence(self) -> None:
        response = "```json\n{\"answer\": 1}\n```"
        self.assertEqual(extract_json_object(response), {"answer": 1})


class ProjectTests(unittest.TestCase):
    def test_project_builds_tasks_and_persists_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "video.mp4"
            video.write_bytes(b"fixture")
            frames = root / "frames"
            frames.mkdir()
            for index in range(6):
                (frames / f"{index:05d}.jpg").write_bytes(b"frame")
            rules = {
                "metric_definitions": {
                    "right_elbow_angle": {
                        "type": "joint_angle",
                        "unit": "degree",
                        "calculation": "angle(a,b,c)",
                    }
                },
                "moves": [
                    {
                        "move_id": 1,
                        "move_name": "qishi",
                        "display_name": "起势",
                        "keyposes": [
                            {
                                "pose_id": "1.1",
                                "stage_name": "预备式",
                                "technique": "自然站立",
                                "checks": [
                                    {
                                        "check_id": "1.1.elbow",
                                        "metric_id": "right_elbow_angle",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
            rules_path = root / "rules.json"
            rules_path.write_text(json.dumps(rules, ensure_ascii=False), encoding="utf-8")
            config = {
                "schema_version": "0.1",
                "project_id": "fixture",
                "title": "Fixture",
                "video_id": "fixture-video",
                "video_path": str(video),
                "frames_dir": str(frames),
                "rules_path": str(rules_path),
                "annotations_path": str(root / "annotations.json"),
                "sample_fps": 5.0,
                "move_ids": [1],
                "move_boundaries_seconds": [
                    {"move_id": 1, "start": 0.0, "end": 1.0}
                ],
            }
            config_path = root / "project.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            project = AnnotationProject(config_path)
            self.assertEqual(len(project.tasks), 1)
            self.assertEqual(project.tasks[0]["move_end_frame_5fps"], 5)
            annotation = project.get_annotation("1.1")
            annotation["review_status"] = "draft"
            annotation["gold"]["start_frame_5fps"] = 1
            annotation["gold"]["end_frame_5fps"] = 3
            project.save_annotation("1.1", annotation)

            reloaded = AnnotationProject(config_path)
            self.assertEqual(
                reloaded.get_annotation("1.1")["gold"]["end_frame_5fps"], 3
            )

    def test_prompt_contains_task_and_metric_catalog(self) -> None:
        task = {
            "move_id": 1,
            "move_display_name": "起势",
            "technique_id": "1.1",
            "stage_name": "预备式",
            "technique": "自然站立",
            "existing_checks": [{"metric_id": "right_elbow_angle"}],
            "unsupported_observations": [],
        }
        metrics = {
            "right_elbow_angle": {
                "type": "joint_angle",
                "unit": "degree",
                "calculation": "angle(a,b,c)",
            }
        }
        prompt = build_user_prompt(task, metrics)
        self.assertIn("自然站立", prompt)
        self.assertIn("right_elbow_angle", prompt)
        self.assertIn("不直接输出绝对时间戳", prompt)


if __name__ == "__main__":
    unittest.main()
