from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from aqa3d.check_planning import (
    build_input, parse_plan, queries_payload, query_signature, stability,
    strict_loads, validate_response,
)
from aqa3d.pose_measurements import MeasurementQuery as Q
from run_check_plan_experiment import (
    MODEL, analyze, call_plans, digest, load_key, prepare,
)
from run_f_group_api import request_payload, run_case
from run_pose_measurements import write_json


def manifest():
    return {"endpoint_keyposes": [
        {"move_id": mid, "move_name_zh": "fixture", "final_technique_step": "bend knees;relax",
         "student_values": "never transmit", "teacher_ranges": "never transmit"}
        for mid in (1, 2, 3)]}


def valid_plan(mid=1):
    return {"schema_version": "1.0", "move_id": mid, "scope": "endpoint_geometry_only",
            "queries": [asdict(Q("knee", "joint_angle", ("L_Hip", "L_Knee", "L_Ankle")))],
            "checks": [
                {"clause_id": "C01", "support": "direct", "endpoint_interpretation": "knee geometry",
                 "evidence": [{"query_id": "knee", "components": ["value"], "purpose": "describe bending",
                               "expected_relation": "bent, not straight"}], "limitations": ["not weight bearing"]},
                {"clause_id": "C02", "support": "unsupported", "endpoint_interpretation": "not visible in joints",
                 "evidence": [], "limitations": ["cannot measure relaxation"]}]}


def config():
    return dict(model=MODEL, base_url="https://example.maas.aliyuncs.com/compatible-mode/v1",
                temperature=0.2, top_p=0.8, max_tokens=16384, enable_thinking=True,
                thinking_budget=4000, timeout_seconds=300, enable_search=False)


class CheckPlanningTests(unittest.TestCase):
    def setUp(self):
        self.request = build_input(manifest(), 1)

    def test_input_whitelist_and_no_example_query_leakage(self):
        text = json.dumps(self.request)
        for excluded in ("student_values", "teacher_ranges", "never transmit", "wrists_to_neck", "queries"):
            self.assertNotIn('"'+excluded+'"', text)
        contract = self.request["measurement_tools"]
        self.assertEqual(contract["operations"]["relative_position"]["components"], ["right", "up", "forward"])
        self.assertIn("point[1] is vertex", contract["operations"]["joint_angle"]["formula"])
        self.assertEqual(len(self.request["clauses"]), 2)

    def test_valid_plan_exports_actual_measurement_queries(self):
        plan = parse_plan(json.dumps(valid_plan()), self.request)
        self.assertEqual(queries_payload(plan)["queries"][0]["points"], ("L_Hip", "L_Knee", "L_Ankle"))
        validation, _ = validate_response(json.dumps(plan), self.request)
        self.assertEqual(validation["semantic_correctness"], "not_reviewed")
        self.assertFalse(validation["can_diagnose"])

    def test_bad_tool_scope_fields_and_identity_rejected(self):
        mutations = [
            lambda p: p.update(move_id=True), lambda p: p.update(scope="whole_video"),
            lambda p: p.update(diagnosis="bad knee"),
            lambda p: p["queries"][0].update(points=["Eye", "L_Knee", "L_Ankle"]),
            lambda p: p["queries"][0].update(operation="eval"),
            lambda p: p["queries"][0].update(coordinate_frame="world"),
            lambda p: p["checks"][0]["evidence"][0].update(components=["up"]),
            lambda p: p["checks"][0]["evidence"][0].update(query_id="invented"),
            lambda p: p["checks"][0].update(limitations=[]),
            lambda p: p["checks"][0].update(support=["direct"]),
        ]
        for mutation in mutations:
            plan = valid_plan()
            mutation(plan)
            with self.subTest(plan=plan):
                record, result = validate_response(json.dumps(plan), self.request)
                self.assertFalse(record["schema_valid"])
                self.assertIsNone(result)

    def test_missing_duplicate_and_unused_queries_rejected(self):
        mutations = [lambda p: p["checks"].pop(),
                     lambda p: p["checks"].append(deepcopy(p["checks"][0])),
                     lambda p: p["checks"][0].update(support="unsupported"),
                     lambda p: p["queries"].append(asdict(Q("other", "joint_angle", ("R_Hip", "R_Knee", "R_Ankle")))),
                     lambda p: p["queries"].append(asdict(Q("alias", "joint_angle", ("L_Ankle", "L_Knee", "L_Hip"))))]
        for mutation in mutations:
            plan = valid_plan()
            mutation(plan)
            self.assertFalse(validate_response(json.dumps(plan), self.request)[0]["schema_valid"])

    def test_strict_json_no_silent_answer_repair(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '```json\n{}\n```'):
            with self.assertRaises(ValueError):
                strict_loads(text)

    def test_query_canonicalization_preserves_direction_and_frame(self):
        a = Q("a", "distance", ("L_Wrist", "Head"), "torso_length")
        b = Q("b", "distance", ("Head", "L_Wrist"), "torso_length")
        self.assertEqual(query_signature(a), query_signature(b))
        a = Q("a", "relative_position", ("L_Wrist", "Head"), "torso_length", "torso_v1")
        b = Q("b", "relative_position", ("Head", "L_Wrist"), "torso_length", "torso_v1")
        self.assertNotEqual(query_signature(a), query_signature(b))

    def test_stability_ignores_query_names_not_support_classification(self):
        a, b = valid_plan(), valid_plan()
        b["queries"][0]["query_id"] = "renamed"
        b["checks"][0]["evidence"][0]["query_id"] = "renamed"
        b["checks"][0]["support"] = "proxy"
        row = stability([("a", a), ("b", b)])[0]
        self.assertEqual(row["query_jaccard"], 1)
        self.assertEqual(row["support_agreement"], .5)
        self.assertEqual(row["clause_evidence_exact_agreement"], 1)

    def prepare_fixture(self, root):
        source = root/"source"
        source.mkdir()
        write_json(source/"reference_manifest_snapshot.json", manifest())
        write_json(source/"summary.json", {"reference_manifest_sha256": digest(source/"reference_manifest_snapshot.json")})
        out = root/"output"
        frozen = prepare(source, out, config())
        return source, out, frozen

    def test_freeze_nine_identical_independent_requests_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out, frozen = self.prepare_fixture(Path(tmp))
            self.assertEqual(len(frozen["cases"]), 9)
            self.assertEqual(prepare(source, out, config()), frozen)
            payloads = [request_payload(config(), "prompt", (out/e["input_file"]).read_text()) for e in frozen["cases"]]
            self.assertEqual(payloads[0], payloads[1])
            self.assertEqual(payloads[0], payloads[2])
            self.assertNotEqual(payloads[0], payloads[3])
            self.assertTrue(all(len(p["messages"]) == 2 and p["model"] == MODEL for p in payloads))
            (out/"inputs/M01.json").write_text("{}")
            with self.assertRaises(ValueError):
                prepare(source, out, config())

    def test_call_count_and_first_failure_stops_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out, frozen = self.prepare_fixture(Path(tmp))
            def completed(e, *args, **kwargs):
                return {"case_id": e["case_id"], "status": "completed"}
            with patch("run_check_plan_experiment.run_case", side_effect=completed) as run:
                self.assertEqual(len(call_plans(frozen, out, "test-token")), 9)
                self.assertEqual(run.call_count, 9)
            with patch("run_check_plan_experiment.run_case", return_value={"case_id": "M01_R01", "status": "failed"}) as run:
                self.assertEqual(len(call_plans(frozen, out, "test-token")), 1)
                self.assertEqual(run.call_count, 1)

    def test_transport_uses_planner_prompt_without_f_prompt_or_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out, frozen = self.prepare_fixture(Path(tmp))
            response = MagicMock()
            response.__enter__.return_value = response
            response.status_code = 200
            response.iter_lines.return_value = iter([
                b'data: {"choices":[{"delta":{"content":"{}"},"finish_reason":"stop"}]}',
                b'data: [DONE]'])
            with patch("run_f_group_api.requests.post", return_value=response) as post:
                status = run_case(frozen["cases"][0], out, config(), "test-secret", retries=0)
            self.assertEqual(status["status"], "completed")
            payload = post.call_args.kwargs["json"]
            self.assertEqual(payload["messages"][0]["content"], (out/"planner_prompt.md").read_text())
            self.assertEqual(len(payload["messages"]), 2)
            for path in out.rglob("*"):
                if path.is_file():
                    self.assertNotIn("test-secret", path.read_text())

    def test_snapshot_mismatch_is_not_eligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out, frozen = self.prepare_fixture(Path(tmp))
            e = frozen["cases"][0]
            answer = out/e["response_file"]
            answer.write_text(json.dumps(valid_plan()))
            status = out/"api_calls"/e["case_id"]/"status.json"
            status.parent.mkdir(parents=True)
            write_json(status, {"status": "completed", "response_sha256": digest(answer),
                                "attempts": [{"model": "unexpected-model"}]})
            _, summary = analyze(frozen, out, execute=False)
            self.assertEqual(summary["schema_valid_plans"], 1)
            self.assertEqual(summary["eligible_plans"], 0)
            self.assertEqual(summary["selected_by_first_valid_repeat"], {})

    def test_key_only_owner_readable_and_missing_key_message_redacted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"credential"
            path.write_text("test-secret")
            path.chmod(0o600)
            self.assertEqual(load_key(path), "test-secret")
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                load_key(path)
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            load_key(None)

    def test_analyze_preserves_raw_and_selects_first_valid_not_best_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out, frozen = self.prepare_fixture(Path(tmp))
            for e in frozen["cases"]:
                content = json.dumps(valid_plan(e["move_id"]))
                if e["case_id"] == "M01_R01":
                    content = "not JSON"
                answer = out/e["response_file"]
                answer.write_text(content)
                status = out/"api_calls"/e["case_id"]/"status.json"
                status.parent.mkdir(parents=True)
                write_json(status, {"status": "completed", "response_sha256": digest(answer),
                                    "attempts": [{"model": MODEL}]})
            with patch("run_check_plan_experiment.export_experiment") as execute:
                folder, summary = analyze(frozen, out, execute=False)
                execute.assert_not_called()
            self.assertEqual(summary["eligible_plans"], 8)
            self.assertEqual(summary["selected_by_first_valid_repeat"]["1"], "M01_R02")
            self.assertIsNone(summary["semantic_accuracy"])
            self.assertEqual((out/"responses/M01_R01.json").read_text(), "not JSON")
            self.assertTrue((folder/"plan_review.csv").exists())

    def test_analyze_does_not_count_missing_calls_as_bad_plans_or_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out, frozen = self.prepare_fixture(Path(tmp))
            _, summary = analyze(frozen, out, execute=False)
            self.assertEqual(summary["completed_calls"], 0)
            self.assertEqual(summary["selected_by_first_valid_repeat"], {})
            self.assertIsNone(summary["semantic_accuracy"])


if __name__ == "__main__":
    unittest.main()
