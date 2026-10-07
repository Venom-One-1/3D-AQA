from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aqa3d.check_plan_repair import extract_plan, repair_response
from aqa3d.pose_measurements import MeasurementQuery as Q
from run_check_plan_repair import digest, run_repair, snapshot
from run_pose_measurements import write_json


def plan():
    return {"schema_version": "1.0", "move_id": 1, "scope": "endpoint_geometry_only",
            "queries": [json.loads(json.dumps(asdict(Q("knee", "joint_angle", ("L_Hip", "L_Knee", "L_Ankle")))))],
            "checks": [{"clause_id": "C01", "support": "proxy", "endpoint_interpretation": "knee geometry",
                        "evidence": [{"query_id": "knee", "components": ["value"], "purpose": "bending",
                                      "expected_relation": "bent, not straight"}],
                        "limitations": ["not actual weight bearing"]}]}


def request():
    return {"schema_version": "1.0", "move_id": 1, "move_name": "fixture",
            "scope": "endpoint_geometry_only", "clauses": [{"clause_id": "C01", "text": "bend knees"}]}


def reviewed(text, **kwargs):
    return {"source_sha256": hashlib.sha256(text.encode()).hexdigest(), **kwargs}


class CheckPlanRepairTests(unittest.TestCase):
    def repair(self, document, policy=None):
        return repair_response(json.dumps(document), request(), policy)

    def test_valid_input_preserved_and_no_actions(self):
        original = plan()
        result = self.repair(original)
        self.assertEqual(result["plan"], original)
        self.assertEqual(result["changes"], [])
        self.assertEqual(result["repaired_validation"]["semantic_correctness"], "not_reviewed")

    def test_extract_unique_json_preserves_strings_and_fences_in_strings(self):
        original = plan()
        original["checks"][0]["limitations"] = ["braces { } and ``` are literal text"]
        text = "explanation\n```json\n" + json.dumps(original) + "\n```\nend"
        result = repair_response(text, request())
        self.assertEqual(result["plan"], original)
        self.assertEqual(result["changes"][0]["kind"], "extract_json")

    def test_ambiguous_fragments_duplicate_keys_nan_refused(self):
        text = json.dumps(plan())
        for bad in (text+"\n"+text, text[:-2], "prefix\n"+text.replace('"move_id": 1', '"move_id": 1, "move_id": 1'),
                    "prefix\n"+text.replace('"move_id": 1', '"move_id": NaN')):
            with self.subTest(bad=bad):
                self.assertIsNone(repair_response(bad, request())["plan"])

    def test_reviewed_quote_escape_keeps_decoded_text(self):
        original = plan()
        original["checks"][0]["endpoint_interpretation"] = 'keep "quoted" text'
        text = json.dumps(original).replace('keep \\"quoted\\" text', 'keep "quoted" text')
        result = repair_response(text, request(), reviewed(text, escape_literals=['keep "quoted" text']))
        self.assertEqual(result["plan"], original)
        self.assertEqual(len(result["changes"]), 1)

    def test_reviewed_policy_bound_to_source_hash_and_exact_before(self):
        text = json.dumps(plan())
        edit = {"path": ["checks", 0, "evidence", 0, "query_id"], "before": "typo", "after": "knee",
                "kind": "reviewed_reference_correction", "reason": "verified fixture"}
        self.assertIsNone(repair_response(text, request(), {"source_sha256": "bad", "edits": [edit]})["plan"])
        self.assertIsNone(repair_response(text, request(), reviewed(text, edits=[edit]))["plan"])

    def test_reference_repair_checks_target_geometry(self):
        p = plan()
        p["checks"][0]["evidence"][0]["query_id"] = "typo"
        text = json.dumps(p)
        edit = {"path": ["checks", 0, "evidence", 0, "query_id"], "before": "typo", "after": "knee",
                "target_query": json.loads(json.dumps(p["queries"][0])),
                "kind": "reviewed_reference_correction", "reason": "same knee definition"}
        self.assertIsNotNone(repair_response(text, request(), reviewed(text, edits=[edit]))["plan"])
        edit["target_query"]["points"][0] = "R_Hip"
        self.assertIsNone(repair_response(text, request(), reviewed(text, edits=[edit]))["plan"])

    def test_no_fuzzy_reference_fix(self):
        p = plan()
        p["checks"][0]["evidence"][0]["query_id"] = "kneee"
        self.assertIsNone(self.repair(p)["plan"])

    def test_duplicate_query_ids_refused_even_when_same_geometry(self):
        p = plan()
        p["queries"].append(deepcopy(p["queries"][0]))
        self.assertIsNone(self.repair(p)["plan"])

    def test_identical_queries_relinked_and_evidence_union_keeps_both_reasons(self):
        p = plan()
        p["queries"] = [asdict(Q("a", "relative_position", ("R_Wrist", "Head"), "torso_length", "torso_v1")),
                        asdict(Q("b", "relative_position", ("R_Wrist", "Head"), "torso_length", "torso_v1"))]
        first = p["checks"][0]["evidence"][0]
        first.update(query_id="a", components=["right"], purpose="first purpose")
        second = dict(first, query_id="b", components=["up"], purpose="second purpose")
        p["checks"][0]["evidence"].append(second)
        result = self.repair(p)
        self.assertEqual(len(result["plan"]["queries"]), 1)
        evidence = result["plan"]["checks"][0]["evidence"]
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["components"], ["right", "up"])
        self.assertIn("first purpose", evidence[0]["purpose"])
        self.assertIn("second purpose", evidence[0]["purpose"])

    def test_unused_queries_removed_but_clauses_not_removed(self):
        p = plan()
        p["queries"].append(asdict(Q("other", "joint_angle", ("R_Hip", "R_Knee", "R_Ankle"))))
        result = self.repair(p)
        self.assertEqual(len(result["plan"]["queries"]), 1)
        self.assertEqual(result["plan"]["checks"], p["checks"])

    def test_opposite_relative_positions_are_not_merged(self):
        p = plan()
        p["queries"] = [asdict(Q("a", "relative_position", ("R_Wrist", "Head"), "torso_length", "torso_v1")),
                        asdict(Q("b", "relative_position", ("Head", "R_Wrist"), "torso_length", "torso_v1"))]
        p["checks"][0]["evidence"] = [
            {"query_id": qid, "components": ["right"], "purpose": "fixture", "expected_relation": "fixture"}
            for qid in ("a", "b")]
        self.assertEqual(len(self.repair(p)["plan"]["queries"]), 2)

    def test_unrecognized_components_or_missing_clauses_not_invented(self):
        p = plan()
        p["checks"][0]["evidence"][0]["components"] = ["left"]
        self.assertIsNone(self.repair(p)["plan"])
        p = plan()
        p["checks"] = []
        self.assertIsNone(self.repair(p)["plan"])

    def test_reviewed_policy_cannot_change_measurement_definition(self):
        text = json.dumps(plan())
        edit = {"path": ["queries", 0, "points"], "before": ["L_Hip", "L_Knee", "L_Ankle"],
                "after": ["R_Hip", "R_Knee", "R_Ankle"], "kind": "incorrect", "reason": "not allowed"}
        self.assertIsNone(repair_response(text, request(), reviewed(text, edits=[edit]))["plan"])

    def test_real_coordinate_policy_flips_left_sign_not_only_component_name(self):
        policy = json.loads((Path(__file__).resolve().parents[1]/"prompt/check_plan_repair_policy_v1.json").read_text())
        edits = policy["cases"]["M02_R01"]["edits"]
        relations = [e for e in edits if e["path"][-1] == "expected_relation"]
        self.assertEqual(len(relations), 2)
        for e in relations:
            self.assertIn("left分量为正值", e["before"])
            self.assertIn("right分量为负值", e["after"])

    def test_export_separate_and_source_unchanged_without_api_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/"source"
            (source/"responses").mkdir(parents=True)
            (source/"inputs").mkdir()
            (source/"api_calls/M01_R01").mkdir(parents=True)
            write_json(source/"responses/M01_R01.json", plan())
            write_json(source/"inputs/M01.json", request())
            write_json(source/"api_calls/M01_R01/status.json", {
                "status": "completed", "response_sha256": digest(source/"responses/M01_R01.json"),
                "attempts": [{"model": "fixture"}]})
            write_json(source/"experiment_manifest.json", {
                "api_config": {"model": "fixture"}, "input_root": str(root/"cache"),
                "frozen_file_sha256": {"inputs/M01.json": digest(source/"inputs/M01.json")},
                "cases": [{"case_id": "M01_R01", "move_id": 1, "repeat": 1,
                           "response_file": "responses/M01_R01.json", "input_file": "inputs/M01.json"}]})
            policy = root/"policy.json"
            write_json(policy, {"version": "1.0", "cases": {}})
            original = snapshot(source)
            with patch("run_check_plan_repair.export_experiment") as execute, patch("requests.post") as api:
                result = run_repair(source, root/"repaired", policy, execute=False)
                execute.assert_not_called()
                api.assert_not_called()
            self.assertEqual(snapshot(source), original)
            self.assertEqual(result["repaired_valid"], 1)
            self.assertEqual(result["selected_first_valid_repaired"], {"1": "M01_R01"})
            self.assertEqual(result["api_calls"], 0)
            self.assertIsNone(result["semantic_accuracy"])
            with self.assertRaises(FileExistsError):
                run_repair(source, root/"repaired", policy, execute=False)
            with self.assertRaises(ValueError):
                run_repair(source, source/"child", policy, execute=False)


if __name__ == "__main__":
    unittest.main()
