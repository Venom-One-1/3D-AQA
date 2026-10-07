import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aqa3d.f_group_review import ComparisonStore
from run_f_group_review_app import main


class ComparisonReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.metric = self.root / "metric"
        self.exp = self.root / "experiment"
        self.state = self.root / "state.json"
        def write(path, data):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data), encoding="utf-8")
        entry = dict(case_id="F001", student_id="01", move_id=1,
                     input_file="inputs/F001.json", response_file="responses/F001.json")
        write(self.exp / "experiment_manifest.json", {"cases": [entry]})
        write(self.exp / entry["input_file"], dict(technique="technique", move={"move_name_zh": "qishi"}, metrics=[]))
        write(self.exp / entry["response_file"], dict(case_id="F001", coach_summary="summary", findings=[dict(finding_id="P1", problem="problem")]))
        row = dict(record_id="01:1:x", subject_id="01", move_id=1, final_technique_step="technique",
                   metric_id="x", metric_label_zh="knee", decision="feedback_candidate", decision_zh="candidate",
                   feedback="feedback", review_reasons_zh=[], technique_aspect="aspect",
                   student_evidence={"boundary_time_seconds": 12})
        write(self.exp / "evaluation/baseline_snapshot.json", [row])
        write(self.metric / "reference_manifest_snapshot.json", dict(reference={"video_id": "BV1WE411W7JB"},
              endpoint_keyposes=[dict(move_id=1, boundary_time_seconds=16)]))
        for name in ["figures/01_reference.jpg", "student/01/01_endpoint.jpg"]:
            path = self.metric / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"image fixture")

    def store(self):
        return ComparisonStore(self.exp, self.metric, self.state)

    def ablation_fixture(self):
        from test_f_group_experiment import fixture, answer
        case = fixture()
        case["technique"] = "technique"
        response = answer(case)
        response["findings"] = [{"finding_id": "P1", "assessment": "possible_issue",
            "problem": "problem", "explanation": "evidence", "uncertainty": "limits",
            "guidance": "guidance", "evidence_metric_ids": [case["metrics"][0]["metric_id"]]}]
        other = self.root / "ablation"
        for root in (self.exp, other):
            for folder in ("inputs", "responses"):
                (root / folder).mkdir(parents=True, exist_ok=True)
            (root / "inputs/F001.json").write_text(json.dumps(case))
            (root / "api_config.json").write_text(json.dumps({"model": "original" if root == self.exp else "snapshot"}))
            (root / "F_prompt.md").write_text("prompt " + root.name)
        (other / "experiment_manifest.json").write_bytes((self.exp / "experiment_manifest.json").read_bytes())
        (self.exp / "responses/F001.json").write_text(json.dumps(response))
        del response["metric_comparisons"]
        (other / "responses/F001.json").write_text(json.dumps(response))
        return other

    def test_ablation_fresh_labels_separate_sources_and_export(self):
        other = self.ablation_fixture()
        original = self.store()
        rid = "F001:F:finding:P1"
        original.update_many([dict(record_id=rid, manual_verdict="correct")], 0)
        before = self.state.read_bytes()
        paired = ComparisonStore(self.exp, self.metric, ablation_root=other)
        self.assertEqual(len(paired.records), 4)
        self.assertEqual(set(r["source"] for r in paired.records.values()), {"F", "F_ablation"})
        self.assertTrue(all(a["manual_verdict"] == "pending" for a in paired.annotations.values()))
        self.assertEqual(paired.state_path, other / "ui_comparison/review_progress.json")
        paired.update_many([dict(record_id=rid, manual_verdict="false_positive", notes="new review"),
                            dict(record_id="F001:F_ablation:finding:P1", manual_verdict="correct")], 0)
        restored = ComparisonStore(self.exp, self.metric, ablation_root=other)
        self.assertEqual(restored.annotations[rid]["manual_verdict"], "false_positive")
        rows = list(csv.DictReader(io.StringIO(restored.export_csv().decode("utf-8-sig"))))
        self.assertEqual(len(rows), 4)
        self.assertEqual(self.state.read_bytes(), before)
        self.assertEqual(paired.client_payload()["mode"], "ablation")
        (other / "responses/F001.json").write_text("{}")
        with self.assertRaises(ValueError):
            restored.update_many([dict(record_id=rid, manual_verdict="correct")], 1)

    def test_ablation_rejects_mismatched_input_and_identity(self):
        other = self.ablation_fixture()
        path = other / "inputs/F001.json"
        case = json.loads(path.read_text())
        case["technique"] = "different"
        path.write_text(json.dumps(case))
        with self.assertRaisesRegex(ValueError, "input mismatch"):
            ComparisonStore(self.exp, self.metric, ablation_root=other)
        path.write_bytes((self.exp / "inputs/F001.json").read_bytes())
        manifest_path = other / "experiment_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["cases"][0]["student_id"] = "10"
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            ComparisonStore(self.exp, self.metric, ablation_root=other)

    def test_live_responses_and_scope(self):
        store = self.store()
        self.assertEqual(len(store.records), 4)
        self.assertEqual(store.cases[0]["response"]["coach_summary"], "summary")
        self.assertEqual(len(store.image_paths), 2)
        self.assertFalse(self.state.exists())

    def test_save_reload_export_and_formula_safety(self):
        store = self.store()
        rid = next(iter(store.records))
        store.update_many([dict(record_id=rid, manual_verdict="correct", notes="=1+1", reviewer="reviewer")], 0)
        restored = self.store()
        self.assertEqual(restored.annotations[rid]["notes"], "=1+1")
        self.assertEqual(restored.revision, 1)
        rows = list(csv.DictReader(io.StringIO(restored.export_csv().decode("utf-8-sig"))))
        self.assertEqual(rows[0]["manual_verdict"], "correct")
        self.assertEqual(rows[0]["notes"], "'=1+1")
        self.assertEqual(rows[0]["student_id"], "01")

    def test_bulk_validation_is_atomic(self):
        store = self.store()
        rid = next(iter(store.records))
        with self.assertRaises(ValueError):
            store.update_many([dict(record_id=rid, manual_verdict="correct"),
                               dict(record_id="unknown", manual_verdict="correct")], 0)
        self.assertEqual(store.annotations[rid]["manual_verdict"], "pending")
        self.assertFalse(self.state.exists())

    def test_write_failure_does_not_commit_memory(self):
        store = self.store()
        rid = next(iter(store.records))
        with patch("aqa3d.f_group_review._atomic_json", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                store.update_many([dict(record_id=rid, manual_verdict="correct")], 0)
        self.assertEqual(store.revision, 0)
        self.assertEqual(store.annotations[rid]["manual_verdict"], "pending")

    def test_stale_tab_and_source_change_rejected(self):
        store = self.store()
        rid = next(iter(store.records))
        store.update_many([dict(record_id=rid, manual_verdict="uncertain")], 0)
        with self.assertRaises(ValueError):
            store.update_many([dict(record_id=rid, manual_verdict="correct")], 0)
        path = self.exp / "responses/F001.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaises(ValueError):
            store.update_many([dict(record_id=rid, manual_verdict="correct")], 1)
        with self.assertRaises(ValueError):
            self.store()

    def test_invalid_enum_and_duplicate_ids(self):
        store = self.store()
        rid = next(iter(store.records))
        with self.assertRaises(ValueError):
            store.update_many([dict(record_id=rid, manual_verdict="yes")], 0)
        with self.assertRaises(ValueError):
            store.update_many([dict(record_id=rid, manual_verdict="correct")] * 2, 0)

    def test_duplicate_server_reports_lock_without_loading_data(self):
        errors = io.StringIO()
        with patch("sys.argv", ["review", "--state-path", str(self.state)]), \
                patch("run_f_group_review_app.fcntl.flock", side_effect=BlockingIOError()), \
                patch("run_f_group_review_app.ComparisonStore") as store, \
                patch("sys.stderr", errors):
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("already using this annotation state", errors.getvalue())
        self.assertNotIn("Traceback", errors.getvalue())
        store.assert_not_called()


if __name__ == "__main__":
    unittest.main()
