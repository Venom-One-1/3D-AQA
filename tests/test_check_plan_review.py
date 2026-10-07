import csv
import http.client
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from aqa3d.check_plan_review import PlanReviewStore
from run_check_plan_review_app import create_server


class PlanReviewTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.exp, self.metric, self.state = self.root / "exp", self.root / "metric", self.root / "state.json"
        self.request = {"move_id": 1, "move_name": "qishi", "technique": "bend knees;relax",
                        "scope": "endpoint_geometry_only", "clauses": [
                            {"clause_id": "C01", "text": "bend knees"},
                            {"clause_id": "C02", "text": "relax"}]}
        plan = {"schema_version": "1.0", "move_id": 1, "scope": "endpoint_geometry_only",
                "queries": [{"query_id": "knee", "operation": "joint_angle", "points": ["L_Hip", "L_Knee", "L_Ankle"]}],
                "checks": [{"clause_id": "C01", "support": "direct", "endpoint_interpretation": "knee bend",
                            "evidence": [{"query_id": "knee", "components": ["value"],
                                          "purpose": "measure bending", "expected_relation": "less than straight"}],
                            "limitations": ["not weight bearing"]},
                           {"clause_id": "C02", "support": "unsupported", "endpoint_interpretation": "not measurable",
                            "evidence": [], "limitations": ["muscle relaxation not observable"]}]}
        self.write(self.exp / "source_manifest_snapshot.json", {"api_config": {"model": "fixture"},
            "cases": [{"case_id": f"M01_R0{r}", "move_id": 1, "repeat": r} for r in (1, 2, 3)]})
        for repeat in (1, 2, 3):
            directory = self.exp / f"M01_R0{repeat}"
            self.write(directory / "input.json", self.request)
            self.write(directory / "plan.json", plan)
            self.write(directory / "repair_log.json", {"changes": [], "semantic_warnings": []})
        self.write(self.metric / "reference_manifest_snapshot.json", {
            "reference": {"video_id": "BV1WE411W7JB"}, "endpoint_keyposes": [
                {"move_id": 1, "final_technique_step": self.request["technique"], "boundary_time_seconds": 16.0}]})
        (self.metric / "figures").mkdir()
        (self.metric / "figures/01_reference.jpg").write_bytes(b"fixture")

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def store(self):
        return PlanReviewStore(self.exp, self.metric, self.state)

    def test_all_repeats_and_unsupported_clauses_preserved(self):
        store = self.store()
        self.assertEqual(len(store.records), 6)
        self.assertEqual(len(store.cases), 3)
        record = store.records["M01_R01:C01"]
        self.assertEqual(record["query_definitions"]["knee"]["points"], ("L_Hip", "L_Knee", "L_Ankle"))
        self.assertIn("point[1] is vertex", record["query_definitions"]["knee"]["formula"])
        self.assertEqual(store.records["M01_R01:C02"]["evidence"], [])
        self.assertTrue(all(a["manual_verdict"] == "pending" for a in store.annotations.values()))
        self.assertFalse(self.state.exists())

    def test_save_resume_export_and_source_immutability(self):
        store = self.store()
        before = {p: Path(p).read_bytes() for p in store.source_hashes}
        store.update_many([{"record_id": "M01_R01:C02", "manual_verdict": "correct", "notes": "=1+1"}], 0)
        resumed = self.store()
        self.assertEqual(resumed.revision, 1)
        self.assertEqual(resumed.annotations["M01_R01:C02"]["manual_verdict"], "correct")
        self.assertEqual(before, {p: Path(p).read_bytes() for p in before})
        rows = list(csv.DictReader(io.StringIO(resumed.export_csv().decode("utf-8-sig"))))
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[1]["notes"], "'=1+1")
        self.assertEqual(rows[1]["support"], "unsupported")
        self.assertEqual(rows[1]["save_status"], "saved")

    def test_invalid_updates_atomic(self):
        store = self.store()
        good = {"record_id": "M01_R01:C01", "manual_verdict": "correct"}
        for bad in (None, [], {}, {**good, "manual_verdict": "wrong"},
                    {**good, "record_id": []}, {**good, "notes": "x" * 4001}):
            with self.subTest(bad=str(bad)[:100]), self.assertRaises(ValueError):
                store.update_many([good, bad], 0)
            self.assertEqual(store.revision, 0)
            self.assertFalse(self.state.exists())
        with self.assertRaises(ValueError):
            store.update_many([good, good], 0)

    def test_optional_judge_never_replaces_annotations_or_base_hashes(self):
        store = self.store()
        store.update_many([{"record_id":"M01_R01:C01", "manual_verdict":"false_positive", "notes":"original"}],0)
        before = self.state.read_bytes()
        records = {rid:{"record_id":rid,"source_text":r["source_text"], "model_verdict":"pass",
                       "triage":"auto_pass_provisional", "audit_sample":False,"human_queue":True,
                       "human_model_disagreement":True, "judgment":{"summary":"pre-review","issues":[]}}
                   for rid,r in store.records.items()}
        overlay = self.root/"overlay.json"
        self.write(overlay,{"schema_version":1,"summary":{"total":6},"source_hashes":store.source_hashes,"records":records})
        judged = PlanReviewStore(self.exp,self.metric,self.state,overlay)
        self.assertEqual(judged.annotations,store.annotations)
        self.assertEqual(judged.dataset_id,store.dataset_id)
        self.assertEqual(self.state.read_bytes(),before)
        rows=list(csv.DictReader(io.StringIO(judged.export_csv().decode("utf-8-sig"))))
        self.assertEqual(rows[0]["manual_verdict"],"false_positive")
        self.assertEqual(rows[0]["model_verdict"],"pass")
        self.assertEqual(judged.client_payload()["judge"],{"total":6})
        overlay.write_text("{}")
        with self.assertRaisesRegex(ValueError,"Judge files changed"):
            judged.update_many([{"record_id":"M01_R01:C01","manual_verdict":"correct"}],1)
        self.assertEqual(self.state.read_bytes(),before)

    def test_revision_conflict_and_disk_failure(self):
        store = self.store()
        update = [{"record_id": "M01_R01:C01", "manual_verdict": "uncertain"}]
        with self.assertRaises(ValueError):
            store.update_many(update, True)
        with patch("aqa3d.check_plan_review._atomic_json", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                store.update_many(update, 0)
        self.assertEqual(store.revision, 0)
        self.assertEqual(store.annotations["M01_R01:C01"]["manual_verdict"], "pending")
        store.update_many(update, 0)
        with self.assertRaises(ValueError):
            store.update_many(update, 0)

    def test_changes_to_sources_block_save_and_resume(self):
        store = self.store()
        update = [{"record_id": "M01_R01:C01", "manual_verdict": "correct"}]
        store.update_many(update, 0)
        path = self.exp / "M01_R01/repair_log.json"
        self.write(path, {"changes": [], "semantic_warnings": ["changed"]})
        with self.assertRaisesRegex(ValueError, "Source files changed"):
            store.update_many(update, 1)
        with self.assertRaisesRegex(ValueError, "sources changed"):
            self.store()

    def test_frozen_state_and_wrong_technique_rejected(self):
        for state in (self.exp / "new.json", self.metric / "new.json"):
            with self.assertRaisesRegex(ValueError, "outside"):
                PlanReviewStore(self.exp, self.metric, state)
        self.request["technique"] = "wrong"
        self.write(self.exp / "M01_R01/input.json", self.request)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.store()

    def test_http_assets_head_save_csrf_and_no_path_traversal(self):
        server = create_server(self.store(), port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            def request(method, path, payload=None, token=None):
                connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
                headers = {"Content-Type": "application/json"}
                if token:
                    headers["X-Review-Token"] = token
                connection.request(method, path, json.dumps(payload) if payload is not None else None, headers)
                response = connection.getresponse()
                status, body = response.status, response.read()
                connection.close()
                return status, body
            status, raw = request("GET", "/api/data")
            self.assertEqual(status, 200)
            token = json.loads(raw)["token"]
            for path in ("/", "/app.js", "/styles.css", "/lucide.js", "/api/image/reference-1"):
                self.assertEqual(request("GET", path)[0], 200)
                self.assertEqual(request("HEAD", path), (200, b""))
            self.assertEqual(request("GET", "/api/image/../../source_manifest_snapshot.json")[0], 404)
            payload = {"revision": 0, "updates": [{"record_id": "M01_R01:C01", "manual_verdict": "correct"}]}
            self.assertEqual(request("POST", "/api/review", payload)[0], 403)
            self.assertEqual(request("POST", "/api/review", payload, token)[0], 200)
            self.assertEqual(request("POST", "/api/review", payload, token)[0], 400)
            status, data = request("GET", "/api/export")
            self.assertEqual(status, 200)
            self.assertIn(b"correct", data)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
