import copy
import csv
import contextlib
import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from aqa3d.plan_feedback import compile_packet, direction, extract_response, select_checks, validate_response


def fixture():
    source = {"move_id": 1, "move_name": "qishi", "technique": "technique",
              "clauses": [{"clause_id": "C01", "text": "clause"}]}
    plan = {"checks": [{"clause_id": "C01", "support": "proxy", "limitations": ["proxy only"],
                       "evidence": [{"query_id": "q", "components": ["right"],
                                     "purpose": "invented quality target", "expected_relation": "90 degrees"}]}]}
    reviews = {"M01_R01:C01": {"source_text": "clause", "move_id": "1", "manual_verdict": "correct",
                               "model_verdict": "fail"}}
    values = {"value": -0.8, "window_median": -0.8, "center_value": -0.79,
              "center_minus_window_median": 0.01, "window_min": -0.81, "window_max": -0.79}
    teachers = [{"subject_id": str(i), "status": "valid", "value": -0.2,
                 "center_value": -0.2, "window_min": -0.21, "window_max": -0.19,
                 "valid_frame_count": 7, "requested_frame_count": 7, "endpoint": {}} for i in range(10)]
    reference = {"status": "valid", "median": -0.2, "mad": 0.05,
                 "median_minus_2mad": -0.3, "median_plus_2mad": -0.1,
                 "p10": -0.31, "p90": -0.09, "teacher_values": teachers,
                 "valid_teacher_count": 10}
    measurement = {"student_id": "01", "move_id": 1, "technique": "technique",
        "student_endpoint": {"source_frame_0based": 123}, "reference_anchor": {},
        "measurements": [{"query_id": "q", "student": {
            "definition": {"operation": "relative_position", "points": ["L_Wrist", "Pelvis"], "unit": "ratio"},
            "components": {"right": values}, "status": "valid", "valid_frame_count": 7, "requested_frame_count": 7},
            "teachers": {"components": {"right": reference}}}]}
    return source, plan, reviews, measurement


def packet():
    s, p, r, m = fixture()
    return compile_packet("PF_01_01", select_checks(p, s, r, "M01_R01"), m)


def response():
    return {"case_id": "PF_01_01", "findings": [{
        "finding_id": "P1", "priority": 1, "clause_ids": ["C01"], "observation": "difference",
        "evidence": [{"evidence_id": "q:right", "student_value": -0.8,
                      "teacher_median": -0.2, "p10": -0.31, "p90": -0.09, "direction": "below"}],
        "synthesis": "combined evidence", "why_priority": "relevant", "training_guidance": "check posture",
        "caveat": "proxy only", "certainty": "supported"}],
        "not_selected": [], "coach_summary": "Check posture."}


class PlanFeedbackTests(unittest.TestCase):
    def test_exact_hash_bound_quote_escape_preserves_text(self):
        original = '{"case_id":"x","findings":[],"coach_summary":"check "word" only"}'
        policy = {'source_sha256': hashlib.sha256(original.encode()).hexdigest(),
                  'escape_fragments': ['"word"']}
        parsed, actions = extract_response(original, policy)
        self.assertEqual(parsed['coach_summary'], 'check "word" only')
        self.assertFalse(actions[0]['decoded_text_changed'])
        with self.assertRaises(ValueError):
            extract_response(original+' ', policy)

    def test_extracts_only_one_unchanged_complete_answer(self):
        original = json.dumps(response())
        answer, events = extract_response('preamble\n```json\n'+original+'\n```')
        self.assertEqual(answer, response())
        self.assertFalse(events[0]['json_content_changed'])
        self.assertEqual(extract_response(original)[1], [])
        with self.assertRaises(ValueError):
            extract_response(original+'\n'+original)
        with self.assertRaises(ValueError):
            extract_response('{"case_id":"a","case_id":"b","findings":[],"coach_summary":"x"}')
        with self.assertRaises(ValueError):
            extract_response('incomplete '+original[:-1])

    def test_human_mapping_overrides_judge_but_omits_unvalidated_targets(self):
        s, p, r, m = fixture()
        selected = select_checks(p, s, r, "M01_R01")
        self.assertEqual(len(selected["checks"]), 1)
        self.assertNotIn("expected_relation", str(selected))
        self.assertNotIn("90 degrees", str(selected))
        self.assertNotIn("invented quality", str(selected))

    def test_excludes_human_false_and_uncertain(self):
        for verdict in ("false_positive", "uncertain"):
            s, p, r, m = fixture()
            r["M01_R01:C01"]["manual_verdict"] = verdict
            r["M01_R01:C01"]["model_verdict"] = "pass"
            self.assertEqual(select_checks(p, s, r, "M01_R01")["checks"], [])

    def test_pending_requires_judge_pass(self):
        s, p, r, m = fixture()
        r["M01_R01:C01"]["manual_verdict"] = "pending"
        self.assertEqual(select_checks(p, s, r, "M01_R01")["checks"], [])
        r["M01_R01:C01"]["model_verdict"] = "pass"
        self.assertEqual(select_checks(p, s, r, "M01_R01")["checks"][0]["acceptance"], "astra_pass_provisional")

    def test_unsupported_stays_excluded_even_human_correct(self):
        s, p, r, m = fixture()
        p["checks"][0]["support"] = "unsupported"
        self.assertEqual(select_checks(p, s, r, "M01_R01")["checks"], [])

    def test_wrong_review_text_rejected(self):
        s, p, r, m = fixture()
        r["M01_R01:C01"]["source_text"] = "different clause"
        with self.assertRaises(ValueError):
            select_checks(p, s, r, "M01_R01")

    def test_signed_left_value_and_all_ten_teachers_preserved(self):
        p = packet()
        e = p["evidence"][0]
        self.assertEqual(e["interval_directions"]["p10_p90"], "below")
        self.assertEqual(e["student"]["value"], -0.8)
        self.assertEqual(len(e["teacher_values"]), 10)
        self.assertEqual(p["student_endpoint"]["source_frame_0based"], 123)
        self.assertEqual(validate_response(response(), p), [])

    def test_missing_component_fails_closed(self):
        s, p, r, m = fixture()
        p["checks"][0]["evidence"][0]["components"] = ["left"]
        with self.assertRaises(ValueError):
            compile_packet("case", select_checks(p, s, r, "M01_R01"), m)

    def test_stability_warns_without_changing_values_or_deleting_teacher(self):
        s, p, r, m = fixture()
        m["measurements"][0]["teachers"]["components"]["right"]["teacher_values"][0]["center_value"] = 0.5
        data = compile_packet("PF_01_01", select_checks(p, s, r, "M01_R01"), m)
        e = data["evidence"][0]
        self.assertIn("unstable_teacher_reference", e["warnings"])
        self.assertTrue(e["usable"])
        self.assertFalse(e["supported_claim_allowed"])
        self.assertTrue(validate_response(response(), data))
        a = response(); a["findings"][0]["certainty"] = "tentative"
        self.assertEqual(validate_response(a, data), [])

    def test_degenerate_interval_not_repaired_into_fake_range(self):
        s, p, r, m = fixture()
        ref = m["measurements"][0]["teachers"]["components"]["right"]
        ref["p90"] = ref["p10"]
        data = compile_packet("case", select_checks(p, s, r, "M01_R01"), m)
        self.assertFalse(data["evidence"][0]["usable"])
        self.assertEqual(direction(-0.3, -0.2, -0.2), "unavailable")

    def test_numeric_direction_and_scope_citations_are_checked(self):
        for key, value in (("student_value", 3), ("p90", float('nan')), ("direction", "above"),
                           ("evidence_id", "unknown")):
            a = response(); a["findings"][0]["evidence"][0][key] = value
            self.assertTrue(validate_response(a, packet()), key)
        a = response(); a["findings"][0]["clause_ids"] = ["C99"]
        self.assertTrue(validate_response(a, packet()))

    def test_no_finding_is_valid_not_forced_to_diagnose(self):
        a = response(); a["findings"] = []
        self.assertEqual(validate_response(a, packet()), [])

    def test_duplicate_findings_and_over_budget_rejected(self):
        a = response(); b = copy.deepcopy(a['findings'][0]); b.update(finding_id="P2", priority=2)
        a['findings'].append(b)
        self.assertTrue(validate_response(a, packet()))
        a['findings'] *= 2
        self.assertTrue(validate_response(a, packet()))

    def test_non_numeric_bool_not_accepted(self):
        a = response(); a['findings'][0]['evidence'][0]['student_value'] = True
        self.assertTrue(validate_response(a, packet()))

    def test_permission_is_not_mandatory_even_if_review_accepts(self):
        s, p, r, m = fixture()
        s['move_id'] = 3
        s['clauses'][0]['clause_id'] = 'C12'
        p['checks'][0]['clause_id'] = 'C12'
        row = r.pop('M01_R01:C01'); row['move_id'] = '3'
        r['M03_R01:C12'] = row
        selected = select_checks(p, s, r, 'M03_R01')
        self.assertEqual(selected['checks'], [])
        self.assertIn('permission', selected['excluded_clauses'][0]['reason'])

    def test_report_regeneration_preserves_manual_reviews(self):
        from run_plan_feedback_experiment import analyze, sha, write_csv, write_json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = packet()
            p['move_name'] = 'qishi'
            p['plan_id'] = 'M01_R01'
            p['reference_anchor']['boundary_time_seconds'] = 16
            p['student_endpoint']['boundary_time_seconds'] = 20
            entry = {'case_id': 'PF_01_01', 'student_id': '01', 'move_id': 1,
                     'input_file': 'inputs/case.json', 'response_file': 'responses/case.json'}
            write_json(root/entry['input_file'], p)
            write_json(root/entry['response_file'], response())
            write_json(root/'api_calls/PF_01_01/status.json', {
                'status': 'completed', 'response_sha256': sha(root/entry['response_file']),
                'attempts': [{'model': 'test-model'}]})
            write_json(root/'comparison_snapshot.json', {'PF_01_01': {
                'old_f_response': {'coach_summary': 'old summary', 'findings': []},
                'rule_baseline_summary': 'baseline summary', 'rule_baseline': []}})
            write_json(root/'frozen_plans.json', {'1': {
                'move_name': 'qishi', 'checks': p['checks'], 'excluded_clauses': []}})
            manifest = {'cases': [entry], 'api_config': {'model': 'test-model'},
                        'old_f_model': 'old-model', 'source_sha256': {}}
            with contextlib.redirect_stdout(io.StringIO()):
                summary = analyze(root, manifest)
            self.assertEqual(summary['valid_responses'], 1)
            path = root/'manual_feedback_review.csv'
            with path.open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))
            rows[0]['manual_verdict'] = 'correct'
            rows[0]['notes'] = 'keep my note'
            write_csv(path, rows)
            with contextlib.redirect_stdout(io.StringIO()):
                analyze(root, manifest)
            with path.open(encoding='utf-8-sig', newline='') as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual(saved[0]['manual_verdict'], 'correct')
            self.assertEqual(saved[0]['notes'], 'keep my note')
            self.assertEqual(len(saved), len(rows))
            self.assertIn('old summary', (root/'reports/PF_01_01.md').read_text())


if __name__ == "__main__":
    unittest.main()
