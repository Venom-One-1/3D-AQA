import copy
import json
from pathlib import Path
import tempfile
import unittest

from aqa3d.check_plan_judge import DIMENSIONS, model_verdict, probe_plan, route_check, signed_axis_probes, validate_judgment
from run_check_plan_judge import analyze, audit_sample, dump, import_answers, prepare, read, sha


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.request = {"move_id":1, "move_name":"qishi", "scope":"endpoint_geometry_only",
                        "technique":"bend knee", "clauses":[{"clause_id":"C01", "text":"bend knee"}]}
        self.plan = {"schema_version":"1.0", "move_id":1, "scope":"endpoint_geometry_only",
                     "queries":[{"query_id":"knee", "operation":"joint_angle", "points":["L_Hip","L_Knee","L_Ankle"]}],
                     "checks":[{"clause_id":"C01", "support":"direct", "endpoint_interpretation":"bend knee",
                                "evidence":[{"query_id":"knee", "components":["value"], "purpose":"knee bend", "expected_relation":"below 180"}],
                                "limitations":["not weight bearing"]}]}
        self.packet = {"case_id":"J001", "request":self.request, "candidate_plan":self.plan}
        self.answer = {"schema_version":"1.0", "case_id":"J001", "checks":[{"clause_id":"C01", "summary":"OK",
                        "issues":[], "dimensions":{d:{"verdict":"pass", "reason":"supported", "query_ids":["knee"]} for d in DIMENSIONS}}]}

    def test_strict_judge_schema_and_grounding(self):
        validate_judgment(json.dumps(self.answer), self.packet)
        for change in (lambda a:a.update(case_id="other"), lambda a:a["checks"].clear(),
                       lambda a:a["checks"][0]["dimensions"]["direction"].update(query_ids=["unknown"]),
                       lambda a:a["checks"][0]["dimensions"]["coverage"].update(verdict="uncertain")):
            a=copy.deepcopy(self.answer); change(a)
            with self.assertRaises(ValueError): validate_judgment(json.dumps(a), self.packet)
        a=copy.deepcopy(self.answer)
        a["checks"][0]["dimensions"]["direction"]["verdict"]="fail"
        issue={"dimension":"direction", "code":"coordinate_sign", "query_ids":["knee"],
               "source_quote":"bend", "plan_quote":"below 180", "reason":"test issue"}
        a["checks"][0]["issues"]=[issue]
        validate_judgment(json.dumps(a),self.packet)
        issue["plan_quote"]="invented"
        with self.assertRaisesRegex(ValueError,"quoted text"):validate_judgment(json.dumps(a),self.packet)

    def test_numeric_mirror_and_axis_probes(self):
        probes=probe_plan(self.plan)+signed_axis_probes()
        self.assertTrue(all(p["status"]=="pass" for p in probes))
        vector={"query_id":"elbow", "operation":"relative_position", "points":["L_Elbow","L_Shoulder"],
                "normalizer":"torso_length", "coordinate_frame":"torso_v1"}
        self.assertTrue(all(p["status"]=="pass" for p in probe_plan({"queries":[vector]})))

    def test_route_fail_closed(self):
        check=self.answer["checks"][0]
        self.assertEqual(route_check(True,[],check)[0],"auto_pass_provisional")
        self.assertEqual(route_check(True,[],None)[0],"pending_judge")
        self.assertEqual(route_check(False,[],check)[0],"auto_return")
        self.assertEqual(route_check(True,[{"status":"fail"}],check)[0],"auto_return")
        self.assertEqual(route_check(True,[{"status":"not_testable"}],check)[0],"needs_review")
        check["dimensions"]["direction"]["verdict"]="fail"
        self.assertEqual(model_verdict(check),"fail")
        self.assertEqual(route_check(True,[],check)[0],"needs_review")

    def test_sampling_deterministic_stratified_only_passes(self):
        rows=[{"record_id":str(i),"move_id":i%3,"triage":"auto_pass_provisional" if i<20 else "needs_review"} for i in range(25)]
        a=audit_sample(rows,.1,5)
        self.assertEqual(a,audit_sample(list(reversed(rows)),.1,5))
        self.assertEqual(len(a),3)
        self.assertTrue(all(int(x)<20 for x in a))
        with self.assertRaises(ValueError):audit_sample(rows,1.1,1)

    def test_pipeline_immutable_blind_and_missing_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);source=base/"source";root=base/"review";answers=base/"answers"
            dump(source/"source_manifest_snapshot.json",{"cases":[{"case_id":"M01_R01","move_id":1}]})
            dump(source/"M01_R01/input.json",self.request);dump(source/"M01_R01/plan.json",self.plan)
            prompt=base/"prompt.md";prompt.write_text("judge rubric")
            prepare(source,root,prompt,4)
            self.assertEqual(set(read(root/"inputs/J001.json")),{"case_id","request","candidate_plan"})
            missing=analyze(root,base/"missing")
            self.assertEqual(missing["triage"],{"pending_judge":1})
            dump(answers/"J001.json",self.answer)
            self.assertEqual(import_answers(root,answers),{"J001":"valid"})
            snapshot={p:sha(p) for p in source.rglob("*.json")}
            summary=analyze(root,base/"analysis")
            self.assertEqual(summary["triage"],{"auto_pass_provisional":1})
            self.assertEqual(summary["audit_sample_count"],1)
            self.assertEqual(snapshot,{p:sha(p) for p in snapshot})
            with self.assertRaisesRegex(ValueError,"exists"):analyze(root,base/"analysis")
            (source/"M01_R01/plan.json").write_text("{}")
            with self.assertRaisesRegex(ValueError,"changed"):analyze(root,base/"changed")


if __name__=="__main__":unittest.main()
