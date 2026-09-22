import copy
import json
from pathlib import Path
import tempfile
import unittest

from aqa3d.endpoint_metrics import MOVE_METRICS
from run_f_group_experiment import make_case, relation, validate_response, validate, write_json, digest


def fixture():
    mids=MOVE_METRICS[1]
    event={"move_id":1,"pose_id":"1.end","move_name_pinyin":"qishi","move_name_zh":"起势",
           "final_technique_step":"两腿屈膝，两掌下按。"}
    rows={};teachers=[f"PRIVATE_TEACHER_{i}" for i in range(10)]
    for code in ["student"]+teachers:
        for metric in mids:
            value=120. if code=="student" else 100.+teachers.index(code)
            rows[(code,metric)]={"pose_id":"1.end","final_technique_step":event["final_technique_step"],
                "unit":"degree","status":"valid","value":value,"window_median":value,"center_value":value,
                "window_min":value-.1,"window_max":value+.1,"valid_frame_count":7,"requested_frame_count":7,
                "feedback":"SECRET_BASELINE_TEXT","direction":"above","subject_id":code,"image":"/private/path.jpg"}
    defs={m:{"unit":"degree","label_zh":m,"formula":"angle"} for m in mids}
    summary={"minimum_teachers":7,"window_seconds_each_side":.2,"minimum_valid_frames":3,"minimum_window_valid_ratio":.7}
    return make_case("F001",event,defs,{},"body axes",rows,teachers,summary)


def answer(case):
    comparisons=[]
    for m in case["metrics"]:
        r={"metric_id":m["metric_id"],"student_value":m["student"]["value"]}
        for method in ("median_2mad","p10_p90"):
            bounds=m["teacher"][method]
            r[method+"_interval"]=[bounds["lower"],bounds["upper"]]
            r[method+"_relation"]=relation(m["student"]["value"],bounds,10)
        comparisons.append(r)
    return {"case_id":case["case_id"],"metric_comparisons":comparisons,"findings":[],"coach_summary":"尚需结合数据限制复核。"}


class FGroupTests(unittest.TestCase):
    def test_case_is_whitelisted_complete_and_retains_all_teachers(self):
        case=fixture();text=json.dumps(case)
        self.assertEqual(len(case["metrics"]),9)
        for m in case["metrics"]:
            self.assertEqual(len(m["teacher"]["samples"]),10)
            self.assertEqual(m["teacher"]["median"],104.5)
            self.assertEqual(m["teacher"]["mad_unscaled"],2.5)
        for excluded in ("PRIVATE_TEACHER","SECRET_BASELINE_TEXT","/private/path.jpg","direction","subject_id"):
            self.assertNotIn(excluded,text)

    def test_boundary_inclusive_and_degenerate_not_evaluable(self):
        self.assertEqual(relation(1,{"lower":1,"upper":2},10),"within")
        self.assertEqual(relation(2,{"lower":1,"upper":2},10),"within")
        self.assertEqual(relation(3,{"lower":1,"upper":2},10),"above")
        for value,interval,count in [(None,{"lower":1,"upper":2},10),(1,{"lower":1,"upper":1},10),(1,{"lower":0,"upper":2},6)]:
            self.assertEqual(relation(value,interval,count),"not_evaluable")

    def test_correct_answer_and_incorrect_numeric_quotes(self):
        case=fixture();response=answer(case)
        errors,checks=validate_response(case,response)
        self.assertEqual(errors,[]);self.assertEqual(len(checks),18)
        response["metric_comparisons"][0]["student_value"]=119
        errors,checks=validate_response(case,response)
        self.assertTrue(errors);self.assertFalse(checks[0]["numbers_correct"])

    def test_unknown_duplicate_missing_metrics_and_fabricated_evidence(self):
        case=fixture();response=answer(case)
        response["metric_comparisons"][0]=copy.deepcopy(response["metric_comparisons"][1])
        response["findings"]=[{"finding_id":"P1","assessment":"possible_issue","problem":"p",
            "evidence_metric_ids":["nonexistent"],"explanation":"e","uncertainty":"u","guidance":"g"}]
        errors,_=validate_response(case,response)
        self.assertTrue(any("duplicate" in e for e in errors))
        self.assertTrue(any("Missing metric" in e for e in errors))
        self.assertTrue(any("Invalid evidence" in e for e in errors))

    def test_abstention_cannot_give_training_advice(self):
        case=fixture();response=answer(case)
        response["findings"]=[{"finding_id":"P1","assessment":"insufficient_evidence","problem":"p",
            "evidence_metric_ids":[case["metrics"][0]["metric_id"]],"explanation":"e","uncertainty":"u","guidance":"g"}]
        errors,_=validate_response(case,response)
        self.assertTrue(any("Abstention" in e for e in errors))

    def test_missing_responses_not_success_and_partial_validation_keeps_raw(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/"inputs").mkdir();(root/"responses").mkdir();(root/"evaluation").mkdir()
            (root/"F_prompt.md").write_text("prompt")
            case=fixture();write_json(root/"inputs/F001.json",case)
            write_json(root/"evaluation/baseline_snapshot.json",[])
            write_json(root/"experiment_manifest.json",{"prompt_sha256":digest(root/"F_prompt.md"),
                "cases":[{"case_id":"F001","student_id":"01","move_id":1,"move_name":"qishi","metric_count":9,
                    "input_file":"inputs/F001.json","response_file":"responses/F001.json","input_sha256":digest(root/"inputs/F001.json")}]})
            result=validate(root,root/"empty_check")
            self.assertEqual(result["status"],"awaiting_responses")
            self.assertIsNone(result["diagnosis_accuracy"])
            response=root/"responses/F001.json";write_json(response,answer(case));raw=response.read_bytes()
            result=validate(root,root/"answered_check")
            self.assertEqual(result["correct_relations"],18)
            self.assertEqual(response.read_bytes(),raw)
            with self.assertRaises(FileExistsError):validate(root,root/"answered_check")


if __name__=="__main__":
    unittest.main()
