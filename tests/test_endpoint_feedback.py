import csv
import json
from pathlib import Path
import tempfile
import unittest
import re
import shutil

from PIL import Image

from aqa3d.endpoint_feedback import (
    FeedbackConfig, coach_summary, evaluate_feedback, make_rules, stability_reasons, validate_rules,
)
from aqa3d.endpoint_metrics import MOVE_METRICS


def row(value=120., center=None, subject="01", mid=1, metric="left_knee_angle"):
    return {"subject_id":subject,"role":"student","move_id":mid,"pose_id":f"{mid}.end",
        "metric_id":metric,"metric_label_zh":metric,"move_name_zh":"测试招式","move_name_pinyin":"test",
        "unit":"degree","status":"valid","value":value,"window_median":value,
        "center_value":value if center is None else center,
        "window_min":min(value,value if center is None else center)-.1,
        "window_max":max(value,value if center is None else center)+.1,
        "valid_frame_count":7,"requested_frame_count":7,"final_technique_step":f"technique {mid}",
        "boundary_time_seconds":1.,"source_frame_time_seconds":1.,"center_source_frame_0based":30,
        "center_phalp_frame_1based":31,"window_start_source_frame_0based":24,
        "window_end_source_frame_0based":30,"image":""}


def teachers():
    result = [row(100.+i,subject=f"teacher{i}") for i in range(10)]
    for r in result:
        r["role"]="teacher"
    return result


def metric_rules():
    return {"poses":[{"move_id":mid,"pose_id":f"{mid}.end","final_technique_step":f"technique {mid}",
        "checks":[{"metric_id":metric,"technique_aspect":"test aspect"} for metric in metrics]}
        for mid,metrics in MOVE_METRICS.items()],
        "metric_definitions":{metric:{"unit":"degree"} for metrics in MOVE_METRICS.values() for metric in metrics}}


class EndpointFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.config=FeedbackConfig()
        self.rules=validate_rules(make_rules(metric_rules()),metric_rules())
        self.rule=self.rules[(1,"left_knee_angle")]

    def evaluate(self, student=None, reference=None, rule=None):
        return evaluate_feedback(student or row(),reference or teachers(),rule or self.rule,self.config)

    def test_stable_supported_direction_produces_candidate_not_confirmed_error(self):
        result=self.evaluate()
        self.assertEqual(result["decision"],"feedback_candidate")
        self.assertEqual(result["manual_review"]["verdict"],"pending")
        self.assertIn("可能",result["feedback"])

    def test_within_is_not_overall_pass(self):
        result=self.evaluate(row(105.))
        self.assertEqual(result["decision"],"within_reference")
        self.assertEqual(result["feedback"],"")
        self.assertIn("不代表动作已经标准",coach_summary([result]))

    def test_unsupported_direction_records_difference_only(self):
        result=self.evaluate(row(90.))
        self.assertEqual(result["decision"],"observed_difference")
        self.assertFalse(result["feedback"])

    def test_interval_disagreement_abstains(self):
        result=self.evaluate(row(109.))
        self.assertIn("interval_disagreement",result["review_reasons"])
        self.assertFalse(result["feedback"])

    def test_center_value_never_replaces_window_median(self):
        result=self.evaluate(row(120.,center=104.))
        self.assertEqual(result["direction"],"above")
        self.assertEqual(result["center_directions"]["median_2mad"],"within")
        self.assertIn("center_shift",result["review_reasons"])
        self.assertIn("center_direction_disagreement",result["review_reasons"])
        self.assertEqual(result["decision"],"needs_review")

    def test_window_spread_abstains_even_if_center_is_stable(self):
        student=row(); student["window_max"]=140.
        self.assertIn("window_spread",self.evaluate(student)["review_reasons"])

    def test_missing_center_and_nan_do_not_generate_feedback(self):
        for field in ("center_value","value","window_min"):
            student=row(); student[field]=float("nan")
            self.assertEqual(self.evaluate(student)["decision"],"needs_review")

    def test_insufficient_window_abstains(self):
        student=row();student["valid_frame_count"]=2
        self.assertIn("invalid_window",self.evaluate(student)["review_reasons"])

    def test_teacher_warning_preserves_all_values_and_blocks_suggestion(self):
        reference=teachers();reference[0]["window_max"]=140.
        result=self.evaluate(reference=reference)
        self.assertEqual(result["teacher_reference"]["valid_teacher_count"],10)
        self.assertEqual(result["teacher_reference"]["median"],104.5)
        self.assertIn("unstable_teacher_reference",result["review_reasons"])
        self.assertEqual(len(result["teacher_evidence"]),10)
        self.assertFalse(result["feedback"])

    def test_insufficient_teachers_and_zero_width_abstain(self):
        short=self.evaluate(reference=teachers()[:6])
        self.assertIn("invalid_reference",short["review_reasons"])
        constant=teachers()
        for r in constant:
            r.update(value=100.,center_value=100.,window_min=100.,window_max=100.)
        self.assertIn("invalid_reference",self.evaluate(reference=constant)["review_reasons"])

    def test_ratio_threshold_is_independent_of_angle_threshold(self):
        r=row(1.,center=1.06);r["unit"]="ratio"
        self.assertIn("center_shift",stability_reasons(r,self.config))
        r["unit"]="degree"
        self.assertNotIn("center_shift",stability_reasons(r,self.config))

    def test_baihe_hand_above_teacher_is_not_lowering_instruction(self):
        rule=self.rules[(3,"right_wrist_head_up_torso_ratio")]
        self.assertNotIn("above",rule["feedback_by_direction"])
        rule=self.rules[(3,"right_elbow_angle")]
        self.assertIn("不等于",rule["feedback_by_direction"]["above"])

    def test_wrist_hip_distance_is_not_bound_to_stance_instruction(self):
        self.assertIn("右手落于右胯旁", self.rules[(2,"right_wrist_hip_distance_torso_ratio")]["technique_aspect"])
        self.assertIn("左手落于左胯前", self.rules[(3,"left_wrist_hip_distance_torso_ratio")]["technique_aspect"])

    def test_summary_contains_only_eligible_feedback(self):
        candidate=self.evaluate(); withheld=self.evaluate(row(120.,center=100.))
        withheld["feedback"]="BLOCKED_TEXT"
        text=coach_summary([candidate,withheld])
        self.assertIn(candidate["feedback"],text)
        self.assertNotIn("BLOCKED_TEXT",text)

    def test_rules_reject_duplicates_missing_rules_and_wrong_endpoint_text(self):
        rules=make_rules(metric_rules());rules["rules"].append(rules["rules"][0])
        with self.assertRaises(ValueError):validate_rules(rules,metric_rules())
        rules=make_rules(metric_rules());rules["rules"].pop()
        with self.assertRaises(ValueError):validate_rules(rules,metric_rules())
        rules=make_rules(metric_rules());rules["rules"][0]["final_technique_step"]="old image pose"
        with self.assertRaises(ValueError):validate_rules(rules,metric_rules())

    def test_configuration_rejects_invalid_thresholds(self):
        for args in ({"center_delta_degree":-1},{"minimum_valid_ratio":2},{"window_spread_ratio":float("nan")}):
            with self.assertRaises(ValueError):FeedbackConfig(**args)

    def test_export_integration_is_complete_and_does_not_overwrite_review(self):
        from run_endpoint_feedback import run, refresh_reports
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);src=root/"input";src.mkdir();out=root/"output"
            metadata={"teacher_ids":[f"teacher{i}" for i in range(10)],"reference_manifest_sha256":"fixture"}
            (src/"summary.json").write_text(json.dumps(metadata))
            (src/"endpoint_metric_rules_first3.json").write_text(json.dumps(metric_rules()))
            (src/"reference_manifest_snapshot.json").write_text(json.dumps({"endpoint_keyposes":metric_rules()["poses"]}))
            for mid in (1,2,3):
                for name in (f"student/01/{mid:02d}_endpoint.jpg",
                             f"figures/{mid:02d}_student_keyposes.jpg",
                             f"figures/{mid:02d}_teacher_keyposes.jpg"):
                    path=src/name;path.parent.mkdir(parents=True,exist_ok=True)
                    Image.new("RGB",(40,30),(mid*50,80,120)).save(path)
            rows=[]
            for mid,metrics in MOVE_METRICS.items():
                for metric in metrics:
                    rows.append(row(mid=mid,metric=metric))
                    for i in range(10):
                        r=row(100.+i,subject=f"teacher{i}",mid=mid,metric=metric)
                        r["role"]="teacher";rows.append(r)
            with (src/"all_endpoint_metrics.csv").open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            result=run(src,out,["01"],self.config)
            self.assertEqual(result["record_count"],26)
            self.assertTrue((out/"01/feedback_report.md").exists())
            records=json.loads((out/"feedback.json").read_text())
            self.assertEqual(len(records[0]["teacher_evidence"]),10)
            with (out/"manual_review.csv").open(encoding="utf-8-sig") as f:
                self.assertEqual({r["manual_verdict"] for r in csv.DictReader(f)},{"pending"})
            manual=out/"manual_review.csv"
            manual.write_text(manual.read_text(encoding="utf-8-sig").replace("pending","correct"),encoding="utf-8-sig")
            preserved={p:p.read_bytes() for p in out.rglob("*") if p.suffix in (".csv",".json")}
            refresh_reports(src,out,["01"])
            for path,data in preserved.items():
                self.assertEqual(path.read_bytes(),data)
            report=(out/"01/feedback_report.md").read_text()
            links=re.findall(r"!\[[^\]]*\]\(([^)]+)\)",report)
            self.assertEqual(len(links),9)
            self.assertNotIn(str(src),report)
            relocated=root/"moved report with spaces";shutil.copytree(out,relocated)
            for link in links:
                self.assertFalse(Path(link).is_absolute())
                with Image.open(relocated/"01"/link) as image:
                    image.load()
            refresh_reports(root/"missing input",relocated,["01"])
            with self.assertRaises(FileExistsError):run(src,out,["01"],self.config)
            rows.append(rows[0])
            with (src/"all_endpoint_metrics.csv").open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            with self.assertRaisesRegex(ValueError,"Duplicate"):run(src,root/"duplicate",["01"],self.config)

    def test_report_images_relocates_stale_evidence_paths(self):
        from run_endpoint_feedback import report_images
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);src=root/"input";out=root/"output/01"
            records=[]
            for mid in (1,2,3):
                path=src/"student/01"/f"{mid:02d}_endpoint.jpg"
                path.parent.mkdir(parents=True,exist_ok=True)
                Image.new("RGB",(10,10)).save(path)
                records.append({"move_id":mid,"subject_id":"01","student_evidence":{"image":"/old/server/absent.jpg"}})
            links=report_images(records,src,out)
            self.assertEqual(len(links),3)
            for link in links.values():self.assertTrue((out/link).is_file())

    def test_report_images_rejects_corrupt_jpeg(self):
        from run_endpoint_feedback import report_images
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);bad=root/"bad.jpg";bad.write_bytes(b"not a JPEG")
            records=[{"move_id":mid,"subject_id":"01","student_evidence":{"image":str(bad)}} for mid in (1,2,3)]
            with self.assertRaises(OSError):report_images(records,root/"input",root/"output/01")


if __name__=="__main__":
    unittest.main()
