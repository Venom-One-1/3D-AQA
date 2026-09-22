import csv
from http.client import HTTPConnection
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest

from PIL import Image

from aqa3d.feedback_review import FeedbackReviewStore
from run_feedback_review_app import create_server


FIELDS = ["record_id", "student_id", "move_id", "move_name", "metric_id",
          "manual_verdict", "suspected_cause", "reviewer", "notes"]


def feedback(record_id, decision, metric="left_knee_angle", move=1):
    student, _, _ = record_id.partition(":")
    return {"record_id":record_id,"subject_id":student,"move_id":move,"move_name_zh":"起势",
        "move_name_pinyin":"qishi","final_technique_step":"屈膝下蹲，两掌下按。",
        "metric_id":metric,"metric_label_zh":"左膝夹角","technique_aspect":"两腿屈膝",
        "unit":"degree","decision":decision,"decision_zh":"待确认",
        "direction":"above","review_reasons_zh":["窗口波动"],"feedback":"请检查屈膝幅度。",
        "limitation":"几何代理","teacher_reference":{"median":120.,"mad":3.,
        "median_minus_2mad":114.,"median_plus_2mad":126.,"p10":115.,"p90":125.,
        "valid_teacher_count":10,"status":"valid"},
        "student_evidence":{"value":130.,"center_value":129.,"boundary_time_seconds":16.,
        "center_source_frame_0based":480}}


def build_fixture(root: Path):
    result = root/"endpoint_feedback_results/fixture";result.mkdir(parents=True)
    metric = root/"endpoint_metric_results/fixture"
    records=[feedback("01:1.end:left_knee_angle","feedback_candidate"),
             feedback("01:1.end:right_knee_angle","needs_review","right_knee_angle"),
             feedback("01:1.end:left_elbow_angle","within_reference","left_elbow_angle"),
             feedback("01:1.end:right_elbow_angle","observed_difference","right_elbow_angle")]
    (result/"feedback.json").write_text(json.dumps(records),encoding="utf-8")
    with (result/"manual_review.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS);writer.writeheader()
        for record in records:
            writer.writerow({"record_id":record["record_id"],"student_id":"01","move_id":"1",
                "move_name":"qishi","metric_id":record["metric_id"],"manual_verdict":"pending",
                "suspected_cause":"","reviewer":"","notes":""})
    for path in (metric/"student/01/01_endpoint.jpg",metric/"figures/01_reference.jpg",
                 metric/"figures/01_teacher_keyposes.jpg"):
        path.parent.mkdir(parents=True,exist_ok=True);Image.new("RGB",(32,24),(10,80,120)).save(path)
    return result


class FeedbackReviewTests(unittest.TestCase):
    def test_filters_review_scope_and_exports_full_compatible_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            store=FeedbackReviewStore(build_fixture(Path(temporary)))
            self.assertEqual(len(store.all_records),4)
            self.assertEqual(len(store.review_records),2)
            self.assertEqual(store.progress()["total"],2)
            store.update("01:1.end:left_knee_angle","correct","","tester","looks right")
            exported=list(csv.DictReader(io.StringIO(store.export_csv().decode("utf-8-sig"))))
            self.assertEqual(len(exported),4)
            self.assertEqual(exported[0]["manual_verdict"],"correct")
            self.assertEqual(exported[2]["manual_verdict"],"pending")

    def test_progress_resumes_without_modifying_source_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            result=build_fixture(Path(temporary));source=result.joinpath("manual_review.csv").read_bytes()
            store=FeedbackReviewStore(result)
            store.update("01:1.end:right_knee_angle","uncertain","reconstruction","A","occluded")
            self.assertEqual(result.joinpath("manual_review.csv").read_bytes(),source)
            resumed=FeedbackReviewStore(result)
            annotation=resumed.annotations["01:1.end:right_knee_angle"]
            self.assertEqual(annotation["manual_verdict"],"uncertain")
            self.assertEqual(resumed.progress()["completed"],1)

    def test_rejects_unknown_records_enums_and_stale_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            result=build_fixture(Path(temporary));store=FeedbackReviewStore(result)
            for args in (("unknown","correct","","", ""),
                         ("01:1.end:left_knee_angle","yes","","", ""),
                         ("01:1.end:left_knee_angle","correct","bad-cause","", "")):
                with self.assertRaises(ValueError):store.update(*args)
            store.update("01:1.end:left_knee_angle","correct","","","")
            records=json.loads((result/"feedback.json").read_text());records[0]["feedback"]="changed"
            (result/"feedback.json").write_text(json.dumps(records))
            with self.assertRaisesRegex(ValueError,"changed after review began"):
                FeedbackReviewStore(result)

    def test_client_payload_uses_whitelisted_images_and_no_raw_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            store=FeedbackReviewStore(build_fixture(Path(temporary)));payload=store.client_payload()
            self.assertEqual(len(payload["records"]),2)
            self.assertEqual(payload["records"][0]["images"]["reference"],"/api/image/reference-1")
            self.assertEqual(set(store.image_paths),{"student-01-1","reference-1","teachers-1"})

    def test_http_review_and_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            server,store=create_server(build_fixture(Path(temporary)),"127.0.0.1",0)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                connection=HTTPConnection("127.0.0.1",server.server_port,timeout=3)
                connection.request("GET","/api/data");response=connection.getresponse()
                self.assertEqual(response.status,200);self.assertEqual(len(json.loads(response.read())["records"]),2)
                body=json.dumps({"record_id":"01:1.end:left_knee_angle","manual_verdict":"false_positive",
                    "suspected_cause":"rule","reviewer":"B","notes":"not supported"})
                connection.request("POST","/api/review",body=body,headers={"Content-Type":"application/json"})
                response=connection.getresponse();self.assertEqual(response.status,200);response.read()
                connection.request("GET","/api/export");response=connection.getresponse()
                self.assertEqual(response.status,200);rows=list(csv.DictReader(io.StringIO(response.read().decode("utf-8-sig"))))
                self.assertEqual(rows[0]["manual_verdict"],"false_positive")
                connection.request("GET","/api/image/reference-1");response=connection.getresponse()
                self.assertEqual(response.status,200);self.assertGreater(len(response.read()),20)
                self.assertEqual(store.progress()["completed"],1)
            finally:
                server.shutdown();server.server_close();thread.join(timeout=3)


if __name__=="__main__":
    unittest.main()
