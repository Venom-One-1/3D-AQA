import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from aqa3d.check_plan_review_analysis import classify, group_statistics, read_review, summarize
from analyze_check_plan_review import analyze


def row(number=1, model="pass", manual="correct", audit=False):
    return {"record_id":f"M01_R01:C{number:02d}","case_id":"M01_R01","move_id":"1",
            "clause_id":f"C{number:02d}","source_text":"test clause", "manual_verdict":manual,
            "notes":"", "model_verdict":model,"audit_sample":str(audit).lower(),
            "human_queue":str(model != "pass" or audit).lower(),"human_model_disagreement":"false"}


def write_csv(path, rows):
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ReviewAgreementTests(unittest.TestCase):
    def test_binary_truth_table_and_no_relabeling(self):
        for model,manual,expected in (("pass","correct","agree"),("fail","false_positive","agree"),
                                       ("pass","false_positive","disagree"),("fail","correct","disagree")):
            with self.subTest(model=model,manual=manual):
                source=row(model=model,manual=manual)
                before=copy.deepcopy(source)
                result=classify(source)
                self.assertEqual(result["judge_agreement"],expected)
                self.assertEqual(source,before)
                self.assertEqual(result["manual_verdict"],manual)
                self.assertEqual(result["human_model_disagreement"],"false")

    def test_uncertain_unavailable_pending_never_binary(self):
        for model,manual,reason in (("uncertain","correct","astra_uncertain"),
                                    ("uncertain","false_positive","astra_uncertain"),
                                    ("fail","uncertain","human_uncertain"),
                                    ("pass","pending","human_pending"),
                                    ("unavailable","correct","astra_unavailable")):
            with self.subTest(model=model,manual=manual):
                result=classify(row(model=model,manual=manual))
                self.assertEqual(result["judge_agreement"],"excluded")
                self.assertIn(reason,result["agreement_exclusion"])
                self.assertIsNone(group_statistics([result])["agreement_rate"])

    def test_group_rates_and_denominators(self):
        rows=[]
        counts=[("fail","false_positive",False,24),("fail","correct",False,6),
                ("fail","uncertain",False,1),("uncertain","correct",False,1),
                ("uncertain","false_positive",False,1),("pass","correct",True,6),
                ("pass","false_positive",True,2),("pass","correct",False,12),("pass","pending",False,64)]
        for model,manual,audit,count in counts:
            for _ in range(count):
                rows.append(classify(row(len(rows)+1,model,manual,audit)))
        summary=summarize(rows)
        self.assertEqual(summary["total"],117)
        reject=summary["groups"]["astra_reject"]
        self.assertEqual((reject["total"],reject["comparable"],reject["agree"],reject["excluded"]),(31,30,24,1))
        self.assertEqual(reject["agreement_rate"],.8)
        self.assertEqual(summary["groups"]["random_audit"]["agreement_rate"],.75)
        self.assertEqual(summary["groups"]["other_pass"]["excluded"],64)
        self.assertEqual(summary["human_queue"]["total"],41)
        self.assertEqual(summary["all_records_descriptive_only"]["comparable"],50)

    def test_historical_provenance_not_double_counted(self):
        source=row()
        prior={source["record_id"]:dict(source)}
        self.assertEqual(classify(source,prior)["review_history"],"previously_reviewed_unchanged")
        prior[source["record_id"]]["notes"]="old note"
        self.assertEqual(classify(source,prior)["review_history"],"previously_reviewed_updated")
        prior[source["record_id"]]["manual_verdict"]="pending"
        self.assertEqual(classify(source,prior)["review_history"],"newly_reviewed")
        self.assertEqual(classify(row(manual="pending"),prior)["review_history"],"not_reviewed")

    def test_invalid_rows_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"input.csv"
            for rows in ([row(),row()], [dict(row(),manual_verdict="yes")],
                         [dict(row(),model_verdict="bad")], [dict(row(),audit_sample="maybe")],
                         [row(model="fail",audit=True)], [dict(row(),record_id="wrong")]):
                with self.subTest(rows=rows):
                    write_csv(path,rows)
                    with self.assertRaises(ValueError):read_review(path)
            write_csv(path,[dict(row(),notes="first line, with comma\nsecond line")])
            self.assertEqual(read_review(path)[1][0]["notes"],"first line, with comma\nsecond line")

    def test_export_snapshots_unchanged_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory); source=base/"review.csv";prior=base/"prior.csv";output=base/"analysis"
            write_csv(source,[dict(row(audit=True,manual="false_positive"),notes="=formula"),row(2,manual="pending")])
            write_csv(prior,[row(manual="pending"),row(2,manual="pending")])
            before=source.read_bytes(); old=prior.read_bytes()
            summary=analyze(source,output,prior)
            self.assertEqual(source.read_bytes(),before)
            self.assertEqual(prior.read_bytes(),old)
            self.assertEqual((output/"input_review.csv").read_bytes(),before)
            self.assertEqual((output/"prior_review.csv").read_bytes(),old)
            self.assertEqual(summary["review_history"]["newly_reviewed"],1)
            with (output/"disagreements.csv").open(encoding="utf-8-sig",newline="") as stream:
                exported=list(csv.DictReader(stream))
            self.assertEqual(len(exported),1)
            self.assertEqual(exported[0]["notes"],"'=formula")
            self.assertEqual(exported[0]["judge_agreement"],"disagree")
            self.assertEqual(json.loads((output/"summary.json").read_text())["total"],2)
            self.assertIn("不合并报告为总体准确率",(output/"report.md").read_text())
            with self.assertRaises(FileExistsError):analyze(source,output,prior)
            self.assertEqual(source.read_bytes(),before)


if __name__=="__main__":
    unittest.main()
