import csv
import hashlib
from http.client import HTTPConnection
import io
import json
from pathlib import Path
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch

from aqa3d.plan_feedback_review import PlanFeedbackReviewStore
from run_plan_feedback_review_app import create_server, main
from test_plan_feedback import packet, response


class PlanReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'experiment'
        self.state = Path(self.temp.name)/'state.json'
        self.root.mkdir()
        self.packet = packet()
        self.packet.update(plan_id='M01_R01', move_name='qishi')
        self.packet['student_endpoint']['boundary_time_seconds'] = 20
        self.packet['reference_anchor'] = {'reference_video_id':'BV1WE411W7JB','boundary_time_seconds':16}
        self.answer = response()
        self.write('inputs/case.json', self.packet)
        self.write('responses/case.json', self.answer)
        raw = (self.root/'responses/case.json').read_bytes()
        self.write('api_calls/PF_01_01/status.json', {'status':'completed','response_sha256':hashlib.sha256(raw).hexdigest()})
        self.write('comparison_snapshot.json', {'PF_01_01': {
            'old_f_case_id':'F001', 'old_f_response':{'case_id':'F001','coach_summary':'old summary',
                'findings':[{'finding_id':'P1','problem':'old problem','evidence_metric_ids':['angle']}]},
            'rule_baseline':[{'subject_id':'01','move_id':1,'final_technique_step':self.packet['technique'],
                'metric_id':'angle','metric_label_zh':'angle','unit':'degree','student_evidence':{},'teacher_reference':{}}]}})
        for role in ('reference','student'):
            p=self.root/f'images/PF_01_01_{role}.jpg';p.parent.mkdir(exist_ok=True);p.write_bytes(b'image fixture')
        self.write('experiment_manifest.json', {'cases':[{'case_id':'PF_01_01','student_id':'01','move_id':1,
            'input_file':'inputs/case.json','response_file':'responses/case.json'}],
            'api_config':{'model':'new-model'},'old_f_model':'old-model','frozen_sha256':{}})

    def write(self, name, obj):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj),encoding='utf-8')

    def store(self):
        return PlanFeedbackReviewStore(self.root, self.state)

    def test_separate_findings_and_summaries_fresh_annotations(self):
        s=self.store()
        self.assertEqual(len(s.records),4)
        self.assertEqual({r['source'] for r in s.records.values()},{'plan_feedback','F'})
        self.assertEqual({r['review_scope'] for r in s.records.values()},{'finding','coach_summary'})
        self.assertTrue(all(a['manual_verdict']=='pending' for a in s.annotations.values()))
        self.assertFalse(self.state.exists())

    def test_flagged_response_is_not_omitted_or_prelabelled(self):
        self.packet['evidence'][0]['supported_claim_allowed']=False
        self.packet['evidence'][0]['warnings']=['unstable_teacher_reference']
        self.write('inputs/case.json',self.packet)
        s=self.store()
        self.assertEqual(len(s.records),4)
        self.assertTrue(s.cases[0]['validation_errors'])
        self.assertEqual(s.annotations['PF_01_01:plan_feedback:finding:P1']['manual_verdict'],'pending')

    def test_save_reload_export_and_unchanged_sources(self):
        s=self.store();rid='PF_01_01:plan_feedback:finding:P1'
        s.update_many([{'record_id':rid,'manual_verdict':'correct','notes':'=1+1'}],0)
        restored=self.store()
        self.assertEqual(restored.revision,1)
        self.assertEqual(restored.annotations[rid]['notes'],'=1+1')
        rows=list(csv.DictReader(io.StringIO(s.export_csv().decode('utf-8-sig'))))
        self.assertEqual(len(rows),4)
        self.assertEqual(next(r for r in rows if r['record_id']==rid)['notes'],"'=1+1")
        self.assertEqual(json.loads((self.root/'responses/case.json').read_text()),self.answer)

    def test_source_changes_and_stale_revisions_refuse_save(self):
        s=self.store();rid=next(iter(s.records))
        s.update_many([{'record_id':rid,'manual_verdict':'uncertain'}],0)
        with self.assertRaises(ValueError):s.update_many([{'record_id':rid,'manual_verdict':'correct'}],0)
        (self.root/'responses/case.json').write_text('{}')
        with self.assertRaises(ValueError):s.update_many([{'record_id':rid,'manual_verdict':'correct'}],1)

    def test_frozen_input_hash_and_path_escape_rejected(self):
        p=self.root/'experiment_manifest.json';m=json.loads(p.read_text())
        m['frozen_sha256']={'inputs/case.json':'bad hash'};self.write('experiment_manifest.json',m)
        with self.assertRaises(ValueError):self.store()
        m['frozen_sha256']={'../outside':'bad'};self.write('experiment_manifest.json',m)
        with self.assertRaisesRegex(ValueError,'escapes'):self.store()

    def test_state_cannot_overwrite_experiment_artifact(self):
        with self.assertRaisesRegex(ValueError,'ui_comparison'):
            PlanFeedbackReviewStore(self.root,self.root/'experiment_manifest.json')

    def test_duplicate_finding_ids_rejected(self):
        self.answer['findings']*=2;self.write('responses/case.json',self.answer)
        self.write('api_calls/PF_01_01/status.json',{'status':'completed',
            'response_sha256':hashlib.sha256((self.root/'responses/case.json').read_bytes()).hexdigest()})
        with self.assertRaisesRegex(ValueError,'Duplicate review ID'):self.store()

    def test_http_images_head_export_and_authenticated_save(self):
        server=create_server(self.store(),port=0)
        thread=Thread(target=server.serve_forever,daemon=True);thread.start()
        self.addCleanup(lambda:(server.shutdown(),server.server_close(),thread.join()))
        conn=HTTPConnection('127.0.0.1',server.server_port);self.addCleanup(conn.close)
        conn.request('GET','/api/data');r=conn.getresponse();self.assertEqual(r.status,200);data=json.loads(r.read())
        for method,path in [('HEAD','/'),('GET','/api/image/PF_01_01-reference'),('GET','/app.js'),('GET','/review.css'),('GET','/api/export')]:
            conn.request(method,path);r=conn.getresponse();self.assertEqual(r.status,200);r.read()
        conn.request('GET','/api/image/../../experiment_manifest.json');r=conn.getresponse();self.assertEqual(r.status,404);r.read()
        rid=next(iter(data['records']));body=json.dumps({'revision':0,'updates':[{'record_id':rid,'manual_verdict':'correct'}]})
        conn.request('POST','/api/review',body,{'Content-Type':'application/json'});r=conn.getresponse();self.assertEqual(r.status,403);r.read()
        conn.request('POST','/api/review',body,{'Content-Type':'application/json','X-Review-Token':data['token']})
        r=conn.getresponse();self.assertEqual(r.status,200);self.assertEqual(json.loads(r.read())['revision'],1)

    def test_lock_conflict_is_readable(self):
        with patch('sys.argv',['review','--state-path',str(self.state)]), \
             patch('run_plan_feedback_review_app.fcntl.flock',side_effect=BlockingIOError()), \
             patch('run_plan_feedback_review_app.PlanFeedbackReviewStore') as store, \
             patch('sys.stderr',io.StringIO()) as err:
            with self.assertRaises(SystemExit) as exc:main()
            self.assertEqual(exc.exception.code,2);self.assertIn('already using',err.getvalue());store.assert_not_called()


if __name__=='__main__':unittest.main()
