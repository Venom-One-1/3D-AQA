import unittest
import json
import tempfile
from pathlib import Path
import numpy as np

from aqa3d.angle_metrics import SMPL_24_JOINTS as J
from aqa3d.endpoint_metrics import (
    aggregate_window, body_axes, compare_interval, compute_endpoint_metrics,
    endpoint_window, teacher_interval,
)


def skeleton():
    p=np.zeros((24,3),dtype=float)
    p[J['Neck']]=[0,1,0];p[J['Head']]=[0,1.3,0]
    for side, sign in [('L',1),('R',-1)]:
        for name, value in [('Hip',[sign*.2,0,0]),('Knee',[sign*.2,-.5,0]),
                            ('Ankle',[sign*.2,-1,0]),('Shoulder',[sign*.3,.8,0]),
                            ('Elbow',[sign*.6,.8,0]),('Wrist',[sign*.9,.8,.2])]:
            p[J[side+'_'+name]]=value
    return p


class EndpointMetricsTests(unittest.TestCase):
    def test_angles_and_distances_are_rotation_translation_scale_invariant(self):
        p=skeleton()[None]
        rotation=np.array([[0,0,1],[0,1,0],[-1,0,0]],dtype=float)
        original=compute_endpoint_metrics(p,[10]).set_index('metric_id').value
        transformed=compute_endpoint_metrics((p@rotation.T)*3+[8,5,-3],[10]).set_index('metric_id').value
        np.testing.assert_allclose(original,transformed,atol=1e-6)
        self.assertAlmostEqual(original['left_knee_angle'],180.)
        self.assertAlmostEqual(original['stance_width_shoulder_ratio'],2/3)

    def test_body_axes_forward_and_right_signs(self):
        up,left,forward,valid=body_axes(skeleton()[None])
        np.testing.assert_allclose(up,[[0,1,0]])
        np.testing.assert_allclose(left,[[1,0,0]])
        np.testing.assert_allclose(forward,[[0,0,1]])
        values=compute_endpoint_metrics(skeleton()[None],[0]).set_index('metric_id').value
        self.assertGreater(values['right_wrist_head_right_torso_ratio'],0)
        self.assertGreater(values['right_wrist_head_forward_torso_ratio'],0)

    def test_degenerate_body_frame_does_not_invalidate_knee(self):
        p=skeleton();p[J['Neck']]=p[J['Pelvis']]
        rows=compute_endpoint_metrics(p[None],[0]).set_index('metric_id')
        self.assertEqual(rows.loc['left_knee_angle','status'],'valid')
        self.assertTrue(np.isnan(rows.loc['stance_width_shoulder_ratio','value']))

    def test_window_clips_at_form_end_and_fractional_fps(self):
        np.testing.assert_array_equal(endpoint_window(100,90,100,30),np.arange(94,101))
        np.testing.assert_array_equal(endpoint_window(100,90,100,29.97),np.arange(95,101))
        np.testing.assert_array_equal(endpoint_window(100,99,100,30),[99,100])

    def test_center_is_retained_not_reselected_and_invalid_count_excluded(self):
        data=aggregate_window([1,2,3,4,100],['valid']*5,[6,7,8,9,10],10)
        self.assertEqual(data['value'],3)
        self.assertEqual(data['center_value'],100)
        self.assertEqual(data['center_minus_window_median'],97)
        invalid=aggregate_window([1,2,3,4,100],['valid']*2+['near_track_switch']*3,[6,7,8,9,10],10)
        self.assertIsNone(invalid['value']); self.assertIsNone(invalid['center_value'])

    def test_teacher_mad_is_unscaled_and_all_valid_values_used(self):
        ref=teacher_interval(list(range(10)))
        self.assertEqual(ref['valid_teacher_count'],10)
        self.assertEqual(ref['median'],4.5);self.assertEqual(ref['mad'],2.5)
        self.assertEqual(ref['median_minus_2mad'],-.5)
        self.assertAlmostEqual(ref['p10'],.9)

    def test_missing_teachers_and_degenerate_interval_do_not_produce_verdict(self):
        ref=teacher_interval([1,2,3,None,None])
        self.assertEqual(compare_interval(100,ref,'median_2mad')[0],'insufficient_teachers')
        ref=teacher_interval([2]*10)
        self.assertEqual(compare_interval(100,ref,'median_2mad')[0],'degenerate_reference_interval')

    def test_comparison_boundary_inclusive_and_signed_deviation(self):
        ref=teacher_interval(list(range(10)))
        self.assertEqual(compare_interval(-.5,ref,'median_2mad'),('within',0.))
        self.assertEqual(compare_interval(-1.,ref,'median_2mad'),('below',-.5))
        self.assertEqual(compare_interval(10.,ref,'median_2mad'),('above',.5))

    def test_cached_path_rejects_changed_tracking_costs(self):
        from run_endpoint_metrics_experiment import cached_alignment
        from aqa3d.smpl_dtw import VideoSampling
        from aqa3d.tracking import TrackPoseSequence
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);video=p/'S.mp4';tracking=p/'S.pkl'
            reference={'video_path':str(p/'R.mp4'),'tracking_path':str(p/'R.pkl')}
            sampling=VideoSampling(video,30.,60,5.,np.arange(0,60,6))
            rotations=np.tile(np.eye(3),(60,23,1,1))
            track=TrackPoseSequence(np.arange(1,61),rotations,1,np.ones(60,dtype=int))
            positions=np.arange(0,60,6)
            summary={'status':'ok','input_video':str(video),'input_tracking':str(tracking),
                'reference_video':reference['video_path'],'reference_tracking':reference['tracking_path'],
                'sample_fps':5.,'input_source_fps':30.,'input_source_frame_count':60,
                'reference_source_fps':30.,'dtw_coefficient':1.,'max_tracking_gap_seconds':0.}
            (p/'summary.json').write_text(json.dumps(summary))
            def save(cost):
                np.savez(p/'dtw_path.npz',target_sample_indices_0based=np.arange(10),
                    reference_sample_indices_0based=np.arange(10),
                    target_source_frame_indices_0based=positions,
                    reference_source_frame_indices_0based=positions,
                    local_geodesic_radians=np.full(10,cost))
            save(0.)
            self.assertIsNotNone(cached_alignment(p,video,tracking,sampling,track,reference,sampling,track,positions))
            save(0.5)
            with self.assertRaisesRegex(ValueError,'costs do not match'):
                cached_alignment(p,video,tracking,sampling,track,reference,sampling,track,positions)


if __name__=='__main__':
    unittest.main()
