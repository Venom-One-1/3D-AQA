import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from aqa3d.fastsam_smpl import FastSamPoseSequence
from aqa3d.fastsam_student_high_fps import maximum_native_student_pair
from aqa3d.fastsam_trimmed import FastSamTrimmedSequence
from aqa3d.smpl_dtw import ReferenceFrameMatch, VideoSampling
from run_fastsam_student_endpoints_gpu import bind_student_rows, export_segments


class NativeStudentHighFpsTests(unittest.TestCase):
    def setUp(self):
        self.reference = VideoSampling(Path('ref.mp4'), 30., 300, 5., np.arange(0, 300, 6))
        self.student = VideoSampling(Path('05.mp4'), 30.0123, 106, 5., np.arange(0, 106, 6))
        self.native = np.arange(60, 160)
        self.output = np.delete(np.arange(106), [6, 13, 20, 45, 60, 90])
        self.required = np.array([60, 120, 180])

    def sample(self, native=None, output=None, student=None, fps=30.):
        return maximum_native_student_pair(self.reference, student or self.student, 30., fps,
                    self.native if native is None else native, self.output if output is None else output, self.required)

    def test_30fps_uses_all_native_rows_and_keeps_six_normal_video_skips(self):
        ref, student, positions, audit = self.sample()
        np.testing.assert_array_equal(positions, np.arange(100))
        np.testing.assert_array_equal(student.source_indices, self.output)
        self.assertEqual(student.sample_fps, 30.)
        self.assertTrue(set(self.required).issubset(ref.source_indices))
        self.assertFalse(audit['interpolation_or_nearest_substitution'])

    def test_crop_native_origin_is_not_rebased_as_a_video_frame(self):
        _, student, positions, audit = self.sample(native=self.native + 147)
        self.assertEqual(student.source_indices[0], 0)
        self.assertEqual(positions[0], 0)
        self.assertEqual(audit['target_first_native_sample_index_0based'], 207)

    def test_fractional_rate_cap_decimates_native_clock_not_video_indices(self):
        base = VideoSampling(Path('v.mp4'), 25., 106, 5., np.arange(0, 106, 5))
        _, student, positions, _ = self.sample(student=base)
        expected = np.floor(np.arange(round(100 / 30 * 25)) * (30 / 25)).astype(int)
        np.testing.assert_array_equal(positions, expected)
        np.testing.assert_array_equal(student.source_indices, self.output[expected])
        self.assertEqual(student.sample_fps, 25.)

    def test_missing_native_reconstruction_row_is_not_silently_removed(self):
        with self.assertRaisesRegex(ValueError, 'complete and consecutive'):
            self.sample(native=np.delete(self.native, 7), output=np.delete(self.output, 7))

    def test_invalid_native_to_video_mapping_is_rejected(self):
        for output in (self.output[:-1], self.output[::-1], self.output + 7):
            with self.subTest(output=output[:2]), self.assertRaisesRegex(ValueError, 'frame mapping'):
                self.sample(output=output)

    def test_invalid_native_rate_is_rejected(self):
        for fps in (float('nan'), 0., -30.):
            with self.subTest(fps=fps), self.assertRaisesRegex(ValueError, 'rates'):
                self.sample(fps=fps)


class StudentEndpointTimeTests(unittest.TestCase):
    def setUp(self):
        self.required = np.arange(1, 25) * 6
        ref = VideoSampling(Path('ref.mp4'), 30., 300, 5., np.arange(0, 300, 6))
        base = VideoSampling(Path('student.mp4'), 29., 202, 5., np.arange(0, 202, 6))
        output = np.delete(np.arange(202), [6, 103])
        native = np.arange(207, 407)
        self.ref_sampling, self.sampling, self.positions, _ = maximum_native_student_pair(ref, base, 30., 30., native, output, self.required)
        self.pts = np.r_[0., np.cumsum(np.linspace(.03, .04, 201))]
        poses = np.tile(np.eye(3), (300, 23, 1, 1))
        self.reference = FastSamPoseSequence(Path('ref.npz'), np.arange(300), np.arange(300) / 30., 30.,
                                            poses, np.ones(300, dtype=bool), np.zeros(300, dtype=int), None, None, {})
        self.student = FastSamTrimmedSequence(Path('student.npz'), output, self.pts[output], 30., poses[:200],
                                             np.ones(200, dtype=bool), np.zeros(200, dtype=int), None, None, {},
                                             output + 207, self.pts[output] + 7., self.pts)
        self.ref_positions = self.ref_sampling.source_indices
        self.matches = [ReferenceFrameMatch(int(np.searchsorted(self.ref_sampling.source_indices, frame)), i * 5 + 2, 1, .25)
                        for i, frame in enumerate(self.required)]
        self.path = np.array([(m.target_index, m.reference_index) for m in self.matches])
        self.manifest = {'endpoint_keyposes': [dict(move_id=i + 1, move_name_pinyin=f'move{i}', move_name_zh=f'move{i}',
                          pose_id=f'{i + 1}.end', keypose_role='form_end_boundary_event', final_technique_step='technique',
                          technique_binding_status='bound', boundary_time_seconds=float(frame / 30.),
                          source_frame_index_0based=int(frame)) for i, frame in enumerate(self.required)]}

    def rows(self):
        return bind_student_rows(self.manifest, self.matches, self.path, self.reference, self.student,
                                  self.ref_sampling, self.sampling, self.ref_positions, self.positions, '12')

    def test_actual_vfr_pts_and_original_frame_provenance_are_preserved(self):
        for row in self.rows():
            frame = row['student_source_frame_0based']
            self.assertEqual(row['student_source_time_seconds'], self.pts[frame])
            self.assertNotAlmostEqual(row['student_source_time_seconds'], frame / 29., places=6)
            self.assertEqual(row['student_original_source_frame_0based'], frame + 207)
            self.assertEqual(row['student_original_source_timestamp_seconds'], self.pts[frame] + 7.)
            self.assertEqual(row['student_tracking_match'], 'exact_output_video_frame')
            self.assertFalse(row['automatic_endpoint_rounding_to_review_grid'])

    def test_reference_annotations_remain_exact(self):
        np.testing.assert_array_equal([r['reference_source_frame_0based'] for r in self.rows()], self.required)

    def test_duplicate_student_endpoints_are_rejected(self):
        previous = self.matches[1]
        self.matches[1] = ReferenceFrameMatch(previous.reference_index, self.matches[0].target_index, 1, .25)
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            self.rows()

    def test_segments_use_actual_pts_and_contiguous_source_frames(self):
        with tempfile.TemporaryDirectory() as name:
            export_segments(Path(name), self.rows(), self.sampling, self.pts, float(self.pts[-1] + .04))
            with (Path(name) / 'segments.csv').open() as handle:
                rows = list(csv.DictReader(handle))
        for row in rows:
            self.assertEqual(float(row['source_interval_start_time_seconds']), self.pts[int(row['start_frame_0based'])])
        for previous, current in zip(rows, rows[1:]):
            self.assertEqual(int(current['start_frame_0based']), int(previous['end_frame_0based_inclusive']) + 1)


if __name__ == '__main__':
    unittest.main()
