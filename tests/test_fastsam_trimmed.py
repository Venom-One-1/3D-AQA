from pathlib import Path
import tempfile
import unittest

import numpy as np

from aqa3d.fastsam_trimmed import load_trimmed_sequence, sampling_from_native_smpl, sampling_from_video_pts


class FastSamTrimmedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / '05.npz'
        self.pts = np.arange(100, dtype=float) / 30.
        self.sampling = sampling_from_video_pts(Path('05.mp4'), self.pts, 30., 100 / 30.)
        self.data = {'schema_id': np.array('fastsam_smpl_primary_v1'),
                     'trim_schema': np.array('reviewed_video_smpl_trim_v1'),
                     'source_frame_indices': np.arange(100) + 60,
                     'output_video_frame_indices': np.arange(100),
                     'trim_source_start_frame': np.array(60), 'input_fps': np.array(29.97),
                     'timestamps': self.pts.copy(), 'source_video_timestamps': self.pts + 2,
                     'body_pose': np.tile(np.eye(3, dtype=np.float32), (100, 23, 1, 1)),
                     'valid_mask': np.ones(100, dtype=bool), 'ignore_mask': np.zeros(100, dtype=bool),
                     'source_track_ids': np.zeros(100, dtype=int),
                     'conversion_mean_vertex_error_mm': np.zeros(100),
                     'conversion_p95_vertex_error_mm': np.zeros(100)}
        self.manifest = {'counts': {'video_frames': 100, 'smpl_samples': 100}}
        self.save()

    def save(self):
        np.savez_compressed(self.path, **self.data)

    def load(self):
        return load_trimmed_sequence(self.path, self.sampling, self.pts, self.manifest)

    def test_original_frame_offset_is_preserved_but_not_used_as_output_index(self):
        sequence = self.load()
        positions = sequence.sample_positions(self.sampling)
        self.assertEqual(sequence.source_indices[0], 0)
        self.assertEqual(sequence.original_source_indices[0], 60)
        p = sequence.provenance(int(positions[1]), self.sampling)
        self.assertEqual(p['original_source_frame_0based'], p['source_frame_0based'] + 60)
        self.assertEqual(p['source_time_seconds'], self.pts[p['source_frame_0based']])

    def test_original_average_fps_does_not_override_verified_pts(self):
        self.data['input_fps'] = np.array(25.)
        self.save()
        sequence = self.load()
        self.assertEqual(sequence.audit['original_input_fps'], 25.)
        self.assertTrue(len(sequence.sample_positions(self.sampling)))

    def test_variable_pts_selects_actual_frame_before_time_grid(self):
        pts = np.array([0., .03, .07, .12, .19, .23, .31, .42, .46, .53])
        sampling = sampling_from_video_pts(Path('v.mp4'), pts, 16., .6)
        np.testing.assert_array_equal(sampling.source_indices, [0, 4, 6])

    def test_wrong_pts_are_rejected(self):
        self.data['timestamps'][12] += .01
        self.save()
        with self.assertRaisesRegex(ValueError, 'actual trimmed video PTS'):
            self.load()

    def test_original_mapping_mismatch_is_rejected(self):
        self.data['source_frame_indices'][5] += 1
        self.save()
        with self.assertRaisesRegex(ValueError, 'crop offset'):
            self.load()

    def drop_frame(self, frame):
        for key, value in list(self.data.items()):
            if value.ndim and len(value) == self.manifest['counts']['smpl_samples']:
                self.data[key] = np.delete(value, frame, axis=0)
        self.manifest['counts']['smpl_samples'] -= 1
        self.save()

    def test_missing_sample_is_strict_by_default(self):
        self.drop_frame(6)
        sequence = self.load()
        with self.assertRaisesRegex(ValueError, 'Missing exact'):
            sequence.sample_positions(self.sampling)
        self.assertEqual(sequence.audit['missing_output_video_frames'], [6])

    def test_explicit_one_frame_tolerance_is_bounded_and_tie_breaks_earlier(self):
        self.drop_frame(6)
        sequence = self.load()
        positions = sequence.sample_positions(self.sampling, max_gap_frames=1)
        self.assertEqual(sequence.source_indices[positions[1]], 5)
        self.assertEqual(sequence.provenance(int(positions[1]), self.sampling)['source_time_seconds'], self.pts[5])

    def test_invalid_selected_frame_is_never_replaced_by_another_person(self):
        self.data['ignore_mask'][6] = True
        self.save()
        with self.assertRaisesRegex(ValueError, 'Invalid or ignored'):
            self.load().sample_positions(self.sampling, max_gap_frames=1)

    def test_duplicate_pts_are_rejected(self):
        pts = self.pts.copy()
        pts[2] = pts[1]
        with self.assertRaisesRegex(ValueError, 'presentation timeline'):
            sampling_from_video_pts(Path('v.mp4'), pts, 30., 100 / 30.)


class NativeSmplSamplingTests(unittest.TestCase):
    def setUp(self):
        self.sampling = sampling_from_video_pts(Path('05.mp4'), np.arange(101) / 30.0123,
                                               30.0123, 101 / 30.0123)
        self.native = np.arange(60, 160)
        self.output = np.delete(np.arange(101), 6)

    def test_expected_video_skip_is_not_a_missing_native_pose(self):
        sampling = sampling_from_native_smpl(self.sampling, self.native, self.output, 30.)
        np.testing.assert_array_equal(sampling.source_indices, self.output[::6])
        self.assertEqual(sampling.source_indices[1], 7)
        self.assertEqual(sampling.sample_fps, 5.)
        self.assertEqual(sampling.source_fps, 30.0123)
        self.assertEqual(sampling.source_frame_count, 101)

    def test_crop_phase_uses_first_retained_native_sample(self):
        sampling = sampling_from_native_smpl(self.sampling, self.native + 1, self.output, 30.)
        np.testing.assert_array_equal(sampling.source_indices, self.output[::6])
        self.assertEqual(sampling.source_indices[0], 0)

    def test_actual_missing_native_pose_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'complete and consecutive'):
            sampling_from_native_smpl(self.sampling, np.delete(self.native, 8),
                                      np.delete(self.output, 8), 30.)

    def test_noninteger_stride_is_not_rounded(self):
        with self.assertRaisesRegex(ValueError, 'integer multiple'):
            sampling_from_native_smpl(self.sampling, self.native, self.output, 29.97)

    def test_invalid_native_rate_is_rejected(self):
        for rate in (0., -30., float('nan'), float('inf')):
            with self.subTest(rate=rate), self.assertRaisesRegex(ValueError, 'sample rate'):
                sampling_from_native_smpl(self.sampling, self.native, self.output, rate)

    def test_video_mapping_must_be_ordered_and_within_video(self):
        for output in (self.output[:-1], self.output[::-1], self.output + 3):
            with self.subTest(output=output[:3]), self.assertRaisesRegex(ValueError, 'frame mapping'):
                sampling_from_native_smpl(self.sampling, self.native, output, 30.)


if __name__ == '__main__':
    unittest.main()
