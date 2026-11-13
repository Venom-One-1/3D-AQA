from pathlib import Path
import tempfile
import unittest

import numpy as np

from aqa3d.fastsam_high_fps import maximum_fps_pair
from aqa3d.fastsam_smpl import FastSamPoseSequence
from aqa3d.smpl_dtw import ReferenceFrameMatch, VideoSampling
from run_fastsam_teacher_endpoints_gpu import bind_endpoint_rows, save_segments


class HighFpsEndpointTests(unittest.TestCase):
    def setUp(self):
        self.required = np.arange(24) * 12 + 6
        ref = VideoSampling(Path('reference.mp4'), 30., 300, 5., np.arange(0, 300, 6))
        target = VideoSampling(Path('teacher.mp4'), 24000 / 1001, 240, 5., np.arange(0, 240, 5))
        self.ref_sampling, self.target_sampling, _ = maximum_fps_pair(ref, target, 30., target.source_fps, self.required)

        def sequence(count, fps):
            indices = np.arange(count)
            return FastSamPoseSequence(Path('source.npz'), indices, indices / fps, fps,
                                       np.tile(np.eye(3), (count, 23, 1, 1)), np.ones(count, dtype=bool),
                                       np.zeros(count, dtype=int), None, None, {})
        self.reference, self.target = sequence(300, 30.), sequence(240, target.source_fps)
        indices = np.searchsorted(self.ref_sampling.source_indices, self.required)
        self.matches = [ReferenceFrameMatch(int(ref_sample), int(i * 8 + 3), 1, .25)
                        for i, ref_sample in enumerate(indices)]
        self.path = np.asarray([(m.target_index, m.reference_index) for m in self.matches])
        self.manifest = {'endpoint_keyposes': [
            {'move_id': i + 1, 'move_name_pinyin': f'move{i + 1}', 'move_name_zh': f'move{i + 1}',
             'pose_id': f'{i + 1}.end', 'keypose_role': 'form_end_boundary_event',
             'final_technique_step': 'technique', 'technique_binding_status': 'bound',
             'boundary_time_seconds': float(frame / 30.), 'source_frame_index_0based': int(frame)}
            for i, frame in enumerate(self.required)]}

    def rows(self):
        return bind_endpoint_rows(self.manifest, self.matches, self.path, self.reference, self.target,
                                  self.ref_sampling, self.target_sampling, self.ref_sampling.source_indices,
                                  self.target_sampling.source_indices, 'teacher')

    def test_exact_annotation_frames_and_native_times_are_not_grid_rounded(self):
        rows = self.rows()
        self.assertEqual(len(rows), 24)
        np.testing.assert_array_equal([r['reference_source_frame_0based'] for r in rows], self.required)
        for row in rows:
            self.assertEqual(row['reference_source_time_seconds'], row['reference_source_frame_0based'] / 30.)
            self.assertEqual(row['teacher_source_time_seconds'], row['teacher_source_frame_0based'] / (24000 / 1001))
            self.assertFalse(row['automatic_endpoint_rounding_to_review_grid'])
            self.assertNotIn('reference_grid_time_seconds', row)
        self.assertNotEqual(rows[0]['teacher_source_time_seconds'] * 5,
                            round(rows[0]['teacher_source_time_seconds'] * 5))

    def test_duplicate_transferred_endpoints_are_rejected(self):
        old = self.matches[1]
        self.matches[1] = ReferenceFrameMatch(old.reference_index, self.matches[0].target_index, 1, .25)
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            self.rows()

    def test_segments_have_contiguous_closed_source_intervals(self):
        import csv
        with tempfile.TemporaryDirectory() as name:
            save_segments(Path(name), self.rows(), self.target_sampling)
            with (Path(name) / 'segments.csv').open() as handle:
                segments = list(csv.DictReader(handle))
        self.assertEqual(int(segments[0]['start_frame_0based']), 0)
        for previous, current in zip(segments, segments[1:]):
            self.assertEqual(int(current['start_frame_0based']), int(previous['end_frame_0based_inclusive']) + 1)
        self.assertNotIn('start_frame_5fps', segments[0])


if __name__ == '__main__':
    unittest.main()
