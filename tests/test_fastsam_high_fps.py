from pathlib import Path
import unittest

import numpy as np

from aqa3d.fastsam_high_fps import maximum_fps_pair
from aqa3d.smpl_dtw import VideoSampling


class MaximumFpsTests(unittest.TestCase):
    def sampling(self, fps, frames):
        return VideoSampling(Path('video.mp4'), fps, frames, 5., np.arange(0, frames, 6))

    def test_thirty_fps_uses_every_frame_and_exact_anchor(self):
        ref, target, policy = maximum_fps_pair(self.sampling(30., 300), self.sampling(30., 240),
                                              30., 30., np.array([60, 150, 270]))
        np.testing.assert_array_equal(ref.source_indices, np.arange(300))
        np.testing.assert_array_equal(target.source_indices, np.arange(240))
        self.assertEqual(policy['pair_sample_fps'], 30.)
        self.assertEqual(policy['reference_extra_annotation_frames'], [])
        self.assertFalse(policy['automatic_endpoint_rounding_to_review_grid'])

    def test_fractional_fps_does_not_duplicate_native_frames(self):
        fps = 24000 / 1001
        ref, target, policy = maximum_fps_pair(self.sampling(30., 300), self.sampling(fps, 240),
                                              30., fps, np.array([60, 150, 270]))
        np.testing.assert_array_equal(target.source_indices, np.arange(240))
        self.assertEqual(policy['pair_sample_fps'], fps)
        self.assertTrue(set((60, 150, 270)).issubset(ref.source_indices))
        self.assertEqual(ref.source_fps, 30.)

    def test_off_grid_annotation_is_inserted_not_rounded(self):
        ref, _, policy = maximum_fps_pair(self.sampling(30., 300), self.sampling(25., 250),
                                         30., 25., np.array([5, 155, 275]))
        np.testing.assert_array_equal(policy['reference_extra_annotation_frames'], [5, 155, 275])
        self.assertTrue(set((5, 155, 275)).issubset(ref.source_indices))
        self.assertTrue(np.all(np.diff(ref.source_indices) > 0))

    def test_native_smpl_rate_limits_common_rate(self):
        _, _, policy = maximum_fps_pair(self.sampling(60., 600), self.sampling(60., 600),
                                        30., 25., np.array([60, 150, 270]))
        self.assertEqual(policy['pair_sample_fps'], 25.)

    def test_bad_rate_and_annotations_rejected(self):
        for fps in (float('nan'), 0., -1.):
            with self.subTest(fps=fps), self.assertRaisesRegex(ValueError, 'rates'):
                maximum_fps_pair(self.sampling(30., 300), self.sampling(30., 300),
                                 fps, 30., np.array([60]))
        for required in (np.array([], dtype=int), np.array([5, 5]), np.array([-1]), np.array([300]), np.array([5.])):
            with self.subTest(required=required), self.assertRaisesRegex(ValueError, 'Anchor frames'):
                maximum_fps_pair(self.sampling(30., 300), self.sampling(30., 300), 30., 30., required)


if __name__ == '__main__':
    unittest.main()
