import unittest

import numpy as np

from run_teacher_1fps_smpl_dtw_batch import uniform_1fps_keyframes


class Uniform1FpsKeyframeTests(unittest.TestCase):
    def test_includes_first_and_last_frames(self):
        source_indices = np.arange(0, 91, dtype=np.int64)
        keyframes = uniform_1fps_keyframes(source_indices, source_fps=30.0)
        np.testing.assert_array_equal(keyframes, [0, 30, 60, 90])

    def test_uses_nearest_sampled_frame_when_sampling_is_sparse(self):
        source_indices = np.arange(0, 181, 2, dtype=np.int64)
        keyframes = uniform_1fps_keyframes(source_indices, source_fps=30.0)
        np.testing.assert_array_equal(source_indices[keyframes], [0, 30, 60, 90, 120, 150, 180])


if __name__ == "__main__":
    unittest.main()
