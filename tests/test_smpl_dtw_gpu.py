import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from aqa3d.smpl_dtw import backtrack_dtw_path, dtw_from_cost_matrix, pairwise_geodesic_costs, select_reference_frame_matches
from aqa3d.smpl_dtw_gpu import cuda_device, dtw_from_cost_matrix_gpu, pairwise_geodesic_costs_gpu


def poses(count, seed):
    return Rotation.random(count * 23, random_state=seed).as_matrix().reshape(count, 23, 3, 3)


class GpuInputTests(unittest.TestCase):
    def test_cpu_device_is_not_a_silent_fallback(self):
        with self.assertRaisesRegex(ValueError, 'CUDA device'):
            cuda_device('cpu')

    def test_unavailable_gpu_is_explicit(self):
        with patch('torch.cuda.is_available', return_value=False), self.assertRaisesRegex(RuntimeError, 'does not silently'):
            cuda_device()

    def test_invalid_pose_shapes_rejected_before_cuda(self):
        for shape in ((0, 23, 3, 3), (2, 24, 3, 3), (2, 23, 3)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, 'nonempty target'):
                pairwise_geodesic_costs_gpu(np.zeros(shape), poses(1, 1))

    def test_nonfinite_poses_are_rejected(self):
        array = poses(2, 1)
        array[0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            pairwise_geodesic_costs_gpu(array, poses(1, 2))

    def test_invalid_tiles(self):
        for kwargs in ({'target_chunk_size': 0}, {'reference_chunk_size': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, 'chunk sizes'):
                pairwise_geodesic_costs_gpu(poses(1, 1), poses(1, 2), **kwargs)

    def test_invalid_costs_and_coefficient(self):
        for values in (np.array([]), np.zeros((0, 1)), np.array([[np.nan]]), np.array([[-1.]])):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, 'costs must'):
                dtw_from_cost_matrix_gpu(values)
        for coefficient in (0., -1., float('nan'), float('inf')):
            with self.subTest(coefficient=coefficient), self.assertRaisesRegex(ValueError, 'coefficient'):
                dtw_from_cost_matrix_gpu(np.ones((2, 2)), coefficient)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for numerical equivalence tests')
class GpuEquivalenceTests(unittest.TestCase):
    def test_random_distance_matrices_and_tile_boundaries(self):
        target, reference = poses(19, 10), poses(17, 11)
        expected = pairwise_geodesic_costs(target, reference, chunk_size=3)
        profile = {}
        for target_chunk, ref_chunk in ((1, 1), (7, 6), (128, 128)):
            actual = pairwise_geodesic_costs_gpu(target, reference, target_chunk_size=target_chunk,
                                                 reference_chunk_size=ref_chunk, profile=profile)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
            self.assertEqual(actual.dtype, np.float64)
        self.assertGreater(profile['wall_seconds'], 0)

    def test_identity_near_zero_and_pi_clamping(self):
        reference = np.tile(np.eye(3), (1, 23, 1, 1))
        target = np.tile(reference, (4, 1, 1, 1))
        for index, angle in enumerate((0., 1e-8, np.pi - 1e-8, np.pi)):
            target[index, 0] = Rotation.from_rotvec([angle, 0, 0]).as_matrix()
        expected = pairwise_geodesic_costs(target, reference)
        actual = pairwise_geodesic_costs_gpu(target, reference)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-9)
        self.assertAlmostEqual(actual[-1, 0], np.pi / 23)

    def test_random_dtw_accumulated_values_and_paths_match_exactly(self):
        for shape in ((1, 1), (1, 9), (8, 1), (11, 7), (6, 13), (17, 17)):
            for coefficient in (.7, 1., 1.3):
                costs = np.random.default_rng(42).uniform(0, 2, size=shape)
                expected, matching, accumulated = dtw_from_cost_matrix(costs, coefficient)
                actual, new_matching, new_accumulated = dtw_from_cost_matrix_gpu(costs, coefficient)
                with self.subTest(shape=shape, coefficient=coefficient):
                    np.testing.assert_array_equal(new_accumulated, accumulated)
                    self.assertEqual(new_matching, matching)
                    self.assertEqual(actual, expected)

    def test_ties_retain_original_backtrack_priority(self):
        for shape in ((9, 7), (7, 9), (5, 5)):
            costs = np.zeros(shape)
            _, _, expected = dtw_from_cost_matrix(costs)
            _, _, actual = dtw_from_cost_matrix_gpu(costs)
            np.testing.assert_array_equal(backtrack_dtw_path(actual), backtrack_dtw_path(expected))

    def test_end_to_end_paths_and_selected_frames(self):
        reference = poses(24, 91)
        target = np.repeat(reference, 2, axis=0)
        cpu_costs = pairwise_geodesic_costs(target, reference)
        gpu_costs = pairwise_geodesic_costs_gpu(target, reference)
        np.testing.assert_allclose(gpu_costs, cpu_costs, rtol=0, atol=2e-8)
        _, _, expected = dtw_from_cost_matrix(cpu_costs)
        _, _, actual = dtw_from_cost_matrix_gpu(gpu_costs)
        path = backtrack_dtw_path(expected)
        new_path = backtrack_dtw_path(actual)
        np.testing.assert_array_equal(new_path, path)
        cpu_matches = select_reference_frame_matches(cpu_costs, path, range(24))
        gpu_matches = select_reference_frame_matches(gpu_costs, new_path, range(24))
        self.assertEqual([m.target_index for m in cpu_matches], [m.target_index for m in gpu_matches])


if __name__ == '__main__':
    unittest.main()
