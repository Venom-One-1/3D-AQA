import math
import unittest

import numpy as np

from aqa3d.alignment import dtw_alignment
from run_teacher_keyframe_smpl_dtw_batch import dtw_from_cost_matrix, pairwise_geodesic_costs


class SmplDtwTests(unittest.TestCase):
    def test_pairwise_geodesic_costs_use_mean_joint_error(self):
        identity = np.broadcast_to(np.eye(3), (2, 23, 3, 3)).copy()
        teacher = np.broadcast_to(np.eye(3), (1, 23, 3, 3)).copy()
        quarter_turn = np.array(
            (
                (0.0, -1.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        identity[1, 0] = quarter_turn

        costs = pairwise_geodesic_costs(identity, teacher, chunk_size=1)
        self.assertEqual(costs.shape, (2, 1))
        self.assertAlmostEqual(float(costs[0, 0]), 0.0, places=7)
        self.assertAlmostEqual(float(costs[1, 0]), (math.pi / 2.0) / 23.0, places=7)

    def test_dtw_from_cost_matrix_matches_existing_dtw_for_precomputed_costs(self):
        student = np.array([[[0.0]], [[2.0]], [[4.0]]], dtype=np.float64)
        teacher = np.array([[[0.0]], [[1.0]], [[4.0]]], dtype=np.float64)
        local_costs = np.linalg.norm(student[:, None] - teacher[None, :], axis=-1).mean(axis=-1)

        old_distance, old_matching, old_accumulated = dtw_alignment(student, teacher)
        new_distance, new_matching, new_accumulated = dtw_from_cost_matrix(local_costs)

        self.assertAlmostEqual(new_distance, old_distance, places=7)
        self.assertEqual(new_matching, old_matching)
        np.testing.assert_allclose(new_accumulated, old_accumulated, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
