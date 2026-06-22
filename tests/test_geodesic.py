import math
import unittest

import numpy as np

from aqa3d.geodesic import geodesic_distance


class GeodesicDistanceTests(unittest.TestCase):
    def test_identical_rotations_have_zero_error(self):
        rotations = np.broadcast_to(np.eye(3), (2, 23, 3, 3)).copy()
        errors, distance = geodesic_distance(rotations, rotations)
        np.testing.assert_allclose(errors, 0.0, atol=1e-7)
        self.assertAlmostEqual(float(distance), 0.0, places=7)

    def test_quarter_turn_has_expected_distance(self):
        teacher = np.zeros((1, 1, 3))
        student = np.array([[[0.0, 0.0, math.pi / 2.0]]])
        errors, distance = geodesic_distance(student, teacher)
        np.testing.assert_allclose(errors, math.pi / 2.0, atol=1e-7)
        self.assertAlmostEqual(float(distance), math.pi / 2.0, places=7)

    def test_joint_weights_change_the_aggregate_only(self):
        teacher = np.zeros((1, 2, 3))
        student = teacher.copy()
        student[0, 0, 0] = math.pi / 2.0
        errors, distance = geodesic_distance(student, teacher, joint_weights=[3.0, 1.0])
        np.testing.assert_allclose(errors[0], [math.pi / 2.0, 0.0], atol=1e-7)
        self.assertAlmostEqual(float(distance), 3.0 * math.pi / 8.0, places=7)

    def test_batch_dimensions_are_preserved(self):
        teacher = np.zeros((2, 1, 1, 3))
        student = teacher.copy()
        student[0, 0, 0, 2] = math.pi / 2.0
        student[1, 0, 0, 2] = math.pi
        errors, distances = geodesic_distance(student, teacher)
        self.assertEqual(errors.shape, (2, 1, 1))
        np.testing.assert_allclose(distances, [math.pi / 2.0, math.pi], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
