import unittest

import numpy as np

from aqa3d.pose_score import (
    apply_pause_penalty,
    calculate_pose_score,
    calibrate_distances,
    unexpected_pause_events,
)


class PoseScoreTests(unittest.TestCase):
    def test_calibration_uses_robust_median_mad_and_scale_floor(self):
        calibration = calibrate_distances(np.asarray([5.0, 6.0, 7.0, 100.0]))

        self.assertAlmostEqual(calibration.median_degrees, 6.5)
        self.assertAlmostEqual(calibration.mad_degrees, 1.0)
        self.assertAlmostEqual(calibration.scale_degrees, 1.4826)

    def test_pose_score_combines_keyframe_and_path_z_values(self):
        calibration = calibrate_distances(np.asarray([8.0, 9.0, 10.0, 11.0, 12.0]))
        score = calculate_pose_score(11.0, 12.0, calibration, calibration)

        expected_z = 0.4 * ((11.0 - 10.0) / 1.4826) + 0.6 * ((12.0 - 10.0) / 1.4826)
        self.assertAlmostEqual(score.pose_score, 95.0 - 10.0 * expected_z)

    def test_only_pauses_dominated_by_teacher_activity_are_counted(self):
        pause = np.asarray([0, 1, 1, 0, 1, 1, 1, 0], dtype=bool)
        active = np.asarray([0, 1, 0, 0, 1, 1, 1, 0], dtype=bool)

        events = unexpected_pause_events(
            pause,
            active,
            fps=2.0,
            minimum_teacher_active_fraction=0.70,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual((events[0].start_index, events[0].end_index), (4, 6))
        self.assertAlmostEqual(events[0].duration_seconds, 1.5)

    def test_pause_penalty_is_capped(self):
        final_score, penalty = apply_pause_penalty(
            82.0,
            5,
            points_per_pause=5.0,
            maximum_penalty=15.0,
        )

        self.assertEqual(penalty, 15.0)
        self.assertEqual(final_score, 67.0)


if __name__ == "__main__":
    unittest.main()
