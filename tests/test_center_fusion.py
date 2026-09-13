import unittest

import numpy as np

from aqa3d.center_fusion import (
    MOVE_METRICS,
    calibrated_excess,
    center_metric_error,
    center_penalty,
    pelvis_lowering,
    robust_calibration,
    validate_move_metrics,
)
from aqa3d.pelvis_quality import SIGNAL_NAMES


def _profile(points: int = 11) -> np.ndarray:
    values = np.zeros((len(SIGNAL_NAMES), points), dtype=np.float64)
    signal = {name: index for index, name in enumerate(SIGNAL_NAMES)}
    values[signal["support_height"]] = np.linspace(0.65, 0.55, points)
    values[signal["support_lateral"]] = np.linspace(0.0, 0.2, points)
    values[signal["support_forward"]] = np.linspace(0.1, -0.1, points)
    return values


class CenterFusionTests(unittest.TestCase):
    def test_move_weights_sum_to_one(self):
        validate_move_metrics(MOVE_METRICS)

    def test_pelvis_lowering_is_positive_when_height_decreases(self):
        self.assertAlmostEqual(
            pelvis_lowering(np.asarray([0.65, 0.62, 0.58, 0.55]), endpoint_fraction=0.25),
            0.10,
        )

    def test_vertical_range_error_is_symmetric_in_log_space(self):
        teacher = _profile()
        larger = teacher.copy()
        smaller = teacher.copy()
        signal = SIGNAL_NAMES.index("support_height")
        center = np.mean(teacher[signal])
        larger[signal] = center + 2.0 * (teacher[signal] - center)
        smaller[signal] = center + 0.5 * (teacher[signal] - center)
        larger_error, _ = center_metric_error(larger, teacher, "vertical_range_error")
        smaller_error, _ = center_metric_error(smaller, teacher, "vertical_range_error")
        self.assertAlmostEqual(larger_error, smaller_error)

    def test_root_vertical_lowering_uses_root_signal(self):
        teacher = _profile()
        student = teacher.copy()
        signal = SIGNAL_NAMES.index("root_vertical")
        teacher[signal] = np.linspace(0.0, -0.10, teacher.shape[1])
        student[signal] = np.linspace(0.0, -0.04, student.shape[1])
        error, details = center_metric_error(
            student,
            teacher,
            "root_vertical_lowering_error",
        )
        self.assertAlmostEqual(error, 0.06)
        self.assertAlmostEqual(details["student_root_vertical_lowering"], 0.04)
        self.assertAlmostEqual(details["teacher_root_vertical_lowering"], 0.10)

    def test_teacher_tolerance_produces_no_penalty(self):
        calibration = robust_calibration(
            np.asarray([0.01, 0.02, 0.03, 0.04, 0.05]),
            minimum_scale=0.005,
        )
        _, inside = calibrated_excess(calibration.median, calibration)
        _, outside = calibrated_excess(
            calibration.median + 2.0 * calibration.scale,
            calibration,
        )
        self.assertEqual(inside, 0.0)
        self.assertAlmostEqual(outside, 1.0)
        self.assertAlmostEqual(
            center_penalty(outside, points_per_scale=3.0, maximum_penalty=15.0),
            3.0,
        )

    def test_center_penalty_respects_cap(self):
        self.assertEqual(
            center_penalty(10.0, points_per_scale=3.0, maximum_penalty=15.0),
            15.0,
        )


if __name__ == "__main__":
    unittest.main()
