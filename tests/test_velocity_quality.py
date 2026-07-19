import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from aqa3d.velocity_quality import (
    TeacherMotionModel,
    analyze_against_teacher_model,
    count_acceleration_peaks,
    motion_fragment_count,
    profile_statistics,
    resample_profile,
)


def simple_teacher_model() -> TeacherMotionModel:
    grid = np.linspace(0.0, 1.0, 5)
    speed = np.asarray((5.0, 10.0, 15.0, 10.0, 5.0))
    acceleration = np.asarray((0.0, 4.0, 0.0, -4.0, 0.0))
    profile_shape = (1, 1, 1, len(grid))
    statistic_shape = (1, 1, len(grid))
    return TeacherMotionModel(
        teacher_ids=("teacher",),
        move_ids=np.asarray((1,)),
        move_names=("move",),
        regions=("whole_body",),
        progress_grid=grid,
        speed_profiles=speed.reshape(profile_shape),
        acceleration_profiles=acceleration.reshape(profile_shape),
        active_profiles=np.ones(profile_shape, dtype=bool),
        speed_median=speed.reshape(statistic_shape),
        speed_mad=np.full(statistic_shape, 1.0),
        speed_p10=(speed - 2.0).reshape(statistic_shape),
        speed_p90=(speed + 2.0).reshape(statistic_shape),
        acceleration_median=acceleration.reshape(statistic_shape),
        acceleration_mad=np.full(statistic_shape, 1.0),
        acceleration_p10=(acceleration - 2.0).reshape(statistic_shape),
        acceleration_p90=(acceleration + 2.0).reshape(statistic_shape),
        acceleration_abs_p95=np.full(statistic_shape, 8.0),
        active_probability=np.ones(statistic_shape),
        duration_median=np.asarray(((1.0,),)),
        amplitude_median=np.asarray(((10.0,),)),
        peak_rate_median=np.asarray(((1.0,),)),
        fragment_count_median=np.asarray(((1.0,),)),
    )


class VelocityQualityTests(unittest.TestCase):
    def test_teacher_model_round_trip_preserves_profiles(self):
        model = simple_teacher_model()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            loaded = TeacherMotionModel.load(path)

        self.assertEqual(loaded.teacher_ids, model.teacher_ids)
        np.testing.assert_allclose(loaded.speed_profiles, model.speed_profiles)

    def test_matching_profile_has_zero_deviation_and_unit_correlation(self):
        model = simple_teacher_model()
        progress = model.progress_grid
        speed = model.speed_median[0, 0]
        acceleration = model.acceleration_median[0, 0]

        metrics, _ = analyze_against_teacher_model(
            model,
            "move",
            "whole_body",
            progress,
            speed,
            acceleration,
            np.ones(len(progress), dtype=bool),
            np.ones(len(progress), dtype=bool),
            fps=5.0,
        )

        self.assertAlmostEqual(metrics.active_mean_speed_ratio, 1.0)
        self.assertAlmostEqual(metrics.velocity_profile_nmae, 0.0)
        self.assertAlmostEqual(metrics.velocity_profile_correlation, 1.0)
        self.assertAlmostEqual(metrics.acceleration_profile_nmae, 0.0)

    def test_resample_profile_collapses_dtw_progress_plateau_with_median(self):
        progress = np.asarray((0.0, 0.5, 0.5, 1.0))
        values = np.asarray((0.0, 4.0, 8.0, 10.0))
        grid = np.asarray((0.0, 0.5, 1.0))

        result = resample_profile(progress, values, grid)

        np.testing.assert_allclose(result, (0.0, 6.0, 10.0))

    def test_profile_statistics_use_neighboring_phase_window(self):
        profiles = np.asarray(((0.0, 10.0, 20.0), (2.0, 12.0, 22.0)))

        stats = profile_statistics(profiles, window_radius=1)

        self.assertAlmostEqual(stats["median"][1], 11.0)
        self.assertGreater(stats["mad"][1], 0.0)

    def test_acceleration_peaks_respect_minimum_separation(self):
        acceleration = np.zeros(30)
        acceleration[[5, 7, 20]] = (10.0, 12.0, 15.0)

        peaks = count_acceleration_peaks(
            acceleration,
            threshold=5.0,
            valid=np.ones(30, dtype=bool),
            fps=10.0,
            minimum_separation_seconds=0.5,
        )

        np.testing.assert_array_equal(peaks, (7, 20))

    def test_motion_fragmentation_joins_short_gap_and_keeps_long_gap(self):
        speed = np.concatenate((np.ones(10), np.zeros(1), np.ones(10), np.zeros(6), np.ones(10)))

        count, active = motion_fragment_count(
            speed,
            np.ones(len(speed), dtype=bool),
            fps=10.0,
            threshold=0.5,
            minimum_active_seconds=0.25,
            bridge_gap_seconds=0.10,
        )

        self.assertEqual(count, 2)
        self.assertTrue(active[10])
        self.assertFalse(np.any(active[21:27]))


if __name__ == "__main__":
    unittest.main()
