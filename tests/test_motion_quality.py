import math
import unittest

import numpy as np

from aqa3d.motion_quality import (
    analyze_region,
    build_reference_progress,
    compute_motion_signals,
    detect_pauses,
    tempo_bins,
    tempo_stall_mask,
)
from run_motion_quality_analysis import ranking_value


def rotation_z(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def pose_sequence(angles: np.ndarray, joint: int = 15) -> np.ndarray:
    poses = np.broadcast_to(np.eye(3), (len(angles), 23, 3, 3)).copy()
    poses[:, joint] = np.stack([rotation_z(float(angle)) for angle in angles])
    return poses


class MotionQualityTests(unittest.TestCase):
    def test_amplitude_ranking_can_compare_larger_or_teacher_closeness(self):
        self.assertLess(
            ranking_value("amplitude_ratio", 1.4, amplitude_ranking="larger_is_better"),
            ranking_value("amplitude_ratio", 0.9, amplitude_ranking="larger_is_better"),
        )
        self.assertLess(
            ranking_value("amplitude_ratio", 0.9, amplitude_ranking="teacher_closeness"),
            ranking_value("amplitude_ratio", 1.4, amplitude_ranking="teacher_closeness"),
        )

    def test_amplitude_absolute_difference_is_symmetric_around_one(self):
        below = ranking_value(
            "amplitude_ratio", 0.8, amplitude_ranking="absolute_difference"
        )
        above = ranking_value(
            "amplitude_ratio", 1.2, amplitude_ranking="absolute_difference"
        )
        exact = ranking_value(
            "amplitude_ratio", 1.0, amplitude_ranking="absolute_difference"
        )

        self.assertAlmostEqual(below, above)
        self.assertEqual(exact, 0.0)

    def test_constant_pose_has_pause_and_nearly_zero_amplitude(self):
        fps = 30.0
        poses = pose_sequence(np.zeros(90))
        signals = compute_motion_signals(poses, fps, smoothing_seconds=0.0)
        pauses = detect_pauses(
            signals.region_intensity["left_arm"],
            signals.valid_frames,
            threshold=0.0,
            fps=fps,
        )

        self.assertGreater(pauses.pause_ratio, 0.95)
        self.assertEqual(pauses.pause_count, 1)
        self.assertLess(float(np.nansum(signals.region_intensity["left_arm"]) / fps), 1e-8)

    def test_uniform_rotation_has_expected_speed_and_no_pause(self):
        fps = 30.0
        angles = np.deg2rad(np.arange(120, dtype=np.float64))
        signals = compute_motion_signals(
            pose_sequence(angles),
            fps,
            smoothing_seconds=0.0,
        )
        intensity = signals.region_intensity["left_arm"]
        expected_rms = 30.0 / math.sqrt(3.0)
        self.assertAlmostEqual(float(np.nanmedian(intensity)), expected_rms, places=5)

        pauses = detect_pauses(
            intensity,
            signals.valid_frames,
            threshold=expected_rms * 0.5,
            fps=fps,
        )
        self.assertEqual(pauses.pause_count, 0)
        self.assertAlmostEqual(pauses.pause_ratio, 0.0)
        self.assertLess(
            float(
                np.nanmax(
                    np.abs(signals.joint_speed_change_degrees_per_second2[:, 15])
                )
            ),
            1e-8,
        )

    def test_linearly_increasing_speed_has_constant_speed_change(self):
        fps = 30.0
        frame_count = 120
        time = np.arange(frame_count, dtype=np.float64) / fps
        angular_acceleration = 20.0
        angles = np.deg2rad(0.5 * angular_acceleration * time * time)
        signals = compute_motion_signals(
            pose_sequence(angles),
            fps,
            smoothing_seconds=0.0,
        )
        changes = signals.joint_speed_change_degrees_per_second2[:, 15]

        self.assertAlmostEqual(float(np.nanmedian(changes[3:-3])), angular_acceleration, places=5)

    def test_inserted_pause_increases_aligned_pause_ratio(self):
        fps = 30.0
        teacher_angles = np.deg2rad(np.arange(180, dtype=np.float64))
        student_angles = teacher_angles.copy()
        student_angles[60:90] = student_angles[59]
        student_angles[90:] -= student_angles[90] - student_angles[59]
        teacher = compute_motion_signals(
            pose_sequence(teacher_angles),
            fps,
            smoothing_seconds=0.0,
        )
        student = compute_motion_signals(
            pose_sequence(student_angles),
            fps,
            smoothing_seconds=0.0,
        )
        progress = np.linspace(0.0, 1.0, len(student_angles))

        metrics, events, _, _ = analyze_region(
            "left_arm",
            teacher,
            student,
            progress,
            len(teacher_angles) / fps,
            len(student_angles) / fps,
            fps,
            fps,
            activity_dilation_seconds=0.0,
        )

        self.assertGreater(metrics.aligned_active_pause_ratio, 0.10)
        self.assertGreaterEqual(metrics.student_pause_count, 1)
        self.assertGreaterEqual(metrics.student_longest_pause_seconds, 0.9)
        self.assertTrue(any(event.reason == "aligned_active_pause" for event in events))

    def test_uniform_slow_motion_changes_duration_without_false_pause(self):
        teacher_fps = 30.0
        student_fps = 30.0
        teacher_angles = np.deg2rad(np.arange(90, dtype=np.float64))
        student_angles = np.deg2rad(np.arange(180, dtype=np.float64) * 0.5)
        teacher = compute_motion_signals(
            pose_sequence(teacher_angles),
            teacher_fps,
            smoothing_seconds=0.0,
        )
        student = compute_motion_signals(
            pose_sequence(student_angles),
            student_fps,
            smoothing_seconds=0.0,
        )
        progress = np.linspace(0.0, 1.0, len(student_angles))

        metrics, _, _, _ = analyze_region(
            "left_arm",
            teacher,
            student,
            progress,
            len(teacher_angles) / teacher_fps,
            len(student_angles) / student_fps,
            teacher_fps,
            student_fps,
            activity_dilation_seconds=0.0,
        )

        self.assertAlmostEqual(metrics.duration_ratio, 2.0)
        self.assertAlmostEqual(metrics.aligned_active_pause_ratio, 0.0)
        self.assertAlmostEqual(metrics.tempo_stall_ratio, 0.0)

    def test_teacher_natural_pause_is_not_aligned_active_pause(self):
        fps = 30.0
        moving = np.deg2rad(np.arange(60, dtype=np.float64))
        angles = np.concatenate((moving, np.full(60, moving[-1])))
        teacher = compute_motion_signals(
            pose_sequence(angles),
            fps,
            smoothing_seconds=0.0,
        )
        student = compute_motion_signals(
            pose_sequence(angles),
            fps,
            smoothing_seconds=0.0,
        )
        progress = np.linspace(0.0, 1.0, len(angles))

        metrics, _, _, _ = analyze_region(
            "left_arm",
            teacher,
            student,
            progress,
            len(angles) / fps,
            len(angles) / fps,
            fps,
            fps,
            activity_dilation_seconds=0.0,
        )

        self.assertAlmostEqual(metrics.aligned_active_pause_ratio, 0.0)

    def test_track_switch_frames_do_not_create_speed_peak(self):
        fps = 30.0
        angles = np.zeros(60, dtype=np.float64)
        angles[30:] = math.pi
        track_ids = np.ones(60, dtype=np.int64)
        track_ids[30:] = 2
        signals = compute_motion_signals(
            pose_sequence(angles),
            fps,
            source_track_ids=track_ids,
            smoothing_seconds=0.2,
        )

        self.assertFalse(signals.valid_frames[29])
        self.assertFalse(signals.valid_frames[30])
        self.assertTrue(np.isnan(signals.joint_speed_degrees_per_second[30, 15]))
        self.assertEqual(signals.excluded_track_switch_frames, 2)
        self.assertLess(float(np.nanmax(signals.joint_speed_degrees_per_second[:, 15])), 1e-8)

    def test_dtw_many_to_one_uses_median_and_preserves_student_frames(self):
        path = np.asarray(
            ((0, 0), (1, 1), (1, 2), (1, 3), (2, 4), (3, 5)),
            dtype=np.int64,
        )
        progress, reference = build_reference_progress(
            path,
            target_sample_indices=np.arange(4),
            target_source_indices=np.asarray((0, 6, 12, 18)),
            target_source_frames=np.arange(19),
            reference_start_index=0,
            reference_end_index=5,
        )

        self.assertEqual(len(progress), 19)
        self.assertAlmostEqual(reference[6], 2.0)
        self.assertTrue(np.all(np.diff(progress) >= 0))

    def test_progress_plateau_is_detected_as_tempo_stall(self):
        fps = 30.0
        progress = np.concatenate(
            (
                np.linspace(0.0, 0.3, 60, endpoint=False),
                np.full(30, 0.3),
                np.linspace(0.3, 1.0, 90),
            )
        )
        active = np.ones(len(progress), dtype=bool)
        valid = np.ones(len(progress), dtype=bool)
        stall, _, _ = tempo_stall_mask(
            progress,
            active,
            valid,
            fps,
            student_duration=len(progress) / fps,
            teacher_duration=4.0,
        )

        self.assertGreater(np.sum(stall[60:90]), 10)

    def test_pause_then_catch_up_increases_local_tempo_distortion(self):
        fps = 30.0
        uniform = np.linspace(0.0, 1.0, 180)
        distorted = np.concatenate(
            (
                np.linspace(0.0, 0.3, 45, endpoint=False),
                np.full(45, 0.3),
                np.linspace(0.3, 1.0, 90),
            )
        )

        _, uniform_distortion = tempo_bins(uniform, fps, 6.0, "whole_body")
        _, distorted_distortion = tempo_bins(distorted, fps, 6.0, "whole_body")

        self.assertGreater(distorted_distortion, uniform_distortion)


if __name__ == "__main__":
    unittest.main()
