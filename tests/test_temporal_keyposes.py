import json
from pathlib import Path
import unittest

import numpy as np

from aqa3d.pose_measurements import JOINTS
from aqa3d.temporal_keyposes import (
    MetricSignals,
    compute_raw_metrics,
    locate_events,
    prepare_metric_signals,
    rolling_median,
    rolling_slope,
    validate_temporal_plan,
)


ROOT = Path(__file__).resolve().parents[1]


def load_plan():
    return json.loads((ROOT / "temporal_query_specs" / "qishi_v1.json").read_text())


def skeleton_sequence(count=5):
    joints = np.zeros((count, 24, 3), dtype=np.float64)
    joints[:, JOINTS["Neck"]] = [0, 1, 0]
    joints[:, JOINTS["Head"]] = [0, 1.3, 0]
    for side, sign in (("L", 1), ("R", -1)):
        for name, xyz in (
            ("Hip", [sign * .2, 0, 0]),
            ("Knee", [sign * .2, -.5, 0]),
            ("Ankle", [sign * .2, -1, 0]),
            ("Shoulder", [sign * .3, .8, 0]),
            ("Elbow", [sign * .6, .8, 0]),
            ("Wrist", [sign * .9, .8, .2]),
        ):
            joints[:, JOINTS[f"{side}_{name}"]] = xyz
    return joints


def piecewise(times, start, peak_start, peak_end, end, low, high, final):
    values = np.full(len(times), low, dtype=np.float64)
    rising = (times >= start) & (times < peak_start)
    values[rising] = low + (high - low) * (times[rising] - start) / (peak_start - start)
    values[(times >= peak_start) & (times <= peak_end)] = high
    falling = (times > peak_end) & (times <= end)
    values[falling] = high + (final - high) * (times[falling] - peak_end) / (end - peak_end)
    values[times > end] = final
    return values


class TemporalKeyposeTests(unittest.TestCase):
    def test_frozen_qishi_plan_is_valid(self):
        validate_temporal_plan(load_plan())

    def test_plan_rejects_unknown_metric_and_future_dependency(self):
        plan = load_plan()
        plan["events"][0]["query"]["conditions"][0]["metric_id"] = "made_up"
        with self.assertRaisesRegex(ValueError, "Unknown metric"):
            validate_temporal_plan(plan)
        plan = load_plan()
        plan["events"][0]["after_event"] = {
            "event_id": "press_down_and_squat_transition",
            "anchor": "end",
            "offset_seconds": 0,
        }
        with self.assertRaisesRegex(ValueError, "not earlier"):
            validate_temporal_plan(plan)

    def test_raw_metrics_use_bilateral_mean_and_tracking_mask(self):
        plan = load_plan()
        joints = skeleton_sequence()
        joints[:, JOINTS["L_Wrist"], 1] = [.7, .8, .9, 1., 1.1]
        joints[:, JOINTS["R_Wrist"], 1] = [.5, .6, .7, .8, .9]
        statuses = np.array(["valid", "valid", "missing_tracking", "valid", "valid"])
        metrics = compute_raw_metrics(plan, joints, np.arange(5), statuses)
        self.assertAlmostEqual(metrics["mean_wrist_pelvis_up"][0], .6)
        self.assertTrue(np.isnan(metrics["mean_wrist_pelvis_up"][2]))
        self.assertTrue(np.isfinite(metrics["mean_knee_angle"][[0, 1, 3, 4]]).all())

    def test_rolling_operations_keep_edges_and_missing_values(self):
        times = np.arange(5, dtype=float)
        values = np.array([0., 1., np.nan, 3., 4.])
        np.testing.assert_allclose(
            rolling_median(values, times, 2), [.5, .5, 2., 3.5, 3.5],
        )
        slope = rolling_slope(np.arange(5, dtype=float), times, 4)
        np.testing.assert_allclose(slope, np.ones(5), atol=1e-12)

    def test_locates_stable_completion_and_terminal_transition(self):
        plan = load_plan()
        times = np.arange(0, 16.0 + 1 / 30, 1 / 30)
        frames = np.arange(len(times))
        wrist_up = piecewise(times, 8.5, 11.5, 12.5, 16, -.1, .95, .12)
        wrist_forward = piecewise(times, 8.5, 11.3, 12.5, 16, .02, .92, .30)
        knee = piecewise(times, 0, 0.1, 12.4, 16, 150, 150, 108)
        elbow = piecewise(times, 0, 0.1, 12.4, 16, 145, 145, 105)
        raw = {
            "mean_wrist_pelvis_up": wrist_up,
            "mean_wrist_pelvis_forward": wrist_forward,
            "mean_knee_angle": knee,
            "mean_elbow_angle": elbow,
        }
        signals = prepare_metric_signals(
            raw, times, smoothing_seconds=.2, derivative_window_seconds=.35,
        )
        detections = locate_events(plan, signals, frames, times, 30)
        raised = detections["arms_raised_completion"]["result"]
        transition = detections["press_down_and_squat_transition"]["result"]
        self.assertEqual(raised["status"], "located")
        self.assertLessEqual(11.0, raised["representative_time_seconds"])
        self.assertLessEqual(raised["representative_time_seconds"], 13.0)
        self.assertEqual(transition["status"], "located")
        self.assertGreaterEqual(transition["start_time_seconds"], 12.0)
        self.assertEqual(transition["end_time_seconds"], 16.0)
        self.assertEqual(transition["representative_time_seconds"], 16.0)

    def test_constant_sequence_is_not_forced_into_a_local_maximum(self):
        plan = load_plan()
        times = np.arange(0, 16.0 + .1, .1)
        frames = np.arange(len(times))
        raw = {
            "mean_wrist_pelvis_up": np.zeros(len(times)),
            "mean_wrist_pelvis_forward": np.zeros(len(times)),
            "mean_knee_angle": np.full(len(times), 140.),
            "mean_elbow_angle": np.full(len(times), 140.),
        }
        signals = prepare_metric_signals(
            raw, times, smoothing_seconds=.2, derivative_window_seconds=.35,
        )
        detections = locate_events(plan, signals, frames, times, 10)
        self.assertEqual(detections["arms_raised_completion"]["result"]["status"], "not_located")
        self.assertEqual(
            detections["press_down_and_squat_transition"]["result"]["reason"],
            "unresolved_dependency",
        )

    def test_missing_required_metric_returns_not_located(self):
        plan = load_plan()
        times = np.arange(0, 16.0 + .1, .1)
        frames = np.arange(len(times))
        finite = np.sin(times)
        signals = {
            "mean_wrist_pelvis_up": MetricSignals(
                np.full(len(times), np.nan), np.full(len(times), np.nan),
                np.full(len(times), np.nan), np.full(len(times), np.nan),
            ),
            "mean_wrist_pelvis_forward": MetricSignals(finite, finite, finite, finite),
            "mean_knee_angle": MetricSignals(finite, finite, finite, finite),
            "mean_elbow_angle": MetricSignals(finite, finite, finite, finite),
        }
        result = locate_events(plan, signals, frames, times, 10)
        self.assertEqual(
            result["arms_raised_completion"]["result"]["reason"],
            "insufficient_valid_data",
        )


if __name__ == "__main__":
    unittest.main()
