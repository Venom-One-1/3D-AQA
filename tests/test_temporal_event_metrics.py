import unittest

import numpy as np

from aqa3d.endpoint_metrics import MOVE_METRICS, compute_endpoint_metrics
from aqa3d.pose_measurements import JOINTS as J, reference_statistics
from aqa3d.temporal_event_metrics import (
    compare_measurement, event_metric_plan, interval_relation, metric_catalog,
    process_change, scalar_series, stability_warnings, summarize_scalar,
)
from run_temporal_event_metrics import endpoint_compatibility, validate_events


def skeleton():
    p = np.zeros((24, 3))
    p[J["Neck"]] = [0, 1, 0]
    p[J["Head"]] = [0, 1.3, 0]
    for side, sign in (("L", 1), ("R", -1)):
        for name, xyz in (("Hip", [sign * .2, 0, 0]), ("Knee", [sign * .2, -.5, .1]),
                          ("Ankle", [sign * .2, -1, 0]), ("Shoulder", [sign * .3, .8, 0]),
                          ("Elbow", [sign * .6, .7, 0]), ("Wrist", [sign * .9, .8, .2])):
            p[J[side + "_" + name]] = xyz
    return p


def event(mid, eid):
    return {"move_id": mid, "event_id": eid, "pose_id": f"{mid}.{eid}"}


class TemporalEventMetricsTests(unittest.TestCase):
    def test_keeps_all_twenty_six_f_group_endpoint_slots(self):
        endpoints = ((1, "press_down_and_squat_transition"), (2, "part_mane_3_L"),
                     (3, "separate_hands_to_endpoint"))
        count = 0
        for mid, eid in endpoints:
            plan = event_metric_plan(event(mid, eid))
            self.assertTrue(plan["contains_f_group_endpoint_metrics"])
            self.assertTrue(set(MOVE_METRICS[mid]).issubset(plan["pose_metric_ids"]))
            count += len(MOVE_METRICS[mid])
        self.assertEqual(count, 26)

    def test_all_ten_event_plans_have_distinct_supported_metrics(self):
        events = [(1, "arms_raised_completion"), (1, "press_down_and_squat_transition")]
        events += [(2, f"hold_ball_{i}") for i in range(1, 4)]
        events += [(2, f"part_mane_{i}_{side}") for i, side in ((1, "L"), (2, "R"), (3, "L"))]
        events += [(3, "gather_hands"), (3, "separate_hands_to_endpoint")]
        catalog = metric_catalog()
        for mid, eid in events:
            plan = event_metric_plan(event(mid, eid))
            for field in ("pose_metric_ids", "process_metric_ids"):
                self.assertEqual(len(plan[field]), len(set(plan[field])))
                self.assertTrue(set(plan[field]).issubset(catalog))
        with self.assertRaises(ValueError):
            event_metric_plan(event(4, "unsupported"))

    def test_native_measurements_reproduce_legacy_definitions(self):
        joints = np.repeat(skeleton()[None], 7, axis=0)
        joints[:, J["L_Wrist"], 0] += np.arange(7) * .01
        frames = np.arange(7)
        old = compute_endpoint_metrics(joints, frames)
        catalog = metric_catalog()
        for mid in set(sum((list(x) for x in MOVE_METRICS.values()), [])):
            values, statuses = scalar_series(joints, frames, np.full(7, "valid"), catalog[mid])
            expected = old[old.metric_id == mid].sort_values("frame_id").value.to_numpy()
            np.testing.assert_allclose(values, expected, atol=1e-8, rtol=0)
            self.assertTrue(np.all(statuses == "valid"))

    def test_anatomical_outward_sign_correct_for_both_elbows(self):
        joints, frames = skeleton()[None], np.array([0])
        for side in ("left", "right"):
            values, _ = scalar_series(joints, frames, np.array(["valid"]),
                                      metric_catalog()[f"{side}_elbow_shoulder_outward_torso_ratio"])
            self.assertAlmostEqual(values[0], .3)

    def test_mirrored_bow_and_hand_order(self):
        left = event_metric_plan(event(2, "part_mane_1_L"))["pose_metric_ids"]
        right = event_metric_plan(event(2, "part_mane_2_R"))["pose_metric_ids"]
        self.assertIn("left_wrist_head_up_torso_ratio", left)
        self.assertIn("right_wrist_head_up_torso_ratio", right)
        self.assertIn("L_over_R_wrist_height_torso_ratio", event_metric_plan(event(3, "gather_hands"))["pose_metric_ids"])

    def test_window_median_not_center_drives_value_and_missing_remains_masked(self):
        stats = summarize_scalar(np.array([1., 9., 3.]), np.full(3, "valid"), np.arange(3), 1)
        self.assertEqual(stats["value"], 3)
        self.assertEqual(stats["center_value"], 9)
        self.assertIn("center_window_difference", stability_warnings(stats, "ratio"))
        stats = summarize_scalar(np.array([1., 9., 3.]), np.array(["valid", "missing_tracking", "valid"]), np.arange(3), 1)
        self.assertIsNone(stats["value"])
        self.assertIsNone(stats["center_value"])

    def test_process_uses_disjoint_endpoint_windows_and_signed_change(self):
        values, frames = np.arange(31, dtype=float), np.arange(31)
        result = process_change(values, np.full(31, "valid"), frames, 0, 30, 30, "degree")
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["value"], 24)
        self.assertEqual(result["raw_endpoint_change"], 30)
        self.assertFalse(set(result["start"]["window_source_frames_0based"]) & set(result["end"]["window_source_frames_0based"]))
        reverse = process_change(-values, np.full(31, "valid"), frames, 0, 30, 30, "degree")
        self.assertEqual(reverse["value"], -24)

    def test_collapsed_interval_not_interpolated_into_a_process(self):
        result = process_change(np.ones(4), np.full(4, "valid"), np.arange(4), 2, 2, 30, "ratio")
        self.assertIsNone(result["value"])
        self.assertEqual(result["status"], "insufficient_process_interval")

    def test_process_gaps_preserved_and_sparse_frame_ids_rejected(self):
        values, frames, statuses = np.arange(31, dtype=float), np.arange(31), np.full(31, "valid", dtype="<U32")
        statuses[0:7] = "missing_tracking"
        result = process_change(values, statuses, frames, 0, 30, 30, "degree")
        self.assertIsNone(result["value"])
        self.assertIn("interval_tracking_or_geometry_gaps", result["warnings"])
        for bad in (frames.astype(float), np.delete(frames, 9)):
            with self.assertRaises(ValueError):
                process_change(values[:len(bad)], statuses[:len(bad)], bad, 0, 30, 30, "degree")

    def test_ten_raw_values_retained_even_teacher_has_warning(self):
        samples = [{"subject_id": str(i), "value": float(i), "status": "valid", "warnings": []} for i in range(10)]
        samples[9]["warnings"] = ["window_variation"]
        stats = reference_statistics(samples)
        self.assertEqual(stats["teacher_count"], 10)
        self.assertEqual(stats["valid_teacher_count"], 10)
        self.assertEqual(len(stats["teacher_values"]), 10)
        row = {"kind": "pose", "value": 30, "center_value": 30, "warnings": []}
        result = compare_measurement(row, stats)
        self.assertTrue(result["consistent_outside"])
        self.assertEqual(result["category"], "review_measurement")
        self.assertIsNone(result["is_action_error"])

    def test_stable_outside_is_evidence_not_action_error(self):
        samples = [{"subject_id": str(i), "value": float(i), "status": "valid", "warnings": []} for i in range(10)]
        stats = reference_statistics(samples)
        result = compare_measurement({"kind": "pose", "value": 30, "center_value": 30, "warnings": []}, stats)
        self.assertEqual(result["category"], "difference_evidence")
        self.assertIsNone(result["is_action_error"])
        with self.assertRaises(ValueError):
            interval_relation(1, stats, "invented")

    def test_time_conversion_and_corrected_reference_checked(self):
        reference = [{"pose_id": "3.gather_hands", "technique": "抱手", "interval_start_source_frame_0based": 1328,
                      "representative_source_frame_0based": 1328, "interval_end_source_frame_0based": 1349}]
        mapped = {"pose_id": "3.gather_hands", "technique": "抱手", "status": "mapped",
                  "target_segment_start_source_frame_0based": 20, "target_segment_end_source_frame_0based": 100,
                  "points": {k: {"target_source_frame_0based": 50, "target_phalp_frame_1based": 51,
                                 "target_source_time_seconds": 2,
                                 "reference_source_frame_0based": 1349 if k == "end" else 1328}
                             for k in ("start", "representative", "end")}}
        validate_events([mapped], reference, 25, 101)
        mapped["points"]["representative"]["reference_source_frame_0based"] = 1338
        with self.assertRaises(ValueError):
            validate_events([mapped], reference, 25, 101)

    def test_endpoint_coverage_loss_or_changed_value_fails(self):
        event_row = event(2, "part_mane_3_L")
        with self.assertRaises(ValueError):
            endpoint_compatibility([], [], event_row)


if __name__ == "__main__":
    unittest.main()
