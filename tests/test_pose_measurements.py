import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from aqa3d.pose_measurements import (
    JOINTS as J, MeasurementQuery as Q, measure_pose, measurement_contract,
    reference_statistics, summarize_window,
)
from run_pose_measurements import example_queries, export_experiment, load_queries, write_json


def skeleton():
    p = np.zeros((24, 3))
    p[J["Neck"]] = [0, 1, 0]
    p[J["Head"]] = [0, 1.3, 0]
    for side, sign in (("L", 1), ("R", -1)):
        for name, xyz in (("Hip", [sign*.2, 0, 0]), ("Knee", [sign*.2, -.5, 0]),
                          ("Ankle", [sign*.2, -1, 0]), ("Shoulder", [sign*.3, .8, 0]),
                          ("Elbow", [sign*.6, .8, 0]), ("Wrist", [sign*.9, .8, .2])):
            p[J[side+"_"+name]] = xyz
    return p


class PoseMeasurementsTests(unittest.TestCase):
    def test_native_joint_order_matches_legacy_contract(self):
        from aqa3d.angle_metrics import SMPL_24_JOINTS
        self.assertEqual(J, SMPL_24_JOINTS)

    def test_known_joint_angles(self):
        p = skeleton()
        q = Q("angle", "joint_angle", ("L_Shoulder", "L_Elbow", "L_Wrist"))
        for point, expected in (([.9, .8, 0], 180), ([.6, 1.1, 0], 90), ([.3, .8, 0], 0)):
            p[J["L_Wrist"]] = point
            self.assertAlmostEqual(measure_pose(p, q).values[0, 0], expected)

    def test_vector_angle_and_order(self):
        q = Q("v", "vector_angle", ("L_Shoulder", "L_Elbow", "Pelvis", "Neck"))
        self.assertAlmostEqual(measure_pose(skeleton(), q).values[0, 0], 90)
        q = Q("parallel", "vector_angle", ("L_Shoulder", "L_Elbow", "R_Elbow", "R_Shoulder"))
        self.assertAlmostEqual(measure_pose(skeleton(), q).values[0, 0], 0)
        q = Q("opposite", "vector_angle", ("L_Shoulder", "L_Elbow", "R_Shoulder", "R_Elbow"))
        self.assertAlmostEqual(measure_pose(skeleton(), q).values[0, 0], 180)

    def test_signed_relative_position(self):
        q = Q("r", "relative_position", ("R_Wrist", "Head"), "torso_length", "torso_v1")
        result = measure_pose(skeleton(), q)
        np.testing.assert_allclose(result.values, [[.9, -.5, .2]])
        reverse = Q("r", "relative_position", ("Head", "R_Wrist"), "torso_length", "torso_v1")
        np.testing.assert_allclose(measure_pose(skeleton(), reverse).values, -result.values)

    def test_normalized_distance_and_midpoint(self):
        q = Q("mid", "distance", ("WristMidpoint", "Neck"), "torso_length")
        self.assertAlmostEqual(measure_pose(skeleton(), q).values[0, 0], np.sqrt(.08))
        q = Q("zero", "distance", ("HipMidpoint", "Pelvis"), "average_leg_length")
        self.assertEqual(measure_pose(skeleton(), q).values[0, 0], 0)

    def test_all_queries_invariant_to_rigid_transform_and_uniform_scale(self):
        rotation, _ = np.linalg.qr(np.array([[1., 2, 3], [4, 6, 9], [1, 4, 8]]))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        p = skeleton()
        for q in example_queries():
            with self.subTest(query=q.query_id):
                a = measure_pose(p, q)
                b = measure_pose(2.8 * p @ rotation.T + [4, -3, 9], q)
                np.testing.assert_allclose(a.values, b.values, atol=1e-10)
                self.assertEqual(b.statuses.tolist(), ["valid"])

    def test_unrelated_point_does_not_change_elbow(self):
        q = Q("elbow", "joint_angle", ("R_Shoulder", "R_Elbow", "R_Wrist"))
        p = skeleton()
        baseline = measure_pose(p, q).values.copy()
        p[J["L_Foot"]] = [100, 90, 40]
        np.testing.assert_array_equal(measure_pose(p, q).values, baseline)
        p[J["Head"]] = np.nan
        np.testing.assert_array_equal(measure_pose(p, q).values, baseline)

    def test_local_perturbation_has_expected_direction(self):
        p = skeleton()
        q = Q("r", "relative_position", ("R_Wrist", "Head"), "torso_length", "torso_v1")
        original = measure_pose(p, q).values
        p[J["R_Wrist"], 2] += .1
        np.testing.assert_allclose(measure_pose(p, q).values-original, [[0, 0, .1]])

    def test_degenerate_vectors_and_missing_points(self):
        q = Q("a", "joint_angle", ("L_Hip", "L_Knee", "L_Ankle"))
        p = skeleton()
        p[J["L_Hip"]] = p[J["L_Knee"]]
        self.assertEqual(measure_pose(p, q).statuses[0], "invalid_zero_length")
        p[J["L_Hip"]] = np.nan
        result = measure_pose(p, q)
        self.assertEqual(result.statuses[0], "missing_point")
        self.assertTrue(np.isnan(result.values).all())

    def test_bad_frame_does_not_invalidate_unrelated_angles(self):
        p = skeleton()
        p[J["Neck"]] = p[J["Pelvis"]]
        angle = Q("a", "joint_angle", ("L_Hip", "L_Knee", "L_Ankle"))
        self.assertEqual(measure_pose(p, angle).statuses[0], "valid")
        relative = Q("r", "relative_position", ("L_Wrist", "Head"), "shoulder_width", "torso_v1")
        self.assertEqual(measure_pose(p, relative).statuses[0], "invalid_body_frame")
        distance = Q("d", "distance", ("L_Wrist", "L_Hip"), "torso_length")
        self.assertEqual(measure_pose(p, distance).statuses[0], "invalid_normalizer")

    def test_reversed_shoulder_axis_is_flagged_not_silently_flipped(self):
        p = skeleton()
        p[[J["L_Shoulder"], J["R_Shoulder"]]] = p[[J["R_Shoulder"], J["L_Shoulder"]]]
        q = Q("r", "relative_position", ("L_Wrist", "Head"), "torso_length", "torso_v1")
        self.assertEqual(measure_pose(p, q).statuses[0], "invalid_body_frame")

    def test_query_validation_no_unsupported_proxy_or_frame(self):
        for kwargs in (
            dict(query_id="a", operation="relaxation", points=("Neck",)),
            dict(query_id="a", operation="distance", points=("Forehead", "R_Wrist"), normalizer="torso_length"),
            dict(query_id="a", operation="relative_position", points=("Pelvis", "Neck"), normalizer="torso_length", coordinate_frame="gravity"),
            dict(query_id="a", operation="distance", points=("Pelvis", "Neck")),
            dict(query_id="a", operation="joint_angle", points=("Pelvis", "Neck")),
            dict(query_id="a", operation="vector_angle", points=("Pelvis", "Neck", "L_Hip", "R_Hip"), normalizer="torso_length"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Q(**kwargs)
        with self.assertRaises(ValueError):
            Q.from_dict({"query_id": "x", "operation": "distance", "points": [], "code": "1+2"})

    def test_query_loader_strict_and_unique(self):
        from dataclasses import asdict
        q = asdict(example_queries()[0])
        self.assertEqual(len(load_queries({"version": "1.0", "queries": [q]})), 1)
        for payload in ({"version": "2", "queries": [q]}, {"version": "1.0", "queries": [q, q]},
                        {"version": "1.0", "queries": []}):
            with self.assertRaises(ValueError):
                load_queries(payload)

    def test_frame_validation(self):
        q = example_queries()[0]
        p = np.repeat(skeleton()[None], 2, axis=0)
        for frames in ([0, 0], [1, 0], [-1, 0], [0., 1.], [0]):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                measure_pose(p, q, frame_ids=frames)
        for points in (np.zeros((2, 23, 3)), np.zeros((0, 24, 3))):
            with self.assertRaises(ValueError):
                measure_pose(points, q)

    def test_tracking_mask_and_insufficient_window(self):
        q = example_queries()[0]
        p = np.repeat(skeleton()[None], 5, axis=0)
        result = measure_pose(p, q, frame_ids=[6, 7, 8, 9, 10],
                              tracking_status=["valid"]*2+["near_track_switch"]*3)
        stats = summarize_window(result, 10)
        self.assertEqual(stats["valid_frame_count"], 2)
        self.assertEqual(stats["status"], "insufficient_valid_window")
        self.assertIsNone(stats["components"]["value"]["value"])
        self.assertIsNone(stats["components"]["value"]["center_value"])
        self.assertEqual(stats["components"]["value"]["window_median"], 180)

    def test_componentwise_summary_keeps_center(self):
        q = Q("r", "relative_position", ("R_Wrist", "Head"), "torso_length", "torso_v1")
        p = np.repeat(skeleton()[None], 5, axis=0)
        p[:, J["R_Wrist"], 2] = [0, .1, .2, .3, 1.]
        result = measure_pose(p, q, frame_ids=[6, 7, 8, 9, 10])
        stats = summarize_window(result, 10)
        self.assertAlmostEqual(stats["components"]["forward"]["value"], .2)
        self.assertAlmostEqual(stats["components"]["forward"]["center_value"], 1.)
        with self.assertRaises(ValueError):
            summarize_window(result, 11)

    def test_reference_retains_missing_teachers_and_unscaled_mad(self):
        samples = [{"subject_id": str(i), "value": i, "status": "valid"} for i in range(10)]
        samples.append({"subject_id": "missing", "value": None, "status": "insufficient_valid_window"})
        ref = reference_statistics(samples)
        self.assertEqual(ref["teacher_count"], 11)
        self.assertEqual(ref["valid_teacher_count"], 10)
        self.assertEqual(ref["median"], 4.5)
        self.assertEqual(ref["mad"], 2.5)
        self.assertAlmostEqual(ref["p10"], .9)
        self.assertEqual(ref["teacher_values"], samples)
        self.assertEqual(reference_statistics(samples[:3])["status"], "insufficient_teachers")
        with self.assertRaises(ValueError):
            reference_statistics(samples + samples)

    def test_all_missing_and_constant_reference(self):
        ref = reference_statistics([{"subject_id": "t", "value": None, "status": "missing"}])
        self.assertIsNone(ref["median"])
        ref = reference_statistics([{"subject_id": str(i), "value": 2., "status": "valid"} for i in range(10)])
        self.assertTrue(ref["zero_width_p10_p90"])
        self.assertTrue(ref["zero_width_median_2mad"])
        self.assertNotIn("diagnosis", ref)
        json.dumps(ref, allow_nan=False)

    def test_contract_is_serializable_and_explicit(self):
        contract = measurement_contract()
        json.dumps(contract, allow_nan=False)
        self.assertIn("gravity_frame", contract["unsupported"])
        self.assertEqual(set(contract["operations"]), {"joint_angle", "vector_angle", "distance", "relative_position"})


def create_fixture(root):
    from aqa3d.endpoint_metrics import MOVE_METRICS, aggregate_window, compute_endpoint_metrics
    root.mkdir()
    manifest = {"endpoint_keyposes": [{"move_id": 1, "pose_id": "1.end"}]}
    write_json(root/"reference_manifest_snapshot.json", manifest)
    digest = hashlib.sha256((root/"reference_manifest_snapshot.json").read_bytes()).hexdigest()
    config = {"status": "ok", "teacher_ids": ["T1", "T2"], "student_ids": ["01"],
              "move_ids": [1], "window_seconds_each_side": .2, "window_clipping": "within_current_move",
              "minimum_window_valid_ratio": .7, "minimum_valid_frames": 3, "minimum_teachers": 2,
              "reference_manifest_sha256": digest}
    write_json(root/"summary.json", config)
    frames = np.arange(4, 11)
    for role, sid in (("teacher", "T1"), ("teacher", "T2"), ("student", "01")):
        path = root/role/sid
        path.mkdir(parents=True)
        event = {"move_id": 1, "pose_id": "1.end", "final_technique_step": "test technique",
                 "center_source_frame_0based": 10, "segment_start_source_frame_0based": 0,
                 "segment_end_source_frame_0based": 10, "boundary_time_seconds": .4,
                 "source_frame_time_seconds": 10/30}
        write_json(path/"alignment.json", {"subject_id": sid, "role": role, "source_fps": 30.,
                                           "manifest_sha256": digest, "endpoints": [event]})
        joints = np.repeat(skeleton()[None], len(frames), axis=0)
        np.savez(path/"01_window_joints.npz", source_frames_0based=frames,
                 native_smpl24_joints=joints, tracking_status=np.asarray(["valid"]*len(frames)))
        old = compute_endpoint_metrics(joints, frames)
        summaries = []
        raw = []
        for metric in MOVE_METRICS[1]:
            selected = old[old.metric_id == metric]
            summaries.append({"move_id": 1, "metric_id": metric,
                              **aggregate_window(selected.value, selected.status, frames, 10)})
            raw.extend({"move_id": 1, "metric_id": metric, "source_frame_0based": int(r.frame_id),
                        "value": r.value, "status": r.status} for r in selected.itertuples())
        for name, rows in (("endpoint_metrics.csv", summaries), ("frame_metrics.csv", raw)):
            with (path/name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


class PoseMeasurementExportTests(unittest.TestCase):
    def test_frozen_export_and_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root, out = Path(directory)/"input", Path(directory)/"output"
            create_fixture(root)
            before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            summary = export_experiment(root, out)
            self.assertEqual(summary["case_count"], 1)
            self.assertEqual(summary["legacy_passed"], 27)
            self.assertEqual(summary["legacy_failed"], 0)
            case = json.loads((out/"cases/01_01.json").read_text())
            self.assertEqual(case["student_endpoint"]["center_source_frame_0based"], 10)
            for m in case["measurements"]:
                for ref in m["teachers"]["components"].values():
                    self.assertEqual(len(ref["teacher_values"]), 2)
            for relative, content in before.items():
                self.assertEqual((root/relative).read_bytes(), content)
            with self.assertRaises(FileExistsError):
                export_experiment(root, out)
            with self.assertRaises(ValueError):
                export_experiment(root, root/"new_results")

    def test_wrong_window_cannot_be_silently_shifted(self):
        with tempfile.TemporaryDirectory() as directory:
            root, out = Path(directory)/"input", Path(directory)/"output"
            create_fixture(root)
            path = root/"teacher/T1/01_window_joints.npz"
            np.savez(path, source_frames_0based=np.arange(5, 12),
                     native_smpl24_joints=np.repeat(skeleton()[None], 7, axis=0),
                     tracking_status=np.asarray(["valid"]*7))
            with self.assertRaisesRegex(ValueError, "window disagrees"):
                export_experiment(root, out)

    def test_custom_queries_and_unknown_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root, out = Path(directory)/"input", Path(directory)/"output"
            create_fixture(root)
            query = Path(directory)/"queries.json"
            write_json(query, {"version": "1.0", "queries": [{"query_id": "test", "operation": "distance",
                       "points": ["WristMidpoint", "Neck"], "normalizer": "torso_length"}]})
            summary = export_experiment(root, out, queries_path=query)
            self.assertEqual(summary["query_count"], 1)
            self.assertEqual(summary["legacy_passed"], 27)
            with self.assertRaises(ValueError):
                export_experiment(root, Path(directory)/"bad", students=["99"])


if __name__ == "__main__":
    unittest.main()
