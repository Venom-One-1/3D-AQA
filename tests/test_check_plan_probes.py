import unittest

import numpy as np

from aqa3d.pose_measurements import JOINTS as J, MeasurementQuery as Q
from run_check_plan_probes import dependencies, probe_query


def fixture():
    p = np.zeros((24, 3))
    p[J["Neck"]] = [0, 1, 0]
    p[J["Head"]] = [0, 1.3, 0]
    for s, sign in (("L", 1), ("R", -1)):
        for name, xyz in (("Hip", [sign*.2, 0, 0]), ("Knee", [sign*.2, -.5, 0]),
                          ("Ankle", [sign*.2, -1, 0]), ("Shoulder", [sign*.3, .8, 0]),
                          ("Elbow", [sign*.6, .8, 0]), ("Wrist", [sign*.9, .8, .2])):
            p[J[s+"_"+name]] = xyz
    return p


class CheckPlanProbesTests(unittest.TestCase):
    def test_dependencies_include_frame_scale_and_midpoints(self):
        q = Q("w", "relative_position", ("WristMidpoint", "AnkleMidpoint"), "average_leg_length", "torso_v1")
        required = dependencies(q)
        self.assertTrue({"Pelvis", "Neck", "L_Knee", "L_Wrist", "R_Ankle", "R_Shoulder", "L_Hip"} <= required)
        self.assertNotIn("Head", required)

    def test_all_four_operations_pass_metamorphic_checks(self):
        for q in [Q("k", "joint_angle", ("L_Hip", "L_Knee", "L_Ankle")),
                  Q("r", "relative_position", ("R_Wrist", "Head"), "torso_length", "torso_v1"),
                  Q("d", "distance", ("WristMidpoint", "Neck"), "shoulder_width"),
                  Q("v", "vector_angle", ("L_Hip", "L_Knee", "R_Hip", "R_Knee"))]:
            rows = probe_query(fixture(), q)
            self.assertTrue(all(r["status"] == "pass" for r in rows), rows)

    def test_wrist_raise_has_signed_expected_change(self):
        q = Q("r", "relative_position", ("R_Wrist", "Head"), "torso_length", "torso_v1")
        rows = probe_query(fixture(), q)
        self.assertIn("wrist_up_expected_delta_0_1", {r["probe"] for r in rows})
        self.assertTrue(all(r["max_abs_error"] < 1e-8 for r in rows))

    def test_invalid_geometry_not_counted_as_pass(self):
        q = Q("d", "distance", ("L_Wrist", "Head"), "torso_length")
        rows = probe_query(np.zeros((24, 3)), q)
        self.assertTrue(all(r["status"] == "not_testable" for r in rows))


if __name__ == "__main__":
    unittest.main()
