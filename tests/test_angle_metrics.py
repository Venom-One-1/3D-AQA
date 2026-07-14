import unittest

import numpy as np
import torch

from aqa3d.angle_metrics import SMPL_24_JOINTS, compute_angle_metrics


def _neutral_pose() -> torch.Tensor:
    joints = torch.zeros((24, 3), dtype=torch.float64)
    points = {
        "Pelvis": (0.0, 0.0, 0.0),
        "L_Hip": (-1.0, 0.0, 0.0),
        "R_Hip": (1.0, 0.0, 0.0),
        "Spine1": (0.0, 1.0, 0.0),
        "L_Knee": (-1.0, -1.0, 0.0),
        "R_Knee": (1.0, -1.0, 0.0),
        "Spine2": (0.0, 2.0, 0.0),
        "L_Ankle": (-1.0, -2.0, 0.0),
        "R_Ankle": (1.0, -2.0, 0.0),
        "Spine3": (0.0, 3.0, 0.0),
        "L_Foot": (-1.0, -2.0, 1.0),
        "R_Foot": (1.0, -2.0, 1.0),
        "Neck": (0.0, 4.0, 0.0),
        "L_Collar": (-0.5, 3.5, 0.0),
        "R_Collar": (0.5, 3.5, 0.0),
        "Head": (0.0, 5.0, 0.0),
        "L_Shoulder": (-1.0, 3.0, 0.0),
        "R_Shoulder": (1.0, 3.0, 0.0),
        "L_Elbow": (-2.0, 3.0, 0.0),
        "R_Elbow": (2.0, 3.0, 0.0),
        "L_Wrist": (-3.0, 3.0, 0.0),
        "R_Wrist": (3.0, 3.0, 0.0),
        "L_Hand": (-4.0, 3.0, 0.0),
        "R_Hand": (4.0, 3.0, 0.0),
    }
    for name, coordinate in points.items():
        joints[SMPL_24_JOINTS[name]] = torch.tensor(coordinate, dtype=torch.float64)
    return joints


class AngleMetricTests(unittest.TestCase):
    def test_single_frame_outputs_all_first_pass_metrics(self):
        result = compute_angle_metrics(_neutral_pose())
        self.assertEqual(result["frame_id"].nunique(), 1)
        self.assertEqual(len(result), 26)
        self.assertIn("left_knee_angle", set(result["metric_id"]))
        self.assertIn("normalized_pelvis_height", set(result["metric_id"]))

    def test_joint_angle_is_in_degrees(self):
        pose = _neutral_pose()
        pose[SMPL_24_JOINTS["L_Ankle"]] = torch.tensor((0.0, -1.0, 0.0), dtype=torch.float64)
        result = compute_angle_metrics(pose)
        row = result[result["metric_id"] == "left_knee_angle"].iloc[0]
        self.assertEqual(row["unit"], "degree")
        self.assertEqual(row["status"], "valid")
        self.assertAlmostEqual(row["value"], 90.0, places=6)

    def test_zero_length_bone_marks_angle_invalid(self):
        pose = _neutral_pose()
        pose[SMPL_24_JOINTS["L_Hip"]] = pose[SMPL_24_JOINTS["L_Knee"]]
        result = compute_angle_metrics(pose)
        row = result[result["metric_id"] == "left_knee_angle"].iloc[0]
        self.assertEqual(row["status"], "invalid_zero_length")
        self.assertTrue(np.isnan(row["value"]))

    def test_frame_ids_are_preserved(self):
        poses = torch.stack((_neutral_pose(), _neutral_pose()), dim=0)
        result = compute_angle_metrics(poses, frame_ids=[10, 25])
        self.assertEqual(sorted(result["frame_id"].unique().tolist()), [10, 25])

    def test_rejects_phalp_45_joint_output(self):
        with self.assertRaisesRegex(ValueError, "24, 3"):
            compute_angle_metrics(torch.zeros((1, 45, 3)))


if __name__ == "__main__":
    unittest.main()
