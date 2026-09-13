import unittest

import numpy as np

from aqa3d.angle_metrics import SMPL_24_JOINTS
from aqa3d.pelvis_quality import compute_pelvis_signals


def _synthetic_sequence(frame_count: int = 61) -> tuple[np.ndarray, np.ndarray]:
    index = SMPL_24_JOINTS
    joints = np.zeros((frame_count, 24, 3), dtype=np.float64)
    for frame in range(frame_count):
        joints[frame, index["Pelvis"]] = [0.0, -1.0, 0.0]
        joints[frame, index["Neck"]] = [0.0, -1.6, 0.0]
        joints[frame, index["L_Hip"]] = [0.2, -0.95, 0.0]
        joints[frame, index["R_Hip"]] = [-0.2, -0.95, 0.0]
        joints[frame, index["L_Knee"]] = [0.2, -0.5, 0.0]
        joints[frame, index["R_Knee"]] = [-0.2, -0.5, 0.0]
        joints[frame, index["L_Ankle"]] = [0.2, 0.0, 0.0]
        joints[frame, index["R_Ankle"]] = [-0.2, 0.0, 0.0]
        joints[frame, index["L_Shoulder"]] = [0.3, -1.5, 0.0]
        joints[frame, index["R_Shoulder"]] = [-0.3, -1.5, 0.0]
    camera = np.zeros((frame_count, 3), dtype=np.float64)
    camera[:, 0] = np.linspace(0.0, -0.4, frame_count)
    return joints, camera


class PelvisQualityTests(unittest.TestCase):
    def test_body_centric_lateral_signal_is_yaw_invariant(self):
        joints, camera = _synthetic_sequence()
        frame_numbers = np.arange(1, len(joints) + 1)
        track_ids = np.ones(len(joints), dtype=np.int64)
        original = compute_pelvis_signals(
            frame_numbers,
            joints,
            camera,
            track_ids,
            30.0,
        )

        angle = np.deg2rad(55.0)
        rotation = np.asarray(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        rotated = compute_pelvis_signals(
            frame_numbers,
            joints @ rotation.T,
            camera @ rotation.T,
            track_ids,
            30.0,
        )
        np.testing.assert_allclose(
            original.root_lateral,
            rotated.root_lateral,
            atol=1e-8,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            original.support_height,
            rotated.support_height,
            atol=1e-8,
            equal_nan=True,
        )

    def test_vertical_residual_detects_added_bobbing(self):
        joints, camera = _synthetic_sequence(121)
        progress = np.linspace(0.0, 1.0, len(joints))
        joints[:, SMPL_24_JOINTS["Pelvis"], 1] += 0.2 * progress
        common = dict(
            frame_numbers=np.arange(1, len(joints) + 1),
            camera_translations=camera,
            track_ids=np.ones(len(joints), dtype=np.int64),
            fps=30.0,
        )
        smooth = compute_pelvis_signals(joints=joints, **common)
        bobbing_joints = joints.copy()
        bobbing_joints[:, SMPL_24_JOINTS["Pelvis"], 1] += 0.04 * np.sin(
            2.0 * np.pi * 3.0 * progress
        )
        bobbing = compute_pelvis_signals(joints=bobbing_joints, **common)
        smooth_rms = np.sqrt(np.nanmean(np.square(smooth.support_vertical_residual)))
        bobbing_rms = np.sqrt(np.nanmean(np.square(bobbing.support_vertical_residual)))
        self.assertGreater(bobbing_rms, smooth_rms * 2.0)

    def test_track_switch_frames_are_excluded(self):
        joints, camera = _synthetic_sequence()
        track_ids = np.ones(len(joints), dtype=np.int64)
        track_ids[30:] = 2
        signals = compute_pelvis_signals(
            np.arange(1, len(joints) + 1),
            joints,
            camera,
            track_ids,
            30.0,
        )
        self.assertEqual(signals.excluded_track_switch_frames, 3)
        self.assertFalse(signals.valid[29])
        self.assertFalse(signals.valid[30])
        self.assertFalse(signals.valid[31])


if __name__ == "__main__":
    unittest.main()
