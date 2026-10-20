import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from aqa3d.smpl_dtw import VideoSampling, uniform_sample_source_indices
from export_qishi_smpl_renders import (
    CV_TO_GL, fixed_heading_rotation, presentation_vertices,
    build_parser, configure_view, mesh_global_orients, reconstruct,
    segment_samples, shared_camera, validate_rgba, yaw_rotation,
)


class QishiSmplRenderTests(unittest.TestCase):
    def test_reference_sampling_includes_closed_endpoint(self):
        sampling = VideoSampling(Path("ref.mp4"), 30.0, 9210, 5.0,
                                 uniform_sample_source_indices(9210, 30.0, 5.0))
        ids, frames = segment_samples(sampling, 0, 480)
        self.assertEqual(len(frames), 81)
        self.assertEqual(int(ids[-1]), 80)
        self.assertEqual(int(frames[-1]), 480)

    def test_student_grid_and_actual_time_are_distinct(self):
        fps = 29.969808062886592
        sampling = VideoSampling(Path("10.mp4"), fps, 10209, 5.0,
                                 uniform_sample_source_indices(10209, fps, 5.0))
        ids, frames = segment_samples(sampling, 0, 449)
        self.assertEqual(len(frames), 76)
        self.assertEqual(int(ids[-1]), 75)
        self.assertEqual(int(frames[-1]), 449)
        self.assertAlmostEqual(float(frames[-1]) / fps, 14.981744262687608)
        self.assertEqual(float(ids[-1]) / 5, 15.0)

    def test_yaw_correction_preserves_later_turn(self):
        global_rotations = np.stack([CV_TO_GL @ yaw_rotation(t) for t in (0.6, 0.6, 0.9)])
        correction, yaw = fixed_heading_rotation(global_rotations, 2)
        self.assertAlmostEqual(yaw, 0.6)
        np.testing.assert_allclose(correction @ global_rotations[0], np.eye(3), atol=1e-10)
        np.testing.assert_allclose(correction @ global_rotations[-1], yaw_rotation(0.3), atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(correction), 1.0)

    def test_identity_root_is_not_zero_matrix_and_leaves_input_unchanged(self):
        original = np.stack([yaw_rotation(0.6), yaw_rotation(-0.4)])
        snapshot = original.copy()
        result = mesh_global_orients(original, "canonical-front")
        np.testing.assert_array_equal(result, np.repeat(np.eye(3)[None], 2, axis=0))
        np.testing.assert_array_equal(original, snapshot)
        np.testing.assert_array_equal(np.linalg.det(result), [1, 1])
        np.testing.assert_array_equal(mesh_global_orients(original, "initial-heading"), original)

    def test_front_mode_defaults_to_unrotated_camera_and_separate_directory(self):
        args = build_parser().parse_args(["--view-mode", "canonical-front"])
        configure_view(args)
        self.assertEqual((args.azimuth, args.elevation), (0.0, 0.0))
        self.assertEqual(args.output_root.name, "smpl_qishi_5fps_front")
        legacy = build_parser().parse_args([])
        configure_view(legacy)
        self.assertEqual((legacy.azimuth, legacy.elevation), (35.0, 8.0))
        self.assertEqual(legacy.output_root.name, "smpl_qishi_5fps")

    def test_front_mode_rejects_rotated_camera(self):
        args = build_parser().parse_args(["--view-mode", "canonical-front", "--azimuth", "35"])
        with self.assertRaises(ValueError):
            configure_view(args)

    def test_frontal_camera_rotation_is_identity(self):
        vertices = np.array([[[-0.6, 0, -0.2], [0.6, 1.7, 0.4]]])
        pose, _, _ = shared_camera([vertices], 768, 1024, 0, 0)
        np.testing.assert_allclose(pose[:3, :3], np.eye(3), atol=1e-12)

    def test_reconstruction_preserves_local_rotations_and_shape(self):
        import torch

        calls = []

        def fake_smpl(**kwargs):
            calls.append({k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()})
            count = len(kwargs["betas"])
            return SimpleNamespace(
                vertices=torch.tensor([[[-0.5, -1, 0], [0.5, 1, 0], [0, 0, 0.4]]]).repeat(count, 1, 1),
                joints=torch.zeros(count, 24, 3),
            )

        track = SimpleNamespace(frame_numbers=np.array([1, 7]),
                                global_orients=np.stack([yaw_rotation(0.4), yaw_rotation(0.8)]),
                                body_poses=np.tile(np.eye(3), (2, 23, 1, 1)),
                                betas=np.arange(20, dtype=float).reshape(2, 10) / 20)
        track.body_poses[1, 5] = yaw_rotation(0.2)
        original_global = track.global_orients.copy()
        subject = {"track": track}
        args = SimpleNamespace(view_mode="canonical-front", canonical_height=1.7, sample_fps=5.0)
        reconstruct(subject, fake_smpl, args)
        np.testing.assert_array_equal(calls[0]["global_orient"].numpy(), np.repeat(np.eye(3)[None], 2, axis=0))
        np.testing.assert_allclose(calls[0]["body_pose"].numpy(), track.body_poses)
        np.testing.assert_allclose(calls[0]["betas"].numpy(), track.betas)
        np.testing.assert_array_equal(track.global_orients, original_global)
        np.testing.assert_array_equal(subject["presentation"]["mesh_to_presentation_rotation"], np.eye(3))
        self.assertIsNone(subject["presentation"]["initial_yaw_degrees"])
        self.assertEqual(subject["presentation"]["global_orient_policy"], "identity_3x3_per_frame")

    def test_presentation_preserves_geometry_up_to_uniform_scale(self):
        vertices = np.array([[[0, -1, 0], [1, 0, 0], [0, 1, 1]]], dtype=float)
        posed, _ = presentation_vertices(vertices, np.array([[0.3, 0.1, -0.4]]), yaw_rotation(0.4), 2.0)
        self.assertAlmostEqual(float(posed[:, :, 1].min()), 0.0)
        before = np.linalg.norm(vertices[0, 1] - vertices[0, 0])
        after = np.linalg.norm(posed[0, 1] - posed[0, 0])
        self.assertAlmostEqual(after / before, 2.0, places=6)

    def test_shared_camera_contains_both_sequences(self):
        points = np.array([[[-0.8, 0, -0.3], [0.8, 1.7, 0.3], [0, 0.9, 0.7]]])
        sequences = [points, points * 1.05]
        pose, xmag, ymag = shared_camera(sequences, 768, 1024, 35, 8)
        self.assertAlmostEqual(xmag / ymag, 768 / 1024)
        for seq in sequences:
            camera_points = (seq - pose[:3, 3]) @ pose[:3, :3]
            self.assertTrue((abs(camera_points[:, :, 0]) < xmag).all())
            self.assertTrue((abs(camera_points[:, :, 1]) < ymag).all())
            self.assertTrue((camera_points[:, :, 2] < 0).all())

    def test_alpha_validation_rejects_blank_opaque_and_clipped(self):
        for alpha in (0, 255):
            with self.assertRaises(ValueError):
                validate_rgba(Image.new("RGBA", (32, 32), (100, 100, 200, alpha)))
        array = np.zeros((32, 32, 4), dtype=np.uint8)
        array[8:24, 8:24] = [160, 180, 220, 255]
        self.assertEqual(validate_rgba(Image.fromarray(array))["alpha_bbox_xyxy"], [8, 8, 24, 24])
        array[0] = [160, 180, 220, 255]
        with self.assertRaises(ValueError):
            validate_rgba(Image.fromarray(array))


if __name__ == "__main__":
    unittest.main()
