import tempfile
import unittest
from pathlib import Path

import numpy as np

from aqa3d.reference_manifest import (
    MOVE_NAMES_PINYIN,
    attach_tracking_availability,
    build_endpoint_keyposes,
    build_reference_manifest,
    load_final_technique_steps,
)
from aqa3d.smpl_dtw import VideoSampling
from aqa3d.smpl_pose_segmentation import GoldBoundary
from aqa3d.tracking import TrackPoseSequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReferenceManifestTests(unittest.TestCase):
    def test_load_final_technique_steps_keeps_multiline_and_duplicate_numbers(self) -> None:
        sections = []
        for move_id in range(1, 25):
            if move_id == 3:
                action_lines = "1. placeholder\n2. final line one\ncontinuation line"
            elif move_id == 20:
                action_lines = "1. placeholder\n3. first three\n3. final duplicate three"
            else:
                action_lines = f"1. placeholder\n2. final move {move_id}"
            sections.append(
                f"## 第{move_id}式 Move {move_id}\n\n"
                f"### 动作要领\n\n{action_lines}\n\n"
                "### 训练要点\n\n- ignored\n"
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "TechPoint.md"
            path.write_text("\n".join(sections), encoding="utf-8")
            steps = load_final_technique_steps(path)

        self.assertEqual(steps[3], "final line one continuation line")
        self.assertEqual(steps[20], "final duplicate three")
        self.assertEqual(steps[24], "final move 24")

    def test_project_endpoint_descriptions_are_self_contained(self) -> None:
        steps = load_final_technique_steps(PROJECT_ROOT / "TechPoint.md")
        required_terms = {
            2: ("左脚", "左手", "右手"),
            4: ("左脚", "左手", "右手"),
            6: ("左臂", "右臂", "右腿", "左脚"),
            8: ("两手", "右腿"),
            10: ("左手", "右手", "右脚"),
            12: ("右掌", "左手", "左脚"),
            17: ("左腿", "左手", "右手"),
            18: ("左脚", "左手", "右手"),
            23: ("重心", "右脚", "两腿", "两手"),
            24: ("两手", "两腿", "左脚"),
        }
        forbidden_shorthand = ("再按相同方法", "按相反方向完成", "连续完成三次")
        for move_id, terms in required_terms.items():
            with self.subTest(move_id=move_id):
                for term in terms:
                    self.assertIn(term, steps[move_id])
                for shorthand in forbidden_shorthand:
                    self.assertNotIn(shorthand, steps[move_id])

    def test_build_endpoint_keyposes_uses_explicit_index_conventions(self) -> None:
        sampling = self._sampling()
        keyposes = self._keyposes(sampling)

        first = keyposes[0]
        second = keyposes[1]
        self.assertEqual(len(keyposes), 24)
        self.assertEqual(first.move_name_pinyin, MOVE_NAMES_PINYIN[0])
        self.assertEqual(first.pose_id, "1.end")
        self.assertEqual(first.sample_index_0based, 10)
        self.assertEqual(first.sample_frame_1based, 11)
        self.assertEqual(first.source_frame_index_0based, 60)
        self.assertEqual(first.phalp_frame_1based, 61)
        self.assertEqual(first.segment_sample_start_index_0based, 0)
        self.assertEqual(first.segment_sample_end_index_0based, 10)
        self.assertEqual(second.segment_sample_start_index_0based, 11)
        self.assertEqual(second.segment_source_start_index_0based, 61)

    def test_tracking_availability_is_exact_and_preserves_source_track_id(self) -> None:
        keyposes = self._keyposes(self._sampling())
        track = TrackPoseSequence(
            frame_numbers=np.asarray([61, 121], dtype=np.int64),
            body_poses=np.tile(np.eye(3), (2, 23, 1, 1)),
            track_id=7,
            source_track_ids=np.asarray([7, 9], dtype=np.int64),
        )
        annotated = attach_tracking_availability(keyposes, track)

        self.assertTrue(annotated[0].tracking_pose_available)
        self.assertEqual(annotated[0].tracking_source_track_id, 7)
        self.assertTrue(annotated[1].tracking_pose_available)
        self.assertEqual(annotated[1].tracking_source_track_id, 9)
        self.assertFalse(annotated[2].tracking_pose_available)
        self.assertIsNone(annotated[2].tracking_source_track_id)

    def test_manifest_records_reference_identity_and_missing_tracking_count(self) -> None:
        sampling = self._sampling()
        keyposes = self._keyposes(sampling)
        manifest = build_reference_manifest(
            keyposes=keyposes,
            sampling=sampling,
            reference_video_id="reference",
            reference_sample_sequence_id="reference_5FPS",
            reference_video_path=Path("/videos/reference.mp4"),
            reference_tracking_path=Path("/tracking/reference.pkl"),
            gold_boundary_path=Path("/annotations/boundaries.txt"),
            technique_path=Path("/project/TechPoint.md"),
            output_dir=Path("/project/reference_data/reference"),
            metric_window_seconds=0.2,
            tracking_used_track_ids=(1, 2),
        )

        self.assertEqual(manifest["reference"]["video_id"], "reference")
        self.assertEqual(manifest["keypose_policy"]["count"], 24)
        self.assertEqual(manifest["tracking_validation"]["missing_endpoint_pose_count"], 24)
        self.assertEqual(manifest["tracking_validation"]["used_track_ids"], [1, 2])
        self.assertEqual(len(manifest["endpoint_keyposes"]), 24)
        self.assertEqual(manifest["reference"]["tail_after_last_endpoint"]["sample_count"], 9)
        self.assertEqual(
            manifest["reference"]["tail_after_last_endpoint"]["source_frame_count"],
            59,
        )

    @staticmethod
    def _sampling() -> VideoSampling:
        return VideoSampling(
            video_path=Path("reference.mp4"),
            source_fps=30.0,
            source_frame_count=1500,
            sample_fps=5.0,
            source_indices=np.arange(250, dtype=np.int64) * 6,
        )

    @staticmethod
    def _keyposes(sampling: VideoSampling):
        boundaries = [
            GoldBoundary(
                move_id=move_id,
                move_name=f"Move {move_id}",
                end_time_seconds=float(move_id * 2),
                sample_index_0based=move_id * 10,
            )
            for move_id in range(1, 25)
        ]
        return build_endpoint_keyposes(
            boundaries,
            sampling,
            {move_id: f"final {move_id}" for move_id in range(1, 25)},
            reference_video_id="reference",
            reference_sample_sequence_id="reference_5FPS",
        )


if __name__ == "__main__":
    unittest.main()
