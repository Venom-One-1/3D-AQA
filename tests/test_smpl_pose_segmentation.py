import tempfile
import unittest
from pathlib import Path

import numpy as np

from aqa3d.smpl_dtw import ReferenceFrameMatch, VideoSampling
from aqa3d.smpl_pose_segmentation import (
    build_mapped_boundaries,
    build_mapped_segments,
    load_gold_boundaries,
    validate_gold_boundaries_against_sampling,
)


def make_sampling(sample_count: int = 8) -> VideoSampling:
    return VideoSampling(
        video_path=Path("video.mp4"),
        source_fps=30.0,
        source_frame_count=sample_count * 6,
        sample_fps=5.0,
        source_indices=np.arange(sample_count, dtype=np.int64) * 6,
    )


class GoldBoundaryTests(unittest.TestCase):
    def write_boundaries(self, directory: str, end_values: tuple[str, str]) -> Path:
        path = Path(directory) / "boundaries.txt"
        path.write_text(
            "URLID,TagID,Tag,End\n"
            f"reference_5FPS,1,qishi,{end_values[0]}\n"
            f"reference_5FPS,2,yemafenzong,{end_values[1]}\n",
            encoding="utf-8",
        )
        return path

    def test_loads_exact_02_second_grid_as_zero_based_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_boundaries(directory, ("0.4", "1.0"))
            boundaries = load_gold_boundaries(
                path,
                5.0,
                expected_count=2,
                expected_url_id="reference_5FPS",
            )

        self.assertEqual([item.sample_index_0based for item in boundaries], [2, 5])
        self.assertEqual([item.end_time_seconds for item in boundaries], [0.4, 1.0])

    def test_rejects_boundary_outside_02_second_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_boundaries(directory, ("0.3", "1.0"))
            with self.assertRaisesRegex(ValueError, "not on the 5.0 FPS grid"):
                load_gold_boundaries(path, 5.0, expected_count=2)

    def test_validates_last_boundary_against_reference_sample_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_boundaries(directory, ("0.4", "1.0"))
            boundaries = load_gold_boundaries(path, 5.0, expected_count=2)

        with self.assertRaisesRegex(ValueError, "only 5 samples"):
            validate_gold_boundaries_against_sampling(boundaries, make_sampling(5))

    def test_mapped_boundary_uses_point_time_not_interval_end_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_boundaries(directory, ("0.4", "0.8"))
            boundaries = load_gold_boundaries(path, 5.0, expected_count=2)
        sampling = make_sampling()
        matches = [
            ReferenceFrameMatch(reference_index=2, target_index=3, candidate_count=2, local_cost=0.1),
            ReferenceFrameMatch(reference_index=4, target_index=6, candidate_count=1, local_cost=0.2),
        ]

        mapped = build_mapped_boundaries(
            "student",
            boundaries,
            matches,
            sampling,
            sampling,
        )

        self.assertEqual(mapped[0].target_sample_index_0based, 3)
        self.assertAlmostEqual(mapped[0].target_end_time_seconds, 0.6)
        self.assertEqual(mapped[0].target_source_frame_1based, 19)

    def test_segments_are_non_overlapping_and_use_exact_boundary_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_boundaries(directory, ("0.4", "0.8"))
            boundaries = load_gold_boundaries(path, 5.0, expected_count=2)
        sampling = make_sampling()
        matches = [
            ReferenceFrameMatch(reference_index=2, target_index=2, candidate_count=1, local_cost=0.0),
            ReferenceFrameMatch(reference_index=4, target_index=4, candidate_count=1, local_cost=0.0),
        ]
        mapped = build_mapped_boundaries("student", boundaries, matches, sampling, sampling)

        segments = build_mapped_segments("student", mapped, sampling)

        self.assertEqual(segments[0].start_frame_5fps, 1)
        self.assertEqual(segments[0].end_frame_5fps, 3)
        self.assertEqual(segments[1].start_frame_5fps, 4)
        self.assertEqual(segments[1].end_frame_5fps, 5)
        self.assertEqual(segments[0].start_frame, 1)
        self.assertEqual(segments[0].end_frame, 13)
        self.assertEqual(segments[1].start_frame, 14)
        self.assertEqual(segments[1].end_frame, 25)

    def test_duplicate_mapped_boundaries_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_boundaries(directory, ("0.4", "0.8"))
            boundaries = load_gold_boundaries(path, 5.0, expected_count=2)
        sampling = make_sampling()
        matches = [
            ReferenceFrameMatch(reference_index=2, target_index=3, candidate_count=1, local_cost=0.0),
            ReferenceFrameMatch(reference_index=4, target_index=3, candidate_count=1, local_cost=0.0),
        ]

        with self.assertRaisesRegex(ValueError, "not strictly increasing"):
            build_mapped_boundaries("student", boundaries, matches, sampling, sampling)


if __name__ == "__main__":
    unittest.main()
