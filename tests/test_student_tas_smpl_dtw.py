import unittest
from pathlib import Path

import numpy as np

from aqa3d.smpl_dtw import ReferenceFrameMatch, VideoSampling
from run_student_tas_smpl_dtw import (
    build_student_segments,
    sampled_interval_to_source_interval,
)
from run_tas_smpl_dtw_mapping import ReferenceSegment


class StudentTasSmplDtwTests(unittest.TestCase):
    def setUp(self):
        self.sampling = VideoSampling(
            video_path=Path("student.mp4"),
            source_fps=30.0,
            source_frame_count=60,
            sample_fps=5.0,
            source_indices=np.arange(0, 60, 6, dtype=np.int64),
        )

    def test_sampled_intervals_cover_source_frames_without_gaps(self):
        first = sampled_interval_to_source_interval(1, 3, self.sampling)
        second = sampled_interval_to_source_interval(4, 6, self.sampling)
        last = sampled_interval_to_source_interval(7, 10, self.sampling)

        self.assertEqual(first, (1, 18))
        self.assertEqual(second, (19, 36))
        self.assertEqual(last, (37, 60))

    def test_student_segments_export_source_and_sampled_coordinates(self):
        references = [
            ReferenceSegment(1, "qishi", 1, 3),
            ReferenceSegment(2, "yemafenzong", 4, 6),
        ]
        matches = [
            ReferenceFrameMatch(2, 2, 1, 0.1),
            ReferenceFrameMatch(5, 5, 2, 0.2),
        ]

        segments = build_student_segments("00", references, matches, self.sampling)

        self.assertEqual(
            (
                segments[0].start_frame,
                segments[0].end_frame,
                segments[0].start_frame_5fps,
                segments[0].end_frame_5fps,
            ),
            (1, 18, 1, 3),
        )
        self.assertEqual(
            (
                segments[1].start_frame,
                segments[1].end_frame,
                segments[1].start_frame_5fps,
                segments[1].end_frame_5fps,
            ),
            (19, 36, 4, 6),
        )
        self.assertAlmostEqual(segments[0].start_time, 0.0)
        self.assertAlmostEqual(segments[0].end_time, 0.6)
        self.assertAlmostEqual(segments[1].start_time, 0.6)
        self.assertAlmostEqual(segments[1].end_time, 1.2)

    def test_rejects_duplicate_student_boundaries(self):
        references = [ReferenceSegment(1, "a", 1, 2), ReferenceSegment(2, "b", 3, 4)]
        matches = [ReferenceFrameMatch(1, 3, 1, 0.1), ReferenceFrameMatch(3, 3, 1, 0.2)]

        with self.assertRaisesRegex(ValueError, "not strictly increasing"):
            build_student_segments("00", references, matches, self.sampling)


if __name__ == "__main__":
    unittest.main()
