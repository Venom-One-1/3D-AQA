import unittest

import numpy as np

from aqa3d.smpl_dtw import (
    ReferenceFrameMatch,
    select_reference_frame_matches,
    uniform_sample_source_indices,
)
from run_tas_smpl_dtw_mapping import ReferenceSegment, build_mapped_segments


class TasSmplDtwMappingTests(unittest.TestCase):
    def test_uniform_sampling_uses_exact_interval_count(self):
        indices = uniform_sample_source_indices(9330, 30.0, 5.0)
        self.assertEqual(len(indices), 1555)
        self.assertEqual(int(indices[0]), 0)
        self.assertEqual(int(indices[-1]), 9324)

    def test_uniform_sampling_handles_2997_fps_without_extra_endpoint(self):
        indices = uniform_sample_source_indices(8362, 30000.0 / 1001.0, 5.0)
        self.assertEqual(len(indices), 1395)
        self.assertTrue(np.all(np.diff(indices) > 0))
        self.assertLess(int(indices[-1]), 8362)

    def test_boundary_candidate_uses_minimum_local_cost(self):
        costs = np.full((5, 4), 9.0)
        costs[1, 2] = 0.5
        costs[2, 2] = 0.2
        path = np.asarray(((0, 0), (1, 1), (1, 2), (2, 2), (3, 3), (4, 3)))
        match = select_reference_frame_matches(costs, path, [2])[0]
        self.assertEqual(match.target_index, 2)
        self.assertEqual(match.candidate_count, 2)
        self.assertAlmostEqual(match.local_cost, 0.2)

    def test_mapped_segments_are_closed_and_non_overlapping(self):
        references = [
            ReferenceSegment(1, "qishi", 1, 3),
            ReferenceSegment(2, "yemafenzong", 4, 6),
        ]
        matches = [
            ReferenceFrameMatch(2, 4, 2, 0.1),
            ReferenceFrameMatch(5, 8, 1, 0.2),
        ]
        mapped = build_mapped_segments("target", references, matches, 5.0)
        self.assertEqual((mapped[0].start_frame, mapped[0].end_frame), (1, 5))
        self.assertEqual((mapped[1].start_frame, mapped[1].end_frame), (6, 9))
        self.assertAlmostEqual(mapped[1].start_time, 1.0)
        self.assertAlmostEqual(mapped[1].end_time, 1.8)

    def test_duplicate_mapped_boundaries_are_rejected(self):
        references = [ReferenceSegment(1, "a", 1, 2), ReferenceSegment(2, "b", 3, 4)]
        matches = [ReferenceFrameMatch(1, 3, 1, 0.1), ReferenceFrameMatch(3, 3, 2, 0.2)]
        with self.assertRaisesRegex(ValueError, "not strictly increasing"):
            build_mapped_segments("target", references, matches, 5.0)


if __name__ == "__main__":
    unittest.main()
