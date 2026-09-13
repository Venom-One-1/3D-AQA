import tempfile
import unittest
from pathlib import Path

import numpy as np

from aqa3d.keypose_alignment import (
    align_keyposes_to_reference_segment,
    discover_keypose_images,
    format_timestamp,
    time_interval_to_inclusive_frames,
)


def _z_rotation(degrees: float) -> np.ndarray:
    radians = np.radians(degrees)
    cosine, sine = np.cos(radians), np.sin(radians)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    )


def _pose(degrees: float) -> np.ndarray:
    return np.repeat(_z_rotation(degrees)[None], 23, axis=0)


class KeyPoseAlignmentTests(unittest.TestCase):
    def test_discovers_moves_and_numeric_frames_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for move_name, frame_names in (
                ("1_qishi", ("00013.jpg", "00004.jpg")),
                ("2_yemafenzong", ("00009.jpg",)),
            ):
                move_dir = root / move_name
                move_dir.mkdir()
                for frame_name in frame_names:
                    (move_dir / frame_name).touch()
            items = discover_keypose_images(root, expected_move_count=2)
        self.assertEqual(
            [(item.move_id, item.source_frame_number) for item in items],
            [(1, 4), (1, 13), (2, 9)],
        )
        self.assertEqual([item.keypose_order for item in items], [1, 2, 1])

    def test_aligns_sparse_keyposes_monotonically(self):
        reference = np.stack([_pose(value) for value in (0, 5, 10, 15, 20)])
        keyposes = np.stack([_pose(value) for value in (5, 15)])
        result = align_keyposes_to_reference_segment(reference, keyposes)
        selected = [match.target_index for match in result.matches]
        self.assertEqual(selected, [1, 3])
        self.assertTrue(np.all(np.diff(selected) >= 0))
        self.assertEqual(result.local_costs.shape, (5, 2))

    def test_interval_conversion_keeps_both_boundaries(self):
        self.assertEqual(
            time_interval_to_inclusive_frames(1.0, 2.0, 30.0, 100),
            (30, 60),
        )

    def test_timestamp_format(self):
        self.assertEqual(format_timestamp(19.0 + 65.0 / 30.0), "00:00:21.167")


if __name__ == "__main__":
    unittest.main()
