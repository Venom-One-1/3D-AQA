import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from aqa3d.tracking import load_primary_smpl_track, load_stitched_primary_track


def _smpl_record(value: float) -> dict:
    return {
        "global_orient": np.eye(3, dtype=np.float64)[None],
        "body_pose": np.broadcast_to(np.eye(3, dtype=np.float64), (23, 3, 3)).copy(),
        "betas": np.full((10,), value, dtype=np.float64),
    }


class TrackingSmplTests(unittest.TestCase):
    def test_loads_longest_track_and_maps_source_frames(self):
        data = {
            "/tmp/img/000001.jpg": {"tid": [2, 1], "smpl": [_smpl_record(2.0), _smpl_record(1.0)]},
            "/tmp/img/000002.jpg": {"tid": [1], "smpl": [_smpl_record(1.0)]},
            "/tmp/img/000003.jpg": {"tid": [1], "smpl": [_smpl_record(1.0)]},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo.pkl"
            joblib.dump(data, path)
            track = load_primary_smpl_track(path)
            self.assertEqual(track.track_id, 1)
            np.testing.assert_array_equal(track.frame_numbers, [1, 2, 3])

            selected = track.at_source_frames([0, 2])
            np.testing.assert_array_equal(selected.frame_numbers, [1, 3])
            self.assertEqual(selected.global_orients.shape, (2, 3, 3))
            self.assertEqual(selected.body_poses.shape, (2, 23, 3, 3))
            self.assertEqual(selected.betas.shape, (2, 10))

    def test_explicit_track_id_is_supported(self):
        data = {
            "/tmp/img/000001.jpg": {"tid": [2, 1], "smpl": [_smpl_record(2.0), _smpl_record(1.0)]},
            "/tmp/img/000002.jpg": {"tid": [1], "smpl": [_smpl_record(1.0)]},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo.pkl"
            joblib.dump(data, path)
            track = load_primary_smpl_track(path, track_id=2)
            self.assertEqual(track.track_id, 2)
            np.testing.assert_array_equal(track.frame_numbers, [1])
            np.testing.assert_allclose(track.betas[0], np.full((10,), 2.0))

    def test_stitches_contiguous_track_id_change(self):
        data = {
            "/tmp/img/000001.jpg": {"tid": [1], "smpl": [_smpl_record(1.0)]},
            "/tmp/img/000002.jpg": {"tid": [1, 2], "smpl": [_smpl_record(1.0), _smpl_record(2.0)]},
            "/tmp/img/000003.jpg": {"tid": [2], "smpl": [_smpl_record(2.0)]},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo.pkl"
            joblib.dump(data, path)
            track = load_stitched_primary_track(path)
            np.testing.assert_array_equal(track.frame_numbers, [1, 2, 3])
            np.testing.assert_array_equal(track.source_track_ids, [1, 1, 2])
            self.assertEqual(track.used_track_ids, (1, 2))


if __name__ == "__main__":
    unittest.main()
