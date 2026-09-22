import copy
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from aqa3d.endpoint_transfer import (
    manifest_boundaries, resolve_tracking_samples, tracking_provenance,
    transferred_keypose_rows,
)
from aqa3d.reference_manifest import build_endpoint_keyposes
from aqa3d.smpl_dtw import VideoSampling, ReferenceFrameMatch
from aqa3d.smpl_pose_segmentation import GoldBoundary, build_mapped_boundaries
from aqa3d.tracking import TrackPoseSequence


class EndpointTransferTests(unittest.TestCase):
    def setUp(self):
        self.sampling = VideoSampling(Path("test.mp4"), 30., 300, 5., np.arange(0, 300, 6))
        gold = [GoldBoundary(i, f"move{i}", i / 5, i) for i in range(1, 25)]
        events = build_endpoint_keyposes(gold, self.sampling,
            {i: f"final technique {i}" for i in range(1, 25)},
            reference_video_id="R", reference_sample_sequence_id="R_5FPS")
        self.manifest = {"schema_version": "1.0", "reference": {
            "video_id": "R", "sample_sequence_id": "R_5FPS", "source_fps": 30.,
            "source_frame_count": 300, "sample_fps": 5., "sample_count": 50},
            "endpoint_keyposes": [asdict(e) for e in events]}
        self.track = TrackPoseSequence(np.arange(1, 301),
            np.tile(np.eye(3), (300, 23, 1, 1)), 1, np.ones(300, dtype=int))

    def test_rejects_stale_fps_and_frame_indices(self):
        for field, value in (("source_fps", 29.97), ("source_frame_count", 299)):
            manifest = copy.deepcopy(self.manifest)
            manifest["reference"][field] = value
            with self.assertRaises(ValueError):
                manifest_boundaries(manifest, self.sampling)
        manifest = copy.deepcopy(self.manifest)
        manifest["endpoint_keyposes"][0]["source_frame_index_0based"] += 1
        with self.assertRaises(ValueError):
            manifest_boundaries(manifest, self.sampling)

    def test_rejects_duplicate_events_and_empty_text(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["endpoint_keyposes"][1] = manifest["endpoint_keyposes"][0]
        with self.assertRaises(ValueError):
            manifest_boundaries(manifest, self.sampling)
        self.manifest["endpoint_keyposes"][2]["final_technique_step"] = " "
        with self.assertRaises(ValueError):
            manifest_boundaries(self.manifest, self.sampling)

    def test_strict_missing_and_nearest_tie_match_existing_loader(self):
        keep = self.track.frame_numbers != 7
        track = TrackPoseSequence(self.track.frame_numbers[keep], self.track.body_poses[keep], 1)
        with self.assertRaisesRegex(ValueError, "Missing tracked samples"):
            resolve_tracking_samples(track, self.sampling)
        positions = resolve_tracking_samples(track, self.sampling, 1 / 30)
        self.assertEqual(track.frame_numbers[positions[1]], 6)
        np.testing.assert_array_equal(track.body_poses[positions],
            track.at_source_frames(self.sampling.source_indices, nearest_max_distance_frames=1))
        p = tracking_provenance(track, self.sampling, int(positions[1]), 6)
        self.assertEqual(p["requested_phalp_frame_1based"], 7)
        self.assertEqual(p["actual_source_frame_0based"], 5)
        self.assertEqual(p["status"], "nearest_substitution")

    def test_switch_flag(self):
        self.track.source_track_ids[10:] = 2
        self.assertTrue(tracking_provenance(self.track, self.sampling, 9, 9)["near_track_id_switch"])
        self.assertFalse(tracking_provenance(self.track, self.sampling, 20, 20)["near_track_id_switch"])

    def test_transfer_preserves_text_and_indices(self):
        gold = manifest_boundaries(self.manifest, self.sampling)
        matches = [ReferenceFrameMatch(i, i + 1, 1, 0.1) for i in range(1, 25)]
        boundaries = build_mapped_boundaries("S", gold, matches, self.sampling, self.sampling)
        path = np.array([[i + 1, i] for i in range(1, 25)])
        positions = resolve_tracking_samples(self.track, self.sampling)
        rows = transferred_keypose_rows(self.manifest, boundaries, path,
            self.sampling, self.sampling, self.track, self.track, positions, positions)
        self.assertEqual(len(rows), 24)
        for i, row in enumerate(rows, 1):
            self.assertEqual(row["final_technique_step"], f"final technique {i}")
            self.assertEqual(row["student_source_frame_0based"], (i + 1) * 6)
            self.assertEqual(row["student_tracking_actual_phalp_frame_1based"], (i + 1) * 6 + 1)
            self.assertFalse(row["review_required"])
            self.assertEqual(row["boundary_annotation_status"], "predicted")

    def test_duplicate_target_endpoints_rejected(self):
        gold = manifest_boundaries(self.manifest, self.sampling)
        matches = [ReferenceFrameMatch(i, 1, 1, 0.1) for i in range(1, 25)]
        with self.assertRaises(ValueError):
            build_mapped_boundaries("S", gold, matches, self.sampling, self.sampling)

    def test_render_keeps_portrait_source_and_separate_labels(self):
        from run_student_endpoint_keyposes import render_keyposes
        portrait = Image.new("RGB", (100, 300), "white")
        portrait.paste("red", (0, 0, 100, 20))
        portrait.paste("blue", (0, 280, 100, 300))
        row = {"move_id": 1, "move_name_zh": "test", "pose_id": "1.end",
               "reference_source_frame_0based": 0, "student_source_frame_0based": 0,
               "reference_boundary_time_seconds": 0., "student_boundary_time_seconds": 0.,
               "reference_tracking_actual_phalp_frame_1based": 1,
               "student_tracking_actual_phalp_frame_1based": 1,
               "local_geodesic_degrees": 0., "candidate_count": 1,
               "review_required": True, "diagnostic_flags": "nearest_substitution;" * 8}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("run_student_endpoint_keyposes.read_video_frames", return_value={0: portrait}):
                render_keyposes(root, [row], Path("R.mp4"), Path("S.mp4"))
            with Image.open(root / "boundary_frames.jpg") as grid:
                self.assertEqual(grid.size, (1160, 484))
                # Both the top and bottom of the portrait survive letterboxing.
                self.assertGreater(grid.getpixel((290, 141))[0], 200)
                self.assertGreater(grid.getpixel((290, 408))[2], 200)
            with Image.open(root / row["student_image"]) as original:
                self.assertEqual(original.size, (100, 300))


if __name__ == "__main__":
    unittest.main()
