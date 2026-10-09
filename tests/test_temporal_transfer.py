import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from aqa3d.smpl_dtw import VideoSampling
from aqa3d.temporal_transfer import (
    apply_correspondence_reviews, apply_reference_corrections, flag_event_order,
    flatten_event, map_event, nearest_reference_sample,
    path_pose_costs, track_frame_quality, validate_path, verify_path_archive,
)
from aqa3d.tracking import TrackPoseSequence


class TemporalTransferTests(unittest.TestCase):
    def setUp(self):
        self.reference = VideoSampling(Path("R.mp4"), 30., 60, 5., np.arange(0, 60, 6))
        self.target = VideoSampling(Path("S.mp4"), 30., 72, 5., np.arange(0, 72, 6))
        self.path = np.array([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 4),
                              (6, 4), (7, 5), (8, 6), (9, 7), (10, 8), (11, 9)])
        self.costs = np.full(len(self.path), 0.2)
        self.costs[6] = 0.01
        self.track = TrackPoseSequence(np.arange(1, 73), np.tile(np.eye(3), (72, 23, 1, 1)),
                                       1, np.ones(72, dtype=int))
        self.positions = self.target.source_indices.copy()
        self.quality = track_frame_quality(self.track, self.target)
        self.ref_quality = {"status": np.full(60, "valid"), "track_ids": np.ones(60)}
        self.endpoint = {"segment_start_source_frame_0based": 0,
                         "segment_end_source_frame_0based": 66,
                         "sample_index_0based": 11, "reference_source_frame_0based": 54}
        self.event = {"pose_id": "1.event", "event_id": "event", "move_id": 1,
                      "move_name": "test", "label": "test", "technique": "test",
                      "scope": "stable_window", "representative_source": "signal_query",
                      "annotation_status": "candidate", "auxiliary_reference_image": "",
                      "auxiliary_correspondence": "none",
                      "reference_segment_start_source_frame_0based": 0,
                      "reference_segment_end_source_frame_0based": 54,
                      "interval_start_source_frame_0based": 24,
                      "representative_source_frame_0based": 24,
                      "interval_end_source_frame_0based": 24}

    def map(self, **kwargs):
        return map_event(self.event, self.path, self.costs, self.reference, self.target,
                         self.endpoint, self.positions, self.track, self.ref_quality,
                         self.quality, **kwargs)

    def test_internal_median_does_not_cherry_pick_lowest_cost_pose(self):
        row, candidates = self.map()
        self.assertEqual(row["points"]["start"]["target_source_frame_0based"], 24)
        self.assertEqual(row["points"]["representative"]["target_source_frame_0based"], 30)
        self.assertEqual(row["points"]["end"]["target_source_frame_0based"], 36)
        self.assertEqual(row["points"]["representative"]["candidate_count"], 3)
        self.assertEqual(len(candidates), 9)

    def test_endpoint_reuses_boundary_minimum_cost(self):
        self.event.update(representative_source="known_move_end",
                          representative_source_frame_0based=54, interval_end_source_frame_0based=54)
        row, _ = self.map()
        self.assertEqual(row["points"]["representative"]["target_source_frame_0based"], 66)
        self.assertEqual(row["points"]["representative"]["selection_policy"], "known_move_end_minimum_local_cost")
        self.endpoint["sample_index_0based"] = 10
        with self.assertRaisesRegex(ValueError, "minimum local cost"):
            self.map()

    def test_span_and_outside_candidates_remain_visible(self):
        self.endpoint["segment_end_source_frame_0based"] = 30
        row, candidates = self.map(review_span_seconds=0.3)
        self.assertIn("representative_wide_candidate_span", row["diagnostic_flags"])
        self.assertIn("representative_candidates_outside_form", row["diagnostic_flags"])
        self.assertEqual(row["points"]["representative"]["target_source_frame_0based"], 24)
        self.assertEqual(sum(not c["inside_form"] for c in candidates), 3)

    def test_no_candidate_in_form_does_not_fallback(self):
        self.endpoint["segment_start_source_frame_0based"] = 42
        row, candidates = self.map()
        self.assertEqual(row["status"], "unmapped")
        self.assertIsNone(row["points"]["representative"]["target_source_frame_0based"])
        self.assertEqual(len(candidates), 9)
        self.assertIsNone(flatten_event(row)["representative_image"])

    def test_missing_exact_tracking_and_old_substitution_reported(self):
        self.quality["status"][30] = "missing_tracking"
        self.positions[5] = 29
        row, _ = self.map()
        point = row["points"]["representative"]
        self.assertEqual(point["target_source_frame_0based"], 30)
        self.assertEqual(point["tracking_actual_phalp_frame_1based"], 30)
        self.assertEqual(point["tracking_frame_offset"], -1)
        self.assertIn("representative_alignment_nearest_substitution", row["diagnostic_flags"])
        self.assertIn("representative_missing_tracking", row["diagnostic_flags"])
        self.assertLess(row["interval_valid_tracking_ratio"], 1.)

    def test_track_switch_excludes_three_native_frames(self):
        self.track.source_track_ids[30:] = 2
        quality = track_frame_quality(self.track, self.target)
        np.testing.assert_array_equal(np.flatnonzero(quality["status"] != "valid"), [29, 30, 31])
        self.assertEqual(quality["track_ids"][30], 2)

    def test_reference_quantization_uses_source_frames_and_form_limits(self):
        self.assertEqual(nearest_reference_sample(27, self.reference, 0, 54), 4)
        self.assertEqual(nearest_reference_sample(27, self.reference, 25, 54), 5)
        self.event["representative_source_frame_0based"] = 25
        self.event["interval_end_source_frame_0based"] = 25
        row, _ = self.map()
        point = row["points"]["representative"]
        self.assertEqual(point["reference_source_frame_0based"], 25)
        self.assertEqual(point["reference_grid_source_frame_0based"], 24)
        self.assertAlmostEqual(point["reference_quantization_offset_seconds"], -1 / 30)

    def test_duplicate_representatives_are_not_silently_changed(self):
        row, _ = self.map()
        other = copy.deepcopy(row)
        flag_event_order([row, other])
        self.assertIn("duplicate_event_representative", row["diagnostic_flags"])
        self.assertEqual(other["points"]["representative"]["target_source_frame_0based"], 30)

    def test_invalid_paths_are_rejected(self):
        validate_path(self.path, self.costs, self.reference, self.target)
        for path in (self.path[::-1], self.path[1:], np.vstack([self.path[:1], self.path])):
            with self.assertRaises(ValueError):
                validate_path(path, np.zeros(len(path)), self.reference, self.target)
        with self.assertRaises(ValueError):
            validate_path(self.path.astype(float), self.costs, self.reference, self.target)

    def test_archive_timing_and_units_and_pose_costs(self):
        archive = {"local_geodesic_degrees": np.degrees(self.costs)}
        for prefix, sampling, column in (("target", self.target, 0), ("reference", self.reference, 1)):
            indices = self.path[:, column]
            source = sampling.source_indices[indices]
            archive.update({f"{prefix}_source_frame_indices_0based": source,
                            f"{prefix}_phalp_frames_1based": source + 1,
                            f"{prefix}_times_seconds": indices / 5.})
        verify_path_archive(archive, self.path, self.costs, self.reference, self.target)
        archive["target_phalp_frames_1based"] = archive["target_phalp_frames_1based"] - 1
        with self.assertRaisesRegex(ValueError, "phalp"):
            verify_path_archive(archive, self.path, self.costs, self.reference, self.target)
        identity = np.tile(np.eye(3), (3, 23, 1, 1))
        np.testing.assert_allclose(path_pose_costs(identity, identity), 0)

    def test_subject_renderer_retains_portrait_and_exports_images(self):
        from run_temporal_keypose_transfer import render_subject
        row, _ = self.map()
        portrait = Image.new("RGB", (100, 300), "white")
        portrait.paste("red", (0, 0, 100, 30))
        portrait.paste("blue", (0, 270, 100, 300))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("run_temporal_keypose_transfer.read_video_frames", return_value={i: portrait for i in (24, 30, 36)}):
                render_subject(root, "student", "S", [row], Path("S.mp4"), {24: portrait})
            with Image.open(root / row["points"]["representative"]["image"]) as image:
                self.assertEqual(image.size, (100, 300))
            with Image.open(root / "keypose_overview.jpg") as image:
                self.assertEqual(image.size, (1344, 357))
                self.assertGreater(image.getpixel((831, 130))[0], 180)
                self.assertGreater(image.getpixel((831, 272))[2], 180)

    def test_snapshot_can_differ_in_json_whitespace_but_not_content(self):
        from run_temporal_keypose_transfer import verify_plan_snapshot
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, snapshot = root / "original.json", root / "snapshot.json"
            original.write_text(json.dumps({"a": 1}))
            snapshot.write_text(json.dumps({"a": 1}, indent=2) + "\n")
            record = {"path": str(original), "sha256": hashlib.sha256(original.read_bytes()).hexdigest()}
            self.assertEqual(verify_plan_snapshot(snapshot, record), {"a": 1})
            snapshot.write_text(json.dumps({"a": 2}))
            with self.assertRaisesRegex(ValueError, "snapshot differs"):
                verify_plan_snapshot(snapshot, record)
            original.write_text(json.dumps({"a": 2}))
            with self.assertRaisesRegex(ValueError, "different plan"):
                verify_plan_snapshot(snapshot, record)

    def test_noninteger_fps_grid_and_native_time_are_separate(self):
        self.target = VideoSampling(Path("S.mp4"), 29.97, 72, 5., np.arange(0, 72, 6))
        row, _ = self.map()
        point = row["points"]["representative"]
        self.assertEqual(point["target_grid_time_seconds"], 1.)
        self.assertAlmostEqual(point["target_source_time_seconds"], 30 / 29.97)
        self.assertEqual(point["target_phalp_frame_1based"], 31)

    def test_cached_alignment_rejects_changed_poses_and_boundary_indices(self):
        from run_temporal_keypose_transfer import verified_alignment
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            indices = [1, 4, 7]
            manifest = {"endpoint_keyposes": [{"sample_index_0based": i} for i in indices]}
            endpoints, previous = [], -1
            for mid, index in enumerate(indices, 1):
                target_index = int(self.path[self.path[:, 1] == index, 0][0])
                source = int(self.target.source_indices[target_index])
                endpoints.append({"move_id": mid, "sample_index_0based": target_index,
                                  "reference_source_frame_0based": int(self.reference.source_indices[index]),
                                  "segment_start_source_frame_0based": previous + 1,
                                  "segment_end_source_frame_0based": source})
                previous = source
            metadata = {"manifest_sha256": "h", "input_video": "S.mp4", "source_fps": 30.,
                        "alignment_tracking_gap_seconds": 0., "endpoints": endpoints}
            (root / "alignment.json").write_text(json.dumps(metadata))
            data = {"target_sample_indices_0based": self.path[:, 0],
                    "reference_sample_indices_0based": self.path[:, 1],
                    "local_geodesic_radians": np.zeros(len(self.path)),
                    "local_geodesic_degrees": np.zeros(len(self.path))}
            for prefix, sampling, col in (("target", self.target, 0), ("reference", self.reference, 1)):
                samples = self.path[:, col]
                frames = sampling.source_indices[samples]
                data.update({f"{prefix}_source_frame_indices_0based": frames,
                             f"{prefix}_phalp_frames_1based": frames + 1,
                             f"{prefix}_times_seconds": samples / 5.})
            np.savez(root / "dtw_path.npz", **data)
            args = (root, manifest, "h", self.reference, self.target, self.track, self.track,
                    self.reference.source_indices)
            verified_alignment(*args)
            metadata["endpoints"][1]["sample_index_0based"] += 1
            (root / "alignment.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "boundary/index"):
                verified_alignment(*args)
            changed = copy.deepcopy(self.track)
            angle = 0.3
            changed.body_poses[24, :, :, :] = np.array([[np.cos(angle), -np.sin(angle), 0],
                                                        [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            with self.assertRaisesRegex(ValueError, "current tracking"):
                verified_alignment(root, manifest, "h", self.reference, self.target,
                                   self.track, changed, self.reference.source_indices)

    def correction_record(self):
        return {"schema_version": "1.0", "reference_video_id": "R", "reference_corrections": [{
            "pose_id": "1.event", "expected_original_source_frame_0based": 24,
            "representative_source_frame_0based": 18, "reason": "user chose interval start",
        }]}

    def test_manual_correction_preserves_original_and_updates_frame_conventions(self):
        self.event.update(interval_start_source_frame_0based=18, interval_end_source_frame_0based=36,
                          representative_score=0.9, sensitivity={"representative_span": 0.2})
        corrected = apply_reference_corrections([self.event], self.correction_record(), self.reference, "R")[0]
        self.assertEqual(self.event["representative_source_frame_0based"], 24)
        self.assertEqual(corrected["representative_source_frame_0based"], 18)
        self.assertEqual(corrected["representative_source_frame_1based"], 19)
        self.assertEqual(corrected["representative_time_seconds"], 0.6)
        self.assertEqual(corrected["representative_sample_index_5fps_0based"], 3)
        self.assertEqual(corrected["original_query_representative"]["representative_source_frame_0based"], 24)
        self.assertEqual(corrected["original_query_representative"]["representative_score"], 0.9)
        self.assertIsNone(corrected["representative_score"])
        self.assertIsNone(corrected["sensitivity"])
        self.assertEqual(corrected["interval_end_source_frame_0based"], 36)

    def test_correction_rejects_stale_original_outside_interval_and_form_endpoint(self):
        self.event.update(interval_start_source_frame_0based=18, interval_end_source_frame_0based=36)
        for field, value in (("expected_original_source_frame_0based", 25),
                             ("representative_source_frame_0based", 12),
                             ("representative_source_frame_0based", 18.0)):
            record = self.correction_record()
            record["reference_corrections"][0][field] = value
            with self.assertRaises(ValueError):
                apply_reference_corrections([self.event], record, self.reference, "R")
        self.event["representative_source"] = "known_move_end"
        with self.assertRaisesRegex(ValueError, "form endpoints"):
            apply_reference_corrections([self.event], self.correction_record(), self.reference, "R")

    def test_correction_rejects_different_reference_and_duplicates(self):
        record = self.correction_record()
        with self.assertRaisesRegex(ValueError, "identity"):
            apply_reference_corrections([self.event], record, self.reference, "Other")
        record["reference_corrections"].append(copy.deepcopy(record["reference_corrections"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            apply_reference_corrections([self.event], record, self.reference, "R")

    def test_corrected_reference_really_changes_target_mapping(self):
        self.event.update(interval_start_source_frame_0based=18, interval_end_source_frame_0based=36)
        old, _ = self.map()
        self.event = apply_reference_corrections([self.event], self.correction_record(), self.reference, "R")[0]
        new, _ = self.map()
        self.assertEqual(old["points"]["representative"]["target_source_frame_0based"], 30)
        self.assertEqual(new["points"]["representative"]["target_source_frame_0based"], 18)
        self.assertEqual(new["reference_original_representative_source_frame_0based"], 24)
        self.assertEqual(new["reference_representative_source"], "manual_reference_correction")
        self.assertEqual(new["points"]["start"], old["points"]["start"])
        self.assertEqual(new["points"]["end"], old["points"]["end"])

    def review_record(self):
        return {"correspondence_reviews": [{"role": "student", "subject_id": "S", "pose_id": "1.event",
                "expected_target_source_frames_0based": [24, 30, 36],
                "expected_reference_representative_source_frame_0based": 24,
                "manual_verdict": "correct", "notes": "frame accepted", "review_date": "2026-10-08"}]}

    def test_manual_review_does_not_remove_automatic_diagnostics(self):
        row, _ = self.map(review_span_seconds=0.1)
        flags = row["diagnostic_flags"].copy()
        apply_correspondence_reviews([row], self.review_record(), "student", "S")
        self.assertEqual(row["manual_review_status"], "correct")
        self.assertEqual(row["diagnostic_flags"], flags)
        self.assertTrue(row["review_required"])
        self.assertEqual(row["manual_review_scope"], "event_correspondence_not_action_quality")

    def test_old_review_not_reused_after_target_or_reference_frame_changes(self):
        for point, key in (("representative", "target_source_frame_0based"),
                           ("representative", "reference_source_frame_0based")):
            row, _ = self.map()
            row["points"][point][key] += 1
            apply_correspondence_reviews([row], self.review_record(), "student", "S")
            self.assertEqual(row["manual_review_status"], "pending")
            self.assertIn("not applicable", row["manual_review_note"])


if __name__ == "__main__":
    unittest.main()
