from dataclasses import asdict
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from aqa3d.fastsam_smpl import (
    export_endpoint_smpl, load_fastsam_pose_sequence, transferred_fastsam_rows,
)
from aqa3d.reference_manifest import build_endpoint_keyposes
from aqa3d.smpl_dtw import (
    ReferenceFrameMatch, VideoSampling, pairwise_geodesic_costs,
    uniform_sample_source_indices,
)
from aqa3d.smpl_pose_segmentation import GoldBoundary, build_mapped_boundaries


class FastSamSmplTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'R.npz'
        self.count = 100
        self.fps = 10.
        self.arrays = {
            'schema_id': np.array('fastsam_smpl_primary_v1'),
            'source_frame_indices': np.arange(self.count),
            'timestamps': np.arange(self.count) / self.fps,
            'input_fps': np.array(self.fps), 'fps': np.array(self.fps),
            'body_pose': np.tile(np.eye(3, dtype=np.float32), (self.count, 23, 1, 1)),
            'global_orient': np.tile(np.eye(3, dtype=np.float32), (self.count, 1, 1, 1)),
            'valid_mask': np.ones(self.count, dtype=bool),
            'ignore_mask': np.zeros(self.count, dtype=bool),
            'source_track_ids': np.zeros(self.count, dtype=np.int64),
            'betas': np.arange(self.count * 10, dtype=np.float32).reshape(self.count, 10),
            'joints': np.arange(self.count * 72, dtype=np.float32).reshape(self.count, 24, 3),
            'image_size': np.array([1920, 1080]),
            'conversion_mean_vertex_error_mm': np.arange(self.count, dtype=np.float32),
            'conversion_p95_vertex_error_mm': np.arange(self.count, dtype=np.float32) * 2,
        }
        self.save()
        self.sampling = VideoSampling(Path('R.mp4'), self.fps, self.count, 5., np.arange(0, self.count, 2))

    def save(self):
        np.savez_compressed(self.path, **self.arrays)

    def test_exact_source_rows_and_unmodified_file(self):
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        sequence = load_fastsam_pose_sequence(self.path)
        positions = sequence.sample_positions(self.sampling)
        np.testing.assert_array_equal(positions, self.sampling.source_indices)
        np.testing.assert_array_equal(sequence.body_poses, self.arrays['body_pose'])
        self.assertEqual(sequence.provenance(12, self.sampling)['source_frame_1based'], 13)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), before)

    def test_invalid_rows_are_retained_but_not_used(self):
        self.arrays['valid_mask'][2] = False
        self.arrays['body_pose'][2] = np.nan
        self.save()
        sequence = load_fastsam_pose_sequence(self.path)
        self.assertEqual(len(sequence.source_indices), self.count)
        self.assertEqual(sequence.audit['usable_frame_count'], self.count - 1)
        with self.assertRaisesRegex(ValueError, 'Invalid or ignored'):
            sequence.sample_positions(self.sampling)

    def test_ignored_row_is_not_nearest_substituted(self):
        self.arrays['ignore_mask'][2] = True
        self.save()
        with self.assertRaisesRegex(ValueError, 'Invalid or ignored'):
            load_fastsam_pose_sequence(self.path).sample_positions(self.sampling)

    def test_missing_exact_frame_rejected(self):
        for key, value in list(self.arrays.items()):
            if value.ndim > 0 and value.shape[0] == self.count:
                self.arrays[key] = np.delete(value, 2, axis=0)
        self.save()
        with self.assertRaisesRegex(ValueError, 'Missing exact source frames'):
            load_fastsam_pose_sequence(self.path).sample_positions(self.sampling)

    def test_duplicate_source_indices_rejected(self):
        self.arrays['source_frame_indices'][2] = 1
        self.save()
        with self.assertRaisesRegex(ValueError, 'unique and increasing'):
            load_fastsam_pose_sequence(self.path)

    def test_timestamp_mismatch_rejected(self):
        self.arrays['timestamps'][2] += 0.1
        self.save()
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            load_fastsam_pose_sequence(self.path)

    def test_fps_and_source_range_mismatch_rejected(self):
        sequence = load_fastsam_pose_sequence(self.path)
        wrong_fps = VideoSampling(Path('R.mp4'), 12., self.count, 5., np.arange(0, self.count, 2))
        with self.assertRaisesRegex(ValueError, 'input_fps'):
            sequence.sample_positions(wrong_fps)
        short = VideoSampling(Path('R.mp4'), self.fps, 90, 5., np.arange(0, 90, 2))
        with self.assertRaisesRegex(ValueError, 'outside the video'):
            sequence.sample_positions(short)

    def test_non_rotation_and_nonfinite_values_rejected(self):
        self.arrays['body_pose'][2, 0, 0, 0] = 2
        self.save()
        with self.assertRaisesRegex(ValueError, 'rotation matrices'):
            load_fastsam_pose_sequence(self.path)
        self.arrays['body_pose'][2, 0, 0, 0] = np.nan
        self.save()
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            load_fastsam_pose_sequence(self.path)

    def test_unknown_schema_and_mask_type_rejected(self):
        self.arrays['schema_id'] = np.array('unknown')
        self.save()
        with self.assertRaisesRegex(ValueError, 'schema'):
            load_fastsam_pose_sequence(self.path)
        self.arrays['schema_id'] = np.array('fastsam_smpl_primary_v1')
        self.arrays['valid_mask'] = np.ones(self.count, dtype=int)
        self.save()
        with self.assertRaisesRegex(ValueError, 'boolean'):
            load_fastsam_pose_sequence(self.path)

    def test_global_orientation_does_not_change_local_cost(self):
        first = load_fastsam_pose_sequence(self.path)
        self.arrays['global_orient'][:, 0] = np.diag([-1., -1., 1.])
        self.save()
        second = load_fastsam_pose_sequence(self.path)
        costs = pairwise_geodesic_costs(first.body_poses[:2], second.body_poses[:2])
        np.testing.assert_array_equal(costs, np.zeros((2, 2)))

    def test_export_preserves_original_pose_shape_and_geometry(self):
        positions = np.array([2, 12, 24])
        output = self.root / 'endpoints.npz'
        export_endpoint_smpl(self.path, output, positions, [1, 2, 3], 'source-hash')
        with np.load(output, allow_pickle=False) as data:
            for key in ('body_pose', 'global_orient', 'betas', 'joints', 'source_frame_indices', 'timestamps'):
                np.testing.assert_array_equal(data[key], self.arrays[key][positions])
            np.testing.assert_array_equal(data['image_size'], [1920, 1080])
            self.assertEqual(data['export_kind'].item(), 'exact_endpoint_subset')
            self.assertEqual(data['source_npz_sha256'].item(), 'source-hash')

    def test_noninteger_fps_keeps_grid_and_source_time_distinct(self):
        self.arrays['input_fps'] = np.array(29.97002997002997)
        self.arrays['timestamps'] = np.arange(self.count) / float(self.arrays['input_fps'])
        self.save()
        fps = float(self.arrays['input_fps'])
        indices = uniform_sample_source_indices(self.count, fps, 5.)
        sampling = VideoSampling(Path('R.mp4'), fps, self.count, 5., indices)
        sequence = load_fastsam_pose_sequence(self.path)
        positions = sequence.sample_positions(sampling)
        self.assertEqual(sequence.provenance(int(positions[1]), sampling)['source_frame_0based'], int(indices[1]))
        self.assertNotEqual(float(indices[1]) / fps, 1 / 5)

    def test_transfer_retains_event_text_and_explicit_npz_provenance(self):
        sequence = load_fastsam_pose_sequence(self.path)
        positions = sequence.sample_positions(self.sampling)
        gold = [GoldBoundary(i, f'move{i}', i / 5, i) for i in range(1, 25)]
        events = build_endpoint_keyposes(gold, self.sampling,
            {i: f'unchanged technique {i}' for i in range(1, 25)},
            reference_video_id='R', reference_sample_sequence_id='R_5FPS')
        manifest = {'reference': {'video_id': 'R'}, 'endpoint_keyposes': [asdict(e) for e in events]}
        matches = [ReferenceFrameMatch(i, i + 1, 1, .1) for i in range(1, 25)]
        boundaries = build_mapped_boundaries('T', gold, matches, self.sampling, self.sampling)
        path = np.array([[i + 1, i] for i in range(1, 25)])
        rows = transferred_fastsam_rows(manifest, boundaries, path, self.sampling, self.sampling,
                                       sequence, sequence, positions, positions)
        self.assertEqual(len(rows), 24)
        for i, row in enumerate(rows, 1):
            self.assertEqual(row['teacher_source_frame_0based'], (i + 1) * 2)
            self.assertEqual(row['teacher_npz_row_0based'], (i + 1) * 2)
            self.assertEqual(row['final_technique_step'], f'unchanged technique {i}')
            self.assertEqual(row['manual_review_status'], 'pending')
            self.assertTrue(row['review_required'])
            self.assertFalse(any('phalp' in key.lower() or 'student' in key for key in row))

    def test_source_track_switch_is_diagnostic_not_reselection(self):
        self.arrays['source_track_ids'][10:] = 2
        self.save()
        sequence = load_fastsam_pose_sequence(self.path)
        self.assertTrue(sequence.provenance(9, self.sampling)['near_source_track_switch'])
        self.assertEqual(sequence.provenance(10, self.sampling)['source_track_id'], 2)
        self.assertFalse(sequence.provenance(20, self.sampling)['near_source_track_switch'])


if __name__ == '__main__':
    unittest.main()
