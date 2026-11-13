"""Read single-person FastSAM SMPL without changing its source timeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .smpl_dtw import VideoSampling


@dataclass(frozen=True)
class FastSamPoseSequence:
    source_path: Path
    source_indices: np.ndarray
    timestamps: np.ndarray
    input_fps: float
    body_poses: np.ndarray
    usable_mask: np.ndarray
    source_track_ids: np.ndarray
    conversion_mean_mm: np.ndarray | None
    conversion_p95_mm: np.ndarray | None
    audit: dict

    def sample_positions(self, sampling: VideoSampling) -> np.ndarray:
        if not np.isclose(self.input_fps, sampling.source_fps, rtol=0, atol=1e-6):
            raise ValueError('NPZ input_fps differs from the source video.')
        if self.source_indices[-1] >= sampling.source_frame_count:
            raise ValueError('NPZ source-frame index is outside the video.')
        positions = np.searchsorted(self.source_indices, sampling.source_indices)
        clipped = positions.clip(0, len(self.source_indices) - 1)
        exact = (positions < len(self.source_indices)) & (
            self.source_indices[clipped] == sampling.source_indices)
        missing = sampling.source_indices[~exact]
        if len(missing):
            raise ValueError(f'Missing exact source frames: {missing[:10].tolist()}')
        invalid = sampling.source_indices[~self.usable_mask[positions]]
        if len(invalid):
            raise ValueError(f'Invalid or ignored sampled source frames: {invalid[:10].tolist()}')
        return positions

    def provenance(self, position: int, sampling: VideoSampling) -> dict:
        source = int(self.source_indices[position])
        around = self.source_track_ids[max(0, position - 1):position + 2]
        result = {
            'npz_row_0based': int(position),
            'source_frame_0based': source,
            'source_frame_1based': source + 1,
            'source_time_seconds': source / sampling.source_fps,
            'npz_timestamp_seconds': float(self.timestamps[position]),
            'source_track_id': int(self.source_track_ids[position]),
            'near_source_track_switch': bool(np.any(around != self.source_track_ids[position])),
            'tracking_match': 'exact_source_frame',
        }
        for name, values in (('conversion_mean_vertex_error_mm', self.conversion_mean_mm),
                             ('conversion_p95_vertex_error_mm', self.conversion_p95_mm)):
            result[name] = float(values[position]) if values is not None else None
        return result


def load_fastsam_pose_sequence(path: str | Path) -> FastSamPoseSequence:
    source = Path(path)
    required = {'schema_id', 'source_frame_indices', 'timestamps', 'input_fps',
                'body_pose', 'valid_mask', 'ignore_mask', 'source_track_ids'}
    with np.load(source, allow_pickle=False) as data:
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f'Missing FastSAM fields: {sorted(missing)}')
        if data['schema_id'].shape != () or data['schema_id'].item() != 'fastsam_smpl_primary_v1':
            raise ValueError('Expected schema fastsam_smpl_primary_v1.')
        indices = data['source_frame_indices'].copy()
        if (indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer)
                or indices[0] < 0 or np.any(np.diff(indices) <= 0)):
            raise ValueError('Source indices must be nonnegative, unique and increasing integers.')
        count = len(indices)
        timestamps = data['timestamps'].astype(np.float64)
        fps_array = data['input_fps']
        if fps_array.shape != () or not np.isfinite(fps_array) or fps_array <= 0:
            raise ValueError('input_fps must be a finite positive scalar.')
        fps = float(fps_array)
        if (timestamps.shape != (count,) or not np.isfinite(timestamps).all()
                or not np.allclose(timestamps, indices / fps, rtol=0, atol=1e-6)):
            raise ValueError('NPZ timestamps do not match source indices/input_fps.')
        valid, ignored = data['valid_mask'], data['ignore_mask']
        if any(a.shape != (count,) or a.dtype != np.bool_ for a in (valid, ignored)):
            raise ValueError('Validity masks must be boolean arrays on the complete timeline.')
        usable = valid & ~ignored
        if not usable.any():
            raise ValueError('No usable SMPL frames.')
        poses = data['body_pose'].copy()
        if poses.shape != (count, 23, 3, 3) or not np.issubdtype(poses.dtype, np.floating):
            raise ValueError('body_pose must have shape (N, 23, 3, 3).')
        rotations = poses[usable].astype(np.float64)
        if not np.isfinite(rotations).all():
            raise ValueError('Non-finite rotations in usable SMPL frames.')
        orth_error = float(np.max(abs(rotations @ np.swapaxes(rotations, -1, -2) - np.eye(3))))
        det_error = float(np.max(abs(np.linalg.det(rotations) - 1)))
        # Allow float32 export roundoff, but never project or repair rotations.
        if max(orth_error, det_error) > 1e-3:
            raise ValueError('body_pose is not a valid set of rotation matrices.')
        track_ids = data['source_track_ids'].copy()
        if track_ids.shape != (count,) or not np.issubdtype(track_ids.dtype, np.integer):
            raise ValueError('source_track_ids must preserve one integer ID per source row.')
        errors = []
        for name in ('conversion_mean_vertex_error_mm', 'conversion_p95_vertex_error_mm'):
            values = data[name].copy() if name in data.files else None
            if values is not None and (values.shape != (count,)
                                       or not np.isfinite(values[usable]).all()):
                raise ValueError('Invalid conversion diagnostic: ' + name)
            errors.append(values)
        audit = {
            'schema_id': 'fastsam_smpl_primary_v1', 'source_row_count': count,
            'usable_frame_count': int(usable.sum()), 'ignored_frame_count': int(ignored.sum()),
            'invalid_frame_count': int((~valid).sum()), 'input_fps': fps,
            'first_source_frame_0based': int(indices[0]), 'last_source_frame_0based': int(indices[-1]),
            'max_timestamp_error_seconds': float(np.max(abs(timestamps - indices / fps))),
            'max_rotation_orthogonality_error': orth_error,
            'max_rotation_determinant_error': det_error,
            'source_rows_removed': 0, 'interpolation_or_pose_repair': False,
        }
    return FastSamPoseSequence(source, indices, timestamps, fps, poses, usable, track_ids,
                               errors[0], errors[1], audit)


def export_endpoint_smpl(source: Path, output: Path, positions: np.ndarray,
                         move_ids: list[int], source_sha256: str) -> None:
    """Preserve every source per-frame field at the exact selected rows."""
    with np.load(source, allow_pickle=False) as data:
        count = len(data['source_frame_indices'])
        values = {}
        for key in data.files:
            value = data[key]
            # image_size is global metadata, even for a synthetic two-row file.
            per_frame = value.ndim > 0 and value.shape[0] == count and key != 'image_size'
            values[key] = value[positions] if per_frame else value
    values.update(export_kind=np.array('exact_endpoint_subset'),
                  endpoint_move_ids=np.asarray(move_ids, dtype=np.int64),
                  source_npz_sha256=np.array(source_sha256))
    np.savez_compressed(output, **values)


def transferred_fastsam_rows(manifest: dict, boundaries: list, path: np.ndarray,
                            reference_sampling: VideoSampling, teacher_sampling: VideoSampling,
                            reference: FastSamPoseSequence, teacher: FastSamPoseSequence,
                            reference_positions: np.ndarray, teacher_positions: np.ndarray,
                            *, target_role: str = 'teacher') -> list[dict]:
    if target_role not in ('teacher', 'student'):
        raise ValueError('Unsupported endpoint target role.')
    events = manifest['endpoint_keyposes']
    if len(events) != len(boundaries) or len(events) != 24:
        raise ValueError('Expected the 24 reference endpoint events.')
    rows = []
    for event, boundary in zip(events, boundaries):
        if event['move_id'] != boundary.move_id:
            raise ValueError('Reference and transferred event IDs differ.')
        row = {key: event[key] for key in (
            'move_id', 'move_name_pinyin', 'move_name_zh', 'pose_id',
            'keypose_role', 'final_technique_step', 'technique_binding_status')}
        row.update(video_id=boundary.video_id, reference_video_id=manifest['reference']['video_id'],
                   boundary_annotation_status='predicted', boundary_policy=boundary.boundary_policy,
                   candidate_count=boundary.candidate_count,
                   local_geodesic_radians=boundary.local_geodesic_radians,
                   local_geodesic_degrees=boundary.local_geodesic_degrees,
                   manual_review_status='pending', review_required=True)
        flags = ['multiple_dtw_candidates'] if boundary.candidate_count > 1 else []
        for prefix, sample, sampling, sequence, positions in (
            ('reference', boundary.reference_sample_index_0based, reference_sampling,
             reference, reference_positions),
            (target_role, boundary.target_sample_index_0based, teacher_sampling,
             teacher, teacher_positions),
        ):
            row[prefix + '_sample_index_0based'] = int(sample)
            row[prefix + '_grid_time_seconds'] = sample / sampling.sample_fps
            provenance = sequence.provenance(int(positions[sample]), sampling)
            if provenance['source_frame_0based'] != int(sampling.source_indices[sample]):
                raise ValueError('Endpoint does not address an exact source frame.')
            row.update({prefix + '_' + key: value for key, value in provenance.items()})
            if provenance['near_source_track_switch']:
                flags.append(prefix + '_source_track_switch')
        candidates = path[path[:, 1] == boundary.reference_sample_index_0based, 0]
        row[target_role + '_candidate_start_grid_time_seconds'] = int(candidates.min()) / teacher_sampling.sample_fps
        row[target_role + '_candidate_end_grid_time_seconds'] = int(candidates.max()) / teacher_sampling.sample_fps
        row['diagnostic_flags'] = ';'.join(flags)
        rows.append(row)
    return rows
