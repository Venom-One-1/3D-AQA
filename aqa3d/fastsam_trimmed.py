"""Use explicit cropped-video frame mappings and presentation timestamps."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
import json
import shutil
import subprocess

import numpy as np

from .fastsam_smpl import FastSamPoseSequence
from .smpl_dtw import VideoSampling


def sampling_from_video_pts(video: Path, timestamps: np.ndarray, source_fps: float,
                            duration: float, sample_fps: float = 5.) -> VideoSampling:
    pts = np.asarray(timestamps, dtype=np.float64)
    if (pts.ndim != 1 or not len(pts) or not np.isfinite(pts).all()
            or np.any(np.diff(pts) <= 0) or abs(pts[0]) > 1.5e-6
            or not np.isfinite(duration) or duration <= pts[-1]
            or source_fps <= 0 or sample_fps <= 0):
        raise ValueError('Invalid trimmed video presentation timeline.')
    count = max(int(np.floor(duration * sample_fps + .5)), 1)
    grid = np.arange(count, dtype=np.float64) / sample_fps
    indices = (np.searchsorted(pts, grid, side='right') - 1).clip(0, len(pts) - 1)
    if np.any(np.diff(indices) <= 0):
        raise ValueError('Video cannot provide unique frames at the requested sample rate.')
    return VideoSampling(video, source_fps, len(pts), sample_fps, indices)


def sampling_from_native_smpl(sampling: VideoSampling, native_indices: np.ndarray,
                              output_indices: np.ndarray, native_fps: float) -> VideoSampling:
    """Decimate a complete native SMPL sample clock through its exact video mapping."""
    native = np.asarray(native_indices)
    output = np.asarray(output_indices)
    if (native.ndim != 1 or not len(native) or not np.issubdtype(native.dtype, np.integer)
            or native[0] < 0 or np.any(np.diff(native) != 1)):
        raise ValueError('Native SMPL sample indices must be complete and consecutive.')
    if (output.shape != native.shape or not np.issubdtype(output.dtype, np.integer)
            or output[0] < 0 or output[-1] >= sampling.source_frame_count
            or np.any(np.diff(output) <= 0)):
        raise ValueError('Invalid native SMPL to cropped-video frame mapping.')
    if (not np.isfinite(native_fps) or native_fps <= 0
            or not np.isfinite(sampling.sample_fps) or sampling.sample_fps <= 0):
        raise ValueError('Invalid native or requested sample rate.')
    ratio = native_fps / sampling.sample_fps
    stride = int(round(ratio))
    if stride < 1 or not np.isclose(ratio, stride, rtol=0, atol=1e-8):
        raise ValueError('Native SMPL rate must be an integer multiple of the DTW rate.')
    # Phase starts at the first retained native sample, not the cropped video frame number.
    return replace(sampling, source_indices=output[::stride].copy())


def inspect_trimmed_video(video: Path) -> tuple[VideoSampling, np.ndarray, dict]:
    executable = shutil.which('ffprobe')
    if executable is None:
        raise FileNotFoundError('ffprobe is required to verify the explicit video timestamps.')
    command = [executable, '-v', 'error', '-select_streams', 'v:0', '-show_packets', '-show_streams',
               '-show_entries', 'packet=pts_time,duration_time:stream=codec_name,width,height,avg_frame_rate,nb_frames,duration_ts,time_base',
               '-of', 'json', str(video)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    streams = data.get('streams', [])
    if len(streams) != 1 or streams[0]['codec_name'] != 'h264':
        raise ValueError('Expected the reviewed H.264 trimmed MP4 stream.')
    stream = streams[0]
    pts = np.sort(np.asarray([float(p['pts_time']) for p in data['packets']], dtype=np.float64))
    if len(pts) != int(stream['nb_frames']):
        raise ValueError('Packet/frame counts differ; refuse to infer a video frame mapping.')
    fps = float(Fraction(stream['avg_frame_rate']))
    duration = float(int(stream['duration_ts']) * Fraction(stream['time_base']))
    sampling = sampling_from_video_pts(video, pts, fps, duration)
    metadata = {'source_fps': fps, 'source_frame_count': len(pts), 'duration_seconds': duration,
                'width': int(stream['width']), 'height': int(stream['height']),
                'timestamp_source': 'sorted unique H264 packet presentation timestamps; count equals MP4 frame count',
                'sample_policy': '5 FPS time grid; exact decoded video frame at or immediately before each grid time',
                'sample_count': sampling.sample_count}
    return sampling, pts, metadata


@dataclass(frozen=True)
class FastSamTrimmedSequence(FastSamPoseSequence):
    original_source_indices: np.ndarray
    original_source_timestamps: np.ndarray
    video_timestamps: np.ndarray

    def sample_positions(self, sampling: VideoSampling, *, max_gap_frames: int = 0) -> np.ndarray:
        if max_gap_frames < 0:
            raise ValueError('Gap tolerance must be nonnegative.')
        if sampling.source_frame_count != len(self.video_timestamps):
            raise ValueError('Trimmed video frame count changed.')
        positions = np.searchsorted(self.source_indices, sampling.source_indices)
        safe = positions.clip(0, len(self.source_indices) - 1)
        exact = (positions < len(self.source_indices)) & (self.source_indices[safe] == sampling.source_indices)
        if not exact.all():
            if not max_gap_frames:
                raise ValueError('Missing exact trimmed-video SMPL frames: ' + str(sampling.source_indices[~exact].tolist()))
            right = safe
            left = (right - 1).clip(0, len(self.source_indices) - 1)
            requested = sampling.source_indices
            positions = np.where(abs(self.source_indices[left] - requested) <= abs(self.source_indices[right] - requested), left, right)
            if np.any(abs(self.source_indices[positions] - requested) > max_gap_frames):
                raise ValueError('Missing sample exceeds the explicitly allowed frame gap.')
        if not self.usable_mask[positions].all():
            raise ValueError('Invalid or ignored exact trimmed-video SMPL samples.')
        return positions

    def provenance(self, position: int, sampling: VideoSampling) -> dict:
        result = super().provenance(position, sampling)
        output_frame = int(self.source_indices[position])
        result.update(source_time_seconds=float(self.video_timestamps[output_frame]),
                      original_source_frame_0based=int(self.original_source_indices[position]),
                      original_source_timestamp_seconds=float(self.original_source_timestamps[position]),
                      timestamp_basis='trimmed_video_PTS',
                      tracking_match='exact_output_video_frame')
        return result


def load_trimmed_sequence(path: Path, sampling: VideoSampling, video_pts: np.ndarray,
                          trim_manifest: dict) -> FastSamTrimmedSequence:
    with np.load(path, allow_pickle=False) as data:
        if (data['schema_id'].item() != 'fastsam_smpl_primary_v1'
                or data['trim_schema'].item() != 'reviewed_video_smpl_trim_v1'):
            raise ValueError('Unsupported reviewed trimmed SMPL schema.')
        indices = data['output_video_frame_indices'].copy()
        original = data['source_frame_indices'].copy()
        count = len(indices)
        if (indices.ndim != 1 or not count or not np.issubdtype(indices.dtype, np.integer)
                or indices[0] < 0 or np.any(np.diff(indices) <= 0)
                or indices[-1] >= sampling.source_frame_count):
            raise ValueError('Invalid output_video_frame_indices.')
        offset = int(data['trim_source_start_frame'])
        if original.shape != indices.shape or not np.array_equal(original, indices + offset):
            raise ValueError('Original/cropped frame mapping disagrees with the recorded crop offset.')
        timestamps = data['timestamps'].copy()
        if (timestamps.shape != (count,) or not np.isfinite(timestamps).all()
                or np.any(np.diff(timestamps) <= 0)
                or not np.allclose(timestamps, video_pts[indices], rtol=0, atol=1.5e-6)):
            raise ValueError('SMPL timestamps disagree with actual trimmed video PTS.')
        if (sampling.source_frame_count != trim_manifest['counts']['video_frames']
                or count != trim_manifest['counts']['smpl_samples']):
            raise ValueError('Source trim manifest frame counts do not match the files.')
        valid, ignored = data['valid_mask'], data['ignore_mask']
        if any(a.shape != (count,) or a.dtype != np.bool_ for a in (valid, ignored)):
            raise ValueError('Invalid SMPL validity masks.')
        usable = valid & ~ignored
        if not usable.any():
            raise ValueError('No usable SMPL poses.')
        poses = data['body_pose'].copy()
        if poses.shape != (count, 23, 3, 3) or not np.issubdtype(poses.dtype, np.floating):
            raise ValueError('Expected body_pose (N,23,3,3).')
        rotations = poses[usable].astype(np.float64)
        if not np.isfinite(rotations).all():
            raise ValueError('Non-finite usable body rotations.')
        orth = float(np.max(abs(rotations @ np.swapaxes(rotations, -1, -2) - np.eye(3))))
        det = float(np.max(abs(np.linalg.det(rotations) - 1)))
        if max(orth, det) > 1e-3:
            raise ValueError('Invalid body rotation matrices; no repair was applied.')
        tracks = data['source_track_ids'].copy()
        if tracks.shape != (count,) or not np.issubdtype(tracks.dtype, np.integer):
            raise ValueError('Invalid source track provenance.')
        residuals = [data[key].copy() for key in ('conversion_mean_vertex_error_mm', 'conversion_p95_vertex_error_mm')]
        if any(a.shape != (count,) or not np.isfinite(a[usable]).all() for a in residuals):
            raise ValueError('Invalid conversion residual diagnostics.')
        original_times = data['source_video_timestamps'].copy()
        if original_times.shape != (count,) or not np.isfinite(original_times).all():
            raise ValueError('Invalid original timestamp provenance.')
        input_fps = float(data['input_fps'])
        if not np.isfinite(input_fps) or input_fps <= 0:
            raise ValueError('Invalid original input_fps metadata.')
        missing = np.setdiff1d(np.arange(sampling.source_frame_count), indices)
        audit = {'schema_id': 'fastsam_smpl_primary_v1', 'trim_schema': 'reviewed_video_smpl_trim_v1',
                 'source_row_count': count, 'usable_frame_count': int(usable.sum()),
                 'original_input_fps': input_fps, 'trimmed_video_average_fps': sampling.source_fps,
                 'crop_source_start_frame': offset, 'missing_output_video_frames': missing.tolist(),
                 'max_timestamp_error_seconds': float(np.max(abs(timestamps - video_pts[indices]))),
                 'max_rotation_orthogonality_error': orth, 'max_rotation_determinant_error': det,
                 'timestamp_basis': 'explicit actual video PTS, not source_frame/input_fps',
                 'source_rows_removed': 0, 'interpolation_or_pose_repair': False}
    return FastSamTrimmedSequence(path, indices, timestamps, input_fps, poses, usable, tracks,
                                  residuals[0], residuals[1], audit, original, original_times, video_pts)
