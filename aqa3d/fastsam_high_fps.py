"""Pair-specific maximum-rate sampling with immutable anchor endpoint frames."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .smpl_dtw import VideoSampling


def maximum_fps_pair(reference: VideoSampling, target: VideoSampling,
                     reference_native_fps: float, target_native_fps: float,
                     mandatory_reference_frames: np.ndarray) -> tuple[VideoSampling, VideoSampling, dict]:
    rates = np.asarray([reference.source_fps, target.source_fps, reference_native_fps, target_native_fps])
    if not np.isfinite(rates).all() or np.any(rates <= 0):
        raise ValueError('Video and native reconstruction rates must be finite and positive.')
    fps = float(rates.min())

    def indices(sampling):
        if sampling.source_frame_count <= 0:
            raise ValueError('Empty source video.')
        count = max(int(np.floor(sampling.source_frame_count / sampling.source_fps * fps + .5)), 1)
        frames = np.floor(np.arange(count, dtype=np.float64) * (sampling.source_fps / fps)).astype(np.int64)
        frames = frames.clip(0, sampling.source_frame_count - 1)
        if np.any(np.diff(frames) <= 0):
            raise ValueError('Maximum-rate grid would duplicate source frames.')
        return frames

    required = np.asarray(mandatory_reference_frames)
    if (required.ndim != 1 or not len(required) or not np.issubdtype(required.dtype, np.integer)
            or np.any(np.diff(required) <= 0) or required[0] < 0
            or required[-1] >= reference.source_frame_count):
        raise ValueError('Anchor frames must be ordered unique exact source frames.')
    ref_grid = indices(reference)
    ref_frames = np.union1d(ref_grid, required)
    target_frames = indices(target)
    policy = {'pair_sample_fps': fps, 'reference_video_fps': reference.source_fps,
              'target_video_fps': target.source_fps, 'reference_native_smpl_fps': reference_native_fps,
              'target_native_smpl_fps': target_native_fps,
              'reference_regular_grid_sample_count': len(ref_grid),
              'reference_extra_annotation_frames': np.setdiff1d(required, ref_grid).tolist(),
              'reference_sample_count': len(ref_frames), 'target_sample_count': len(target_frames),
              'reference_sampling': 'maximum common rate grid plus immutable annotated source frames',
              'time_basis': 'exact source frame / verified source video FPS; not sample index / nominal FPS',
              'review_fps': 5., 'automatic_endpoint_rounding_to_review_grid': False,
              'interpolation_or_nearest_substitution': False}
    return (replace(reference, sample_fps=fps, source_indices=ref_frames),
            replace(target, sample_fps=fps, source_indices=target_frames), policy)
