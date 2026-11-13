"""Maximum common-rate sampling through exact native cropped-student mappings."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .fastsam_high_fps import maximum_fps_pair
from .smpl_dtw import VideoSampling


def maximum_native_student_pair(reference: VideoSampling, target_video: VideoSampling,
                                 reference_native_fps: float, student_native_fps: float,
                                 native_indices: np.ndarray, output_video_indices: np.ndarray,
                                 required_reference_frames: np.ndarray):
    native = np.asarray(native_indices)
    output = np.asarray(output_video_indices)
    if (native.ndim != 1 or not len(native) or not np.issubdtype(native.dtype, np.integer)
            or native[0] < 0 or np.any(np.diff(native) != 1)):
        raise ValueError('Native student sampling indices must be complete and consecutive.')
    if (output.shape != native.shape or not np.issubdtype(output.dtype, np.integer)
            or output[0] < 0 or output[-1] >= target_video.source_frame_count
            or np.any(np.diff(output) <= 0)):
        raise ValueError('Invalid native-to-trimmed-video frame mapping.')
    ref_sampling, _, policy = maximum_fps_pair(reference, target_video, reference_native_fps,
                                              student_native_fps, required_reference_frames)
    fps = policy['pair_sample_fps']
    count = max(int(np.floor(len(native) / student_native_fps * fps + .5)), 1)
    # Use the retained native sample clock, never an assumed video-frame = sample-index mapping.
    positions = np.floor(np.arange(count) * (student_native_fps / fps)).astype(np.int64)
    if positions[-1] >= len(native) or np.any(np.diff(positions) <= 0):
        raise ValueError('Native student sampling would duplicate or exceed source rows.')
    target_sampling = replace(target_video, sample_fps=fps, source_indices=output[positions])
    policy.update(target_sampling='maximum common FPS on the complete native SMPL clock; exact stored video mapping',
                  target_sample_count=len(positions), target_native_sample_count=len(native),
                  target_first_native_sample_index_0based=int(native[0]),
                  target_native_sampling_complete=True,
                  time_basis='reference source frame/FPS; student explicit trimmed-video PTS and original timestamp provenance',
                  target_timestamp_basis='trimmed_video_PTS',
                  native_video_frame_skips_are_not_missing_poses=True,
                  student_rows_removed_or_repaired=False)
    return ref_sampling, target_sampling, positions, policy
