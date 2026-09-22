"""Validate manifest events and retain tracking provenance during DTW transfer."""

from __future__ import annotations

import numpy as np

from .smpl_dtw import VideoSampling
from .smpl_pose_segmentation import GoldBoundary, validate_gold_boundaries_against_sampling
from .tracking import TrackPoseSequence


def manifest_boundaries(manifest: dict, sampling: VideoSampling) -> list[GoldBoundary]:
    reference = manifest["reference"]
    rows = manifest["endpoint_keyposes"]
    if manifest.get("schema_version") != "1.0":
        raise ValueError("Unsupported reference manifest version.")
    if [r["move_id"] for r in rows] != list(range(1, 25)):
        raise ValueError("Manifest must contain ordered unique move IDs 1..24.")
    for field, actual in (("source_fps", sampling.source_fps),
                          ("source_frame_count", sampling.source_frame_count),
                          ("sample_fps", sampling.sample_fps),
                          ("sample_count", sampling.sample_count)):
        if not np.isclose(reference[field], actual, rtol=0, atol=1e-8):
            raise ValueError(f"Manifest/video mismatch: {field}.")
    boundaries = [GoldBoundary(r["move_id"], r["move_name_zh"],
                               r["boundary_time_seconds"], r["sample_index_0based"])
                  for r in rows]
    validate_gold_boundaries_against_sampling(boundaries, sampling)
    previous_sample = previous_source = -1
    for r in rows:
        sample = r["sample_index_0based"]
        source = int(sampling.source_indices[sample])
        expected = {
            "pose_id": f'{r["move_id"]}.end',
            "reference_video_id": reference["video_id"],
            "reference_sample_sequence_id": reference["sample_sequence_id"],
            "keypose_role": "form_end_boundary_event",
            "sample_frame_1based": sample + 1,
            "source_frame_index_0based": source,
            "source_frame_1based": source + 1,
            "phalp_frame_1based": source + 1,
            "segment_sample_start_index_0based": previous_sample + 1,
            "segment_sample_end_index_0based": sample,
            "segment_source_start_index_0based": previous_source + 1,
            "segment_source_end_index_0based": source,
        }
        for field, value in expected.items():
            if r[field] != value:
                raise ValueError(f'Move {r["move_id"]}: inconsistent {field}.')
        if sample <= previous_sample or source <= previous_source:
            raise ValueError("Manifest boundaries must be strictly increasing.")
        if not np.isclose(r["source_frame_time_seconds"], source / sampling.source_fps,
                          rtol=0, atol=1e-8):
            raise ValueError("Manifest source-frame time is inconsistent.")
        if not r["final_technique_step"].strip():
            raise ValueError("Empty endpoint technique description.")
        previous_sample, previous_source = sample, source
    return boundaries


def resolve_tracking_samples(track: TrackPoseSequence, sampling: VideoSampling,
                             max_gap_seconds: float = 0.0) -> np.ndarray:
    """Return track positions, matching at_source_frames nearest-frame tie-breaking."""
    frames = track.frame_numbers
    if not len(frames) or np.any(np.diff(frames) <= 0):
        raise ValueError("Tracking frames must be nonempty and strictly increasing.")
    if frames[0] < 1 or frames[-1] > sampling.source_frame_count:
        raise ValueError("Tracking frame range is outside the input video.")
    if not np.isfinite(max_gap_seconds) or max_gap_seconds < 0:
        raise ValueError("Tracking gap tolerance must be finite and nonnegative.")
    requested = sampling.source_indices + 1
    right = np.searchsorted(frames, requested).clip(0, len(frames) - 1)
    left = (right - 1).clip(0, len(frames) - 1)
    positions = np.where(abs(frames[left] - requested) <= abs(frames[right] - requested),
                         left, right)
    gaps = abs(frames[positions] - requested)
    limit = int(round(max_gap_seconds * sampling.source_fps))
    missing = requested[gaps > limit]
    if len(missing):
        raise ValueError(f"Missing tracked samples: {len(missing)}; PHALP frames "
                         f"{missing[:10].tolist()}; tolerance={limit} frames.")
    if not np.isfinite(track.body_poses[positions]).all():
        raise ValueError("Sampled tracking contains non-finite rotations.")
    return positions


def tracking_provenance(track: TrackPoseSequence, sampling: VideoSampling,
                        position: int, source: int) -> dict:
    actual = int(track.frame_numbers[position])
    ids = (track.source_track_ids if track.source_track_ids is not None
           else np.full(len(track.frame_numbers), track.track_id))
    switch_frames = track.frame_numbers[np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    near_switch = bool(np.any(abs(switch_frames - actual) <= 1)
                       or np.any(abs(switch_frames - (source + 1)) <= 1))
    return {
        "requested_phalp_frame_1based": source + 1,
        "actual_phalp_frame_1based": actual,
        "actual_source_frame_0based": actual - 1,
        "actual_source_time_seconds": (actual - 1) / sampling.source_fps,
        "frame_offset": actual - (source + 1),
        "source_track_id": int(ids[position]),
        "status": "exact" if actual == source + 1 else "nearest_substitution",
        "near_track_id_switch": near_switch,
    }


def transferred_keypose_rows(manifest, boundaries, path, reference_sampling,
                             target_sampling, reference_track, target_track,
                             reference_positions, target_positions,
                             review_geodesic_degrees=None):
    if len(boundaries) != 24:
        raise ValueError("Expected 24 mapped endpoints.")
    rows = []
    for event, boundary in zip(manifest["endpoint_keyposes"], boundaries):
        if event["move_id"] != boundary.move_id:
            raise ValueError("Manifest and boundary move IDs differ.")
        row = {key: event[key] for key in (
            "move_id", "move_name_pinyin", "move_name_zh", "pose_id",
            "keypose_role", "final_technique_step", "technique_binding_status")}
        row.update(video_id=boundary.video_id,
                   reference_video_id=manifest["reference"]["video_id"],
                   boundary_annotation_status="predicted",
                   boundary_policy=boundary.boundary_policy,
                   candidate_count=boundary.candidate_count,
                   local_geodesic_degrees=boundary.local_geodesic_degrees,
                   local_geodesic_radians=boundary.local_geodesic_radians)
        flags = []
        for prefix, sample, sampling, track, positions in (
            ("reference", boundary.reference_sample_index_0based, reference_sampling,
             reference_track, reference_positions),
            ("student", boundary.target_sample_index_0based, target_sampling,
             target_track, target_positions),
        ):
            source = int(sampling.source_indices[sample])
            row.update({f"{prefix}_sample_index_0based": sample,
                        f"{prefix}_sample_frame_1based": sample + 1,
                        f"{prefix}_boundary_time_seconds": sample / sampling.sample_fps,
                        f"{prefix}_source_frame_0based": source,
                        f"{prefix}_source_frame_1based": source + 1,
                        f"{prefix}_source_time_seconds": source / sampling.source_fps})
            provenance = tracking_provenance(track, sampling, int(positions[sample]), source)
            row.update({f"{prefix}_tracking_{key}": value for key, value in provenance.items()})
            if provenance["status"] != "exact":
                flags.append(f"{prefix}_nearest_substitution")
            if provenance["near_track_id_switch"]:
                flags.append(f"{prefix}_track_id_switch")
        candidates = path[path[:, 1] == boundary.reference_sample_index_0based, 0]
        row["student_candidate_start_time_seconds"] = int(candidates.min()) / target_sampling.sample_fps
        row["student_candidate_end_time_seconds"] = int(candidates.max()) / target_sampling.sample_fps
        if (review_geodesic_degrees is not None
                and boundary.local_geodesic_degrees > review_geodesic_degrees):
            flags.append("high_geodesic_distance")
        row["review_required"] = bool(flags)
        row["diagnostic_flags"] = ";".join(flags)
        row["manual_review_status"] = "pending"
        rows.append(row)
    return rows
