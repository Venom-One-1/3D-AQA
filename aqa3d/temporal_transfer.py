"""Transfer semantic reference events along a verified full-video DTW path."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .smpl_dtw import VideoSampling
from .tracking import TrackPoseSequence


POINT_FIELDS = {
    "start": "interval_start_source_frame_0based",
    "representative": "representative_source_frame_0based",
    "end": "interval_end_source_frame_0based",
}


def apply_reference_corrections(events: list[dict], record: dict,
                                sampling: VideoSampling, video_id: str) -> list[dict]:
    if record.get("schema_version") != "1.0" or record.get("reference_video_id") != video_id:
        raise ValueError("Manual reference corrections have a different schema/video identity.")
    corrections = record.get("reference_corrections", [])
    if len({r["pose_id"] for r in corrections}) != len(corrections):
        raise ValueError("Duplicate reference corrections.")
    lookup = {r["pose_id"]: r for r in corrections}
    result = deepcopy(events)
    for event in result:
        correction = lookup.get(event["pose_id"])
        if correction is None:
            continue
        original = event["representative_source_frame_0based"]
        source = correction["representative_source_frame_0based"]
        if correction["expected_original_source_frame_0based"] != original:
            raise ValueError("Manual correction does not match the original representative.")
        if event["representative_source"] == "known_move_end":
            raise ValueError("Representative overrides cannot change form endpoints.")
        if (type(source) is not int or not event["interval_start_source_frame_0based"]
                <= source <= event["interval_end_source_frame_0based"]):
            raise ValueError("Manual representative must be an integer inside the event interval.")
        if not correction.get("reason", "").strip():
            raise ValueError("Manual correction requires a reason.")
        sample = nearest_reference_sample(source, sampling,
            event["reference_segment_start_source_frame_0based"],
            event["reference_segment_end_source_frame_0based"])
        original_result = {key: value for key, value in event.items()
                           if key.startswith("representative_") or key in (
                               "sensitivity", "boundary_delta_seconds", "selection_rationale")}
        event.update(
            original_query_representative=original_result,
            representative_source_frame_0based=source,
            representative_source_frame_1based=source + 1,
            representative_time_seconds=source / sampling.source_fps,
            representative_source="manual_reference_correction",
            representative_score=None, sensitivity=None,
            representative_sample_index_5fps_0based=sample,
            representative_sample_frame_5fps_1based=sample + 1,
            representative_grid_time_5fps_seconds=sample / sampling.sample_fps,
            representative_grid_source_frame_0based=int(sampling.source_indices[sample]),
            boundary_delta_seconds=(source - event["reference_segment_end_source_frame_0based"])
                / sampling.source_fps,
            annotation_status="human_corrected_reference_representative",
            selection_rationale=correction["reason"],
            manual_reference_correction=deepcopy(correction),
        )
    return result


def apply_correspondence_reviews(events: list[dict], record: dict, role: str, subject_id: str) -> None:
    reviews = record.get("correspondence_reviews", [])
    keys = [(r["role"], r["subject_id"], r["pose_id"]) for r in reviews]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate manual correspondence reviews.")
    lookup = dict(zip(keys, reviews))
    for event in events:
        review = lookup.get((role, subject_id, event["pose_id"]))
        if review is None:
            continue
        if review["manual_verdict"] not in ("correct", "false_positive", "uncertain"):
            raise ValueError("Invalid manual correspondence verdict.")
        sources = [event["points"][k]["target_source_frame_0based"] for k in POINT_FIELDS]
        ref_source = event["points"]["representative"]["reference_source_frame_0based"]
        if (sources != review["expected_target_source_frames_0based"]
                or ref_source != review["expected_reference_representative_source_frame_0based"]
                or event["status"] != "mapped"):
            event["manual_review_note"] = "Prior review is not applicable to these changed frames."
            continue
        event.update(manual_review_status=review["manual_verdict"],
                     manual_review_note=review["notes"], manual_review_date=review["review_date"],
                     manual_review_scope="event_correspondence_not_action_quality")


def validate_path(path: np.ndarray, costs: np.ndarray,
                  reference: VideoSampling, target: VideoSampling) -> None:
    if path.ndim != 2 or path.shape[1] != 2 or not np.issubdtype(path.dtype, np.integer):
        raise ValueError("DTW path must be an integer (P, 2) array: target, reference.")
    if not len(path) or costs.shape != (len(path),) or not np.isfinite(costs).all():
        raise ValueError("Invalid DTW path costs.")
    if np.any(costs < 0) or np.any(costs > np.pi + 1e-6):
        raise ValueError("Geodesic path costs must be radians in [0, pi].")
    if (not np.array_equal(path[0], [0, 0]) or not np.array_equal(
            path[-1], [target.sample_count - 1, reference.sample_count - 1])):
        raise ValueError("DTW path must cover both full videos.")
    steps = np.diff(path, axis=0)
    if np.any(steps < 0) or np.any(steps > 1) or np.any(steps.sum(axis=1) == 0):
        raise ValueError("DTW path must use monotonic unit steps.")


def verify_path_archive(data: Any, path: np.ndarray, costs: np.ndarray,
                        reference: VideoSampling, target: VideoSampling) -> None:
    validate_path(path, costs, reference, target)
    for prefix, sampling, column in (("target", target, 0), ("reference", reference, 1)):
        indices = path[:, column]
        source = sampling.source_indices[indices]
        for key, expected in (
            (f"{prefix}_source_frame_indices_0based", source),
            (f"{prefix}_phalp_frames_1based", source + 1),
            (f"{prefix}_times_seconds", indices / sampling.sample_fps),
        ):
            if data[key].shape != expected.shape or not np.allclose(
                    data[key], expected, rtol=0, atol=1e-8):
                raise ValueError(f"Stale DTW archive: {key}.")
    if not np.allclose(data["local_geodesic_degrees"], np.degrees(costs),
                       rtol=1e-5, atol=1e-5):
        raise ValueError("DTW archive mixes radians and degrees.")


def path_pose_costs(target_poses: np.ndarray, reference_poses: np.ndarray) -> np.ndarray:
    relative = target_poses @ np.swapaxes(reference_poses, -1, -2)
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1) / 2
    return np.arccos(np.clip(cosine, -1, 1)).mean(axis=-1)


def track_frame_quality(track: TrackPoseSequence, sampling: VideoSampling) -> dict[str, np.ndarray]:
    """Dense native-frame masks; missing exact frames remain invalid."""
    frames = track.frame_numbers
    if (not len(frames) or np.any(np.diff(frames) <= 0) or frames[0] < 1
            or frames[-1] > sampling.source_frame_count):
        raise ValueError("Invalid tracking frame range/order.")
    ids = (track.source_track_ids if track.source_track_ids is not None
           else np.full(len(frames), track.track_id))
    status = np.full(sampling.source_frame_count, "missing_tracking", dtype="<U32")
    dense_ids = np.full(sampling.source_frame_count, -1, dtype=np.int64)
    status[frames - 1] = np.where(np.isfinite(track.body_poses).all(axis=(1, 2, 3)),
                                 "valid", "invalid_rotation")
    dense_ids[frames - 1] = ids
    switches = frames[np.flatnonzero(np.diff(ids) != 0) + 1] - 1
    for offset in (-1, 0, 1):
        selected = switches + offset
        selected = selected[(selected >= 0) & (selected < len(status))]
        status[selected] = "near_track_switch"
    return {"status": status, "track_ids": dense_ids, "switch_source_frames": switches}


def nearest_reference_sample(source_frame: int, sampling: VideoSampling,
                             segment_start: int, segment_end: int) -> int:
    """Snap to the nearest actual sampled source frame; ties choose the earlier."""
    if not segment_start <= source_frame <= segment_end:
        raise ValueError("Reference event lies outside its form.")
    eligible = np.flatnonzero((sampling.source_indices >= segment_start)
                             & (sampling.source_indices <= segment_end))
    if not len(eligible):
        raise ValueError("Reference form has no sampled frames.")
    return int(eligible[np.argmin(abs(sampling.source_indices[eligible] - source_frame))])


def map_event(event: dict, path: np.ndarray, costs: np.ndarray,
              reference: VideoSampling, target: VideoSampling, endpoint: dict,
              target_positions: np.ndarray, target_track: TrackPoseSequence,
              reference_quality: dict, target_quality: dict,
              *, review_span_seconds: float = 0.8) -> tuple[dict, list[dict]]:
    """Map intervals by candidate envelope, internal representatives by lower median.

    Known form endpoints keep the existing minimum-local-cost boundary policy.
    Candidates outside the already mapped form are recorded, never hidden.
    """
    if not np.isfinite(review_span_seconds) or review_span_seconds < 0:
        raise ValueError("Invalid diagnostic candidate-span threshold.")
    segment_start = int(endpoint["segment_start_source_frame_0based"])
    segment_end = int(endpoint["segment_end_source_frame_0based"])
    if not 0 <= segment_start <= segment_end < target.source_frame_count:
        raise ValueError("Invalid target form segment.")
    ref_frames = [int(event[field]) for field in POINT_FIELDS.values()]
    if not ref_frames[0] <= ref_frames[1] <= ref_frames[2]:
        raise ValueError("Reference event interval does not contain its representative.")
    anchored = event["representative_source"] == "known_move_end"
    if anchored and (ref_frames[1] != event["reference_segment_end_source_frame_0based"]
                     or ref_frames[2] != ref_frames[1]):
        raise ValueError("Known-end event does not use the reference form endpoint.")
    points, candidate_rows, flags = {}, [], []
    for name, field in POINT_FIELDS.items():
        source = int(event[field])
        ref_sample = nearest_reference_sample(
            source, reference, event["reference_segment_start_source_frame_0based"],
            event["reference_segment_end_source_frame_0based"],
        )
        path_rows = np.flatnonzero(path[:, 1] == ref_sample)
        candidate_indices = path[path_rows, 0]
        candidate_sources = target.source_indices[candidate_indices]
        eligible = (candidate_sources >= segment_start) & (candidate_sources <= segment_end)
        inside = path_rows[eligible]
        selected = None
        policy = {"start": "first_in_form_candidate", "end": "last_in_form_candidate",
                  "representative": "lower_median_in_form_candidate"}[name]
        if anchored and name in ("representative", "end"):
            policy = "known_move_end_minimum_local_cost"
            if int(endpoint["reference_source_frame_0based"]) != source:
                raise ValueError("Cached endpoint references a different form boundary.")
            best = path_rows[np.argmin(costs[path_rows])]
            if (int(path[best, 0]) != endpoint["sample_index_0based"]
                    or int(target.source_indices[path[best, 0]]) != segment_end):
                raise ValueError("Cached endpoint does not match DTW minimum local cost.")
            selected = int(best)
        elif len(inside):
            index = {"start": 0, "end": len(inside) - 1,
                     "representative": (len(inside) - 1) // 2}[name]
            selected = int(inside[index])
        else:
            flags.append(f"{name}_no_candidate_in_form")
        if not eligible.all():
            flags.append(f"{name}_candidates_outside_form")
        span = float((candidate_sources[-1] - candidate_sources[0]) / target.source_fps)
        if span > review_span_seconds + 1e-8:
            flags.append(f"{name}_wide_candidate_span")
        ref_status = str(reference_quality["status"][source])
        grid_source = int(reference.source_indices[ref_sample])
        grid_status = str(reference_quality["status"][grid_source])
        if ref_status != "valid" or grid_status != "valid":
            flags.append(f"{name}_reference_tracking_invalid")
        point = {
            "reference_source_frame_0based": source,
            "reference_source_time_seconds": source / reference.source_fps,
            "reference_sample_index_0based": ref_sample,
            "reference_grid_time_seconds": ref_sample / reference.sample_fps,
            "reference_grid_source_frame_0based": grid_source,
            "reference_quantization_offset_seconds": (grid_source - source) / reference.source_fps,
            "reference_tracking_status": ref_status,
            "reference_grid_tracking_status": grid_status,
            "candidate_count": len(path_rows), "in_form_candidate_count": len(inside),
            "candidate_start_source_frame_0based": int(candidate_sources[0]),
            "candidate_end_source_frame_0based": int(candidate_sources[-1]),
            "candidate_span_seconds": span, "selection_policy": policy,
            "target_sample_index_0based": None, "target_grid_time_seconds": None,
            "target_source_frame_0based": None, "target_phalp_frame_1based": None,
            "target_source_time_seconds": None, "local_geodesic_degrees": None,
            "tracking_actual_phalp_frame_1based": None, "tracking_frame_offset": None,
            "tracking_status": None, "tracking_source_track_id": None,
            "image": None,
        }
        for row_index, in_form in zip(path_rows, eligible):
            target_index = int(path[row_index, 0])
            target_source = int(target.source_indices[target_index])
            actual = int(target_track.frame_numbers[target_positions[target_index]])
            candidate_rows.append({
                "pose_id": event["pose_id"], "point": name,
                "reference_sample_index_0based": ref_sample,
                "target_sample_index_0based": target_index,
                "target_source_frame_0based": target_source,
                "target_source_time_seconds": target_source / target.source_fps,
                "local_geodesic_degrees": float(np.degrees(costs[row_index])),
                "inside_form": bool(in_form), "selected": selected == row_index,
                "tracking_actual_phalp_frame_1based": actual,
                "tracking_frame_offset": actual - target_source - 1,
                "tracking_status": str(target_quality["status"][target_source]),
            })
        if selected is not None:
            target_index = int(path[selected, 0])
            target_source = int(target.source_indices[target_index])
            position = int(target_positions[target_index])
            actual = int(target_track.frame_numbers[position])
            status = str(target_quality["status"][target_source])
            point.update(
                target_sample_index_0based=target_index,
                target_grid_time_seconds=target_index / target.sample_fps,
                target_source_frame_0based=target_source,
                target_phalp_frame_1based=target_source + 1,
                target_source_time_seconds=target_source / target.source_fps,
                local_geodesic_degrees=float(np.degrees(costs[selected])),
                tracking_actual_phalp_frame_1based=actual,
                tracking_frame_offset=actual - target_source - 1,
                tracking_status=status,
                tracking_source_track_id=int(target_quality["track_ids"][target_source]),
            )
            if actual != target_source + 1:
                flags.append(f"{name}_alignment_nearest_substitution")
            if status != "valid":
                flags.append(f"{name}_{status}")
        points[name] = point
    start, rep, end = [points[k]["target_source_frame_0based"] for k in POINT_FIELDS]
    mapped = all(value is not None for value in (start, rep, end))
    valid_ratio = None
    if mapped:
        if not start <= rep <= end:
            flags.append("representative_outside_interval")
            mapped = False
        else:
            statuses = target_quality["status"][start:end + 1]
            valid_ratio = float(np.mean(statuses == "valid"))
            if valid_ratio < 1:
                flags.append("interval_contains_invalid_tracking")
            if start == end:
                flags.append("interval_collapsed_to_single_sample")
    result = {
        "pose_id": event["pose_id"], "event_id": event["event_id"],
        "move_id": event["move_id"], "move_name": event["move_name"],
        "label": event["label"], "technique": event["technique"], "scope": event["scope"],
        "reference_representative_source": event["representative_source"],
        "reference_original_representative_source_frame_0based": event.get(
            "original_query_representative", {}).get("representative_source_frame_0based",
                event["representative_source_frame_0based"]),
        "reference_correction_note": event.get("manual_reference_correction", {}).get("reason", ""),
        "reference_annotation_status": event["annotation_status"],
        "auxiliary_reference_image": event["auxiliary_reference_image"],
        "auxiliary_correspondence": event["auxiliary_correspondence"],
        "status": "mapped" if mapped else "unmapped", "points": points,
        "target_segment_start_source_frame_0based": segment_start,
        "target_segment_end_source_frame_0based": segment_end,
        "interval_valid_tracking_ratio": valid_ratio,
        "diagnostic_flags": sorted(set(flags)), "review_required": bool(flags),
        "manual_review_status": "pending",
        "manual_review_note": "", "manual_review_date": "", "manual_review_scope": "",
    }
    return result, candidate_rows


def flag_event_order(events: list[dict]) -> None:
    """Keep duplicate/nonmonotonic selections visible without adjusting frames."""
    for previous, current in zip(events, events[1:]):
        p = previous["points"]["representative"]["target_source_frame_0based"]
        c = current["points"]["representative"]["target_source_frame_0based"]
        if p is None or c is None or c > p:
            continue
        flag = "duplicate_event_representative" if p == c else "nonmonotonic_event_representative"
        for event in (previous, current):
            event["diagnostic_flags"] = sorted(set(event["diagnostic_flags"] + [flag]))
            event["review_required"] = True


def flatten_event(event: dict) -> dict:
    row = {key: value for key, value in event.items() if key != "points"}
    row["diagnostic_flags"] = ";".join(event["diagnostic_flags"])
    for name, point in event["points"].items():
        row.update({f"{name}_{key}": value for key, value in point.items()})
    return row
