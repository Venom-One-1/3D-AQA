#!/usr/bin/env python
"""Compare first-three endpoint geometry for five students and ten teachers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from aqa3d.endpoint_metrics import (
    LABELS, MOVE_METRICS, aggregate_window, compare_interval, compute_endpoint_metrics,
    endpoint_window, rules_for_manifest, teacher_interval,
)
from aqa3d.endpoint_transfer import manifest_boundaries, resolve_tracking_samples
from aqa3d.smpl_dtw import (
    ReferenceFrameMatch, backtrack_dtw_path, dtw_from_cost_matrix, inspect_video_sampling,
    pairwise_geodesic_costs,
)
from aqa3d.smpl_pose_segmentation import build_mapped_boundaries
from aqa3d.tracking import load_stitched_primary_track, load_stitched_root_track
from run_smpl_pose_dtw_segmentation import save_dtw_path
from run_student_endpoint_keyposes import write_json
from visualize_tas_boundary_frames import read_video_frames


PROJECT = Path(__file__).resolve().parent
TEACHERS = (
    "QxVvRcRn2TA", "BV1iE411c7Ni_p03", "BV1tk4y1r7Yr_p27", "an5qNCspzUw", "i8kMrJmAfjU",
    "BV1vShdzLEsn_p1", "420p2gYFTa", "BV1MFZKYnEdM_pNA", "BV1svYmz6EFB_pNA", "Bg3kJjFReAQ",
)
STUDENTS = ("01", "02", "03", "04", "10")


def path_geodesic(target, reference):
    relative = target @ np.swapaxes(reference, -1, -2)
    return np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1)-1)/2, -1, 1)).mean(axis=-1)


def cached_alignment(cache_dir, video, tracking_path, sampling, track, reference, ref_sampling,
                     ref_track, ref_positions):
    """Reuse a full path only after checking identities, timing and all path costs."""
    summary_path = cache_dir / "summary.json"
    if not summary_path.is_file() or not (cache_dir / "dtw_path.npz").is_file():
        return None
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "ok":
        return None
    for key, expected in (("input_video", video), ("input_tracking", tracking_path),
                          ("reference_video", Path(reference["video_path"])),
                          ("reference_tracking", Path(reference["tracking_path"]))):
        if key not in summary or Path(summary[key]).resolve() != expected.resolve():
            raise ValueError(f"Stale alignment identity {cache_dir}: {key}")
    for key, expected in (("sample_fps", 5.), ("input_source_fps", sampling.source_fps),
                          ("input_source_frame_count", sampling.source_frame_count),
                          ("reference_source_fps", ref_sampling.source_fps)):
        if not np.isclose(summary[key], expected, rtol=0, atol=1e-8):
            raise ValueError(f"Stale alignment timing {cache_dir}: {key}")
    if summary.get("dtw_coefficient", 1.) != 1.:
        raise ValueError("Cached DTW must use coefficient=1.")
    gap = float(summary.get("max_tracking_gap_seconds", 0.))
    positions = resolve_tracking_samples(track, sampling, gap)
    with np.load(cache_dir / "dtw_path.npz") as data:
        path = np.column_stack((data["target_sample_indices_0based"], data["reference_sample_indices_0based"]))
        saved_costs = data["local_geodesic_radians"].copy()
        if (not np.array_equal(data["target_source_frame_indices_0based"], sampling.source_indices[path[:, 0]])
                or not np.array_equal(data["reference_source_frame_indices_0based"], ref_sampling.source_indices[path[:, 1]])):
            raise ValueError("Cached path has inconsistent source indices.")
    if not np.array_equal(path[0], [0, 0]) or not np.array_equal(path[-1], [sampling.sample_count-1, ref_sampling.sample_count-1]):
        raise ValueError("Cached path does not span both full videos.")
    steps = np.diff(path, axis=0)
    if not np.all((steps >= 0) & (steps <= 1)) or np.any(steps.sum(axis=1) == 0):
        raise ValueError("Invalid cached DTW path steps.")
    costs = path_geodesic(track.body_poses[positions[path[:, 0]]],
                         ref_track.body_poses[ref_positions[path[:, 1]]])
    if not np.allclose(costs, saved_costs, rtol=1e-5, atol=1e-5):
        raise ValueError("Cached path costs do not match the current tracking poses.")
    return path, costs, positions, {"cache_source": str(cache_dir), "cache_path_costs_verified": True,
                                    "alignment_tracking_gap_seconds": gap}


def align_subject(subject_id, video, tracking_path, role, directory, args,
                  manifest, gold, ref_sampling, ref_track, ref_positions):
    sampling = inspect_video_sampling(video, 5.)
    track = load_stitched_primary_track(tracking_path)
    cached = (args.teacher_alignment_root if role == "teacher" else args.student_alignment_root) / subject_id
    alignment = cached_alignment(cached, video, tracking_path, sampling, track, manifest["reference"],
                                 ref_sampling, ref_track, ref_positions)
    if alignment is None:
        positions = resolve_tracking_samples(track, sampling, args.max_tracking_gap_seconds)
        print(f"[{role}/{subject_id}] computing full DTW ({sampling.sample_count} x {ref_sampling.sample_count})", flush=True)
        costs = pairwise_geodesic_costs(track.body_poses[positions], ref_track.body_poses[ref_positions])
        _, _, accumulated = dtw_from_cost_matrix(costs)
        path = backtrack_dtw_path(accumulated)
        save_dtw_path(directory, path, costs, ref_sampling, sampling)
        path_costs = costs[path[:, 0], path[:, 1]]
        provenance = {"cache_source": None, "alignment_tracking_gap_seconds": args.max_tracking_gap_seconds}
    else:
        path, path_costs, positions, provenance = alignment
        shutil.copy2(cached / "dtw_path.npz", directory / "dtw_path.npz")
    matches = []
    for b in gold[:3]:
        selected = np.flatnonzero(path[:, 1] == b.sample_index_0based)
        best = selected[np.argmin(path_costs[selected])]
        matches.append(ReferenceFrameMatch(b.sample_index_0based, int(path[best, 0]),
                                          len(selected), float(path_costs[best])))
    boundaries = build_mapped_boundaries(subject_id, gold[:3], matches, ref_sampling, sampling)
    events = []
    previous = -1
    for b in boundaries:
        original = manifest["endpoint_keyposes"][b.move_id-1]
        event = {k: original[k] for k in ("move_id", "move_name_pinyin", "move_name_zh", "pose_id", "final_technique_step")}
        source = b.target_source_frame_0based
        pos = positions[b.target_sample_index_0based]
        event.update(center_source_frame_0based=source, segment_start_source_frame_0based=previous+1,
            segment_end_source_frame_0based=source, sample_index_0based=b.target_sample_index_0based,
            boundary_time_seconds=b.target_end_time_seconds,
            actual_phalp_frame_1based=int(track.frame_numbers[pos]),
            actual_track_id=int(track.source_track_ids[pos]),
            local_geodesic_degrees=b.local_geodesic_degrees, candidate_count=b.candidate_count,
            source_frame_time_seconds=source/sampling.source_fps,
            reference_source_frame_0based=b.reference_source_frame_0based,
            reference_boundary_time_seconds=b.reference_end_time_seconds)
        events.append(event)
        previous = source
    write_json(directory / "alignment.json", {"subject_id": subject_id, "role": role,
        "input_video": str(video), "input_tracking": str(tracking_path),
        "manifest_sha256": args.manifest_hash, "source_fps": sampling.source_fps,
        "manual_review_status": "pending",
        "substituted_sample_count": int(np.count_nonzero(track.frame_numbers[positions] != sampling.source_indices+1)),
        **provenance, "endpoints": events})
    return sampling, track, events


def measure_subject(subject_id, role, video, tracking_path, sampling, primary, events, directory, args):
    root = load_stitched_root_track(tracking_path)
    primary_index = {int(f): i for i, f in enumerate(primary.frame_numbers)}
    root_index = {int(f): i for i, f in enumerate(root.frame_numbers)}
    switches = primary.frame_numbers[np.flatnonzero(np.diff(primary.source_track_ids) != 0)+1]
    raw_rows, summaries = [], []
    frames = read_video_frames(video, [e["center_source_frame_0based"] for e in events])
    for event in events:
        mid = event["move_id"]; center = event["center_source_frame_0based"]
        requested = endpoint_window(center, event["segment_start_source_frame_0based"],
            event["segment_end_source_frame_0based"], sampling.source_fps, args.window_seconds)
        joints = np.full((len(requested), 24, 3), np.nan)
        statuses, available, indices = [], [], []
        ids = []
        for i, frame in enumerate(requested):
            phalp = int(frame)+1
            pi, ri = primary_index.get(phalp), root_index.get(phalp)
            ids.append(int(primary.source_track_ids[pi]) if pi is not None else None)
            if pi is None or ri is None:
                statuses.append("missing_tracking")
            elif (root.source_track_ids[ri] != primary.source_track_ids[pi]
                    or not np.allclose(root.body_poses[ri], primary.body_poses[pi], atol=1e-7, rtol=0)):
                statuses.append("tracking_selection_mismatch")
            elif np.any(abs(switches-phalp) <= 1):
                statuses.append("near_track_switch")
            else:
                statuses.append("valid"); available.append(int(frame)); indices.append(i)
        if available:
            native = root.at_source_frames(available).to_smpl24_joints(device=args.device)
            joints[indices] = native
        metrics = compute_endpoint_metrics(joints, requested)
        image_path = directory / f"{mid:02d}_endpoint.jpg"
        frames[center].save(image_path, quality=95)
        np.savez_compressed(directory / f"{mid:02d}_window_joints.npz", source_frames_0based=requested,
                            native_smpl24_joints=joints, tracking_status=np.asarray(statuses))
        for metric in MOVE_METRICS[mid]:
            selected = metrics[metrics.metric_id == metric].copy()
            for i, status in enumerate(statuses):
                if status != "valid":
                    selected.iloc[i, selected.columns.get_loc("value")] = np.nan
                    selected.iloc[i, selected.columns.get_loc("status")] = status
            agg = aggregate_window(selected.value, selected.status, requested, center, args.minimum_window_valid_ratio)
            common = {"role": role, "subject_id": subject_id, "move_id": mid,
                "move_name_pinyin": event["move_name_pinyin"], "move_name_zh": event["move_name_zh"],
                "pose_id": event["pose_id"], "metric_id": metric, "metric_label_zh": LABELS[metric],
                "unit": selected.unit.iloc[0], "center_source_frame_0based": center,
                "center_phalp_frame_1based": center+1, "boundary_time_seconds": event["boundary_time_seconds"],
                "source_frame_time_seconds": center/sampling.source_fps,
                "window_start_source_frame_0based": int(requested[0]),
                "window_end_source_frame_0based": int(requested[-1]),
                "local_geodesic_degrees": event["local_geodesic_degrees"],
                "image": str(image_path.absolute()), "final_technique_step": event["final_technique_step"]}
            summaries.append({**common, **agg})
            for index, (_, row) in enumerate(selected.iterrows()):
                raw_rows.append({"role": role, "subject_id": subject_id, "move_id": mid,
                    "pose_id": event["pose_id"], "source_frame_0based": int(row.frame_id),
                    "phalp_frame_1based": int(row.frame_id)+1, "source_time_seconds": row.frame_id/sampling.source_fps,
                    "track_id": ids[index], "is_center_frame": row.frame_id == center,
                    "metric_id": metric, "value": row.value, "unit": row.unit, "status": row.status})
    pd.DataFrame(raw_rows).to_csv(directory / "frame_metrics.csv", index=False)
    pd.DataFrame(summaries).to_csv(directory / "endpoint_metrics.csv", index=False)
    return summaries


def compare_all(all_rows, args):
    lookup = {(r["role"], r["subject_id"], r["move_id"], r["metric_id"]): r for r in all_rows}
    references, comparisons = [], []
    for mid, metrics in MOVE_METRICS.items():
        for metric in metrics:
            raw = [lookup[("teacher", tid, mid, metric)] for tid in TEACHERS]
            stats = teacher_interval([r["value"] for r in raw], args.minimum_teachers)
            record = {"move_id": mid, "pose_id": f"{mid}.end", "metric_id": metric,
                "label_zh": LABELS[metric], "unit": raw[0]["unit"], **stats,
                "teacher_values": [{k: r[k] for k in ("subject_id", "value", "status", "center_value",
                    "window_median", "valid_frame_count", "requested_frame_count", "image", "boundary_time_seconds")} for r in raw]}
            references.append(record)
            for sid in args.students:
                student = lookup[("student", sid, mid, metric)]
                result = dict(student)
                result.update({"reference_valid_teacher_count": stats["valid_teacher_count"],
                    "reference_status": stats["status"], "teacher_median": stats["median"],
                    "teacher_mad": stats["mad"], "teacher_p10": stats["p10"], "teacher_p90": stats["p90"],
                    "teacher_median_minus_2mad": stats["median_minus_2mad"],
                    "teacher_median_plus_2mad": stats["median_plus_2mad"],
                    "student_minus_teacher_median": student["value"]-stats["median"] if student["value"] is not None and stats["median"] is not None else None})
                for method in ("median_2mad", "p10_p90"):
                    status, deviation = compare_interval(student["value"], stats, method)
                    result[method+"_status"] = status; result[method+"_outside_distance"] = deviation
                result["interval_methods_agree"] = result["median_2mad_status"] == result["p10_p90_status"]
                comparisons.append(result)
    return references, comparisons


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-manifest", type=Path, default=PROJECT/"reference_data/BV1WE411W7JB/reference_manifest.json")
    p.add_argument("--data-root", type=Path, default=Path("/home/sqw/VisualSearch/aqa"))
    p.add_argument("--teacher-alignment-root", type=Path, default=PROJECT/"smpl_pose_dtw_segmentation_results")
    p.add_argument("--student-alignment-root", type=Path, default=PROJECT/"student_keypose_results")
    p.add_argument("--output-root", type=Path, default=PROJECT/"endpoint_metric_results/first3_five_students")
    p.add_argument("--students", nargs="+", default=list(STUDENTS))
    p.add_argument("--window-seconds", type=float, default=0.2)
    p.add_argument("--minimum-window-valid-ratio", type=float, default=0.7)
    p.add_argument("--minimum-teachers", type=int, default=7)
    p.add_argument("--max-tracking-gap-seconds", type=float, default=0.)
    p.add_argument("--device", default="cpu")
    return p


def main():
    args = build_parser().parse_args()
    if not math.isfinite(args.window_seconds) or args.window_seconds < 0:
        raise ValueError("Window must be finite and nonnegative.")
    if not 0 < args.minimum_window_valid_ratio <= 1 or not 1 <= args.minimum_teachers <= 10:
        raise ValueError("Invalid minimum valid-data thresholds.")
    if len(set(args.students)) != len(args.students) or any(not x.isdigit() for x in args.students):
        raise ValueError("Student IDs must be unique numeric filenames.")
    out = args.output_root.absolute()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh output-root to preserve prior runs.")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    raw_manifest = args.reference_manifest.read_bytes(); manifest = json.loads(raw_manifest)
    args.manifest_hash = hashlib.sha256(raw_manifest).hexdigest()
    rules = rules_for_manifest(manifest)
    write_json(out/"endpoint_metric_rules_first3.json", rules)
    write_json(out/"reference_manifest_snapshot.json", manifest)
    reference = manifest["reference"]
    ref_sampling = inspect_video_sampling(reference["video_path"], 5.)
    gold = manifest_boundaries(manifest, ref_sampling)
    ref_track = load_stitched_primary_track(reference["tracking_path"])
    ref_positions = resolve_tracking_samples(ref_track, ref_sampling)
    all_rows, failures = [], []
    subjects = [("teacher", tid, args.data_root/"teach_trimmed"/(tid+".mp4"),
                 args.data_root/"Tracking/teach_trimmed"/tid/"results"/("demo_"+tid+".pkl")) for tid in TEACHERS]
    subjects += [("student", sid, args.data_root/"student"/(sid+".mp4"),
                  args.data_root/"Tracking/student_full"/sid/"results"/("demo_"+sid+".pkl")) for sid in args.students]
    for role, sid, video, tracking in subjects:
        begin = time.perf_counter(); directory = out/role/sid; directory.mkdir(parents=True)
        print(f"[{role}/{sid}] loading alignment and endpoint windows", flush=True)
        try:
            sampling, track, events = align_subject(sid, video, tracking, role, directory, args,
                manifest, gold, ref_sampling, ref_track, ref_positions)
            rows = measure_subject(sid, role, video, tracking, sampling, track, events, directory, args)
            all_rows.extend(rows)
            print(f"[{role}/{sid}] {sum(r['status']=='valid' for r in rows)}/{len(rows)} valid metrics; {time.perf_counter()-begin:.1f}s", flush=True)
        except Exception as exc:
            failures.append({"role": role, "subject_id": sid, "error": repr(exc)})
            write_json(directory/"failure.json", failures[-1])
            print(f"[{role}/{sid}] failed: {exc}", flush=True)
            # Failed teachers retain explicit null values in every reference record.
            for mid, metrics in MOVE_METRICS.items():
                event = manifest["endpoint_keyposes"][mid-1]
                for metric in metrics:
                    all_rows.append({"role": role, "subject_id": sid, "move_id": mid, "metric_id": metric,
                        "pose_id": event["pose_id"], "move_name_zh": event["move_name_zh"],
                        "move_name_pinyin": event["move_name_pinyin"], "metric_label_zh": LABELS[metric],
                        "final_technique_step": event["final_technique_step"], "unit": rules["metric_definitions"][metric]["unit"],
                        "value": None, "status": "subject_failed", "center_value": None,
                        "window_median": None, "valid_frame_count": 0, "requested_frame_count": 0,
                        "image": "", "boundary_time_seconds": None})
    references, comparisons = compare_all(all_rows, args)
    write_json(out/"teacher_reference_values.json", references)
    pd.DataFrame([{k: v for k,v in r.items() if k != "teacher_values"} for r in references]).to_csv(out/"teacher_reference_ranges.csv", index=False)
    pd.DataFrame(all_rows).to_csv(out/"all_endpoint_metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out/"student_teacher_comparison.csv", index=False)
    from plot_endpoint_metrics import export_report
    export_report(out, all_rows, references, comparisons, rules, args.students)
    write_json(out/"summary.json", {"status": "partial" if failures else "ok", "failures": failures,
        "reference_manifest_sha256": args.manifest_hash, "teacher_ids": list(TEACHERS),
        "student_ids": args.students, "move_ids": [1,2,3], "comparison_count": len(comparisons),
        "valid_teacher_metric_count": sum(r['role']=='teacher' and r['status']=='valid' for r in all_rows),
        "valid_student_metric_count": sum(r['role']=='student' and r['status']=='valid' for r in all_rows),
        "window_seconds_each_side": args.window_seconds, "window_clipping": "within_current_move",
        "minimum_window_valid_ratio": args.minimum_window_valid_ratio, "minimum_valid_frames": 3,
        "minimum_teachers": args.minimum_teachers, "default_interval": "median_2mad",
        "secondary_interval": "p10_p90", "mad_scaled": False,
        "reference_ranges_manual_review": "pending", "elapsed_seconds": time.perf_counter()-started})
    print(f"Completed -> {out}; failures={len(failures)}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
