#!/usr/bin/env python
"""Export neutral pose evidence from frozen endpoint joint caches; no tracking, DTW or LLM."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from aqa3d.pose_measurements import (
    VERSION, MeasurementQuery, MeasurementResult, measure_pose, measurement_contract,
    reference_statistics, summarize_window,
)


PROJECT = Path(__file__).resolve().parent


def example_queries() -> list[MeasurementQuery]:
    """A shared diagnostic query set, NOT a move-specific prescription or LLM plan."""
    queries = []
    for side in ("L", "R"):
        for name, points in (("knee", ("Hip", "Knee", "Ankle")),
                             ("elbow", ("Shoulder", "Elbow", "Wrist"))):
            queries.append(MeasurementQuery(f"{side}_{name}", "joint_angle", tuple(f"{side}_{p}" for p in points)))
        for origin in ("Pelvis", "Head", f"{side}_Hip", f"{side}_Wrist"):
            target = f"{side}_Elbow" if origin == f"{side}_Wrist" else f"{side}_Wrist"
            queries.append(MeasurementQuery(f"{target}_relative_{origin}", "relative_position",
                                             (target, origin), "torso_length", "torso_v1"))
        queries.append(MeasurementQuery(f"{side}_wrist_hip_distance", "distance",
                                         (f"{side}_Wrist", f"{side}_Hip"), "torso_length"))
        queries.append(MeasurementQuery(f"{side}_upper_arm_trunk", "vector_angle",
                                         (f"{side}_Shoulder", f"{side}_Elbow", "Pelvis", "Neck")))
    for scale in ("shoulder_width", "average_leg_length"):
        queries.append(MeasurementQuery(f"ankles_{scale}", "relative_position",
                                         ("L_Ankle", "R_Ankle"), scale, "torso_v1"))
    queries.append(MeasurementQuery("wrists_to_neck", "distance", ("WristMidpoint", "Neck"), "torso_length"))
    return queries


def load_queries(payload: Any) -> list[MeasurementQuery]:
    if not isinstance(payload, dict) or set(payload) != {"version", "queries"} or payload["version"] != VERSION:
        raise ValueError("Query file must contain version='1.0' and queries")
    if not isinstance(payload["queries"], list) or not 1 <= len(payload["queries"]) <= 128:
        raise ValueError("Provide 1 to 128 queries")
    queries = [MeasurementQuery.from_dict(item) for item in payload["queries"]]
    if len({q.query_id for q in queries}) != len(queries):
        raise ValueError("Duplicate query IDs")
    return queries


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


class FrozenInputs:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.hashes: dict[str, str] = {}

    def read(self, relative: str) -> bytes:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Source path escapes input root")
        value = path.read_bytes()
        digest = hashlib.sha256(value).hexdigest()
        if relative in self.hashes and self.hashes[relative] != digest:
            raise ValueError(f"Input changed while reading: {relative}")
        self.hashes[relative] = digest
        return value

    def json(self, relative: str) -> Any:
        return json.loads(self.read(relative))

    def csv(self, relative: str) -> list[dict[str, str]]:
        return list(csv.DictReader(io.StringIO(self.read(relative).decode("utf-8-sig"))))

    def verify(self) -> None:
        for relative in list(self.hashes):
            self.read(relative)


def _ids(values: Any, name: str) -> list[str]:
    if (not isinstance(values, list) or not values or len(set(values)) != len(values)
            or any(not isinstance(v, str) or not v or not all(c.isalnum() or c in "_-" for c in v) for v in values)):
        raise ValueError(f"Invalid {name}")
    return values


def load_window(inputs: FrozenInputs, prefix: str, event: dict[str, Any], fps: float,
                half_window: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = ("center_source_frame_0based", "segment_start_source_frame_0based", "segment_end_source_frame_0based")
    if any(type(event[n]) is not int for n in names):
        raise ValueError("Source frame indices must be integers")
    center, start, end = (event[n] for n in names)
    if not 0 <= start <= center <= end or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid frozen endpoint bounds or FPS")
    radius = int(np.floor(half_window * fps + 1e-8))
    expected = np.arange(max(start, center-radius), min(end, center+radius)+1)
    with np.load(io.BytesIO(inputs.read(f"{prefix}/{event['move_id']:02d}_window_joints.npz")), allow_pickle=False) as cache:
        frames = cache["source_frames_0based"].copy()
        joints = cache["native_smpl24_joints"].copy()
        status = cache["tracking_status"].copy()
    if frames.dtype.kind not in "iu" or not np.array_equal(frames, expected):
        raise ValueError("Cached window disagrees with frozen bounds/FPS; no silent reindexing")
    if joints.shape != (len(frames), 24, 3) or status.shape != (len(frames),):
        raise ValueError("Invalid joint cache shape")
    if not np.isclose(event["source_frame_time_seconds"], center/fps, atol=1e-8, rtol=0):
        raise ValueError("Source frame time does not match index/FPS")
    return joints, frames, status


def legacy_mapping() -> dict[str, tuple[MeasurementQuery, int, bool]]:
    queries = {q.query_id: q for q in example_queries()}
    mapping = {}
    for name, side in (("left", "L"), ("right", "R")):
        for joint in ("knee", "elbow"):
            mapping[f"{name}_{joint}_angle"] = (queries[f"{side}_{joint}"], 0, False)
        for origin in ("Pelvis", "Head"):
            for axis, index in (("up", 1), ("forward", 2)):
                mapping[f"{name}_wrist_{origin.lower()}_{axis}_torso_ratio"] = (
                    queries[f"{side}_Wrist_relative_{origin}"], index, False)
        mapping[f"{name}_wrist_hip_distance_torso_ratio"] = (queries[f"{side}_wrist_hip_distance"], 0, False)
    mapping["right_wrist_head_right_torso_ratio"] = (queries["R_Wrist_relative_Head"], 0, False)
    mapping["stance_width_shoulder_ratio"] = (queries["ankles_shoulder_width"], 0, True)
    mapping["stance_length_leg_ratio"] = (queries["ankles_average_leg_length"], 2, True)
    return mapping


def check_legacy(
    joints: np.ndarray, frames: np.ndarray, tracking: np.ndarray, center: int,
    old_rows: list[dict[str, str]], old_frames: list[dict[str, str]], thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    """Check only unchanged definitions, with abs applied per frame for legacy stance metrics."""
    mapping, records = legacy_mapping(), []
    for old in old_rows:
        metric = old["metric_id"]
        if metric not in mapping:
            records.append({"metric_id": metric, "status": "not_comparable", "max_abs_error": None})
            continue
        query, component, absolute = mapping[metric]
        result = measure_pose(joints, query, frame_ids=frames, tracking_status=tracking)
        values = result.values[:, component:component+1]
        if absolute:
            values = abs(values)
        # A scalar view only for compatibility aggregation; not a new measurement definition.
        scalar = MeasurementQuery("compatibility", "distance", ("Pelvis", "Neck"), "torso_length")
        stats = summarize_window(MeasurementResult(scalar, frames, values, result.statuses), center, **thresholds)
        now = {**stats, **stats["components"]["value"]}
        errors, ok = [], old["status"] == now["status"]

        def compare(a: str, b: float | int | None) -> None:
            nonlocal ok
            previous = float(a) if a else np.nan
            if b is None or not np.isfinite(b):
                ok &= not np.isfinite(previous)
            elif not np.isfinite(previous):
                ok = False
            else:
                errors.append(abs(previous-b))
                ok &= abs(previous-b) <= 1e-6

        for key in ("center_value", "window_median", "value", "window_min", "window_max",
                    "valid_frame_count", "requested_frame_count"):
            compare(old[key], now[key])
        old_series = {int(r["source_frame_0based"]): r for r in old_frames if r["metric_id"] == metric}
        ok &= set(old_series) == set(frames)
        for i, frame in enumerate(frames):
            if int(frame) in old_series:
                saved = old_series[int(frame)]
                ok &= (saved["status"] == "valid") == (result.statuses[i] == "valid")
                compare(saved["value"], float(values[i, 0]))
        records.append({"metric_id": metric, "status": "pass" if ok else "fail",
                        "max_abs_error": max(errors) if errors else None})
    return records


def export_experiment(input_root: Path, output_root: Path, *, queries_path: Path | None = None,
                      students: list[str] | None = None, moves: list[int] | None = None) -> dict[str, Any]:
    inputs = FrozenInputs(input_root)
    output_root = output_root.resolve()
    if output_root.is_relative_to(inputs.root) or inputs.root.is_relative_to(output_root):
        raise ValueError("Output must be separate from frozen input data")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("Use an empty output directory; previous results are never overwritten")
    config = inputs.json("summary.json")
    if config["status"] != "ok":
        raise ValueError("Requires a complete frozen endpoint experiment")
    teacher_ids = _ids(config["teacher_ids"], "teacher IDs")
    all_students = _ids(config["student_ids"], "student IDs")
    students = _ids(students if students is not None else all_students, "selected students")
    moves = config["move_ids"] if moves is None else moves
    if not set(students) <= set(all_students) or not moves or len(set(moves)) != len(moves):
        raise ValueError("Unknown students or duplicate/empty moves")
    if any(type(mid) is not int or mid not in config["move_ids"] for mid in moves):
        raise ValueError("Unknown move ID")
    half_window = config["window_seconds_each_side"]
    if not np.isfinite(half_window) or half_window < 0 or config["window_clipping"] != "within_current_move":
        raise ValueError("Invalid frozen window settings")
    thresholds = {"minimum_valid_ratio": config["minimum_window_valid_ratio"],
                  "minimum_valid_frames": config["minimum_valid_frames"]}
    manifest = inputs.json("reference_manifest_snapshot.json")
    manifest_hash = inputs.hashes["reference_manifest_snapshot.json"]
    if manifest_hash != config["reference_manifest_sha256"]:
        raise ValueError("Frozen reference manifest checksum mismatch")
    query_raw = queries_path.read_bytes() if queries_path is not None else None
    queries = load_queries(json.loads(query_raw)) if query_raw is not None else example_queries()
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root/"summary.json", {"status": "in_progress", "version": VERSION})
    write_json(output_root/"measurement_contract.json", measurement_contract())
    write_json(output_root/"queries.json", {"version": VERSION, "queries": [asdict(q) for q in queries]})
    write_json(output_root/"reference_manifest_snapshot.json", manifest)
    measured, events, rows, compatibility = {}, {}, [], []
    for role, ids in (("teacher", teacher_ids), ("student", students)):
        for sid in ids:
            prefix = f"{role}/{sid}"
            alignment = inputs.json(prefix+"/alignment.json")
            if alignment["subject_id"] != sid or alignment["role"] != role or alignment["manifest_sha256"] != manifest_hash:
                raise ValueError(f"Frozen subject identity mismatch: {prefix}")
            listed = alignment["endpoints"]
            if len({e["move_id"] for e in listed}) != len(listed):
                raise ValueError("Duplicate endpoint move IDs")
            event_map = {e["move_id"]: e for e in listed}
            old_rows = inputs.csv(prefix+"/endpoint_metrics.csv")
            old_frames = inputs.csv(prefix+"/frame_metrics.csv")
            bundle = {"version": VERSION, "role": role, "subject_id": sid,
                      "source_alignment": alignment, "endpoints": []}
            for mid in moves:
                event = event_map[mid]
                joints, frames, tracking = load_window(inputs, prefix, event, alignment["source_fps"], half_window)
                center = event["center_source_frame_0based"]
                endpoint = {"event": event, "measurements": []}
                for query in queries:
                    result = measure_pose(joints, query, frame_ids=frames, tracking_status=tracking)
                    stats = summarize_window(result, center, **thresholds)
                    evidence = {"definition": query.definition(), **stats,
                                "frame_values": [[float(v) if np.isfinite(v) else None for v in row] for row in result.values]}
                    measured[(role, sid, mid, query.query_id)] = evidence
                    endpoint["measurements"].append(evidence)
                    for component, numbers in stats["components"].items():
                        rows.append({"role": role, "subject_id": sid, "move_id": mid,
                                     "query_id": query.query_id, "component": component,
                                     "unit": query.definition()["unit"], "status": stats["status"], **numbers,
                                     "valid_frame_count": stats["valid_frame_count"],
                                     "requested_frame_count": stats["requested_frame_count"],
                                     "center_source_frame_0based": center, "center_phalp_frame_1based": center+1,
                                     "boundary_time_seconds": event["boundary_time_seconds"],
                                     "source_frame_time_seconds": event["source_frame_time_seconds"]})
                events[(role, sid, mid)] = event
                bundle["endpoints"].append(endpoint)
                for check in check_legacy(joints, frames, tracking, center,
                        [r for r in old_rows if int(r["move_id"]) == mid],
                        [r for r in old_frames if int(r["move_id"]) == mid], thresholds):
                    compatibility.append({"role": role, "subject_id": sid, "move_id": mid, **check})
            directory = output_root/role/sid
            directory.mkdir(parents=True)
            write_json(directory/"measurements.json", bundle)
            print(f"{prefix}: {len(moves)} frozen endpoints measured", flush=True)
    references = {}
    reference_rows = []
    for mid in moves:
        references[str(mid)] = {}
        for query in queries:
            component_refs = {}
            for component in measured[("teacher", teacher_ids[0], mid, query.query_id)]["components"]:
                samples = []
                for tid in teacher_ids:
                    ev = measured[("teacher", tid, mid, query.query_id)]
                    samples.append({"subject_id": tid, "status": ev["status"],
                                    **ev["components"][component], "valid_frame_count": ev["valid_frame_count"],
                                    "requested_frame_count": ev["requested_frame_count"],
                                    "endpoint": events[("teacher", tid, mid)]})
                ref = reference_statistics(samples, config["minimum_teachers"])
                component_refs[component] = ref
                reference_rows.append({"move_id": mid, "query_id": query.query_id, "component": component,
                                       **{k: v for k, v in ref.items() if k != "teacher_values"}})
            references[str(mid)][query.query_id] = {"definition": query.definition(), "components": component_refs}
    cases_dir = output_root/"cases"
    cases_dir.mkdir()
    for sid in students:
        for mid in moves:
            event = events[("student", sid, mid)]
            write_json(cases_dir/f"{sid}_{mid:02d}.json", {
                "version": VERSION, "student_id": sid, "move_id": mid,
                "scope": "endpoint_geometry_only; no diagnosis, feedback or quality ranking",
                "technique": event["final_technique_step"], "student_endpoint": event,
                "reference_anchor": next(e for e in manifest["endpoint_keyposes"] if e["move_id"] == mid),
                "measurements": [{"query_id": q.query_id,
                                  "student": measured[("student", sid, mid, q.query_id)],
                                  "teachers": references[str(mid)][q.query_id]} for q in queries]})
    write_json(output_root/"teacher_references.json", references)
    write_csv(output_root/"measurement_summary.csv", rows)
    write_csv(output_root/"teacher_reference_ranges.csv", reference_rows)
    write_csv(output_root/"legacy_compatibility.csv", compatibility)
    inputs.verify()
    write_json(output_root/"provenance.json", {
        "input_root": str(inputs.root), "source_sha256": inputs.hashes,
        "queries_sha256": hashlib.sha256(query_raw).hexdigest() if query_raw is not None else None,
        "implementation_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                  (Path(__file__), PROJECT/"aqa3d/pose_measurements.py")},
        "source_files_unchanged_at_end": True, "frozen_config": config,
        "query_origin": "user_file" if query_raw is not None else "diagnostic_examples_not_llm_selected"})
    failed = sum(r["status"] == "fail" for r in compatibility)
    summary = {"status": "compatibility_failed" if failed else "ok", "version": VERSION,
               "student_ids": students, "teacher_ids": teacher_ids, "move_ids": moves,
               "case_count": len(students)*len(moves), "query_count": len(queries),
               "measurement_component_count": len(rows), "legacy_checks": len(compatibility),
               "legacy_passed": sum(r["status"] == "pass" for r in compatibility),
               "legacy_failed": failed,
               "legacy_not_comparable": sum(r["status"] == "not_comparable" for r in compatibility),
               "legacy_max_abs_error": max((r["max_abs_error"] or 0 for r in compatibility), default=0),
               "tracking_rerun": False, "dtw_rerun": False, "llm_calls": 0}
    write_json(output_root/"summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=PROJECT/"endpoint_metric_results/first3_five_students")
    parser.add_argument("--output-root", type=Path, default=PROJECT/"pose_measurement_results/first3_five_students_v1")
    parser.add_argument("--queries", type=Path, help="Versioned JSON query list; omit for diagnostic examples")
    parser.add_argument("--students", nargs="+")
    parser.add_argument("--moves", nargs="+", type=int)
    args = parser.parse_args()
    summary = export_experiment(args.input_root, args.output_root, queries_path=args.queries,
                                students=args.students, moves=args.moves)
    print(json.dumps(summary, indent=2))
    if summary["legacy_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
