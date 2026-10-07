"""Numerical probes for generated queries; never label synthetic poses as action errors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from aqa3d.pose_measurements import JOINTS, MIDPOINTS, MeasurementQuery, measure_pose


def dependencies(query: MeasurementQuery) -> set[str]:
    names = set()
    for point in query.points:
        names.update(MIDPOINTS.get(point, (point,)))
    if query.normalizer == "torso_length" or query.coordinate_frame == "torso_v1":
        names.update(("Pelvis", "Neck"))
    if query.normalizer == "shoulder_width" or query.coordinate_frame == "torso_v1":
        names.update(("L_Shoulder", "R_Shoulder"))
    if query.coordinate_frame == "torso_v1":
        names.update(("L_Hip", "R_Hip"))
    if query.normalizer == "average_leg_length":
        names.update(s+"_"+joint for s in ("L", "R") for joint in ("Hip", "Knee", "Ankle"))
    return names


def probe_query(joints: np.ndarray, query: MeasurementQuery) -> list[dict]:
    base = measure_pose(joints, query)
    p = np.asarray(joints, dtype=float)
    if p.ndim == 2:
        p = p[None]
    rows = []

    def check(name, changed, expected, changed_points):
        result = measure_pose(changed, query)
        valid = (base.statuses == "valid") & (result.statuses == "valid")
        count = int(valid.sum())
        error = float(np.max(np.abs(result.values[valid] - base.values[valid] - expected[valid]))) if count else None
        matching_status = bool(np.array_equal(base.statuses, result.statuses))
        passed = error is not None and error < 1e-7 and matching_status
        rows.append({"query_id": query.query_id, "probe": name, "checked_frames": count,
                     "changed_points": changed_points, "max_abs_error": error,
                     "status_mask_unchanged": matching_status,
                     "status": "pass" if passed else ("not_testable" if not count else "fail")})

    rotation, _ = np.linalg.qr(np.array([[1., 2, 3], [4, 6, 9], [1, 4, 8]]))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    check("rigid_transform_and_uniform_scale", 1.7*p@rotation.T+[2, -3, 4],
          np.zeros_like(base.values), "all")
    unrelated = sorted(set(JOINTS) - dependencies(query))
    if unrelated:
        changed = p.copy()
        changed[:, [JOINTS[n] for n in unrelated]] += [.13, -.07, .11]
        check("unrelated_points", changed, np.zeros_like(base.values), unrelated)
    if (query.operation == "relative_position" and query.normalizer == "torso_length"
            and query.points[0] in ("L_Wrist", "R_Wrist")
            and query.points[1] not in ("L_Wrist", "R_Wrist", "WristMidpoint")):
        # 0.1*(Neck-Pelvis) is exactly 0.1 torso-length along the torso up axis.
        changed = p.copy()
        changed[:, JOINTS[query.points[0]]] += .1*(p[:, JOINTS["Neck"]] - p[:, JOINTS["Pelvis"]])
        expected = np.zeros_like(base.values)
        expected[:, 1] = .1
        check("wrist_up_expected_delta_0_1", changed, expected, [query.points[0]])
    if (query.operation == "joint_angle" and len(set(query.points)) == 3
            and all(point in JOINTS for point in query.points)):
        fixture = np.zeros((24, 3))
        fixture[JOINTS[query.points[0]]] = [1, 0, 0]
        fixture[JOINTS[query.points[2]]] = [0, 1, 0]
        value = measure_pose(fixture, query).values[0, 0]
        rows.append({"query_id": query.query_id, "probe": "synthetic_right_angle_90_degrees",
                     "checked_frames": 1, "changed_points": list(query.points),
                     "max_abs_error": float(abs(value-90)), "status_mask_unchanged": None,
                     "status": "pass" if np.isclose(value, 90) else "fail"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis_root.resolve()
    target = root/"geometry_probes.json"
    if target.exists():
        parser.error("Existing probe results will not be overwritten")
    manifest = json.loads((root.parent/"experiment_manifest.json").read_text())
    records, hashes = [], {}

    def remember(path):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    for plan_path in sorted(root.glob("M*_R*/plan.json")):
        remember(plan_path)
        plan = json.loads(plan_path.read_text())
        cache = Path(manifest["input_root"])/"student/01"/f"{plan['move_id']:02d}_window_joints.npz"
        remember(cache)
        with np.load(cache, allow_pickle=False) as data:
            joints = data["native_smpl24_joints"]
        for definition in plan["queries"]:
            records.extend({"case_id": plan_path.parent.name, **row}
                           for row in probe_query(joints, MeasurementQuery.from_dict(definition)))
    for path, checksum in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != checksum:
            raise ValueError("Probe source changed during execution")
    summary = {"checks": len(records), "passed": sum(r["status"] == "pass" for r in records),
               "failed": sum(r["status"] == "fail" for r in records),
               "not_testable": sum(r["status"] == "not_testable" for r in records)}
    report = {"scope": "Numerical behavior only, not semantic or action-quality validation",
              "baseline": "student/01 frozen endpoint joint windows; no API calls; no source modification",
              "tracking_note": "Only finite geometry is tested here; not tracking accuracy or confidence",
              "synthetic_note": "Perturbations need not be anatomically realizable and are not error labels",
              "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_sha256": hashes, "summary": summary, "records": records}
    with target.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(summary))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
