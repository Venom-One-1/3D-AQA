# Generic Pose Measurements v1

This module measures geometry. It does not judge action correctness, generate
coaching advice, or call an LLM. Legacy `angle_metrics.py`, endpoint metrics,
reference intervals, F-group responses and human reviews are unchanged.

## Contract

`aqa3d.pose_measurements` is NumPy-only. Inputs are native SMPL-24 joint positions
in the registered order, `(24,3)` or `(T,24,3)`, with original **0-based source
frame indices**. PHALP indices are source indices + 1. Use consistent joint
definitions and geometry units for students and teachers.

Four operations are exposed through `MeasurementQuery` and `measure_pose`:

| Operation | Ordered points | Output |
| --- | --- | --- |
| `joint_angle` | first, vertex, third | Unsigned included angle, degrees, 180 = straight |
| `vector_angle` | start1, end1, start2, end2 | Unsigned 3D angle between directed vectors |
| `relative_position` | target, origin | Signed right/up/forward components divided by scale |
| `distance` | first, second | Euclidean distance divided by scale |

Angles are not clinical flexion or axial-rotation measurements. Do not call the
angle between shoulder and hip lines a signed axial twist. Raw rotation matrices,
ground clearance, gravity alignment, palm orientation, muscle relaxation and
actual weight bearing are not provided by this interface.

Allowed derived points are `HipMidpoint`, `ShoulderMidpoint`, `AnkleMidpoint` and
`WristMidpoint`. No arbitrary expressions or aliases are executed. `Head` is a
head-joint proxy, not the eye or forehead; `Wrist` is not palm center and `Ankle`
is not ground contact. Even SMPL Hand/Foot nodes do not establish palm/toe pose.

### Frame and scale

Relative positions require the explicitly named `torso_v1` frame:

- Up is normalized Neck minus Pelvis, **not gravity**.
- Left is the projected combination of 0.7 normalized hip-left and 0.3 normalized
  shoulder-left. Right is its negative; forward is cross(left, up).
- Nonfinite/zero axes or hip/shoulder agreement below cosine 0.5 invalidate the
  relative measurement. No fallback silently changes the frame definition.
- Frame follows the torso; it cannot detect whole-body tilt relative to gravity.

Length queries require an explicit normalizer: `torso_length`, `shoulder_width`,
or `average_leg_length`. All are measured per frame, independently for each
subject. This is a geometric ratio, not physical stature or a pure body-shape
estimate; posture changes can affect the denominator. Freeze each query's scale
before evaluating subjects. The same query (including scale) runs for every
teacher/student. Do not switch scales to obtain a desired conclusion.

### Validity and aggregation

Invalid geometry produces NaN internally, null in exported JSON, and a reason.
Unknown operations/points/frames or malformed requests raise `ValueError`.
Provided tracking exclusions take precedence and are never interpolated away.
`valid` means computable, **not confidence in the reconstruction**.

Window summaries retain the exact center and source frame IDs. The frozen
experiment uses a 0.2 s half-window clipped inside the move, so an endpoint's
window typically contains only preceding frames and the center. Require at
least 3 valid frames and 70% validity by default. Vector components share a
validity mask; their component-wise medians are not necessarily one real pose.

Teacher output preserves all original sample values and statuses (including
nulls), P10/P90, median and **unscaled MAD**. Fewer than 7 valid teachers is
flagged by default. Zero-width intervals are flagged, not expanded. No teacher
is removed for being unusual; no interval is declared a technical pass/fail rule.

## Frozen experiment

Run from the project using the existing `4d-humans` environment:

```bash
python run_pose_measurements.py
python run_pose_measurements.py --students 01 --moves 3 --output-root pose_measurement_results/01_move3_v1
python run_pose_measurements.py --queries my_queries.json --output-root pose_measurement_results/custom_v1
```

The default input is `endpoint_metric_results/first3_five_students`; output is
`pose_measurement_results/first3_five_students_v1`. Existing nonempty output
directories are refused. Cached NPZ joints, tracking exclusions, alignment JSON,
reference manifest, FPS, boundaries and windows are reused **without rerunning
tracking/DTW**. Every source file read is hashed and checked again at the end.

The default 19 queries are identical across all three moves and serve only as
diagnostic examples, not an automatic LLM-selected plan or complete coverage of
the movement requirements. A custom query file has this form:

```json
{
  "version": "1.0",
  "queries": [
    {"query_id": "right_elbow", "operation": "joint_angle", "points": ["R_Shoulder", "R_Elbow", "R_Wrist"]},
    {"query_id": "elbow_relative_wrist", "operation": "relative_position", "points": ["R_Elbow", "R_Wrist"], "normalizer": "torso_length", "coordinate_frame": "torso_v1"}
  ]
}
```

Results include `measurement_contract.json`, `queries.json`, a reference manifest
snapshot, subject `measurements.json` files with raw per-frame values, 15 student
case JSONs with matching teacher samples, `measurement_summary.csv`, teacher
references/ranges, `legacy_compatibility.csv`, `provenance.json` and `summary.json`.
No below/above verdicts, diagnoses, scores, ranking or new feedback are generated.

Compatibility checks cover unchanged definitions only, including old per-frame
and aggregated values (absolute error <= 1e-6). Legacy stance absolute values
are taken **before** temporal aggregation. New signed vectors are not mislabeled
as old absolute stance widths. Legacy pelvis-coordinate "height" and shoulder/
hip "twist" are not promoted into equivalent physical measurements.

## Verification

```bash
python -m unittest discover -s tests -p 'test_pose_measurements.py' -v
python -m unittest discover -s tests -v
```

Tests cover analytic angles/directions, derived points, rigid-transform and scale
invariance, missing/zero-length geometry, unrelated-point independence, tracking
exclusions, exact frame windows, null teachers, strict query validation, source
preservation, and legacy-compatible exports. Perturbation tests check geometry,
not whether a modified posture constitutes a Tai Chi technical error.
