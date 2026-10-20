# Qishi SMPL Renders: Canonical Front View

Real SMPL meshes reconstructed from existing tracking, not AI-generated poses.
Reference: `BV1WE411W7JB`. Student: `10`. Form: Qishi (form 1).

## Contents

| Directory | Samples | Inclusive source frame0 bounds |
| --- | ---: | --- |
| `BV1WE411W7JB/` | 81 | 0 to 480 |
| `10/` | 76 | 0 to 449 |

Each subject directory contains transparent PNGs, `contact_sheet.jpg`,
`frames.csv`, and `frames.json`. The root `render_manifest.json` records the
shared camera, material, source hashes, model hash, and exporter hash.

## Rendering Convention

- Each frame's root `global_orient` is the identity 3x3 rotation matrix, not a
  zero matrix. The original per-frame local `body_pose` and `betas` are retained.
- Camera azimuth and elevation are both zero. Native identity-root SMPL model
  coordinates use Y up and Z forward; no camera-coordinate flip or initial
  heading correction is applied in this mode.
- The orthographic camera and 768 x 1024 canvas are shared by both subjects and
  every frame. Light blue-gray meshes are rendered without video overlays.
- One fixed scale per subject normalizes the median-shape rest height to 1.7.
  There is no per-frame zoom. The pelvis is centered horizontally/depth-wise,
  and the lowest mesh vertex is placed at display Y=0 for each frame. This is
  presentation placement, not estimated ground contact or measured height.
- PNG backgrounds are transparent. Contact sheets use white for inspection.

These are canonicalized local-pose illustrations. They intentionally remove
original global heading, root tilt, and root translation, so they must not be
used to judge those properties of the original performance. Root normalization
does not remove local torso or head rotations from `body_pose`.

## Sampling and Filenames

Sampling uses the existing global 5 FPS grid and frozen form boundaries:
`source_frame0 = floor(sample_index * source_fps / 5)` within closed bounds.
PHALP frame numbers are `source_frame0 + 1`. Every sample has exact tracking;
no nearest-frame substitution is used.

For example, `qishi_00060_t012.000_f000360.png` identifies sample 60, grid time
12.000 seconds, and source frame0 360. Grid timestamps and actual source-frame
timestamps are stored separately in both indexes. Student 10 has a non-integer
source FPS: its last sample is at grid time 15.000 seconds but actual source
time approximately 14.981744 seconds.

## Reproduction

Run in the existing 4d-humans environment with SMPL weights and EGL available:

```bash
cd /home/sqw/Projects/3D-AQA
python export_qishi_smpl_renders.py --view-mode canonical-front \
  --output-root /home/sqw/Projects/3D-AQA/report_assets/smpl_qishi_5fps_front_new
```

The destination must not already exist. The previous oblique-view export in
`report_assets/smpl_qishi_5fps/` is preserved. No source tracking, boundaries,
reference manifest, or evaluation results are modified by this export.
