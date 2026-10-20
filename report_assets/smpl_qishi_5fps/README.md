# Qishi SMPL Figure Assets

Actual SMPL meshes reconstructed from the existing PHALP tracking parameters.
No generative image model, new tracking inference, DTW rerun, or video overlay.

## Outputs

| Subject | PNG count | Source FPS | Closed frame0 interval | Sampling grid |
| --- | ---: | ---: | --- | --- |
| BV1WE411W7JB | 81 | 30.0 | 0..480 | 0.0..16.0 s |
| 10 | 76 | 29.969808062886592 | 0..449 | 0.0..15.0 s |

Each subject directory contains:

- `qishi_XXXXX_tTTT.TTT_fFFFFFF.png`: 768 x 1024 RGBA render, transparent background.
- `frames.csv` and `frames.json`: filenames, sampled-grid timestamps, actual source timestamps, source frame0, PHALP frame1, selected track ID, presentation transforms, and validation data.
- `contact_sheet.jpg`: white-background overview of every rendered sample for selecting figure assets. This overview is not transparent; the individual PNGs are.

The root `render_manifest.json` records input hashes, model hash, camera, material, presentation conventions, and export completion status.

## Timing

Sampling uses the repository's existing `uniform_sample_source_indices`:
`frame0 = floor(sample_index * source_fps / 5)`.
PHALP frame numbers are `frame0 + 1`. No missing-frame substitution is used.
Source-frame boundaries are inclusive. Both known endpoints are included.

For student 10, the final 15.0 s grid point selects frame0 449, whose actual
source timestamp is 14.981744263 s. Both time conventions are retained rather
than labeling the grid timestamp as an exact source timestamp.

## Presentation Conventions

- Identical orthographic camera for both subjects and every frame: azimuth 35 degrees, elevation 8 degrees; fixed canvas and framing.
- Original framewise SMPL body rotations, global rotations, and shape coefficients are used.
- Each clip receives one constant yaw correction based on its initial second, so the initial body heading is comparable. Later turns and pitch/roll are retained, not independently corrected in each frame.
- One uniform scale per clip matches a canonical rest height of 1.7 display units, using that clip's median-shape rest mesh. Original body proportions are retained. There is no per-frame fit-to-canvas zoom.
- Meshes are centered horizontally on the pelvis and their lowest vertex is placed at display Y=0. This is a presentation alignment, not a recovered ground plane or measured world trajectory.
- No camera translation, original room/background, shadows on a floor, labels, or timestamps are baked into the individual PNGs.
- Light blue-gray smooth material, three directional lights and 2x supersampling.
- SMPL reconstruction can contain errors. Standard SMPL hand geometry does not establish accurate finger articulation or palm orientation.

## Reproduce

Use the existing `4d-humans` environment with SMPL model files and a working
EGL offscreen context. The exporter refuses to overwrite an existing directory.

```bash
cd /home/sqw/Projects/3D-AQA
/home/sqw/apps/anaconda3/envs/4d-humans/bin/python export_qishi_smpl_renders.py \
  --output-root /home/sqw/Projects/3D-AQA/report_assets/smpl_qishi_5fps_new
```

For six representative previews, add `--preview-only` and use another new output
directory. `--azimuth`, `--elevation`, `--width`, `--height` and `--supersample`
control rendering without altering the tracked joint parameters.

The exporter validates every PNG for alpha transparency, nonempty foreground,
canvas margins, and non-flat lighting. Focused unit tests cover sampling,
endpoint inclusion, constant heading transforms, shape-preserving scaling,
shared camera framing, and alpha validation.
