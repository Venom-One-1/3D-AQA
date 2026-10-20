# Quantitative Action Quality Assessment

## Deliverables

- `Quantitative_Assessment_two_branch_v2.png`: paper schematic, 3000 x 1600 pixels.
- `Quantitative_Assessment_two_branch_v2.svg`: vector layout and text, with embedded real SMPL PNGs.
- `Quantitative_Assessment_two_branch_v2.pdf`: vector PDF for paper layout.
- `figure_data.json`: frozen sources, SHA-256 hashes, plotted values, exact frame indices, and render settings.
- `Quantitative_Assessment_two_branch_v2_validation.json`: layout and data-source checks.
- Six transparent 768 x 1024 SMPL renders, three per subject.

The unversioned diagram in this directory is the first draft. The v2 diagram clarifies that both branches consume the shared reference/student input sequences.

## Scope

This figure documents the user's approved simplified **two-branch configuration**. It is not a report of a newly executed experiment. No DTW, cohort ranking, score calibration, optimization, or LLM inference was rerun. Existing experimental outputs and algorithm code were not modified.

The illustration covers the entire segmented *Parting the Wild Horse's Mane* form, not just its first or last bow stance. Unlike the qualitative figures, no semantic key-event localization is used in this quantitative pathway.

Reference pose anchor: `QxVvRcRn2TA`. Student pictured: `10`. The final ranking is across a cohort of students performing the same form. No numeric final ranks or absolute quality scores are shown.

## Pose Branch

Source: `run_teacher_1fps_smpl_dtw_batch.py`, `run_teacher_keyframe_smpl_dtw_batch.py`, and the frozen case `teacher_1fps_smpl_dtw_results/10_2_yemafenzong/`.

1. Select uniformly spaced teacher anchors at nominal 1 FPS, retaining first and last frames. The cached case has 26 anchors.
2. SMPL-DTW uses the full sampled pose sequences; **1 FPS applies to anchor selection, not the DTW sequence rate**.
3. The local cost is the mean SO(3) geodesic distance over 23 local SMPL rotations. The global orientation is excluded.
4. For each teacher anchor, choose the student candidate on the DTW path with minimum **local** geodesic cost, not accumulated DTW cost.
5. Average the resulting distances uniformly across all anchors and 23 joints. Smaller is better.

The figure's heatmap is the actual saved 26 x 23 matched-anchor error array, in degrees. It is **not** a DTW cost matrix. The five illustrated correspondence links are a subset of the saved matching results; they are not a newly estimated path.

No visual-frame-difference anchor selection, old 40/60 mixed pose score, or teacher-calibrated pose score is used in this schematic.

## Pelvis Branch

Sources: `aqa3d/pelvis_quality.py`, `aqa3d/center_fusion.py`, `pelvis_quality_results/teacher_pelvis_model.npz`, and `pelvis_quality_results/clips/10_2_yemafenzong/pelvis_signals.npz`.

- Pelvis position is measured relative to the ankle midpoint and normalized by a fixed clip-level body-size estimate. Height uses the implementation's fixed camera-up direction; lateral/forward axes are body-relative. This is not a true center-of-mass or weight-distribution estimate.
- Use the frozen DTW-aligned profile on a 201-point reference-progress grid. Pelvis profiles have their own existing alignment; the figure does not claim that they reuse the dense pose-branch DTW path.
- The reference profile is the componentwise median of **10 teachers**, not only the pictured pose anchor. The plotted lateral trajectory is real cached data; other signals are summarized in the metric list.
- For this form, combine horizontal-trajectory RMSE, horizontal endpoint error, and vertical-trajectory RMSE with existing internal weights 0.50 / 0.30 / 0.20.
- Existing teacher calibration uses leave-one-teacher-out errors. For each component, `scale = max(1.4826 * MAD, 0.005)`, `z = (error - median) / scale`, and `excess = max(0, z - 1)`. The branch output is the weighted sum of these excesses.

This **pelvis** calibration is retained. The user's request to omit score calibration applies to the old **pose** score. Within-pelvis weights are distinct from the equal weights used between branches.

## Equal-Weight Borda

The rank normalization and tie convention follow `aqa3d/rank_fusion.py`:

- Rank each branch's error in ascending order over the same N students; ties receive average ranks.
- Convert each branch rank r to quality `q = (N - r) / (N - 1)`, for N > 1.
- Fuse with `Q = 0.5 * q_pose + 0.5 * q_pelvis`; order students by descending Q, averaging tied final ranks.

The abnormal-pause branch is excluded entirely. No weights are learned and no human rankings are inputs at inference. The existing historical late-fusion experiment still contains its original configuration; this schematic does not retroactively change it or claim an SRCC for the newly approved configuration.

## Real SMPL Assets

Display anchor orders (zero based): 0, 12, 25.

| Subject | Clip-relative frame0 values | PHALP frame numbers |
| --- | --- | --- |
| QxVvRcRn2TA | 0, 360, 749 | 1, 361, 750 |
| Student 10 | 0, 348, 718 | 1, 349, 719 |

These are frame indices within the existing **segmented clips**, not the original full videos. Track ID 1 is loaded explicitly for each subject, matching the frozen pose summary. Missing exact frames are not replaced by nearby frames.

Rendering retains tracking body poses and shape parameters, uses identity root orientation for a canonical front view, and applies a common orthographic camera, canvas size, and rest-height display normalization. These presentation transforms do not modify any measured signal. No image-generation model was used to create the meshes or measurements.

## Reproduce

From the repository root, using the existing `4d-humans` environment:

```bash
PYOPENGL_PLATFORM=egl python report_assets/generate_quantitative_assessment_figure.py \
  --output report_assets/quantitative_assessment_two_branch_new \
  --name Quantitative_Assessment_two_branch_v2
```

To change only the layout, reuse rendered assets and select a new output name:

```bash
python report_assets/generate_quantitative_assessment_figure.py \
  --output report_assets/quantitative_assessment_two_branch_v1 \
  --compose-only --name Quantitative_Assessment_two_branch_next
```

Existing images are not overwritten. The native diagram helper checks all text for clipping and overlap; the asset helper checks transparent alpha and foreground margins. Source hashes are checked before composition.
