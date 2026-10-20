# Quantitative Assessment: Simplified v3

## Deliverables

- `Quantitative_Assessment_two_branch_v3.png`: 3000 x 1600 paper figure.
- `Quantitative_Assessment_two_branch_v3.svg`: editable vector text and shapes; real SMPL images and heatmap are embedded.
- `Quantitative_Assessment_two_branch_v3.pdf`: publication PDF.
- `Quantitative_Assessment_two_branch_v3_validation.json`: layout, source-hash and provenance checks.
- `Figure_caption.txt`: suggested caption and separate preparation note.

The figure retains the blue pose branch and green pelvis branch. Their colors continue through the two branch ranks and the 50% / 50% fusion block. Teacher references are gold; the illustrated student is orange. Video IDs, frame details, calibration equations, implementation notes and rank formulas are omitted from the figure, not from the method.

## Scope and Provenance

This is a layout revision of the frozen v2 figure, not a new experiment. No SMPL reconstruction, DTW, calibration, cohort ranking, model inference or parameter tuning was rerun. The abnormal-pause branch remains excluded. Existing experiments and earlier figures are unchanged.

The example covers the complete segmented Parting the Wild Horse's Mane form. It does not use the first-bow-stance key event to select pose anchors. The two branches process each student; ranking is over the same cohort performing the same form. Anonymous bars do not encode actual student identities, distances, scores or ranks.

Frozen input directory:
`/home/sqw/Projects/3D-AQA/report_assets/quantitative_assessment_two_branch_v1/`

Its `figure_data.json` contains source paths, hashes, plotted arrays and mesh metadata. All eleven source hashes and six rendered-mesh hashes are checked when composing v3. Source artifacts are reused in place, without edits.

- Pose reference and pictured teacher: `QxVvRcRn2TA`.
- Pictured student: `10`.
- Reference clip-relative frames: 0, 360, 749.
- Student clip-relative frames: 0, 348, 718.
- The original transparent SMPL renders retain the common canonical front view and scale.
- Pelvis reference: the pointwise median of ten teachers, not only the pictured pose reference.
- Pelvis plot: the real `support_lateral` profile at 201 aligned progress points, using a common vertical scale for both curves.

## Details Moved Out of the Figure

### Pose Branch

Uniform reference-anchor sampling is nominally 1 FPS and includes the first and last frames. The cached example has 26 anchors. This does not downsample the DTW input to 1 FPS: alignment uses the full sampled pose sequences. For each anchor, the student frame is the candidate with minimum local geodesic cost along the DTW path. The five displayed correspondence links are a subset of those saved matches.

The heatmap is the actual 26 x 23 matched-anchor joint-error array, not a DTW cost matrix. Its rows are reference anchors and columns are local SMPL joints. The pose deviation is the uniform mean of SO(3) geodesic distances across all anchors and 23 local joints, excluding global orientation. Smaller deviation gives a better branch rank. No old mixed pose score or pose-score calibration is included.

### Pelvis Branch

Pelvis displacement is measured relative to the ankle midpoint and normalized by the existing clip-level body-size estimate. The diagram's three-axis icon is conceptual: lateral and forward directions are body-relative; height follows the implementation's fixed up direction. This is not world-space motion, true center of mass, or a weight-distribution estimate.

The cached pelvis signals use their own existing DTW-based progress alignment; the figure does not imply reuse of the dense pose-branch path. For this form, horizontal-trajectory RMSE, horizontal endpoint error and vertical-trajectory RMSE are combined using internal weights 0.50 / 0.30 / 0.20 after teacher calibration. The short label "Endpoint deviation" refers to the horizontal endpoint error.

Teacher calibration uses leave-one-teacher-out errors. For each component, `scale = max(1.4826 * MAD, 0.005)`, `z = (error - median) / scale`, and `excess = max(0, z - 1)`. The branch output is the weighted sum of these excesses. Pelvis calibration remains part of the method; omitting its explanation from the figure does not remove it. Within-pelvis weights are distinct from the equal weights between branches.

### Ranking

Rank each branch's deviation in ascending order over the same N students. Ties receive average ranks. With `N > 1`, convert rank r to `q = (N - r) / (N - 1)` and fuse `Q = 0.5 * q_pose + 0.5 * q_pelvis`. Sort by descending Q for the final relative ranking, averaging tied final ranks. This is not an absolute action-quality score.

The illustrated simplified configuration has not been rerun as a combined evaluation. Historical experiment configurations and SRCC results must not be attributed to it without a corresponding run.

## Reproduce

Use a new output directory or filename; existing figure files are never overwritten.

```bash
TMPDIR=/home/sqw/tmp PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/home/sqw/tmp/quantitative-v3-mpl \
  /home/sqw/apps/anaconda3/envs/4d-humans/bin/python \
  /home/sqw/Projects/3D-AQA/report_assets/generate_quantitative_assessment_figure.py \
  --compose-only \
  --output /home/sqw/Projects/3D-AQA/report_assets/quantitative_assessment_two_branch_v1 \
  --figure-output-root /home/sqw/tmp/quantitative-v3-reproduction \
  --name Quantitative_Assessment_two_branch_v3
```

The pre-revision generator is backed up at:
`/home/sqw/tmp/generate_quantitative_assessment_figure_before_v3.py`.
