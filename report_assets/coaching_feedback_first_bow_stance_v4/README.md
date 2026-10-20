# Coaching feedback v4: single teacher range

Outputs: `Coaching_Feedback_first_bow_stance_v4.png`, `.svg`, and `.pdf`.
The PNG is 3000 x 1600. SVG/PDF retain vector text and graphics with embedded
original SMPL renders. The earlier v3 outputs are preserved.

Each of the three checkpoints now shows only its frozen P10-P90 teacher
range (the former B range), labeled by the shared `Teacher Ranges` legend.
The A range and A/B labels are removed. Student values are unchanged;
each row uses a linear axis fitted to its retained range and student value.
All pixels outside the Measurement & Structured Evidence panel match v3.

This is a display-only change. It does not change experimental measurements
or the evidence originally supplied to the LLM. Values, range endpoints,
plot coordinates, source hashes, and layout checks are in the audit JSON.

Use `Figure_caption.txt` alongside the figure. Feedback remains the
user-authorized illustrative abridgment of endpoint response F014, not a
new inference for the displayed first event. Detailed provenance is in
`../coaching_feedback_first_bow_stance_v3/README.md`; the dual-range display
description there applies only to v3. In particular, the first-event knee
and wrist values are within the displayed range, so the borrowed advice
must not be presented as a verified diagnosis of these measurements.

## Reproduce

From the project root, choose a new output directory:

```bash
TMPDIR=/home/sqw/tmp MPLCONFIGDIR=/home/sqw/tmp/matplotlib-coaching-v3 \
PYTHONDONTWRITEBYTECODE=1 \
/home/sqw/apps/anaconda3/envs/4d-humans/bin/python \
  report_assets/generate_coaching_feedback_figure.py \
  --compose-only \
  --output-root report_assets/first_bow_stance_figures_v2 \
  --figure-output-root report_assets/coaching_feedback_first_bow_stance_v4_reproduced \
  --figure-name Coaching_Feedback_first_bow_stance_v4
```
