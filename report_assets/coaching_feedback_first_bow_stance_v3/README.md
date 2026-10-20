# Coaching feedback: simplified three-checkpoint figure

Outputs: `Coaching_Feedback_first_bow_stance_v3.png`, `.svg`, and `.pdf`.
PNG dimensions are 3000 x 1600. SVG/PDF retain vector text and graphics;
the two original transparent SMPL renders are embedded as images.

`Figure_caption.txt` contains the accompanying English caption, including
the provenance of the illustrative coaching text. Keep that provenance
when using the figure in a paper or presentation.

## Color correspondence

| Color | Metric | Technique cue | Feedback excerpt |
| --- | --- | --- | --- |
| `#786092` | Left knee angle | Bend the left knee into a bow stance | Deepen the front-knee bend |
| `#E4523F` | Normalized stance length | Step forward into a wide stance | Lengthen your stance |
| `#1670E8` | Left wrist height relative to head | Raise the left hand to eye level | Insufficient evidence; no height correction |

Only these three representative checkpoints are displayed. The full eight
checkpoints and the underlying experimental pipeline are unchanged. The
right-arm finding and the right-arm portion of the summary are omitted.

The two teacher ranges are displayed for each metric: A is median +/- 2 MAD,
B is P10-P90. Both use the same linear axis within each metric, with an
independent axis domain for different metrics. Marker positions come from
the frozen window medians. Numeric ranges, values, drawing coordinates,
and source hashes are recorded in the audit JSON.

## Evidence and feedback provenance

Assets and measurements come from `../first_bow_stance_figures_v2/`.
Reference: BV1WE411W7JB, frame0 686. Student: 10, frame0 623.
The full source hashes and the two exact mesh-image hashes are verified
before drawing. No reconstruction, event detection, DTW, or LLM call is rerun.

The user authorized borrowing endpoint response F014 while the first-event
Qwen run is pending. The two retained findings are abridged translations
of F014 P1 and P3; the summary retains only its bow-stance recommendation.
They are not diagnoses newly generated from the first-event measurements.
In particular, the first-event left knee and wrist values lie within both
displayed teacher ranges. The deeper-knee recommendation belongs to the
borrowed endpoint response and must not be presented as a verified inference
from this figure's current numbers.

The wrist measurement is relative to the SMPL head joint, normalized by
torso length. It is a proxy for the verbal hand-at-eye-level requirement,
not an eye landmark. Stance length uses the feedback module's normalized
stance metric, not the KEL ankle-separation signal.

## Reproduce

From the project root, choose a new output directory:

```bash
TMPDIR=/home/sqw/tmp MPLCONFIGDIR=/home/sqw/tmp/matplotlib-coaching-v3 \
PYTHONDONTWRITEBYTECODE=1 \
/home/sqw/apps/anaconda3/envs/4d-humans/bin/python \
  report_assets/generate_coaching_feedback_figure.py \
  --compose-only \
  --output-root report_assets/first_bow_stance_figures_v2 \
  --figure-output-root report_assets/coaching_feedback_first_bow_stance_v3_reproduced \
  --figure-name Coaching_Feedback_first_bow_stance_v3
```

The generator refuses to overwrite existing output directories and images.
The earlier figures are retained in their original versioned directories.
