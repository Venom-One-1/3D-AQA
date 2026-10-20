# Key Event Localization v8

Publication figure for the first left bow-stance event in Parting the Wild
Horse's Mane. Outputs are `KEL_first_bow_stance_v8.png`, `.svg`, and `.pdf`.
The PNG is 3000 x 1600; SVG/PDF retain vector text, lines, and editable shapes
with embedded SMPL renders and DTW heatmap.

## Display changes

- Remove the explanations, source identifiers, student number, timestamps,
  and frame indices marked for deletion in the reviewed v7 image.
- Retain event bands, representative dots, and interval-to-interval DTW mapping.
- Show three query cues with matching colors across technique phrases, query
  metrics, conditions, curves, and legend text:

| Color | Technique phrase | Metric | Displayed condition |
| --- | --- | --- | --- |
| `#1670E8` | left hand up | Left wrist height | High wrist (local maximum) |
| `#E4523F` | Step forward | Ankle separation | Wide step (local maximum) |
| `#786092` | bend the left knee | Left knee angle | Bent knee (local minimum) |

## Frozen evidence

Assets come from `../first_bow_stance_figures_v2/`. Their hashes are verified
before composition. No SMPL reconstruction, event localization, DTW, or LLM
inference is rerun. The full query still includes its original additional
conditions and ordering constraints; this figure displays only the three
plotted cues. The three signals are independently min-max scaled over the
visible 16-32 second interval for plotting, not for changing detection.

The reference is BV1WE411W7JB, event interval 21.466667-24.500000 seconds,
representative frame0 686. The target is student 10, mapped interval
19.386177-22.389199 seconds, representative frame0 623.

Event-transfer timelines are schematic and independently laid out, not to
scale. Their drawn width difference does not report an actual duration ratio.
The three dashed arrows map the interval start, representative frame, and
interval end, respectively. Source details are retained in the audit JSON.

## Reproduce

Run from the project root using the existing 4d-humans environment. Choose
a new destination directory; existing output directories are not overwritten.

```bash
TMPDIR=/home/sqw/tmp MPLCONFIGDIR=/home/sqw/tmp/matplotlib-kel-v8 \
PYTHONDONTWRITEBYTECODE=1 \
/home/sqw/apps/anaconda3/envs/4d-humans/bin/python \
  report_assets/generate_first_bow_stance_figures.py \
  --compose-only --kel-only \
  --output-root report_assets/first_bow_stance_figures_v2 \
  --figure-output-root report_assets/key_event_localization_v8_reproduced \
  --suffix _v8
```
