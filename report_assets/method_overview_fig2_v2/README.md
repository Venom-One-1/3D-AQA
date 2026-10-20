# Fig. 2, Revision 2

Files: `Fig2_Method_Overview.png` (3200 x 1450), editable `.svg`, vector `.pdf`,
and `Fig2_Method_Overview.audit.json` with source hashes and layout verification.

## Approved Visual Changes

The revised figure preserves the original composition, typography, round corners,
white internal panels, teacher/student colors, and genuine tracked SMPL inserts.

The supplied one-page palette PDF informed the warm gold and muted lavender.
Tints and darker title colors were adapted for readable white titles and visual
consistency with the existing KEL and coaching figures.

| Region | Background | Title | Outline |
| --- | --- | --- | --- |
| Technique-Guided Inspection | `#E4EFFF` | `#3974D7` | `#7599CC` |
| Geometric Evidence | `#FFF5DC` | `#A47C27` | `#D4AE53` |
| Coaching Feedback | `#F4ECF4` | `#8E6195` | `#C9A1CA` |
| Relative Ranking | `#E5F5DF` | `#15931C` | `#69A653` |

Offline specification grows from 1060 x 180 to 1060 x 220. The coaching LLM panel
grows from 290 x 80 to 460 x 180. Both contain matching vector chip/network icons
and explicit LLM labels. The advice bubble shrinks from 550 x 234 to 490 x 156;
its wording and font size are retained. Arrows are rerouted to the resized panels.

The following six labels, marked by the user, are removed from the artwork:

- LLM-assisted, reviewed
- Reference anchor shown
- Student 10 shown
- Matched semantic events
- Illustrative feedback
- Shared geometric evidence for individual coaching and cohort-level ranking

Other gray labels, including 3D Pose, Form segmentation, Technique + measurements,
Teacher range, and Student value, remain visible.

## Source And Interpretation

The source and illustrative status follow `method_overview_fig2_v1/README.md`.
The pictured first left bow-stance event is `2.part_mane_1_L`: reference
BV1WE411W7JB frame0 686, student 10 frame0 623. The three interval plots use the
same frozen first-event measurements and ten-teacher median +/- 2 MAD ranges;
the pictured anchor is not part of these ten-teacher statistics.

The feedback sentences remain illustrative abridgments of the previously
authorized borrowed endpoint F014 response. Removing the in-figure annotation
does not change their provenance: they are not a fresh first-event model result.
Anonymous ranking bars remain schematic. The unified checkpoint-based ranking
design has not been experimentally validated by this figure revision.

Offline specifications remain authored and reviewed offline. The two icons
communicate LLM participation in specification authoring and feedback generation;
they do not imply automatic online query generation.

No model calls or assessment experiments are made by this revision. Revision 1
images, data, and its original drawing script remain available.

## Reproduction

```bash
python report_assets/generate_method_overview_fig2_v2.py --output-root <new-directory>
```

Use the existing `4d-humans` environment. The generator refuses to overwrite an
existing output directory and checks asset hashes, text width, canvas bounds,
and text overlap before exporting all three formats.
