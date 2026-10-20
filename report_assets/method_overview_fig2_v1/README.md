# Fig. 2: Technique-Guided Tai Chi Assessment

## Deliverables

- `Fig2_Method_Overview.png`: 3200 x 1450 pixels, white background.
- `Fig2_Method_Overview.svg`: editable vector labels and shapes, embedded real images.
- `Fig2_Method_Overview.pdf`: vector publication export with embedded fonts.
- `Fig2_Method_Overview.audit.json`: source hashes, plotted values, and layout checks.

The figure follows the approved collection-input, single-example, dual-output design.
Reference demonstrations and student videos are collections. The visible example is
the first left bow-stance event in Parting the Wild Horse's Mane. Inspection repeats
for each student. Evidence branches into per-student language feedback and numerical
aggregation followed by ranking within the student cohort. Ranking does not use the
LLM output as its input.

## Source And Scope

- Reference anchor: BV1WE411W7JB, original video frame0 686.
- Student example: student 10, original video frame0 623.
- Event: `2.part_mane_1_L`.
- Real SMPL PNGs are reused unchanged from `first_bow_stance_figures_v2`; their
  recorded hashes are verified before drawing. Presentation uses the previously
  agreed canonical frontal orientation, shared camera, and light blue-gray material.
- Video thumbnails are decoded at the exact source frames above. Back sheets depict
  collections; they are not additional video observations or repeated teacher samples.
- The three interval illustrations use existing first-event measurements and
  ten-teacher median +/- 2 MAD ranges. Each metric has its own display scale.
  No normative pass/fail threshold, score, or rank is implied. The reference anchor
  is not one of the ten teachers used for the interval statistics.
- The timelines and form divisions are schematic, not to time scale. Joint circles
  identify anatomical regions for illustration; they are not projected joint data.
- Offline specification means LLM-assisted authoring followed by review and reuse,
  not unconstrained automatic query generation during every inference.

## Draft Boundaries

This is the approved high-level method schematic, not a new experiment report.

The shared checkpoint-evidence ranking branch is the newly agreed design and has
not been experimentally validated by this drawing. The old independent pose/pelvis
ranking branches are intentionally absent. Green bars are anonymous illustrations;
they do not show an actual cohort result.

The two short feedback sentences are illustrative abridgments of the previously
authorized borrowed endpoint F014 feedback. They are not a fresh Qwen inference
for the first bow-stance event. A small 'Illustrative feedback' label is retained.
The first-event feedback should be generated and checked before using this panel
as a case-specific experimental result in a final paper.

No tracking, segmentation, localization, ranking, or paid model inference was rerun.
Existing figures, experiment results, and annotations were not overwritten.

## Suggested Caption

Overview of the technique-guided Tai Chi assessment framework. Offline, reviewed
specifications determine when to inspect a movement and which posture checkpoints
to measure. Reconstructed pose sequences are segmented into forms, and semantic
events are localized in a reference and transferred to the other demonstrations and
students. Event-specific measurements are compared with teacher reference
distributions to form structured geometric evidence. This evidence supports
individual coaching feedback and, after numerical aggregation, relative ranking
within a student cohort. One reference-student pair is expanded for illustration;
the inspection procedure is applied to each student.

## Reproduction

From the project root, using the existing `4d-humans` environment:

```bash
python report_assets/generate_method_overview_fig2.py --output-root <new-directory>
```

The generator refuses to overwrite an existing output directory. It reuses the
project's matplotlib vector diagram API, without image-generation repainting of
the SMPL assets. Export checks reject overlapping labels and out-of-bounds text.
