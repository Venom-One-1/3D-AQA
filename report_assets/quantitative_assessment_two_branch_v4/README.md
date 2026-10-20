# Quantitative Assessment v4: Palette Alignment

This revision changes presentation only. The v3 text, source arrays, SMPL meshes, matching links, fusion weights and method are retained. Previous figures are not overwritten.

## Palette

- Pose branch: unchanged blue, including Pose Rank and its 50% weight.
- Pelvis branch: fill `#F4ECF4`, edge `#C9A1CA`, accent `#8E6195`. This includes the branch header, arrows, direction icon, component bullets, Weighted Sum, Pelvis Rank and its 50% weight.
- Final Relative Ranking: fill `#E5F5DF`, edge `#69A653`, title `#15931C`; anonymous icons and bars use `#338F44`, `#68B477`, `#A6D1A5`. These colors match Fig2's ranking panel.
- Borda fusion background remains neutral. The input/output arrows retain their source module colors.
- Reference stays gold `#CBAB29`; student stays orange `#EF741D`, including both real pelvis curves.

The final ranking area now has the same pale-green panel and deep-green title treatment as Fig2. Its bars are illustrative only, not computed cohort ranks or absolute quality scores.

## Files and Provenance

- `Quantitative_Assessment_two_branch_v4.png`: 3000 x 1600 figure.
- Matching `.svg` and `.pdf`: vector layout and text with embedded real assets.
- `Quantitative_Assessment_two_branch_v4_validation.json`: layout, source hashes and palette.
- `visual_data_verification.json`: comparison against v3 and checks of the new palette.
- `Figure_caption.txt`: caption updated to describe blue/purple branches.

Frozen assets remain in `/home/sqw/Projects/3D-AQA/report_assets/quantitative_assessment_two_branch_v1/`. Full method/provenance details are in `/home/sqw/Projects/3D-AQA/report_assets/quantitative_assessment_two_branch_v3/README.md`.

No reconstruction, DTW, calibration, ranking or model inference was rerun. The figure is the approved two-branch schematic, not a newly validated combined experiment. The pose reference is QxVvRcRn2TA; the illustrated student is 10. The gold pelvis profile is the pointwise median of ten teachers.

## Reproduce

```bash
TMPDIR=/home/sqw/tmp PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/home/sqw/tmp/quantitative-v4-mpl \
  /home/sqw/apps/anaconda3/envs/4d-humans/bin/python \
  /home/sqw/Projects/3D-AQA/report_assets/generate_quantitative_assessment_figure.py \
  --compose-only \
  --output /home/sqw/Projects/3D-AQA/report_assets/quantitative_assessment_two_branch_v1 \
  --figure-output-root /home/sqw/tmp/quantitative-v4-reproduction \
  --name Quantitative_Assessment_two_branch_v4
```

The preceding generator is backed up at `/home/sqw/tmp/generate_quantitative_assessment_figure_before_v4.py`.
