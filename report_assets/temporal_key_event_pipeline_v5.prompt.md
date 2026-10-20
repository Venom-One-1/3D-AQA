# Temporal Key Event Pipeline v5

- Mode: built-in image_gen image editing.
- Base layout: v3-derived initial warping edit, matching the user's annotated screenshot; not the widened-interval v4 correction.
- Edit target: /home/sqw/.codex/generated_images/01a11ea2-2c78-7f91-b0f4-67cecd8861ff/exec-b64b9129-c92e-4ed8-9681-9c84dcb9b8e2.png
- Final output: /home/sqw/.codex/generated_images/01a11ea2-2c78-7f91-b0f4-67cecd8861ff/exec-82207dee-152f-4df4-9034-c6f986f9961e.png
- Project output: report_assets/temporal_key_event_pipeline_v5.png
- Change: redirect only the leftmost dashed correspondence arrow to the existing Target event interval's left boundary, retaining the preceding unshaded time axis and the original highlight extent.
- Previous versions retained. Timeline geometry, pose illustrations and metric curves are schematic, not measurements.

## Edit Prompt

Use case: precise-object-edit.
Input: the provided unannotated academic pipeline diagram, derived from v3. This is the exact layout to retain.
Make ONLY ONE tiny change: redirect the LEFTMOST BLACK DASHED CORRESPONDENCE ARROW inside the Event Transfer module in the lower panel (b).

Currently this arrow slopes down-left from the upper gold highlighted interval's left boundary near (1082,688) toward the beginning of the entire lower orange time axis near (1028,744). DELETE this incorrect down-left arrow completely, leaving the original clean white background.
Replace it with ONE thin BLACK DASHED arrow, of exactly the same stroke style and size, running from the SAME upper gold event left boundary near (1082,688) diagonally DOWN-RIGHT to the lower orange highlighted event LEFT BOUNDARY tick near (1113,744). Its arrowhead must point to the TOP of the existing orange tick at x=1113, which marks where the pale orange highlight starts. It must NOT point to the time-axis beginning at x=1024.
Only one arrow is replaced, leaving exactly three correspondence arrows total, non-crossing.

CRITICAL: Do NOT alter ANY timeline geometry or interval shading. The lower orange full baseline still starts at approximately x=1024 and ends at x=1318. Its shaded EVENT interval starts at the INTERNAL orange tick x=1113 and extends to x=1318. The unshaded orange baseline segment x=1024..1113 MUST remain unshaded. KEEP that internal event boundary tick. Do NOT remove it, do NOT extend the highlight leftward, and do NOT resize, shift, or redraw any interval.
Keep the upper gold interval and dot exactly as they are; keep the lower orange representative dot at about x=1208. Keep the middle and right dashed arrows completely unchanged.

Everything else in the entire image must be unchanged: all typography, all text, positions, colors, mannequins, module borders, panel backgrounds, chart curves, DTW matrix, gold inter-panel arrow, horizontal green connectors, footer. All four main module borders in panel (a) remain black. No red circles or blue annotation arrows; the new arrow is dashed black like its neighbors. Same uncropped composition and aspect ratio (1774 x 887).

