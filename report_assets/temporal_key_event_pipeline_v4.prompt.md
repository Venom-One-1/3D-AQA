# Temporal Key Event Pipeline v4

- Mode: built-in image_gen image editing.
- Input: /home/sqw/Projects/3D-AQA/report_assets/temporal_key_event_pipeline_v3.png
- Initial edit: /home/sqw/.codex/generated_images/01a11ea2-2c78-7f91-b0f4-67cecd8861ff/exec-b64b9129-c92e-4ed8-9681-9c84dcb9b8e2.png
- Final output: /home/sqw/.codex/generated_images/01a11ea2-2c78-7f91-b0f4-67cecd8861ff/exec-d3a6535e-d36c-4fd5-966a-b36ee1ad6034.png
- Project output: report_assets/temporal_key_event_pipeline_v4.png
- Change: Event Transfer shows offset event starts, unequal durations, and unequal relative representative positions. Three diagonal non-crossing arrows map start, representative frame, and end.
- Visual check: target interval highlight begins and ends at the mapped boundary arrows; the obsolete interior target boundary tick is removed.
- Earlier versions retained. Poses, signals, interval widths, and landmark positions are schematic, not experimental measurements.

## Initial Edit Prompt

Use case: precise-object-edit.
Edit target: the provided existing academic pipeline diagram, "Technique-Guided Key Event Localization", 1774 x 887.
Make a single localized revision ONLY inside the bottom-row Event Transfer module (approximately x=902..1345,y=555..828), specifically its two timeline illustrations and their three dashed correspondence arrows. Keep that module title and subtitle exactly as written: "Event Transfer", "Map intervals and representative frames".

Goal: clearly illustrate nonlinear monotonic temporal warping, NOT an already synchronized pair of equal intervals. Both original time axes stay horizontal. Retain gold for Reference, orange for Target, translucent event-interval fills, a black-outlined colored dot for each representative frame, short boundary ticks, and thin black dashed arrows pointing from the upper reference landmarks to the corresponding lower target landmarks.

Precise geometry guide in the original image coordinates:
- Keep the Reference and Target labels at the left of their respective timelines, with no overlaps.
- Keep both full time-axis baselines approximately x=1028..1318, at y=675 (Reference) and y=758 (Target).
- Upper gold event interval: left boundary at x=1108, right boundary at x=1235, gold representative dot at x=1171, approximately halfway within this 127px-wide interval. Gold translucent shading between these two boundaries only.
- Lower orange event interval: left boundary at x=1055, right boundary at x=1300, orange representative dot at x=1220, about two-thirds along this distinctly longer 245px-wide interval. Orange translucent shading between its own two boundaries only.
- Replace the three old VERTICAL dashed arrows with THREE clearly DIAGONAL dashed black correspondence arrows, all pointing downward: upper left boundary (1108,690) to lower left boundary (1055,743); upper representative (1171,690) to lower representative (1220,743); upper right boundary (1235,690) to lower right boundary (1300,743). Keep small gaps from markers so dots and ticks are unobscured. The first arrow slopes down-left, the other two slope down-right at different angles. They MUST NOT cross. Preserve chronological ordering start < representative < end on both axes.
- The lower highlighted event must visibly differ in start time, duration, and relative representative position. This is a schematic correspondence diagram, not measured values, so do not print any pixel coordinates or numerical timestamps.

Preserve EVERYTHING outside this local timeline revision: the entire upper panel (a) with ALL FOUR principal module outlines BLACK; its pale blue outer panel and title; fully continuous blue Wrist height curve and red Forward reach curve; the lower pale green panel; Pose Sequences; SMPL Pose DTW cost matrix and monotonic path; Target Events; all mannequin poses; all text; footer; the green arrows including SMPL Pose DTW -> Event Transfer; and the gold orthogonal arrow from Reference Events into Event Transfer WITHOUT any label or white backing. Keep image dimensions/aspect ratio, framing, typography and colors. No new text, callouts, red annotation boxes, curved time axes, crossed arrows, or redesign.

## Local Correction Prompt

Use case: precise-object-edit. Edit the supplied pipeline diagram, preserving it except for ONE tiny correction inside the Event Transfer module in bottom panel (b).

The new diagonal dashed mapping arrows are good and must stay. However the lower ORANGE highlighted event band has the wrong left boundary: its shading currently starts at an extra internal vertical orange tick around x=1113, while the left dashed arrow correctly lands near x=1028. Correct ONLY the lower orange event illustration as follows:
- The target event START boundary is the leftmost orange vertical tick directly under the left down-left dashed arrowhead, around x=1024..1030 at y=757.
- Extend the pale orange event-interval highlight LEFTWARD to this leftmost tick so it fills continuously from this leftmost boundary all the way to the rightmost boundary at x=1318.
- REMOVE the obsolete extra internal orange vertical tick around x=1113 completely. There must be ONLY TWO orange vertical boundary ticks in this lower timeline, one at each end, and ONE orange representative dot around x=1208. Do not place a tick in the middle.
- Keep the representative dot offset to the right of the highlighted interval's midpoint, not centered. Keep all three diagonal arrows non-crossing and pointing to start, dot, and end respectively.
- Preserve the upper gold interval unchanged, narrower and at a different horizontal location from the lower orange interval.
The target event's highlighted start, representative dot, and highlighted end must match the three arrow destinations visually. No explanatory text or numbers.

Everything else including all text, titles, typography, black outlines in panel (a), complete blue/red curves, all other modules, gold connecting arrow, colors, pose illustrations and image framing MUST remain identical. No red boxes, no new labels, no new arrows. Full uncropped 1774:887 composition.

