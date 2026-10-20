# Temporal Key Event Pipeline Figure

- Tool: built-in image_gen, using the user-supplied Figure 1 as a style reference.
- Deliverable: temporal_key_event_pipeline_v1.png
- Diagram: reference event localization followed by DTW event transfer.
- Scope: authored, reviewed temporal queries; no online LLM query-generation claim.
- The pose illustrations, signal curves, and DTW matrix are schematic, not measured experimental data.
- Final connector audit: reviewed reference events and the DTW path both feed Event Transfer; Target Events receives the result of Event Transfer.

## Initial Generation Prompt

Create a polished, publication-ready scientific pipeline diagram, using the LAST USER-ATTACHED figure strictly as a STYLE REFERENCE, not as the content to copy. Make an entirely new figure about technique-guided temporal key-event localization in Tai Chi. Match the reference's clean white canvas, large rounded containers, thin colored outlines, bold black English sans-serif headings, pale blue and pale green section fills, royal-blue and green title tabs with white lettering, gold reference accents, orange target accents, solid directional arrows, and small light blue-gray 3D SMPL-style human mannequins. No gradients, shadows, dark background, photoreal invented study photographs, branding, decorative artwork, or illegible microtext. Wide landscape canvas, approximately 2:1 ratio, high resolution, generous readable typography and spacing. The figure should look like a careful academic vector flowchart.

Title centered above: "Technique-Guided Key Event Localization"

The algorithm must be EXACTLY this TWO-STAGE pipeline. Top stage locates semantic events ONLY in the reference; bottom stage transfers these events to a target using SMPL Pose DTW. Do not run the temporal query independently on the target. Do not imply selecting a student's best-quality frame. No quality scores, coaching feedback module, GPT logo, automatic language-to-query API call, or fixed count of ten checkpoints.

TOP large pale-blue outlined rounded section with royal-blue header:
"(a) Reference Event Localization"
Within it, FOUR clearly separated white rounded modules arranged left to right:
1. "Technique Description"
   simple document visual and the concise two-line example:
   "Raise both arms forward"
   "Hands near shoulder level"
2. "Temporal Query"
   subtitle: "Authored offline; reviewed"
   Three short fields:
   "Metrics: wrist height + reach"
   "Conditions: local peaks + slow change"
   "Selection: interval + representative"
   A solid blue arrow goes module 1 -> module 2.
3. "Reference Localization"
   At its top show two explicit input labels "Reference SMPL sequence" and "Known form boundaries", with a small strip of three light-gray SMPL mannequin poses under the first label, hands gradually lifted in front.
   Under these show a tiny clean line plot of TWO smoothly varying geometric signals over time, with a shaded search-window span and one highlighted event window, labeled "Metric time series".
   The reference SMPL sequence and known form boundaries must visibly feed the query execution / signal evaluation within this module.
   Solid blue arrow from module 2 -> module 3.
4. "Reference Events"
   one light-gray SMPL mannequin with both arms forward around shoulder height.
   Below it a gold horizontal timeline with an interval bracket, a highlighted representative dot, and these concise labels:
   "Event interval [ts, te]"
   "Representative frame tr"
   "Review / correction"
   Solid blue arrow module 3 -> module 4.
All body-relative wrist positions are normalized; add a short caption beneath the time-series graphic: "Body-relative, normalized signals". No real or fake numerical timestamps.

BOTTOM large pale-green outlined rounded section with green header:
"(b) DTW Event Transfer"
Within it FOUR white rounded modules aligned in columns with the top:
1. "Pose Sequences"
   Two small separate strips of light blue-gray SMPL mannequins:
   first labeled "Reference" with a gold timeline,
   second labeled "Target" with an orange timeline.
   Both represent full pose sequences, not static single poses.
2. "SMPL Pose DTW"
   small clearly drawn cost-matrix visual with a monotonic step path (only right, down, or diagonal steps), axes labeled "Reference" and "Target".
   concise subtitle: "Geodesic pose distance"
   A solid arrow from module 1 -> module 2.
3. "Event Transfer"
   Show two horizontal timelines gold reference and orange target with matching interval brackets and representative dots connected by correspondences.
   short text: "Map intervals and representative frames"
   A solid arrow from module 2 -> module 3.
   CRITICAL: add a DISTINCT routed gold arrow from TOP module 4 "Reference Events" DOWN through the inter-stage gap into TOP EDGE of BOTTOM module 3 "Event Transfer". Its label is "Reviewed reference events". Route cleanly; do NOT pass through any labels, mannequins, or unrelated modules. This arrow must not enter DTW or the output box; the mapping uses both events and the DTW path.
4. "Target Events"
   a small light-gray SMPL mannequin with arms forward, subtly different bend from reference to show correspondence does not force correctness.
   Orange interval timeline and representative dot.
   Exact short labels:
   "Mapped event interval"
   "Representative frame"
   "Timestamps + frame indices"
   Solid arrow module 3 -> module 4.

Footer in small but readable dark gray:
"Localize events in the reference, then transfer their temporal correspondence to the target."
Use consistent capitalization, no repeated confusing arrows. Every arrow must point in the correct direction; layout all connectors in reserved whitespace. Keep all text inside its module, professionally typeset with adequate margins. The visual hierarchy and palette should closely match the supplied academic pipeline figure, while all content matches the algorithm described here.

## Connector Correction Prompt

Edit this scientific pipeline image with minimal local changes. Preserve all layout, typography, boxes, colors, figures, title, text, and all other arrows exactly. The current diagram has TWO incorrect or missing input connectors to the lower "Event Transfer" box. Fix these precisely:

1. The gold "Reviewed reference events" arrow currently descends from the TOP "Reference Events" box into the LOWER "Target Events" output box. DELETE that gold arrow completely, including its existing label white patch. Replace it with a routed gold elbow arrow: start at the BOTTOM CENTER of the upper "Reference Events" box; descend vertically only into the white horizontal gap between the two large stages; turn LEFT and travel in that gap until aligned with the center of the LOWER "Event Transfer" box (the third lower box, with two gold/orange correspondence timelines); then turn DOWN and end with an arrowhead ON the TOP EDGE of the lower "Event Transfer" box. Thus its destination is Event Transfer, NEVER Target Events. Keep the label "Reviewed reference events" next to this gold elbow in available whitespace. Avoid the green stage title tab and all labels. The gold line must not overlap any black box borders except its source/destination.

2. Add the missing green right-pointing arrow across the narrow gap BETWEEN the lower "SMPL Pose DTW" box and lower "Event Transfer" box, horizontally at the same height as the existing lower row arrows. This makes the bottom sequence continuous: Pose Sequences -> SMPL Pose DTW -> Event Transfer -> Target Events. Do not draw any diagonal shortcut or new box.

Also make one text-only correction to the tiny line-plot legend in the upper "Reference Localization" box: change "Left wrist height" to "Wrist height" and change "Right wrist height" to "Forward reach". Preserve the blue and red line samples and all plotted curves.

Keep everything else unchanged. This is a precise scientific connector correction, not a redesign. Ensure the completed graph unambiguously has BOTH the DTW result and the reviewed reference event as inputs to Event Transfer, with Target Events having ONLY the outgoing arrow from Event Transfer.

