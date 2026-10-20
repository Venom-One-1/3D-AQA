# KEL v6: Search Window Fill

Tool: built-in image_gen (edit).
Source: user-attached KEL_v5.png; unannotated image is the edit target,
red-oval copy is the location guide.
Change: fill the missing right-side blue search-window region.

## Prompt

Use case: precise-object-edit.
Asset type: existing scientific pipeline figure, minimal localized correction.
Input images: the two most recently attached copies of the user's updated KEL_v5.png. The clean, unannotated copy is the EDIT TARGET. The copy with a red oval is only a location guide. Work from the user's updated figure with the small real frontal SMPL meshes, not any earlier diagram.
Make exactly one local edit in panel (a), within the Reference Localization box, inside the Metric time series plot: the pale-blue Search window currently has blue shading to the LEFT of the pale-yellow Event window but is missing its matching blue shading on the RIGHT. Fill the narrow rectangular strip BETWEEN the yellow event window's right dashed vertical edge and the blue search window's right dashed vertical edge with the SAME pale translucent light blue as the existing left search-window fill. In the displayed 1959 x 949 image this missing strip is approximately x=1271..1308, y=359..447 (about 64.9%-66.8% across and 37.8%-47.1% down); use the actual plot boundaries rather than extending outside the search window. The fill extends across the FULL plot height from the top of the window to the horizontal time axis, NOT merely below the blue curve. Keep the blue and red metric curves clearly visible on top of the new fill. Keep the yellow Event window yellow and retain every existing dashed boundary.
Strict invariants: preserve the entire original composition, aspect ratio, all text verbatim and its typography, all panels and borders, all small real SMPL meshes and poses, all arrows, all timeline positions and intervals, the heatmap, legend, and both metric curves. Do not recolor or regenerate other regions. Do not add any annotation. Do not include the red oval in the final output. Return the complete clean figure with only this small missing background strip repaired. White opaque overall canvas as in the original.
