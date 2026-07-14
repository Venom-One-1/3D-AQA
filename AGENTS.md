# Repository Guidelines

## Project Structure & Module Organization

This repository implements a 3D action-quality assessment pipeline for Tai Chi clips. It combines YOLO pose sampling/alignment, DTW frame matching, PHALP/4D-Humans SMPL track loading, and local SMPL joint geodesic scoring. The current scoring convention compares the 23 local `body_pose` joints and excludes `global_orient`.

- `aqa3d/` contains reusable library code: `alignment.py` for video sampling, YOLO body vectors, keyframes, and DTW; `geodesic.py` for SO(3) distance utilities; `tracking.py` for PHALP/SMPL track loading; `pipeline.py` for end-to-end scoring and result export.
- `aqa3d/angle_metrics.py` derives interpretable SMPL-24 joint-angle metrics from matched frames.
- Top-level scripts are command-line entry points:
  - `run_3d_aqa.py` scores one student/teacher pair.
  - `run_batch.py` runs the original student-keyframe batch workflow.
  - `run_teacher_keyframe_batch.py` uses teacher motion-peak keyframes as fixed anchors and YOLO 2D-vector DTW.
  - `run_teacher_keyframe_smpl_dtw_batch.py` uses teacher motion-peak anchors with SMPL-geodesic DTW local costs.
  - `run_teacher_1fps_smpl_dtw_batch.py` uses uniform 1 FPS teacher anchors with SMPL-geodesic DTW.
  - `export_teacher_keyframe_angle_metrics.py` exports long-form angle metrics for teacher-keyframe results.
  - `plot_teacher_keyframe_rankings.py` and `plot_student_move_rank_heatmap.py` generate ranking visualizations.
  - `inspect_tracking_pkl.py` and `diagnose_smpl_dtw_alignment.py` are diagnostic utilities.
- `tests/` contains standard-library `unittest` tests for geodesic distance, tracking/SMPL loading helpers, SMPL-DTW, 1 FPS anchors, and angle metrics.
- Generated outputs live under directories such as `results/`, `teacher_keyframe_results/`, `teacher_keyframe_smpl_dtw_results/`, `teacher_1fps_smpl_dtw_results/`, `smpl_dtw_diagnostics/`, and `validate_dtw/results/`. Avoid committing large experiment artifacts unless intentionally requested.

## Build, Test, and Development Commands

Install dependencies inside the intended `4d-humans` environment:

```bash
python -m pip install -r requirements.txt
```

Run the unit test suite:

```bash
python -m unittest discover -s tests -v
```

Run one student/teacher pair:

```bash
python run_3d_aqa.py --student-video <student.mp4> --teacher-video <teacher.mp4> \
  --student-tracking <student.pkl> --teacher-tracking <teacher.pkl> \
  --output-dir results/<case> --device cuda:2 --yolo-batch-size 8
```

Run the configured batch workflow:

```bash
python run_batch.py --output-root results --device cuda:2 --yolo-batch-size 8
```

Run fixed teacher-keyframe ranking with YOLO 2D-vector DTW:

```bash
python run_teacher_keyframe_batch.py --output-root teacher_keyframe_results \
  --device cuda:2 --yolo-batch-size 8
```

Run teacher-keyframe ranking with SMPL-geodesic DTW:

```bash
python run_teacher_keyframe_smpl_dtw_batch.py --output-root teacher_keyframe_smpl_dtw_results
```

Run uniform 1 FPS teacher-anchor SMPL-DTW:

```bash
python run_teacher_1fps_smpl_dtw_batch.py --output-root teacher_1fps_smpl_dtw_results
```

Export angle metrics and plots from teacher-keyframe results:

```bash
python export_teacher_keyframe_angle_metrics.py --result-root teacher_keyframe_results --device cpu
python plot_teacher_keyframe_rankings.py --results-root teacher_keyframe_results
python plot_student_move_rank_heatmap.py --results-root teacher_keyframe_results
```

Validate the new DTW implementation against the legacy AQA implementation from the separate `aqa` environment:

```bash
conda activate aqa
python validate_dtw/validate_implementations.py --device cuda:2
```

## Coding Style & Naming Conventions

Use Python 3 style with 4-space indentation, type hints for public functions, and `pathlib.Path` for filesystem paths. Keep reusable logic in `aqa3d/`; keep CLI parsing, default paths, and file export wiring in top-level scripts. Use `snake_case` for functions, variables, modules, and test names. Prefer explicit validation and clear exceptions for shape, frame-index, and file-path assumptions. Existing comments include English and Chinese; keep new comments short and only where they clarify non-obvious math, indexing, or frame-number conventions.

For numerical code, keep data as `numpy.ndarray` where possible and use vectorized operations for frame/joint cost matrices. The project distinguishes sampled-frame indices, source video frame indices, and PHALP frame IDs; preserve the explicit 0-based to 1-based conversions in result exports.

## Testing Guidelines

Use the standard-library `unittest` framework. Add tests under `tests/test_*.py` and name methods `test_<behavior>`. For numerical code, assert shapes as well as values and use `numpy.testing.assert_allclose` with explicit tolerances. Prefer small synthetic matrices/tracks over real videos or private datasets in tests. Run `python -m unittest discover -s tests -v` before submitting changes.

## Commit & Pull Request Guidelines

Recent commit messages are informal and brief. Prefer concise, imperative summaries that state the change, for example `Add SMPL-DTW tests` or `Fix PHALP frame indexing`. Pull requests should describe the pipeline behavior changed, list test commands run, mention required external data/model paths, and include representative output paths or screenshots for visualization changes.

## Security & Configuration Tips

Default scripts currently document local data roots such as `/home/sqw/VisualSearch/aqa/ActionSegments`, `/home/sqw/VisualSearch/aqa/Tracking`, and YOLO weights under the sibling `aqa/model_weights/` directory. Treat these as local examples, not portable assumptions. Do not hard-code new private dataset locations beyond documented defaults. Keep model weights, videos, tracking `.pkl` files, generated `.npz` files, figures, and large CSV exports out of source unless they are small, intentional fixtures.
