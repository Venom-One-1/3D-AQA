# Repository Guidelines

## Project Structure & Module Organization

This repository implements a 3D action-quality assessment pipeline that combines YOLO pose alignment, DTW keyframe matching, and SMPL local joint geodesic distance scoring.

- `aqa3d/` contains reusable library code: `alignment.py` for video sampling, YOLO body vectors, keyframes, and DTW; `geodesic.py` for SO(3) distance utilities; `tracking.py` for PHALP/SMPL track loading; `pipeline.py` for end-to-end scoring and result export.
- Top-level scripts such as `run_3d_aqa.py`, `run_batch.py`, and `inspect_tracking_pkl.py` are command-line entry points.
- `tests/` contains unit tests, currently focused on geodesic-distance behavior.
- `results/`, `teacher_keyframe_results/`, and `validate_dtw/results/` hold generated outputs; avoid committing large experiment artifacts unless intentionally requested.

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

## Coding Style & Naming Conventions

Use Python 3 style with 4-space indentation, type hints for public functions, and `pathlib.Path` for filesystem paths. Keep reusable logic in `aqa3d/`; keep CLI parsing in top-level scripts. Use `snake_case` for functions, variables, modules, and test names. Prefer explicit validation and clear exceptions for shape, frame-index, and file-path assumptions. Existing comments include English and Chinese; keep new comments short and only where they clarify non-obvious math or indexing.

## Testing Guidelines

Use the standard-library `unittest` framework. Add tests under `tests/test_*.py` and name methods `test_<behavior>`. For numerical code, assert shapes as well as values and use `numpy.testing.assert_allclose` with explicit tolerances. Run `python -m unittest discover -s tests -v` before submitting changes.

## Commit & Pull Request Guidelines

Recent commit messages are informal and brief. Prefer concise, imperative summaries that state the change, for example `Add weighted geodesic tests` or `Fix PHALP frame indexing`. Pull requests should describe the pipeline behavior changed, list test commands run, mention required external data/model paths, and include representative output paths or screenshots for visualization changes.

## Security & Configuration Tips

Do not hard-code private dataset locations beyond documented examples. Keep model weights, videos, tracking `.pkl` files, and generated `.npz` outputs out of source unless they are small, intentional fixtures.
