#!/usr/bin/env python
"""Inspect the frame-wise contents of a PHALP/4D-Humans tracking .pkl file."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def frame_number(key: str) -> int:
    """Extract PHALP's one-based frame number from an image-path key."""
    return int(Path(key).stem)


def describe(value: Any) -> str:
    """Return a compact, terminal-friendly description of nested tracking data."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return f"ndarray shape={value.shape}, dtype={value.dtype}, empty"
        numeric = np.issubdtype(value.dtype, np.number)
        if numeric:
            return (
                f"ndarray shape={value.shape}, dtype={value.dtype}, "
                f"min={float(value.min()):.5g}, max={float(value.max()):.5g}"
            )
        return f"ndarray shape={value.shape}, dtype={value.dtype}"
    if isinstance(value, dict):
        return f"dict keys={list(value.keys())}"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__} length={len(value)}"
    return repr(value)


def print_frame(key: str, record: dict[str, Any]) -> None:
    print(f"\nFrame {frame_number(key)}: {key}")
    print("Top-level fields:")
    for name, value in record.items():
        print(f"  {name}: {describe(value)}")

    track_ids = record.get("tid", [])
    smpls = record.get("smpl", [])
    print(f"Tracked people: {len(track_ids)}")
    for person_index, (track_id, smpl) in enumerate(zip(track_ids, smpls)):
        print(f"  Person {person_index} (tid={track_id}):")
        if not isinstance(smpl, dict):
            print(f"    smpl: {describe(smpl)}")
            continue
        for name, value in smpl.items():
            print(f"    {name}: {describe(np.asarray(value) if not isinstance(value, np.ndarray) else value)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pkl_path", type=Path, help="PHALP result such as demo_*.pkl")
    parser.add_argument(
        "--frame",
        type=int,
        default=1,
        help="One-based PHALP frame number to show; default: 1",
    )
    parser.add_argument(
        "--list-frames",
        action="store_true",
        help="Print every available one-based PHALP frame number.",
    )
    args = parser.parse_args()

    if not args.pkl_path.is_file():
        raise FileNotFoundError(f"Tracking file does not exist: {args.pkl_path}")
    data = joblib.load(args.pkl_path)
    if not isinstance(data, dict) or not data:
        raise ValueError("Expected a non-empty frame-keyed PHALP tracking dictionary.")

    ordered_keys = sorted(data, key=frame_number)
    track_counts = Counter(
        int(track_id) for record in data.values() for track_id in record.get("tid", [])
    )
    print(f"File: {args.pkl_path}")
    print(f"Frames: {len(ordered_keys)} (PHALP {frame_number(ordered_keys[0])} to {frame_number(ordered_keys[-1])})")
    print(f"Track IDs and observation counts: {dict(sorted(track_counts.items()))}")
    print(f"First-frame fields: {list(data[ordered_keys[0]].keys())}")

    if args.list_frames:
        print("Available PHALP frames:")
        print(" ".join(str(frame_number(key)) for key in ordered_keys))

    selected_key = next((key for key in ordered_keys if frame_number(key) == args.frame), None)
    if selected_key is None:
        raise KeyError(
            f"PHALP frame {args.frame} is absent. "
            f"Available range: {frame_number(ordered_keys[0])}-{frame_number(ordered_keys[-1])}."
        )
    print_frame(selected_key, data[selected_key])


if __name__ == "__main__":
    main()
