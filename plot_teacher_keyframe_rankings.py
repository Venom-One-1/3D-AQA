#!/usr/bin/env python
"""Plot per-move student rankings from teacher-keyframe Geodesic results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "teacher_keyframe_results"
MOVE_TITLES = {
    "qishi": "Qishi",
    "yemafenzong": "Yemafenzong",
    "baiheliangchi": "Baiheliangchi",
}


@dataclass(frozen=True)
class RankingEntry:
    student_id: str
    clip_name: str
    move: str
    distance_radians: float
    distance_degrees: float
    teacher_keyframe_count: int
    unique_student_frame_count: int


def infer_move(clip_name: str, summary: dict) -> str:
    return str(summary.get("move") or clip_name.rsplit("_", 1)[-1]).lower()


def infer_student_id(clip_name: str) -> str:
    return clip_name.split("_", 1)[0]


def load_entries(results_root: Path) -> list[RankingEntry]:
    entries: list[RankingEntry] = []
    for summary_path in sorted(results_root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        clip_name = summary_path.parent.name
        distance_radians = float(summary["mean_geodesic_distance_radians"])
        entries.append(
            RankingEntry(
                student_id=infer_student_id(clip_name),
                clip_name=clip_name,
                move=infer_move(clip_name, summary),
                distance_radians=distance_radians,
                distance_degrees=float(summary.get("mean_geodesic_distance_degrees", np.degrees(distance_radians))),
                teacher_keyframe_count=int(summary["teacher_keyframe_count"]),
                unique_student_frame_count=int(summary["unique_matched_student_frame_count"]),
            )
        )
    if not entries:
        raise FileNotFoundError(f"No summary.json files found under {results_root}")
    return entries


def save_ranking_table(entries: list[RankingEntry], output_path: Path) -> None:
    grouped: dict[str, list[RankingEntry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.move].append(entry)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "move", "rank_best_is_1", "student_id", "clip_name", "mean_geodesic_distance_radians",
                "mean_geodesic_distance_degrees", "teacher_keyframe_count", "unique_matched_student_frame_count",
            )
        )
        for move in sorted(grouped):
            for rank, entry in enumerate(sorted(grouped[move], key=lambda value: value.distance_radians), start=1):
                writer.writerow(
                    (
                        move, rank, entry.student_id, entry.clip_name, entry.distance_radians,
                        entry.distance_degrees, entry.teacher_keyframe_count, entry.unique_student_frame_count,
                    )
                )


def plot_move(entries: list[RankingEntry], output_path: Path, dpi: int) -> None:
    ranked = sorted(entries, key=lambda entry: entry.distance_radians)
    positions = np.arange(len(ranked))[::-1]
    distances = np.asarray([entry.distance_degrees for entry in ranked])
    colors = ["#138a72", "#2780c2", "#d48c25", "#be4b5a", "#7559a6"]
    x_max = max(float(distances.max()) * 1.18, 1.0)

    figure, axis = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    for index, (position, entry, distance) in enumerate(zip(positions, ranked, distances)):
        color = colors[index % len(colors)]
        axis.hlines(position, 0, distance, color=color, linewidth=2.4, alpha=0.72)
        axis.scatter(distance, position, color=color, s=115, zorder=3, edgecolor="white", linewidth=1.1)
        axis.annotate(
            f"{distance:.2f} deg",
            (distance, position),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            color="#20252b",
        )

    move = ranked[0].move
    anchor_count = ranked[0].teacher_keyframe_count
    axis.set_title(f"{MOVE_TITLES.get(move, move.title())}: Student Ranking", fontsize=14, pad=12)
    axis.text(
        0.0,
        1.01,
        f"Mean local-SMPL Geodesic Distance | Fixed teacher motion-peak anchors: K={anchor_count} | Rank 1 = Best",
        transform=axis.transAxes,
        fontsize=9.5,
        color="#48515d",
    )
    axis.set_xlabel("Mean Geodesic Distance (degrees; lower is better)", fontsize=11)
    axis.set_yticks(positions)
    axis.set_yticklabels([f"Rank {rank}   Student {entry.student_id}" for rank, entry in enumerate(ranked, start=1)])
    axis.set_xlim(0, x_max)
    axis.set_ylim(-0.7, len(ranked) - 0.3)
    axis.grid(axis="x", color="#d9dee5", linewidth=0.8)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    figure.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, help="Default: <results-root>/ranking_figures")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_root / "ranking_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = load_entries(args.results_root)
    grouped: dict[str, list[RankingEntry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.move].append(entry)
    for move, move_entries in sorted(grouped.items()):
        plot_move(move_entries, output_dir / f"{move}_ranking.png", args.dpi)
    save_ranking_table(entries, output_dir / "ranking_summary.csv")
    print(f"Saved {len(grouped)} per-move ranking figures and ranking_summary.csv to {output_dir}")


if __name__ == "__main__":
    main()
