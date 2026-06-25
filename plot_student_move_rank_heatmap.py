#!/usr/bin/env python
"""Plot a student-by-move rank heatmap from teacher-keyframe results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "teacher_keyframe_results"
MOVE_ORDER = ("qishi", "yemafenzong", "baiheliangchi")
MOVE_LABELS = ("Qishi", "Yemafenzong", "Baiheliangchi")


def load_rankings(results_root: Path) -> tuple[list[str], dict[tuple[str, str], tuple[int, float]]]:
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for summary_path in results_root.glob("*/summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        clip_name = summary_path.parent.name
        student_id = clip_name.split("_", 1)[0]
        move = str(summary.get("move") or clip_name.rsplit("_", 1)[-1]).lower()
        distance_degrees = float(
            summary.get("mean_geodesic_distance_degrees", np.degrees(summary["mean_geodesic_distance_radians"]))
        )
        grouped[move].append((student_id, distance_degrees))
    if not grouped:
        raise FileNotFoundError(f"No teacher-keyframe summary.json files found under {results_root}")

    students = sorted({student for rows in grouped.values() for student, _ in rows}, key=int)
    rankings: dict[tuple[str, str], tuple[int, float]] = {}
    for move, rows in grouped.items():
        for rank, (student_id, distance_degrees) in enumerate(sorted(rows, key=lambda item: item[1]), start=1):
            rankings[(student_id, move)] = (rank, distance_degrees)
    return students, rankings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output", type=Path, help="Default: <results-root>/ranking_figures/student_move_rank_heatmap.png")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    students, rankings = load_rankings(args.results_root)
    available_moves = [move for move in MOVE_ORDER if any((student, move) in rankings for student in students)]
    if not available_moves:
        raise ValueError("None of the expected moves have ranking results.")
    ranks = np.full((len(students), len(available_moves)), np.nan)
    labels = np.full(ranks.shape, "", dtype=object)
    for row, student_id in enumerate(students):
        for column, move in enumerate(available_moves):
            result = rankings.get((student_id, move))
            if result is not None:
                rank, distance_degrees = result
                ranks[row, column] = rank
                labels[row, column] = f"Rank {rank}\n{distance_degrees:.2f} deg"

    colors = ["#138a72", "#2780c2", "#d48c25", "#be4b5a", "#7559a6"]
    color_map = ListedColormap(colors[: int(np.nanmax(ranks))])
    boundaries = np.arange(0.5, int(np.nanmax(ranks)) + 1.5, 1)
    normalizer = BoundaryNorm(boundaries, color_map.N)
    figure, axis = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
    image = axis.imshow(ranks, cmap=color_map, norm=normalizer, aspect="auto")
    for row in range(ranks.shape[0]):
        for column in range(ranks.shape[1]):
            if labels[row, column]:
                axis.text(column, row, labels[row, column], ha="center", va="center", fontsize=11, color="white", weight="semibold")

    axis.set_title("Student x Move Ranking", fontsize=16, pad=15)
    axis.text(
        0.5,
        1.01,
        "Teacher motion-peak anchors | Mean local-SMPL Geodesic Distance | Rank 1 = Best",
        transform=axis.transAxes,
        ha="center",
        fontsize=10,
        color="#48515d",
    )
    axis.set_xticks(np.arange(len(available_moves)), [MOVE_LABELS[MOVE_ORDER.index(move)] for move in available_moves])
    axis.set_yticks(np.arange(len(students)), [f"Student {student}" for student in students])
    axis.tick_params(axis="both", length=0, labelsize=11)
    for edge in np.arange(-0.5, len(students), 1):
        axis.axhline(edge, color="white", linewidth=2)
    for edge in np.arange(-0.5, len(available_moves), 1):
        axis.axvline(edge, color="white", linewidth=2)

    colorbar = figure.colorbar(image, ax=axis, ticks=np.arange(1, color_map.N + 1), shrink=0.88, pad=0.03)
    colorbar.ax.set_yticklabels(["Rank 1 (Best)"] + [f"Rank {rank}" for rank in range(2, color_map.N + 1)])
    colorbar.outline.set_visible(False)
    output = args.output or args.results_root / "ranking_figures" / "student_move_rank_heatmap.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, facecolor="white")
    plt.close(figure)
    print(f"Saved heatmap to {output}")


if __name__ == "__main__":
    main()
