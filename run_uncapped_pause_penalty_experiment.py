#!/usr/bin/env python
"""Rank students with manually reviewed, uncapped pause-event penalties."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from aqa3d.pose_score import apply_pause_penalty


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "pose_score_experiment_results"
DEFAULT_SCORE_CSV = DEFAULT_RESULT_ROOT / "student_pose_scores.csv"
DEFAULT_EVENT_CSV = DEFAULT_RESULT_ROOT / "unexpected_pause_events.csv"
DEFAULT_REVIEW_CSV = PROJECT_ROOT / "pause_event_reviews.csv"
DEFAULT_HUMAN_RANKINGS = PROJECT_ROOT / "human_rankings.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULT_ROOT / "uncapped_pause_penalty"
DEFAULT_POINTS = (2.0, 4.0, 5.0, 6.0, 8.0)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reviewed_pause_counts(
    score_rows: list[dict[str, str]],
    event_rows: list[dict[str, str]],
    review_rows: list[dict[str, str]],
) -> tuple[dict[str, int], list[dict]]:
    reviews = {
        (row["case_id"], int(row["event_index"])): bool(int(row["is_valid"]))
        for row in review_rows
    }
    effective_counts = {row["case_id"]: 0 for row in score_rows}
    audited_events: list[dict] = []
    for event in event_rows:
        key = (event["case_id"], int(event["event_index"]))
        is_valid = reviews.get(key, True)
        effective_counts[event["case_id"]] += int(is_valid)
        audited_events.append(
            {
                **event,
                "reviewed": int(key in reviews),
                "is_valid_unexpected_pause": int(is_valid),
            }
        )
    return effective_counts, audited_events


def rank_uncapped_penalties(
    score_rows: list[dict[str, str]],
    effective_counts: dict[str, int],
    human_ranks: dict[tuple[str, str], int],
    points_per_pause: tuple[float, ...],
) -> tuple[list[dict], list[dict]]:
    scored_rows: list[dict] = []
    correlation_rows: list[dict] = []
    moves = sorted({row["move_name"] for row in score_rows})
    for points in points_per_pause:
        for move_name in moves:
            selected = [row for row in score_rows if row["move_name"] == move_name]
            scores: list[float] = []
            for row in selected:
                count = effective_counts[row["case_id"]]
                final_score, penalty = apply_pause_penalty(
                    float(row["pose_score"]),
                    count,
                    points_per_pause=points,
                    maximum_penalty=float("inf"),
                )
                scores.append(final_score)
                row["_penalty"] = penalty
            predicted_ranks = rankdata(-np.asarray(scores), method="average")
            expected_ranks = np.asarray(
                [human_ranks[(move_name, row["student_id"])] for row in selected],
                dtype=np.float64,
            )
            correlation, p_value = spearmanr(predicted_ranks, expected_ranks)
            correlation_rows.append(
                {
                    "move_name": move_name,
                    "points_per_pause": points,
                    "penalty_cap": "none",
                    "spearman_correlation": float(correlation),
                    "p_value": float(p_value),
                }
            )
            ordered = sorted(
                zip(selected, scores, predicted_ranks, expected_ranks),
                key=lambda item: item[2],
            )
            for row, final_score, predicted_rank, expected_rank in ordered:
                scored_rows.append(
                    {
                        "move_name": move_name,
                        "points_per_pause": points,
                        "penalty_cap": "none",
                        "predicted_rank": float(predicted_rank),
                        "student_id": row["student_id"],
                        "human_rank": int(expected_rank),
                        "pose_score": float(row["pose_score"]),
                        "automatic_pause_count": int(row["unexpected_pause_count"]),
                        "reviewed_pause_count": effective_counts[row["case_id"]],
                        "pause_penalty": float(row["_penalty"]),
                        "final_score": final_score,
                    }
                )
    return scored_rows, correlation_rows


def save_markdown_report(path: Path, scored_rows: list[dict], correlations: list[dict]) -> None:
    correlation_lookup = {
        (row["move_name"], float(row["points_per_pause"])): row["spearman_correlation"]
        for row in correlations
    }
    move_order = ("qishi", "yemafenzong", "baiheliangchi")
    lines = ["# Uncapped pause-penalty rankings", ""]
    for points in sorted({float(row["points_per_pause"]) for row in scored_rows}):
        lines.extend(
            (
                f"## {points:g} points per unexpected pause",
                "",
                "| Move | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 | Spearman |",
                "|---|---:|---:|---:|---:|---:|---:|",
            )
        )
        for move_name in move_order:
            selected = [
                row
                for row in scored_rows
                if row["move_name"] == move_name
                and float(row["points_per_pause"]) == points
            ]
            selected.sort(key=lambda row: float(row["predicted_rank"]))
            students = [row["student_id"] for row in selected]
            rho = correlation_lookup[(move_name, points)]
            lines.append(f"| {move_name} | {' | '.join(students)} | {rho:.2f} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-csv", type=Path, default=DEFAULT_SCORE_CSV)
    parser.add_argument("--event-csv", type=Path, default=DEFAULT_EVENT_CSV)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--human-rankings", type=Path, default=DEFAULT_HUMAN_RANKINGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--points-per-pause", type=float, nargs="+", default=DEFAULT_POINTS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    score_rows = read_rows(args.score_csv)
    event_rows = read_rows(args.event_csv)
    review_rows = read_rows(args.review_csv)
    human_ranks = {
        (row["move"], row["student_id"]): int(row["rank"])
        for row in read_rows(args.human_rankings)
    }
    effective_counts, audited_events = reviewed_pause_counts(
        score_rows,
        event_rows,
        review_rows,
    )
    scored_rows, correlations = rank_uncapped_penalties(
        score_rows,
        effective_counts,
        human_ranks,
        tuple(args.points_per_pause),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "reviewed_pause_events.csv", audited_events)
    write_rows(args.output_dir / "ranking_results.csv", scored_rows)
    write_rows(args.output_dir / "rank_correlations.csv", correlations)
    save_markdown_report(args.output_dir / "ranking_report.md", scored_rows, correlations)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "penalty_cap": None,
                "points_per_pause": list(args.points_per_pause),
                "manual_review_file": str(args.review_csv),
                "manual_invalid_event_count": sum(
                    int(row["is_valid"]) == 0 for row in review_rows
                ),
                "case_count": len(score_rows),
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Uncapped pause-penalty experiment -> {args.output_dir}")


if __name__ == "__main__":
    main()
