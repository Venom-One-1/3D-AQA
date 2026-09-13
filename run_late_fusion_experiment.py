#!/usr/bin/env python
"""Evaluate learnable late rank fusion with leave-one-move-out validation."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aqa3d.rank_fusion import (
    RankingEvaluation,
    evaluate_ranking,
    fit_query_weights,
    leave_one_query_out,
    metric_quality,
    weighted_quality,
)
from run_uncapped_pause_penalty_experiment import reviewed_pause_counts


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_POSE_SCORES = (
    PROJECT_ROOT / "pose_score_experiment_results" / "student_pose_scores.csv"
)
DEFAULT_PAUSE_EVENTS = (
    PROJECT_ROOT / "pose_score_experiment_results" / "unexpected_pause_events.csv"
)
DEFAULT_PAUSE_REVIEWS = PROJECT_ROOT / "pause_event_reviews.csv"
DEFAULT_CENTER_SUMMARY = (
    PROJECT_ROOT
    / "center_fusion_experiment_results"
    / "student_center_summary.csv"
)
DEFAULT_HUMAN_RANKINGS = PROJECT_ROOT / "human_rankings.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "late_fusion_results"
DEFAULT_REGULARIZATION = 0.1
DEFAULT_REGULARIZATION_SENSITIVITY = (0.0, 0.01, 0.1, 1.0, 10.0)
MOVE_ORDER = ("qishi", "yemafenzong", "baiheliangchi")


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    higher_is_better: bool


@dataclass(frozen=True)
class QueryData:
    move_name: str
    case_ids: tuple[str, ...]
    student_ids: tuple[str, ...]
    human_ranks: np.ndarray
    raw_values: np.ndarray
    metric_ranks: np.ndarray
    rank_quality: np.ndarray
    continuous_quality: np.ndarray


METRIC_SPECS = (
    MetricSpec("pose_score", True),
    MetricSpec("reviewed_pause_count", False),
    MetricSpec("weighted_center_excess", False),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    known = set(fieldnames)
    for row in rows[1:]:
        for field in row:
            if field not in known:
                fieldnames.append(field)
                known.add(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def numeric_id_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def build_feature_rows(
    pose_rows: list[dict[str, str]],
    center_rows: list[dict[str, str]],
    effective_pause_counts: dict[str, int],
) -> list[dict]:
    center_by_case = {row["case_id"]: row for row in center_rows}
    if len(center_by_case) != len(center_rows):
        raise ValueError("Center summary contains duplicate case_id values.")
    feature_rows: list[dict] = []
    for pose_row in pose_rows:
        case_id = pose_row["case_id"]
        if case_id not in center_by_case:
            raise KeyError(f"Missing center summary for {case_id}.")
        if case_id not in effective_pause_counts:
            raise KeyError(f"Missing reviewed pause count for {case_id}.")
        values = {
            "pose_score": float(pose_row["pose_score"]),
            "reviewed_pause_count": float(effective_pause_counts[case_id]),
            "weighted_center_excess": float(
                center_by_case[case_id]["weighted_center_excess"]
            ),
        }
        for spec in METRIC_SPECS:
            feature_rows.append(
                {
                    "case_id": case_id,
                    "move_name": pose_row["move_name"],
                    "student_id": pose_row["student_id"],
                    "metric_id": spec.metric_id,
                    "raw_value": values[spec.metric_id],
                    "higher_is_better": int(spec.higher_is_better),
                }
            )
    return feature_rows


def load_human_ranks(rows: list[dict[str, str]]) -> dict[tuple[str, str], int]:
    result: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["move"], row["student_id"])
        if key in result:
            raise ValueError(f"Duplicate human ranking for {key}.")
        result[key] = int(row["rank"])
    return result


def prepare_queries(
    feature_rows: list[dict],
    human_ranks: dict[tuple[str, str], int],
    *,
    expected_students: int = 5,
) -> tuple[dict[str, QueryData], list[dict]]:
    metric_ids = tuple(spec.metric_id for spec in METRIC_SPECS)
    direction = {spec.metric_id: spec.higher_is_better for spec in METRIC_SPECS}
    features: dict[tuple[str, str, str], float] = {}
    case_ids: dict[tuple[str, str], str] = {}
    moves: list[str] = []
    for row in feature_rows:
        move_name = str(row["move_name"])
        student_id = str(row["student_id"])
        metric_id = str(row["metric_id"])
        if move_name not in moves:
            moves.append(move_name)
        if metric_id not in direction:
            raise KeyError(f"Unknown metric_id: {metric_id}")
        if bool(int(row["higher_is_better"])) != direction[metric_id]:
            raise ValueError(f"Inconsistent direction for {metric_id}.")
        key = (move_name, student_id, metric_id)
        if key in features:
            raise ValueError(f"Duplicate feature row for {key}.")
        features[key] = float(row["raw_value"])
        case_ids[(move_name, student_id)] = str(row["case_id"])

    ordered_moves = [move for move in MOVE_ORDER if move in moves]
    ordered_moves.extend(move for move in moves if move not in ordered_moves)
    queries: dict[str, QueryData] = {}
    ranking_rows: list[dict] = []
    for move_name in ordered_moves:
        students = sorted(
            {
                student_id
                for feature_move, student_id, _ in features
                if feature_move == move_name
            },
            key=numeric_id_key,
        )
        if len(students) != expected_students:
            raise ValueError(
                f"{move_name} has {len(students)} students; "
                f"expected {expected_students}."
            )
        missing = [
            (move_name, student_id, metric_id)
            for student_id in students
            for metric_id in metric_ids
            if (move_name, student_id, metric_id) not in features
        ]
        if missing:
            raise ValueError(f"Missing metric observations: {missing}")
        missing_ranks = [
            (move_name, student_id)
            for student_id in students
            if (move_name, student_id) not in human_ranks
        ]
        if missing_ranks:
            raise ValueError(f"Missing human rankings: {missing_ranks}")

        expected = np.asarray(
            [human_ranks[(move_name, student_id)] for student_id in students],
            dtype=np.float64,
        )
        if set(expected.tolist()) != set(range(1, len(students) + 1)):
            raise ValueError(f"{move_name} human ranks must be 1..{len(students)}.")
        raw = np.asarray(
            [
                [
                    features[(move_name, student_id, metric_id)]
                    for metric_id in metric_ids
                ]
                for student_id in students
            ],
            dtype=np.float64,
        )
        metric_ranks = np.empty_like(raw)
        rank_quality = np.empty_like(raw)
        continuous_quality = np.empty_like(raw)
        for metric_index, spec in enumerate(METRIC_SPECS):
            (
                metric_ranks[:, metric_index],
                rank_quality[:, metric_index],
                continuous_quality[:, metric_index],
            ) = metric_quality(
                raw[:, metric_index],
                higher_is_better=spec.higher_is_better,
            )
        query = QueryData(
            move_name=move_name,
            case_ids=tuple(case_ids[(move_name, student_id)] for student_id in students),
            student_ids=tuple(students),
            human_ranks=expected,
            raw_values=raw,
            metric_ranks=metric_ranks,
            rank_quality=rank_quality,
            continuous_quality=continuous_quality,
        )
        queries[move_name] = query
        for student_index, student_id in enumerate(students):
            for metric_index, spec in enumerate(METRIC_SPECS):
                ranking_rows.append(
                    {
                        "case_id": query.case_ids[student_index],
                        "move_name": move_name,
                        "student_id": student_id,
                        "metric_id": spec.metric_id,
                        "higher_is_better": int(spec.higher_is_better),
                        "raw_value": raw[student_index, metric_index],
                        "metric_rank": metric_ranks[student_index, metric_index],
                        "rank_quality": rank_quality[student_index, metric_index],
                        "continuous_quality": continuous_quality[
                            student_index, metric_index
                        ],
                        "human_rank": int(expected[student_index]),
                    }
                )
    if set(human_ranks) != {
        (move_name, student_id)
        for move_name, query in queries.items()
        for student_id in query.student_ids
    }:
        raise ValueError("Human ranking rows and feature cases do not match exactly.")
    return queries, ranking_rows


def add_method_results(
    ranking_rows: list[dict],
    evaluation_rows: list[dict],
    *,
    method_id: str,
    scores_by_move: dict[str, np.ndarray],
    queries: dict[str, QueryData],
    protocol: str,
    notes: str,
) -> None:
    move_evaluations: list[RankingEvaluation] = []
    for move_name, query in queries.items():
        if move_name not in scores_by_move:
            raise KeyError(f"{method_id} is missing scores for {move_name}.")
        scores = np.asarray(scores_by_move[move_name], dtype=np.float64)
        predicted, evaluation = evaluate_ranking(
            query.student_ids,
            scores,
            query.human_ranks,
        )
        move_evaluations.append(evaluation)
        evaluation_rows.append(
            {
                "method_id": method_id,
                "evaluation_scope": "per_move",
                "move_name": move_name,
                "evaluation_protocol": protocol,
                "spearman_correlation": evaluation.spearman,
                "mean_absolute_rank_error": evaluation.mean_absolute_rank_error,
                "exact_rank_count": evaluation.exact_rank_count,
                "case_count": evaluation.case_count,
                "complete_order": evaluation.complete_order,
                "notes": notes,
            }
        )
        for index, student_id in enumerate(query.student_ids):
            ranking_rows.append(
                {
                    "method_id": method_id,
                    "evaluation_protocol": protocol,
                    "move_name": move_name,
                    "case_id": query.case_ids[index],
                    "student_id": student_id,
                    "quality_score": float(scores[index]),
                    "predicted_rank": float(predicted[index]),
                    "human_rank": int(query.human_ranks[index]),
                    "absolute_rank_error": float(
                        abs(predicted[index] - query.human_ranks[index])
                    ),
                }
            )
    evaluation_rows.append(
        {
            "method_id": method_id,
            "evaluation_scope": "aggregate",
            "move_name": "ALL",
            "evaluation_protocol": protocol,
            "spearman_correlation": float(
                np.mean([evaluation.spearman for evaluation in move_evaluations])
            ),
            "mean_absolute_rank_error": float(
                np.average(
                    [
                        evaluation.mean_absolute_rank_error
                        for evaluation in move_evaluations
                    ],
                    weights=[evaluation.case_count for evaluation in move_evaluations],
                )
            ),
            "exact_rank_count": sum(
                evaluation.exact_rank_count for evaluation in move_evaluations
            ),
            "case_count": sum(evaluation.case_count for evaluation in move_evaluations),
            "complete_order": "",
            "notes": notes,
        }
    )


def fit_cross_validated_scores(
    queries: dict[str, QueryData],
    *,
    input_type: str,
    regularization: float,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    if input_type not in {"rank", "continuous"}:
        raise ValueError(f"Unknown input_type: {input_type}")
    features = {
        move_name: (
            query.rank_quality
            if input_type == "rank"
            else query.continuous_quality
        )
        for move_name, query in queries.items()
    }
    human_ranks = {
        move_name: query.human_ranks for move_name, query in queries.items()
    }
    folds = leave_one_query_out(
        features,
        human_ranks,
        regularization=regularization,
    )
    scores: dict[str, np.ndarray] = {}
    fold_rows: list[dict] = []
    for fold in folds:
        held_out = fold.held_out_query
        scores[held_out] = weighted_quality(
            features[held_out],
            fold.fit.weights,
        )
        for metric_index, spec in enumerate(METRIC_SPECS):
            fold_rows.append(
                {
                    "row_type": "fold",
                    "input_type": input_type,
                    "held_out_move": held_out,
                    "training_moves": ",".join(fold.training_queries),
                    "regularization": regularization,
                    "metric_id": spec.metric_id,
                    "weight": float(fold.fit.weights[metric_index]),
                    "training_pair_count": fold.fit.pair_count,
                    "training_objective": fold.fit.objective,
                    "optimizer_success": int(fold.fit.success),
                    "optimizer_iterations": fold.fit.iterations,
                    "optimizer_message": fold.fit.message,
                }
            )
    return scores, fold_rows


def summarize_fold_weights(fold_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for input_type in ("rank", "continuous"):
        for spec in METRIC_SPECS:
            selected = [
                float(row["weight"])
                for row in fold_rows
                if row["row_type"] == "fold"
                and row["input_type"] == input_type
                and row["metric_id"] == spec.metric_id
            ]
            if not selected:
                raise ValueError(
                    f"No fold weights for {input_type}/{spec.metric_id}."
                )
            rows.append(
                {
                    "row_type": "summary",
                    "input_type": input_type,
                    "held_out_move": "ALL",
                    "training_moves": "",
                    "regularization": fold_rows[0]["regularization"],
                    "metric_id": spec.metric_id,
                    "weight": "",
                    "weight_mean": float(np.mean(selected)),
                    "weight_std": float(np.std(selected)),
                    "weight_min": float(np.min(selected)),
                    "weight_max": float(np.max(selected)),
                    "fold_count": len(selected),
                }
            )
    return rows


def evaluate_scores_by_move(
    scores: dict[str, np.ndarray],
    queries: dict[str, QueryData],
) -> tuple[list[RankingEvaluation], list[dict]]:
    evaluations: list[RankingEvaluation] = []
    move_rows: list[dict] = []
    for move_name, query in queries.items():
        _, evaluation = evaluate_ranking(
            query.student_ids,
            scores[move_name],
            query.human_ranks,
        )
        evaluations.append(evaluation)
        move_rows.append(
            {
                "held_out_move": move_name,
                "test_spearman": evaluation.spearman,
                "test_mean_absolute_rank_error": (
                    evaluation.mean_absolute_rank_error
                ),
                "test_exact_rank_count": evaluation.exact_rank_count,
                "test_case_count": evaluation.case_count,
                "test_complete_order": evaluation.complete_order,
            }
        )
    return evaluations, move_rows


def regularization_sensitivity(
    queries: dict[str, QueryData],
    regularizations: tuple[float, ...],
) -> list[dict]:
    rows: list[dict] = []
    for input_type in ("rank", "continuous"):
        for regularization in regularizations:
            scores, fold_weights = fit_cross_validated_scores(
                queries,
                input_type=input_type,
                regularization=regularization,
            )
            evaluations, move_rows = evaluate_scores_by_move(scores, queries)
            weights_by_fold: dict[str, dict[str, float]] = {}
            for row in fold_weights:
                weights_by_fold.setdefault(row["held_out_move"], {})[
                    row["metric_id"]
                ] = float(row["weight"])
            for move_row in move_rows:
                held_out = move_row["held_out_move"]
                rows.append(
                    {
                        "input_type": input_type,
                        "regularization": regularization,
                        "evaluation_scope": "per_move",
                        **move_row,
                        **{
                            f"weight_{metric_id}": weight
                            for metric_id, weight in weights_by_fold[held_out].items()
                        },
                    }
                )
            rows.append(
                {
                    "input_type": input_type,
                    "regularization": regularization,
                    "evaluation_scope": "aggregate",
                    "held_out_move": "ALL",
                    "test_spearman": float(
                        np.mean([evaluation.spearman for evaluation in evaluations])
                    ),
                    "test_mean_absolute_rank_error": float(
                        np.mean(
                            [
                                evaluation.mean_absolute_rank_error
                                for evaluation in evaluations
                            ]
                        )
                    ),
                    "test_exact_rank_count": sum(
                        evaluation.exact_rank_count for evaluation in evaluations
                    ),
                    "test_case_count": sum(
                        evaluation.case_count for evaluation in evaluations
                    ),
                    "test_complete_order": "",
                }
            )
    return rows


def full_data_weight_rows(
    queries: dict[str, QueryData],
    *,
    regularization: float,
) -> list[dict]:
    rows: list[dict] = []
    human = {move: query.human_ranks for move, query in queries.items()}
    for input_type in ("rank", "continuous"):
        features = {
            move: (
                query.rank_quality
                if input_type == "rank"
                else query.continuous_quality
            )
            for move, query in queries.items()
        }
        fit = fit_query_weights(
            features,
            human,
            regularization=regularization,
        )
        for metric_index, spec in enumerate(METRIC_SPECS):
            rows.append(
                {
                    "input_type": input_type,
                    "regularization": regularization,
                    "metric_id": spec.metric_id,
                    "weight": float(fit.weights[metric_index]),
                    "training_moves": ",".join(queries),
                    "training_pair_count": fit.pair_count,
                    "training_objective": fit.objective,
                    "optimizer_success": int(fit.success),
                    "warning": "Exploratory full-data fit; not an out-of-fold result.",
                }
            )
    return rows


def report_markdown(
    path: Path,
    evaluation_rows: list[dict],
    fold_rows: list[dict],
    full_weight_rows: list[dict],
) -> None:
    aggregates = {
        row["method_id"]: row
        for row in evaluation_rows
        if row["evaluation_scope"] == "aggregate"
    }
    method_order = [
        *(f"metric_{spec.metric_id}" for spec in METRIC_SPECS),
        "equal_weight_borda",
        "learned_rank_fusion_oof",
        "learned_continuous_fusion_oof",
        "rule_based_in_sample_reference",
    ]
    lines = [
        "# Learnable late rank-fusion experiment",
        "",
        "## Aggregate results",
        "",
        "| Method | Protocol | Macro Spearman | Mean rank error | Exact ranks |",
        "|---|---|---:|---:|---:|",
    ]
    for method_id in method_order:
        row = aggregates[method_id]
        lines.append(
            f"| {method_id} | {row['evaluation_protocol']} | "
            f"{float(row['spearman_correlation']):.3f} | "
            f"{float(row['mean_absolute_rank_error']):.3f} | "
            f"{row['exact_rank_count']}/{row['case_count']} |"
        )

    per_move = {
        (row["method_id"], row["move_name"]): row
        for row in evaluation_rows
        if row["evaluation_scope"] == "per_move"
    }
    lines.extend(
        (
            "",
            "## Complete rankings",
            "",
            "| Method | Move | Complete order | Spearman | Mean rank error |",
            "|---|---|---|---:|---:|",
        )
    )
    for method_id in method_order:
        for move_name in MOVE_ORDER:
            row = per_move[(method_id, move_name)]
            lines.append(
                f"| {method_id} | {move_name} | {row['complete_order']} | "
                f"{float(row['spearman_correlation']):.3f} | "
                f"{float(row['mean_absolute_rank_error']):.3f} |"
            )

    lines.extend(
        (
            "",
            "## Default cross-validation weights",
            "",
            "| Input | Held-out move | Pose | Pause | Center |",
            "|---|---|---:|---:|---:|",
        )
    )
    fold_lookup: dict[tuple[str, str], dict[str, float]] = {}
    for row in fold_rows:
        if row["row_type"] != "fold":
            continue
        fold_lookup.setdefault(
            (row["input_type"], row["held_out_move"]),
            {},
        )[row["metric_id"]] = float(row["weight"])
    for input_type in ("rank", "continuous"):
        for move_name in MOVE_ORDER:
            weights = fold_lookup[(input_type, move_name)]
            lines.append(
                f"| {input_type} | {move_name} | "
                f"{weights['pose_score']:.3f} | "
                f"{weights['reviewed_pause_count']:.3f} | "
                f"{weights['weighted_center_excess']:.3f} |"
            )

    lines.extend(
        (
            "",
            "## Cross-validation weight stability",
            "",
            "| Input | Metric | Mean | Std | Min | Max |",
            "|---|---|---:|---:|---:|---:|",
        )
    )
    for row in fold_rows:
        if row["row_type"] != "summary":
            continue
        lines.append(
            f"| {row['input_type']} | {row['metric_id']} | "
            f"{float(row['weight_mean']):.3f} | "
            f"{float(row['weight_std']):.3f} | "
            f"{float(row['weight_min']):.3f} | "
            f"{float(row['weight_max']):.3f} |"
        )

    lines.extend(
        (
            "",
            "## Exploratory full-data weights",
            "",
            "| Input | Pose | Pause | Center |",
            "|---|---:|---:|---:|",
        )
    )
    full_lookup: dict[str, dict[str, float]] = {}
    for row in full_weight_rows:
        full_lookup.setdefault(row["input_type"], {})[row["metric_id"]] = float(
            row["weight"]
        )
    for input_type in ("rank", "continuous"):
        weights = full_lookup[input_type]
        lines.append(
            f"| {input_type} | {weights['pose_score']:.3f} | "
            f"{weights['reviewed_pause_count']:.3f} | "
            f"{weights['weighted_center_excess']:.3f} |"
        )
    lines.extend(
        (
            "",
            "The learned results are relative rankings for this five-student cohort. "
            "Only the leave-one-move-out rows are out-of-fold evaluations.",
            "Unreviewed pause events are provisionally treated as valid, so the learned "
            "pause weight must remain exploratory.",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-scores", type=Path, default=DEFAULT_POSE_SCORES)
    parser.add_argument("--pause-events", type=Path, default=DEFAULT_PAUSE_EVENTS)
    parser.add_argument("--pause-reviews", type=Path, default=DEFAULT_PAUSE_REVIEWS)
    parser.add_argument("--center-summary", type=Path, default=DEFAULT_CENTER_SUMMARY)
    parser.add_argument("--human-rankings", type=Path, default=DEFAULT_HUMAN_RANKINGS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-students", type=int, default=5)
    parser.add_argument("--regularization", type=float, default=DEFAULT_REGULARIZATION)
    parser.add_argument(
        "--regularization-sensitivity",
        type=float,
        nargs="+",
        default=DEFAULT_REGULARIZATION_SENSITIVITY,
    )
    parser.add_argument("--reference-points-per-pause", type=float, default=6.0)
    parser.add_argument(
        "--reference-points-per-center-scale",
        type=float,
        default=2.0,
    )
    parser.add_argument("--reference-center-penalty-cap", type=float, default=15.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.expected_students < 2:
        raise ValueError("--expected-students must be at least two.")
    if args.regularization < 0 or any(
        value < 0 for value in args.regularization_sensitivity
    ):
        raise ValueError("Regularization values must be non-negative.")

    pose_rows = read_rows(args.pose_scores)
    pause_event_rows = read_rows(args.pause_events)
    pause_review_rows = read_rows(args.pause_reviews)
    center_rows = read_rows(args.center_summary)
    effective_pause_counts, reviewed_events = reviewed_pause_counts(
        pose_rows,
        pause_event_rows,
        pause_review_rows,
    )
    feature_rows = build_feature_rows(
        pose_rows,
        center_rows,
        effective_pause_counts,
    )
    human_ranks = load_human_ranks(read_rows(args.human_rankings))
    queries, metric_ranking_rows = prepare_queries(
        feature_rows,
        human_ranks,
        expected_students=args.expected_students,
    )

    output_rankings: list[dict] = []
    evaluation_rows: list[dict] = []
    for metric_index, spec in enumerate(METRIC_SPECS):
        add_method_results(
            output_rankings,
            evaluation_rows,
            method_id=f"metric_{spec.metric_id}",
            scores_by_move={
                move: query.rank_quality[:, metric_index]
                for move, query in queries.items()
            },
            queries=queries,
            protocol="no_training",
            notes="Single-metric relative ranking.",
        )

    uniform = np.full(len(METRIC_SPECS), 1.0 / len(METRIC_SPECS))
    add_method_results(
        output_rankings,
        evaluation_rows,
        method_id="equal_weight_borda",
        scores_by_move={
            move: weighted_quality(query.rank_quality, uniform)
            for move, query in queries.items()
        },
        queries=queries,
        protocol="no_training",
        notes="Equal-weight normalized-rank Borda fusion.",
    )

    default_fold_rows: list[dict] = []
    for input_type, method_id in (
        ("rank", "learned_rank_fusion_oof"),
        ("continuous", "learned_continuous_fusion_oof"),
    ):
        scores, fold_rows = fit_cross_validated_scores(
            queries,
            input_type=input_type,
            regularization=args.regularization,
        )
        default_fold_rows.extend(fold_rows)
        add_method_results(
            output_rankings,
            evaluation_rows,
            method_id=method_id,
            scores_by_move=scores,
            queries=queries,
            protocol="leave_one_move_out",
            notes=(
                f"Pairwise logistic fusion with lambda={args.regularization:g}; "
                "each move is scored by weights trained on the other moves."
            ),
        )

    metric_index = {spec.metric_id: index for index, spec in enumerate(METRIC_SPECS)}
    reference_scores = {}
    for move, query in queries.items():
        pose = query.raw_values[:, metric_index["pose_score"]]
        pauses = query.raw_values[:, metric_index["reviewed_pause_count"]]
        center = query.raw_values[:, metric_index["weighted_center_excess"]]
        reference_scores[move] = (
            pose
            - args.reference_points_per_pause * pauses
            - np.minimum(
                args.reference_points_per_center_scale * center,
                args.reference_center_penalty_cap,
            )
        )
    add_method_results(
        output_rankings,
        evaluation_rows,
        method_id="rule_based_in_sample_reference",
        scores_by_move=reference_scores,
        queries=queries,
        protocol="in_sample_tuned_reference",
        notes=(
            "Existing tuned rule: pose - "
            f"{args.reference_points_per_pause:g}*pause - "
            f"min({args.reference_points_per_center_scale:g}*center, "
            f"{args.reference_center_penalty_cap:g})."
        ),
    )

    sensitivity_rows = regularization_sensitivity(
        queries,
        tuple(args.regularization_sensitivity),
    )
    default_fold_rows.extend(summarize_fold_weights(default_fold_rows))
    full_weight_rows = full_data_weight_rows(
        queries,
        regularization=args.regularization,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_root / "fusion_features.csv", feature_rows)
    write_rows(args.output_root / "metric_rankings.csv", metric_ranking_rows)
    write_rows(args.output_root / "fold_weights.csv", default_fold_rows)
    write_rows(args.output_root / "oof_rankings.csv", output_rankings)
    write_rows(args.output_root / "evaluation_summary.csv", evaluation_rows)
    write_rows(
        args.output_root / "regularization_sensitivity.csv",
        sensitivity_rows,
    )
    write_rows(args.output_root / "full_data_weights.csv", full_weight_rows)
    report_markdown(
        args.output_root / "ranking_report.md",
        evaluation_rows,
        default_fold_rows,
        full_weight_rows,
    )
    unreviewed_count = sum(
        int(row["reviewed"]) == 0 for row in reviewed_events
    )
    (args.output_root / "summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "relative_ranking_only": True,
                "metric_ids": [spec.metric_id for spec in METRIC_SPECS],
                "move_names": list(queries),
                "student_count_per_move": args.expected_students,
                "case_count": sum(len(query.student_ids) for query in queries.values()),
                "default_regularization": args.regularization,
                "regularization_sensitivity": list(
                    args.regularization_sensitivity
                ),
                "cross_validation": "leave-one-move-out",
                "reviewed_pause_event_count": len(reviewed_events) - unreviewed_count,
                "unreviewed_pause_event_count": unreviewed_count,
                "unreviewed_pause_policy": "provisionally treated as valid",
                "rule_based_reference": {
                    "points_per_pause": args.reference_points_per_pause,
                    "points_per_center_scale": (
                        args.reference_points_per_center_scale
                    ),
                    "center_penalty_cap": args.reference_center_penalty_cap,
                    "evaluation_protocol": "in_sample_tuned_reference",
                },
                "warning": (
                    "Only three independent move rankings are available. Learned "
                    "weights are exploratory and are not validated absolute scores."
                ),
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Late rank-fusion experiment -> {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
