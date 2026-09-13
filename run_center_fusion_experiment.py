#!/usr/bin/env python
"""Fuse teacher-calibrated pelvis penalties with pose and pause scoring."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from aqa3d.center_fusion import (
    MOVE_METRICS,
    RobustCalibration,
    calibrated_excess,
    center_metric_error,
    center_penalty,
    robust_calibration,
    validate_move_metrics,
)
from aqa3d.pelvis_quality import TeacherPelvisModel
from run_uncapped_pause_penalty_experiment import reviewed_pause_counts


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_POSE_SCORES = PROJECT_ROOT / "pose_score_experiment_results" / "student_pose_scores.csv"
DEFAULT_PAUSE_EVENTS = PROJECT_ROOT / "pose_score_experiment_results" / "unexpected_pause_events.csv"
DEFAULT_PAUSE_REVIEWS = PROJECT_ROOT / "pause_event_reviews.csv"
DEFAULT_HUMAN_RANKINGS = PROJECT_ROOT / "human_rankings.csv"
DEFAULT_PELVIS_ROOT = PROJECT_ROOT / "pelvis_quality_results"
DEFAULT_TEACHER_MODEL = DEFAULT_PELVIS_ROOT / "teacher_pelvis_model.npz"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "center_fusion_experiment_results"
DEFAULT_PAUSE_POINTS = (2.0, 4.0, 5.0, 6.0, 8.0)
DEFAULT_CENTER_POINTS = (0.0, 2.0, 3.0, 5.0)


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


def leave_one_out_teacher_errors(
    model: TeacherPelvisModel,
) -> list[dict]:
    rows: list[dict] = []
    for teacher_index, teacher_id in enumerate(model.teacher_ids):
        keep = np.arange(len(model.teacher_ids)) != teacher_index
        for move_index, move_name in enumerate(model.move_names):
            template = np.nanmedian(model.profiles[keep, move_index], axis=0)
            held_profile = model.profiles[teacher_index, move_index]
            for spec in MOVE_METRICS[move_name]:
                error, details = center_metric_error(
                    held_profile,
                    template,
                    spec.metric_id,
                )
                rows.append(
                    {
                        "teacher_id": teacher_id,
                        "move_id": int(model.move_ids[move_index]),
                        "move_name": move_name,
                        "metric_id": spec.metric_id,
                        "weight": spec.weight,
                        "error": error,
                        **details,
                    }
                )
    return rows


def calibrate_teacher_errors(
    rows: list[dict],
) -> tuple[dict[tuple[str, str], RobustCalibration], list[dict]]:
    calibrations: dict[tuple[str, str], RobustCalibration] = {}
    output_rows: list[dict] = []
    for move_name, specs in MOVE_METRICS.items():
        for spec in specs:
            values = np.asarray(
                [
                    float(row["error"])
                    for row in rows
                    if row["move_name"] == move_name
                    and row["metric_id"] == spec.metric_id
                ],
                dtype=np.float64,
            )
            calibration = robust_calibration(
                values,
                minimum_scale=spec.minimum_scale,
            )
            calibrations[(move_name, spec.metric_id)] = calibration
            output_rows.append(
                {
                    "move_name": move_name,
                    "metric_id": spec.metric_id,
                    "weight": spec.weight,
                    "minimum_scale": spec.minimum_scale,
                    **asdict(calibration),
                    "tolerance_scales": 1.0,
                    "penalty_policy": "weight * max(0, z - 1)",
                }
            )
    return calibrations, output_rows


def load_student_profile(pelvis_root: Path, case_id: str) -> np.ndarray:
    path = pelvis_root / "clips" / case_id / "pelvis_signals.npz"
    with np.load(path) as data:
        return data["profile"].astype(np.float64)


def student_center_errors(
    pose_rows: list[dict[str, str]],
    model: TeacherPelvisModel,
    calibrations: dict[tuple[str, str], RobustCalibration],
    pelvis_root: Path,
) -> tuple[list[dict], list[dict]]:
    component_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    for row in pose_rows:
        case_id = row["case_id"]
        move_name = row["move_name"]
        move_index = model.move_index(move_name)
        profile = load_student_profile(pelvis_root, case_id)
        template = model.median[move_index]
        weighted_excess = 0.0
        case_components: list[dict] = []
        for spec in MOVE_METRICS[move_name]:
            error, details = center_metric_error(profile, template, spec.metric_id)
            calibration = calibrations[(move_name, spec.metric_id)]
            z_value, excess = calibrated_excess(error, calibration)
            weighted_component = spec.weight * excess
            weighted_excess += weighted_component
            component = {
                "case_id": case_id,
                "student_id": row["student_id"],
                "move_id": int(row["move_id"]),
                "move_name": move_name,
                "metric_id": spec.metric_id,
                "weight": spec.weight,
                "error": error,
                "teacher_error_median": calibration.median,
                "teacher_error_mad": calibration.mad,
                "teacher_error_scale": calibration.scale,
                "z_value": z_value,
                "tolerance_scales": 1.0,
                "excess_scales": excess,
                "weighted_excess": weighted_component,
                **details,
            }
            component_rows.append(component)
            case_components.append(component)
        aggregate_rows.append(
            {
                "case_id": case_id,
                "student_id": row["student_id"],
                "move_id": int(row["move_id"]),
                "move_name": move_name,
                "weighted_center_excess": weighted_excess,
                "active_component_count": sum(
                    float(component["excess_scales"]) > 0
                    for component in case_components
                ),
                "maximum_component_z": max(
                    float(component["z_value"]) for component in case_components
                ),
            }
        )
    return component_rows, aggregate_rows


def fusion_sensitivity(
    pose_rows: list[dict[str, str]],
    center_rows: list[dict],
    effective_pause_counts: dict[str, int],
    human_ranks: dict[tuple[str, str], int],
    pause_points: tuple[float, ...],
    center_points: tuple[float, ...],
    center_penalty_cap: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    center_by_case = {row["case_id"]: row for row in center_rows}
    ranking_rows: list[dict] = []
    correlation_rows: list[dict] = []
    configuration_rows: list[dict] = []
    moves = tuple(MOVE_METRICS)
    for pause_point in pause_points:
        for center_point in center_points:
            configuration_correlations: list[float] = []
            configuration_rank_errors: list[float] = []
            configuration_exact = 0
            for move_name in moves:
                selected = [row for row in pose_rows if row["move_name"] == move_name]
                scores: list[float] = []
                penalties: list[tuple[float, float]] = []
                for row in selected:
                    pause_penalty = pause_point * effective_pause_counts[row["case_id"]]
                    center_row = center_by_case[row["case_id"]]
                    pelvis_penalty = center_penalty(
                        float(center_row["weighted_center_excess"]),
                        points_per_scale=center_point,
                        maximum_penalty=center_penalty_cap,
                    )
                    final_score = float(
                        np.clip(
                            float(row["pose_score"]) - pause_penalty - pelvis_penalty,
                            0.0,
                            100.0,
                        )
                    )
                    scores.append(final_score)
                    penalties.append((pause_penalty, pelvis_penalty))
                predicted = rankdata(-np.asarray(scores), method="average")
                expected = np.asarray(
                    [
                        human_ranks[(move_name, row["student_id"])]
                        for row in selected
                    ],
                    dtype=np.float64,
                )
                correlation, p_value = spearmanr(predicted, expected)
                correlation_rows.append(
                    {
                        "move_name": move_name,
                        "points_per_pause": pause_point,
                        "points_per_center_scale": center_point,
                        "center_penalty_cap": center_penalty_cap,
                        "spearman_correlation": float(correlation),
                        "p_value": float(p_value),
                    }
                )
                configuration_correlations.append(float(correlation))
                configuration_rank_errors.extend(
                    np.abs(predicted - expected).tolist()
                )
                configuration_exact += int(np.sum(predicted == expected))
                ordered = sorted(
                    zip(selected, scores, penalties, predicted, expected),
                    key=lambda item: item[3],
                )
                for row, score, penalty_pair, predicted_rank, human_rank in ordered:
                    center_row = center_by_case[row["case_id"]]
                    ranking_rows.append(
                        {
                            "move_name": move_name,
                            "points_per_pause": pause_point,
                            "points_per_center_scale": center_point,
                            "center_penalty_cap": center_penalty_cap,
                            "predicted_rank": float(predicted_rank),
                            "student_id": row["student_id"],
                            "human_rank": int(human_rank),
                            "pose_score": float(row["pose_score"]),
                            "reviewed_pause_count": effective_pause_counts[row["case_id"]],
                            "pause_penalty": penalty_pair[0],
                            "weighted_center_excess": center_row[
                                "weighted_center_excess"
                            ],
                            "center_penalty": penalty_pair[1],
                            "final_score": score,
                        }
                    )
            configuration_rows.append(
                {
                    "points_per_pause": pause_point,
                    "points_per_center_scale": center_point,
                    "center_penalty_cap": center_penalty_cap,
                    "mean_spearman": float(np.mean(configuration_correlations)),
                    "minimum_move_spearman": float(
                        np.min(configuration_correlations)
                    ),
                    "mean_absolute_rank_error": float(
                        np.mean(configuration_rank_errors)
                    ),
                    "exact_rank_count": configuration_exact,
                    "total_rank_count": len(configuration_rank_errors),
                }
            )
    configuration_rows.sort(
        key=lambda row: (
            -float(row["mean_spearman"]),
            float(row["mean_absolute_rank_error"]),
            -int(row["exact_rank_count"]),
        )
    )
    return ranking_rows, correlation_rows, configuration_rows


def save_report(
    path: Path,
    configurations: list[dict],
    correlations: list[dict],
    ranking_rows: list[dict],
) -> None:
    best = configurations[0]
    pause_point = float(best["points_per_pause"])
    center_point = float(best["points_per_center_scale"])
    correlation_lookup = {
        (
            row["move_name"],
            float(row["points_per_pause"]),
            float(row["points_per_center_scale"]),
        ): float(row["spearman_correlation"])
        for row in correlations
    }
    lines = [
        "# Center-fusion sensitivity experiment",
        "",
        "Best observed configuration on the five-student diagnostic set:",
        "",
        f"- points per reviewed pause: {pause_point:g}",
        f"- points per weighted center scale: {center_point:g}",
        f"- center penalty cap: {best['center_penalty_cap']:g}",
        f"- mean Spearman: {best['mean_spearman']:.3f}",
        f"- mean absolute rank error: {best['mean_absolute_rank_error']:.3f}",
        "",
        "| Move | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 | Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for move_name in MOVE_METRICS:
        selected = [
            row
            for row in ranking_rows
            if row["move_name"] == move_name
            and float(row["points_per_pause"]) == pause_point
            and float(row["points_per_center_scale"]) == center_point
        ]
        selected.sort(key=lambda row: float(row["predicted_rank"]))
        students = [row["student_id"] for row in selected]
        rho = correlation_lookup[(move_name, pause_point, center_point)]
        lines.append(f"| {move_name} | {' | '.join(students)} | {rho:.2f} |")
    lines.extend(
        (
            "",
            "This is a sensitivity result, not a fitted final scoring rule. "
            "The same five students were used for interpretation.",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-scores", type=Path, default=DEFAULT_POSE_SCORES)
    parser.add_argument("--pause-events", type=Path, default=DEFAULT_PAUSE_EVENTS)
    parser.add_argument("--pause-reviews", type=Path, default=DEFAULT_PAUSE_REVIEWS)
    parser.add_argument("--human-rankings", type=Path, default=DEFAULT_HUMAN_RANKINGS)
    parser.add_argument("--pelvis-root", type=Path, default=DEFAULT_PELVIS_ROOT)
    parser.add_argument("--teacher-model", type=Path, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--points-per-pause",
        type=float,
        nargs="+",
        default=DEFAULT_PAUSE_POINTS,
    )
    parser.add_argument(
        "--points-per-center-scale",
        type=float,
        nargs="+",
        default=DEFAULT_CENTER_POINTS,
    )
    parser.add_argument("--center-penalty-cap", type=float, default=15.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_move_metrics(MOVE_METRICS)
    if args.center_penalty_cap < 0:
        raise ValueError("--center-penalty-cap must be non-negative.")
    pose_rows = read_rows(args.pose_scores)
    event_rows = read_rows(args.pause_events)
    review_rows = read_rows(args.pause_reviews)
    human_ranks = {
        (row["move"], row["student_id"]): int(row["rank"])
        for row in read_rows(args.human_rankings)
    }
    effective_counts, reviewed_events = reviewed_pause_counts(
        pose_rows,
        event_rows,
        review_rows,
    )
    model = TeacherPelvisModel.load(args.teacher_model)
    teacher_errors = leave_one_out_teacher_errors(model)
    calibrations, calibration_rows = calibrate_teacher_errors(teacher_errors)
    component_rows, center_rows = student_center_errors(
        pose_rows,
        model,
        calibrations,
        args.pelvis_root,
    )
    ranking_rows, correlations, configurations = fusion_sensitivity(
        pose_rows,
        center_rows,
        effective_counts,
        human_ranks,
        tuple(args.points_per_pause),
        tuple(args.points_per_center_scale),
        args.center_penalty_cap,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_root / "teacher_leave_one_out_errors.csv", teacher_errors)
    write_rows(args.output_root / "center_calibration.csv", calibration_rows)
    write_rows(args.output_root / "student_center_components.csv", component_rows)
    write_rows(args.output_root / "student_center_summary.csv", center_rows)
    write_rows(args.output_root / "reviewed_pause_events.csv", reviewed_events)
    write_rows(args.output_root / "ranking_results.csv", ranking_rows)
    write_rows(args.output_root / "rank_correlations.csv", correlations)
    write_rows(args.output_root / "configuration_summary.csv", configurations)
    save_report(
        args.output_root / "ranking_report.md",
        configurations,
        correlations,
        ranking_rows,
    )
    (args.output_root / "summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "teacher_count": len(model.teacher_ids),
                "teacher_calibration": "leave-one-teacher-out against median of other nine",
                "move_metrics": {
                    move: [asdict(spec) for spec in specs]
                    for move, specs in MOVE_METRICS.items()
                },
                "tolerance_scales": 1.0,
                "points_per_pause": list(args.points_per_pause),
                "pause_penalty_cap": None,
                "points_per_center_scale": list(args.points_per_center_scale),
                "center_penalty_cap": args.center_penalty_cap,
                "best_observed_configuration": configurations[0],
                "warning": (
                    "Exploratory five-student sensitivity result; do not treat "
                    "the observed best configuration as a validated final score."
                ),
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Center-fusion experiment -> {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
