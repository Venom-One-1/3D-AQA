"""Relative-ranking features and constrained late-fusion utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata, spearmanr


@dataclass(frozen=True)
class WeightFit:
    weights: np.ndarray
    objective: float
    pair_count: int
    success: bool
    iterations: int
    message: str


@dataclass(frozen=True)
class FoldFit:
    held_out_query: str
    training_queries: tuple[str, ...]
    fit: WeightFit


@dataclass(frozen=True)
class RankingEvaluation:
    spearman: float
    mean_absolute_rank_error: float
    exact_rank_count: int
    case_count: int
    complete_order: str


def metric_quality(
    values: np.ndarray,
    *,
    higher_is_better: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return average ranks, normalized rank quality, and continuous quality."""
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 1 or len(raw) < 2:
        raise ValueError("Metric values must be a one-dimensional array of length >= 2.")
    if not np.all(np.isfinite(raw)):
        raise ValueError("Metric values must all be finite.")

    oriented = raw if higher_is_better else -raw
    ranks = rankdata(-oriented, method="average")
    rank_quality = 1.0 - (ranks - 1.0) / (len(raw) - 1.0)
    value_range = float(np.ptp(oriented))
    if np.isclose(value_range, 0.0):
        continuous_quality = np.full(len(raw), 0.5, dtype=np.float64)
    else:
        continuous_quality = (oriented - np.min(oriented)) / value_range
    return (
        np.asarray(ranks, dtype=np.float64),
        np.asarray(rank_quality, dtype=np.float64),
        np.asarray(continuous_quality, dtype=np.float64),
    )


def weighted_quality(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    coefficients = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or coefficients.shape != (values.shape[1],):
        raise ValueError(
            f"Expected features (N, M) and weights (M,), got "
            f"{values.shape} and {coefficients.shape}."
        )
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(coefficients)):
        raise ValueError("Features and weights must be finite.")
    if np.any(coefficients < -1e-10) or not np.isclose(np.sum(coefficients), 1.0):
        raise ValueError("Weights must be non-negative and sum to one.")
    return values @ coefficients


def pairwise_differences(
    features: np.ndarray,
    human_ranks: np.ndarray,
) -> np.ndarray:
    """Orient all pair differences so the preferred item comes first."""
    values = np.asarray(features, dtype=np.float64)
    ranks = np.asarray(human_ranks, dtype=np.float64)
    if values.ndim != 2 or ranks.shape != (values.shape[0],):
        raise ValueError(
            f"Expected features (N, M) and ranks (N,), got "
            f"{values.shape} and {ranks.shape}."
        )
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(ranks)):
        raise ValueError("Features and human ranks must be finite.")

    differences: list[np.ndarray] = []
    for left in range(len(ranks)):
        for right in range(left + 1, len(ranks)):
            if np.isclose(ranks[left], ranks[right]):
                continue
            better, worse = (
                (left, right) if ranks[left] < ranks[right] else (right, left)
            )
            differences.append(values[better] - values[worse])
    if not differences:
        raise ValueError("At least one strict human preference is required.")
    return np.asarray(differences, dtype=np.float64)


def pairwise_objective(
    weights: np.ndarray,
    differences: np.ndarray,
    *,
    regularization: float,
) -> tuple[float, np.ndarray]:
    coefficients = np.asarray(weights, dtype=np.float64)
    pairs = np.asarray(differences, dtype=np.float64)
    if pairs.ndim != 2 or coefficients.shape != (pairs.shape[1],):
        raise ValueError("Pair differences and weights have incompatible shapes.")
    if regularization < 0:
        raise ValueError("regularization must be non-negative.")

    margins = pairs @ coefficients
    uniform = np.full(len(coefficients), 1.0 / len(coefficients))
    loss = float(
        np.mean(np.logaddexp(0.0, -margins))
        + regularization * np.sum(np.square(coefficients - uniform))
    )
    gradient = (
        np.mean(-expit(-margins)[:, None] * pairs, axis=0)
        + 2.0 * regularization * (coefficients - uniform)
    )
    return loss, np.asarray(gradient, dtype=np.float64)


def learn_simplex_weights(
    differences: np.ndarray,
    *,
    regularization: float = 0.1,
) -> WeightFit:
    pairs = np.asarray(differences, dtype=np.float64)
    if pairs.ndim != 2 or pairs.shape[0] < 1 or pairs.shape[1] < 1:
        raise ValueError("differences must have shape (P, M) with P, M >= 1.")
    if not np.all(np.isfinite(pairs)):
        raise ValueError("Pair differences must be finite.")

    metric_count = pairs.shape[1]
    initial = np.full(metric_count, 1.0 / metric_count, dtype=np.float64)

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        return pairwise_objective(
            weights,
            pairs,
            regularization=regularization,
        )

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        jac=True,
        bounds=[(0.0, 1.0)] * metric_count,
        constraints={
            "type": "eq",
            "fun": lambda weights: float(np.sum(weights) - 1.0),
            "jac": lambda weights: np.ones_like(weights),
        },
        options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"Weight optimization failed: {result.message}")
    weights = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
    weights /= np.sum(weights)
    objective_value, _ = objective(weights)
    return WeightFit(
        weights=weights,
        objective=objective_value,
        pair_count=len(pairs),
        success=bool(result.success),
        iterations=int(result.nit),
        message=str(result.message),
    )


def fit_query_weights(
    features_by_query: Mapping[str, np.ndarray],
    human_ranks_by_query: Mapping[str, np.ndarray],
    *,
    regularization: float = 0.1,
) -> WeightFit:
    query_names = tuple(features_by_query)
    if not query_names or set(query_names) != set(human_ranks_by_query):
        raise ValueError("Feature and human-rank query sets must match and be non-empty.")
    differences = [
        pairwise_differences(
            features_by_query[query_name],
            human_ranks_by_query[query_name],
        )
        for query_name in query_names
    ]
    metric_counts = {values.shape[1] for values in features_by_query.values()}
    if len(metric_counts) != 1:
        raise ValueError("All queries must use the same number of metrics.")
    return learn_simplex_weights(
        np.concatenate(differences, axis=0),
        regularization=regularization,
    )


def leave_one_query_out(
    features_by_query: Mapping[str, np.ndarray],
    human_ranks_by_query: Mapping[str, np.ndarray],
    *,
    regularization: float = 0.1,
) -> tuple[FoldFit, ...]:
    query_names = tuple(features_by_query)
    if len(query_names) < 2:
        raise ValueError("Leave-one-query-out requires at least two queries.")
    if set(query_names) != set(human_ranks_by_query):
        raise ValueError("Feature and human-rank query sets must match.")

    folds: list[FoldFit] = []
    for held_out in query_names:
        training = tuple(name for name in query_names if name != held_out)
        fit = fit_query_weights(
            {name: features_by_query[name] for name in training},
            {name: human_ranks_by_query[name] for name in training},
            regularization=regularization,
        )
        folds.append(
            FoldFit(
                held_out_query=held_out,
                training_queries=training,
                fit=fit,
            )
        )
    return tuple(folds)


def predicted_ranks(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Scores must be a finite one-dimensional array.")
    return np.asarray(rankdata(-values, method="average"), dtype=np.float64)


def format_complete_order(
    item_ids: Sequence[str],
    ranks: np.ndarray,
) -> str:
    values = np.asarray(ranks, dtype=np.float64)
    if values.shape != (len(item_ids),):
        raise ValueError("Item IDs and ranks must have the same length.")

    def item_key(item_id: str) -> tuple[int, int | str]:
        text = str(item_id)
        return (0, int(text)) if text.isdigit() else (1, text)

    grouped: dict[float, list[str]] = {}
    for item_id, rank in zip(item_ids, values):
        grouped.setdefault(float(rank), []).append(str(item_id))
    groups = []
    for rank in sorted(grouped):
        groups.append(" = ".join(sorted(grouped[rank], key=item_key)))
    return " > ".join(groups)


def evaluate_ranking(
    item_ids: Sequence[str],
    scores: np.ndarray,
    human_ranks: np.ndarray,
) -> tuple[np.ndarray, RankingEvaluation]:
    expected = np.asarray(human_ranks, dtype=np.float64)
    predicted = predicted_ranks(scores)
    if expected.shape != predicted.shape:
        raise ValueError("Scores and human ranks must have the same length.")
    correlation, _ = spearmanr(predicted, expected)
    return predicted, RankingEvaluation(
        spearman=float(correlation),
        mean_absolute_rank_error=float(np.mean(np.abs(predicted - expected))),
        exact_rank_count=int(np.sum(np.isclose(predicted, expected))),
        case_count=len(expected),
        complete_order=format_complete_order(item_ids, predicted),
    )
