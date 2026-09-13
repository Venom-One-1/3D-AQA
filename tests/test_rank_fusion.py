import unittest

import numpy as np
from scipy.special import expit

from aqa3d.rank_fusion import (
    evaluate_ranking,
    format_complete_order,
    learn_simplex_weights,
    leave_one_query_out,
    metric_quality,
    pairwise_differences,
    weighted_quality,
)
from run_late_fusion_experiment import (
    METRIC_SPECS,
    build_feature_rows,
    prepare_queries,
)


class RankFusionTests(unittest.TestCase):
    def test_metric_direction_and_normalized_rank_quality(self):
        high_ranks, high_quality, _ = metric_quality(
            np.asarray([1.0, 3.0, 2.0]),
            higher_is_better=True,
        )
        low_ranks, low_quality, _ = metric_quality(
            np.asarray([1.0, 3.0, 2.0]),
            higher_is_better=False,
        )

        np.testing.assert_allclose(high_ranks, [3.0, 1.0, 2.0])
        np.testing.assert_allclose(high_quality, [0.0, 1.0, 0.5])
        np.testing.assert_allclose(low_ranks, [1.0, 3.0, 2.0])
        np.testing.assert_allclose(low_quality, [1.0, 0.0, 0.5])

    def test_ties_use_average_rank_and_are_displayed(self):
        ranks, quality, _ = metric_quality(
            np.asarray([3.0, 3.0, 1.0]),
            higher_is_better=True,
        )

        np.testing.assert_allclose(ranks, [1.5, 1.5, 3.0])
        np.testing.assert_allclose(quality, [0.75, 0.75, 0.0])
        self.assertEqual(
            format_complete_order(("1", "2", "3"), ranks),
            "1 = 2 > 3",
        )

    def test_constant_continuous_metric_is_neutral(self):
        _, _, continuous = metric_quality(
            np.asarray([4.0, 4.0, 4.0]),
            higher_is_better=False,
        )

        np.testing.assert_allclose(continuous, [0.5, 0.5, 0.5])

    def test_equal_weight_borda(self):
        features = np.asarray(
            [
                [1.0, 0.5, 0.0],
                [0.0, 0.5, 1.0],
            ]
        )
        scores = weighted_quality(features, np.full(3, 1.0 / 3.0))

        np.testing.assert_allclose(scores, [0.5, 0.5])

    def test_pairwise_differences_follow_human_preference(self):
        features = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.2, 0.2]])
        human_ranks = np.asarray([1.0, 3.0, 2.0])
        differences = pairwise_differences(features, human_ranks)

        self.assertEqual(differences.shape, (3, 2))
        np.testing.assert_allclose(differences[0], [1.0, -1.0])
        np.testing.assert_allclose(differences[1], [0.8, -0.2])
        np.testing.assert_allclose(differences[2], [0.2, -0.8])

    def test_known_synthetic_weights_are_recovered(self):
        rng = np.random.default_rng(7)
        true_weights = np.asarray([0.60, 0.30, 0.10])
        raw_differences = rng.normal(size=(30000, 3))
        preferred_first = rng.random(len(raw_differences)) < expit(
            raw_differences @ true_weights
        )
        oriented = np.where(
            preferred_first[:, None],
            raw_differences,
            -raw_differences,
        )

        fit = learn_simplex_weights(oriented, regularization=0.0)

        self.assertTrue(fit.success)
        np.testing.assert_allclose(np.sum(fit.weights), 1.0, atol=1e-10)
        self.assertTrue(np.all(fit.weights >= 0.0))
        np.testing.assert_allclose(fit.weights, true_weights, atol=0.04)

    def test_leave_one_query_out_never_trains_on_held_out_query(self):
        features = {
            "a": np.asarray([[1.0, 0.0], [0.0, 1.0], [0.2, 0.2]]),
            "b": np.asarray([[0.9, 0.1], [0.1, 0.9], [0.3, 0.3]]),
            "c": np.asarray([[0.8, 0.2], [0.2, 0.8], [0.4, 0.4]]),
        }
        human = {
            query: np.asarray([1.0, 3.0, 2.0]) for query in features
        }

        folds = leave_one_query_out(features, human)

        self.assertEqual({fold.held_out_query for fold in folds}, set(features))
        for fold in folds:
            self.assertNotIn(fold.held_out_query, fold.training_queries)
            self.assertEqual(len(fold.training_queries), 2)
            self.assertEqual(fold.fit.pair_count, 6)

    def test_ranking_evaluation_reports_all_outputs(self):
        predicted, evaluation = evaluate_ranking(
            ("10", "2", "1"),
            np.asarray([0.9, 0.6, 0.2]),
            np.asarray([1.0, 3.0, 2.0]),
        )

        np.testing.assert_allclose(predicted, [1.0, 2.0, 3.0])
        self.assertAlmostEqual(evaluation.spearman, 0.5)
        self.assertAlmostEqual(evaluation.mean_absolute_rank_error, 2.0 / 3.0)
        self.assertEqual(evaluation.exact_rank_count, 1)
        self.assertEqual(evaluation.complete_order, "10 > 2 > 1")

    def test_feature_table_and_query_validation_cover_all_metrics(self):
        pose_rows = [
            {
                "case_id": "1_1_qishi",
                "student_id": "1",
                "move_name": "qishi",
                "pose_score": "80",
            },
            {
                "case_id": "2_1_qishi",
                "student_id": "2",
                "move_name": "qishi",
                "pose_score": "70",
            },
        ]
        center_rows = [
            {"case_id": "1_1_qishi", "weighted_center_excess": "1.0"},
            {"case_id": "2_1_qishi", "weighted_center_excess": "2.0"},
        ]
        features = build_feature_rows(
            pose_rows,
            center_rows,
            {"1_1_qishi": 0, "2_1_qishi": 1},
        )
        human = {("qishi", "1"): 1, ("qishi", "2"): 2}

        queries, metric_rows = prepare_queries(
            features,
            human,
            expected_students=2,
        )

        self.assertEqual(len(features), 2 * len(METRIC_SPECS))
        self.assertEqual(len(metric_rows), 2 * len(METRIC_SPECS))
        self.assertEqual(queries["qishi"].rank_quality.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
