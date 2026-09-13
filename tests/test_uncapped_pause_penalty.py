import unittest

from run_uncapped_pause_penalty_experiment import (
    rank_uncapped_penalties,
    reviewed_pause_counts,
)


class UncappedPausePenaltyTests(unittest.TestCase):
    def test_manual_false_positive_is_excluded_from_count(self):
        scores = [{"case_id": "sample"}]
        events = [
            {"case_id": "sample", "event_index": "1"},
            {"case_id": "sample", "event_index": "2"},
        ]
        reviews = [
            {"case_id": "sample", "event_index": "2", "is_valid": "0"},
        ]

        counts, audited = reviewed_pause_counts(scores, events, reviews)

        self.assertEqual(counts["sample"], 1)
        self.assertEqual(audited[1]["is_valid_unexpected_pause"], 0)

    def test_uncapped_penalty_can_exceed_fifteen_points(self):
        scores = [
            {
                "case_id": "a",
                "student_id": "1",
                "move_name": "qishi",
                "pose_score": "90",
                "unexpected_pause_count": "4",
            },
            {
                "case_id": "b",
                "student_id": "2",
                "move_name": "qishi",
                "pose_score": "80",
                "unexpected_pause_count": "0",
            },
            {
                "case_id": "c",
                "student_id": "3",
                "move_name": "qishi",
                "pose_score": "70",
                "unexpected_pause_count": "0",
            },
        ]
        human = {("qishi", "1"): 3, ("qishi", "2"): 1, ("qishi", "3"): 2}

        ranked, _ = rank_uncapped_penalties(
            scores,
            {"a": 4, "b": 0, "c": 0},
            human,
            (5.0,),
        )

        sample_a = next(row for row in ranked if row["student_id"] == "1")
        self.assertEqual(sample_a["pause_penalty"], 20.0)
        self.assertEqual(sample_a["final_score"], 70.0)


if __name__ == "__main__":
    unittest.main()
