import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = PROJECT_ROOT / "technique_feedback_rules_first3.json"
FULL_RULES_PATH = PROJECT_ROOT / "technique_feedback_rules.json"


class TechniqueFeedbackRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))

    def test_contains_expected_moves_and_keypose_frames(self) -> None:
        frames = {
            move["move_name"]: [pose["frame"] for pose in move["keyposes"]]
            for move in self.rules["moves"]
        }
        self.assertEqual(
            frames,
            {
                "qishi": [4, 13, 17, 21],
                "yemafenzong": [13, 16, 19, 22, 25, 27, 32, 34, 38, 41, 45],
                "baiheliangchi": [14, 19, 26],
            },
        )
        self.assertEqual(sum(len(items) for items in frames.values()), 18)

    def test_all_checks_reference_defined_metrics(self) -> None:
        metric_ids = set(self.rules["metric_definitions"])
        check_ids: set[str] = set()
        reference_keys: set[str] = set()
        for move in self.rules["moves"]:
            for pose in move["keyposes"]:
                self.assertEqual(
                    pose["image"],
                    f"{move['keypose_dir']}/{pose['frame']:05d}.jpg",
                )
                for check in pose["checks"]:
                    self.assertIn(check["metric_id"], metric_ids)
                    self.assertTrue(check["reference_key"].startswith(f"{pose['pose_id']}:"))
                    self.assertTrue({"below", "above"} & set(check["feedback"]))
                    self.assertNotIn(check["check_id"], check_ids)
                    self.assertNotIn(check["reference_key"], reference_keys)
                    check_ids.add(check["check_id"])
                    reference_keys.add(check["reference_key"])

    def test_distance_metrics_are_normalized(self) -> None:
        for metric_id, metric in self.rules["metric_definitions"].items():
            metric_type = metric["type"]
            if "distance" not in metric_type:
                continue
            self.assertTrue(
                metric_type.startswith("normalized_"),
                f"{metric_id} uses an unnormalized distance type: {metric_type}",
            )
            self.assertIn("normalizer", metric, metric_id)

    def test_teacher_reference_policy_preserves_raw_values(self) -> None:
        policy = self.rules["teacher_reference_policy"]
        self.assertEqual(policy["teacher_count"], 10)
        self.assertEqual(len(policy["teacher_ids"]), 10)
        self.assertEqual(len(set(policy["teacher_ids"])), 10)
        self.assertTrue(policy["raw_values_required"])
        self.assertEqual(policy["default_interval"], "median_plus_or_minus_2mad")
        self.assertEqual(policy["secondary_interval"], "p10_p90")


class FullTechniqueFeedbackRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rules = json.loads(FULL_RULES_PATH.read_text(encoding="utf-8"))
        cls.first_three = json.loads(RULES_PATH.read_text(encoding="utf-8"))

    def test_contains_all_24_moves_and_preserves_reviewed_first_three(self) -> None:
        self.assertEqual(
            [move["move_id"] for move in self.rules["moves"]],
            list(range(1, 25)),
        )
        self.assertEqual(self.rules["moves"][:3], self.first_three["moves"])
        self.assertEqual(len(self.rules["moves"]), 24)

    def test_keypose_frames_match_annotation_order(self) -> None:
        expected = {
            1: [4, 13, 17, 21],
            2: [13, 16, 19, 22, 25, 27, 32, 34, 38, 41, 45],
            3: [14, 19, 26],
            4: [16, 18, 21, 26, 28, 32, 35, 39, 42, 45, 48, 52],
            5: [10, 13, 17],
            6: [9, 12, 15, 19, 22, 25, 29, 32, 36, 39, 43, 46],
            7: [12, 15, 18, 22, 26, 31, 34, 38, 43, 49],
            8: [11, 16, 20, 24, 27, 31, 36, 39, 43, 49, 55],
            9: [13, 19, 25, 29, 33],
            10: [17, 23, 27, 33, 38, 43],
            11: [12, 16, 20],
            12: [14, 19, 24],
            13: [12, 16, 24, 27],
            14: [None, None],
            15: [13, 18, 23, 27],
            16: [None, None, None, None],
            17: [13, 17, 22, 26, 31, 37],
            18: [12, 16, 20, 24, 28, 32, 36, 40],
            19: [15, 20, 26],
            20: [9, 13, 17],
            21: [14, 19, 23, 29, 35, 39],
            22: [13, 17, 24],
            23: [12, 17, 21, 27],
            24: [10, 16, 22],
        }
        actual = {
            move["move_id"]: [pose["frame"] for pose in move["keyposes"]]
            for move in self.rules["moves"]
        }
        self.assertEqual(actual, expected)
        self.assertEqual(sum(len(frames) for frames in actual.values()), 133)
        self.assertEqual(
            sum(frame is not None for frames in actual.values() for frame in frames),
            127,
        )

    def test_missing_keypose_images_are_explicit(self) -> None:
        missing_moves = [
            move["move_id"]
            for move in self.rules["moves"]
            if move.get("keypose_status") == "missing_keypose_images"
        ]
        self.assertEqual(missing_moves, [14, 16])
        for move in self.rules["moves"]:
            for pose in move["keyposes"]:
                if pose["frame"] is None:
                    self.assertIsNone(pose["image"])
                    self.assertEqual(
                        pose["binding_status"],
                        "missing_keypose_image_and_frame",
                    )

    def test_every_check_has_a_unique_valid_metric_reference(self) -> None:
        metric_ids = set(self.rules["metric_definitions"])
        check_ids: set[str] = set()
        reference_keys: set[str] = set()
        check_count = 0
        for move in self.rules["moves"]:
            self.assertTrue(move["keyposes"])
            for pose in move["keyposes"]:
                self.assertTrue(pose["checks"], pose["pose_id"])
                if pose["frame"] is not None:
                    self.assertEqual(
                        pose["image"],
                        f"{move['keypose_dir']}/{pose['frame']:05d}.jpg",
                    )
                for item in pose["checks"]:
                    self.assertIn(item["metric_id"], metric_ids)
                    self.assertEqual(
                        item["reference_key"],
                        f"{pose['pose_id']}:{item['metric_id']}",
                    )
                    self.assertTrue({"below", "above"} & set(item["feedback"]))
                    self.assertNotIn(item["check_id"], check_ids)
                    self.assertNotIn(item["reference_key"], reference_keys)
                    check_ids.add(item["check_id"])
                    reference_keys.add(item["reference_key"])
                    check_count += 1
        self.assertEqual(check_count, 786)

    def test_all_distance_metrics_are_normalized(self) -> None:
        for metric_id, metric in self.rules["metric_definitions"].items():
            if "distance" not in metric["type"]:
                continue
            self.assertTrue(metric["type"].startswith("normalized_"), metric_id)
            self.assertIn("normalizer", metric, metric_id)


if __name__ == "__main__":
    unittest.main()
