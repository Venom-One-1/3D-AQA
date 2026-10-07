import json
from pathlib import Path
import tempfile
import unittest

from prepare_f_group_ablation import prepare
from run_f_group_experiment import digest, write_json


class AblationPreparationTests(unittest.TestCase):
    def test_subset_preserves_inputs_and_protects_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            (source / "inputs").mkdir(parents=True)
            (source / "evaluation").mkdir()
            (source / "F_prompt.md").write_text("Original prompt")
            baseline = source / "evaluation/baseline_snapshot.json"
            write_json(baseline, [])
            entries = []
            for sid in ("01", "02", "03", "04", "10"):
                for mid in (1, 2, 3):
                    cid = f"F{len(entries)+1:03d}"
                    path = source / f"inputs/{cid}.json"
                    write_json(path, {"case_id": cid, "unchanged": [1, 2, 3]})
                    entries.append({"case_id": cid, "student_id": sid, "move_id": mid,
                        "metric_count": 3, "input_file": f"inputs/{cid}.json",
                        "response_file": f"responses/{cid}.json", "input_sha256": digest(path)})
            write_json(source / "experiment_manifest.json", {"cases": entries,
                "prompt_sha256": digest(source / "F_prompt.md"), "baseline_sha256": digest(baseline)})
            before = {str(p): digest(p) for p in source.rglob("*") if p.is_file()}
            prompt = root / "new.md"
            prompt.write_text("New prompt")
            output = root / "output"
            result = prepare(source, output, prompt)
            self.assertEqual(result["case_count"], 9)
            self.assertFalse(result["require_metric_comparisons"])
            self.assertEqual({c["student_id"] for c in result["cases"]}, {"01", "02", "10"})
            self.assertEqual(result["cases"][-1]["case_id"], "F015")
            for entry in result["cases"]:
                self.assertEqual(digest(output / entry["input_file"]), entry["input_sha256"])
            self.assertEqual(before, {str(p): digest(p) for p in source.rglob("*") if p.is_file()})
            with self.assertRaises(FileExistsError):
                prepare(source, output, prompt)
            (source / "inputs/F001.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Original case changed"):
                prepare(source, root / "changed", prompt)
            self.assertFalse((root / "changed").exists())


if __name__ == "__main__":
    unittest.main()
