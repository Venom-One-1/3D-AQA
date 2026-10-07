"""Freeze unchanged F-group cases for the no-metric-comparison ablation."""

import argparse
import copy
import shutil
from pathlib import Path

from run_f_group_experiment import PROJECT, digest, fresh_directory, strict_json, write_csv, write_json


def prepare(source, output, prompt):
    manifest = strict_json(source / "experiment_manifest.json")
    if digest(source / "F_prompt.md") != manifest["prompt_sha256"]:
        raise ValueError("Original prompt changed")
    baseline = source / "evaluation/baseline_snapshot.json"
    if digest(baseline) != manifest["baseline_sha256"]:
        raise ValueError("Original baseline changed")
    selected = [copy.deepcopy(c) for c in manifest["cases"] if c["student_id"] in ("01", "02", "10")]
    expected = {(sid, mid) for sid in ("01", "02", "10") for mid in (1, 2, 3)}
    if len(selected) != 9 or {(c["student_id"], c["move_id"]) for c in selected} != expected:
        raise ValueError("Expected exactly the first three moves of students 01, 02, 10")
    for entry in selected:
        if digest(source / entry["input_file"]) != entry["input_sha256"]:
            raise ValueError(f"Original case changed: {entry['case_id']}")
    prompt.read_text(encoding="utf-8")
    fresh_directory(output)
    for name in ("inputs", "responses", "evaluation"):
        (output / name).mkdir()
    shutil.copy2(prompt, output / "F_prompt.md")
    shutil.copy2(baseline, output / "evaluation/baseline_snapshot.json")
    for entry in selected:
        shutil.copy2(source / entry["input_file"], output / entry["input_file"])
    write_csv(output / "case_index.csv", selected, list(selected[0]))
    child = copy.deepcopy(manifest)
    child.update(cases=selected, case_count=len(selected), metric_count=sum(c["metric_count"] for c in selected),
                 expected_model_label="qwen3.8-max-0902", require_metric_comparisons=False,
                 prompt_sha256=digest(output / "F_prompt.md"),
                 ablation={"name": "no_explicit_metric_comparison", "parent": str(source.resolve()),
                           "parent_manifest_sha256": digest(source / "experiment_manifest.json"),
                           "parent_prompt_sha256": manifest["prompt_sha256"],
                           "inputs_byte_identical": True,
                           "model_confound": "Previous model alias qwen3.8-max is not guaranteed identical to qwen3.8-max-0902"})
    write_json(output / "experiment_manifest.json", child)
    (output / "README.md").write_text(
        "# F-group ablation: no explicit metric comparisons\n\n"
        "Nine unchanged inputs: students 01, 02, 10; moves 1, 2, 3.\n"
        "Keep findings and coach_summary; remove metric_comparisons and explicit per-metric comparison instructions.\n"
        "Teacher ranges, raw samples and all student measurements remain unchanged.\n"
        "Run with run_f_group_api.py using a separate output root and model qwen3.8-max-0902.\n"
        "Keep enable_thinking=true, thinking_budget=4000, temperature=0.2, top_p=0.8, max_tokens=16384.\n"
        "First run --case-id F001; verify success, then run all cases (the completed F001 is reused).\n"
        "Do not infer free billing from HTTP success; check the provider's quota/billing console.\n"
        "The previous alias may differ from this snapshot: not a strict prompt-only causal comparison.\n"
        "Format/evidence-ID validation is not a diagnosis or numerical-claim accuracy assessment.\n",
        encoding="utf-8")
    return child


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=PROJECT / "f_group_experiment/first3_five_students")
    parser.add_argument("--output-root", type=Path, default=PROJECT / "f_group_experiment/no_metric_comparison_01_02_10")
    parser.add_argument("--prompt", type=Path, default=PROJECT / "prompt/f_group_no_metric_comparison.md")
    args = parser.parse_args()
    result = prepare(args.source_root, args.output_root, args.prompt)
    print(f"Prepared {result['case_count']} cases; no API requests sent.")


if __name__ == "__main__":
    main()
