"""Read-only grouped judge-human agreement analysis of an exported review CSV."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from aqa3d.check_plan_review_analysis import DERIVED_FIELDS, classify, read_review, report_markdown, summarize

PROJECT = Path(__file__).resolve().parent


def analyze(review_csv: Path, output: Path, prior_csv: Path | None = None) -> dict:
    review_csv, output = Path(review_csv).resolve(), Path(output).resolve()
    snapshot = review_csv.read_bytes()
    fields, source_rows = read_review(review_csv)
    prior = None
    hashes = {str(review_csv): hashlib.sha256(snapshot).hexdigest()}
    if prior_csv is not None:
        prior_csv = Path(prior_csv).resolve()
        prior_snapshot = prior_csv.read_bytes()
        _, previous = read_review(prior_csv, require_judge=False)
        prior = {r["record_id"]: r for r in previous}
        for row in source_rows:
            old = prior.get(row["record_id"])
            if old and any(row[k] != old[k] for k in ("case_id", "move_id", "clause_id", "source_text")):
                raise ValueError("Historical row identity changed; use matching prior data")
        hashes[str(prior_csv)] = hashlib.sha256(prior_snapshot).hexdigest()
    rows = [classify(row, prior) for row in source_rows]
    summary = summarize(rows)
    summary["source_sha256"] = hashes
    summary["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary["library_sha256"] = hashlib.sha256((PROJECT / "aqa3d/check_plan_review_analysis.py").read_bytes()).hexdigest()
    for filename, digest in hashes.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
            raise ValueError("Input changed during analysis")
    # Refuse to replace an earlier analysis or any existing annotation directory.
    output.mkdir(parents=True, exist_ok=False)
    (output / "input_review.csv").write_bytes(snapshot)
    if prior_csv is not None:
        (output / "prior_review.csv").write_bytes(prior_snapshot)
    for filename, subset in (("agreement_details.csv", rows),
                             ("disagreements.csv", [r for r in rows if r["judge_agreement"] == "disagree"]),
                             ("excluded_cases.csv", [r for r in rows if r["judge_agreement"] == "excluded"])):
        with (output / filename).open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields+DERIVED_FIELDS)
            writer.writeheader()
            for row in subset:
                writer.writerow({k:("'"+v if isinstance(v,str) and v.lstrip().startswith(("=", "+", "-", "@")) else v)
                                 for k,v in row.items()})
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (output / "report.md").write_text(report_markdown(summary, rows), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--prior-review-csv", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT / "check_plan_judge_results/first3_v1_astra/manual_agreement_v1")
    args = parser.parse_args()
    summary = analyze(args.review_csv, args.output_dir, args.prior_review_csv)
    print(json.dumps({"output":str(args.output_dir), "groups":summary["groups"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
