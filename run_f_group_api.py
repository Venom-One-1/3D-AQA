"""Batch frozen F-group cases through Bailian; preserve first model answers."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import getpass
import json
import os
from pathlib import Path
import shutil
import time
from urllib.parse import urlsplit
import uuid

import requests

from run_f_group_experiment import PROJECT, digest, strict_json, validate, write_json


def validate_base_url(value):
    url = urlsplit(value)
    host = url.hostname or ""
    if (url.scheme != "https" or url.username or url.password or url.query or url.fragment
            or url.port not in (None, 443)
            or not (host == "dashscope.aliyuncs.com" or host.endswith(".maas.aliyuncs.com"))
            or url.path.rstrip("/") != "/compatible-mode/v1"):
        raise ValueError("Use an official Bailian HTTPS compatible-mode/v1 Base URL")
    return value.rstrip("/")


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def freeze_experiment(source, output, config):
    manifest = strict_json(source / "experiment_manifest.json")
    files = {"F_prompt.md": manifest["prompt_sha256"],
             "evaluation/baseline_snapshot.json": manifest["baseline_sha256"],
             "experiment_manifest.json": digest(source / "experiment_manifest.json"),
             "case_index.csv": digest(source / "case_index.csv")}
    files.update({c["input_file"]: c["input_sha256"] for c in manifest["cases"]})
    for name, checksum in files.items():
        if digest(source / name) != checksum:
            raise ValueError(f"Frozen source changed: {name}")
    config = dict(config, source_files_sha256=files)
    config_path = output / "api_config.json"
    if config_path.exists():
        if strict_json(config_path) != config:
            raise ValueError("Run settings changed; use a new output directory")
        for name, checksum in files.items():
            if digest(output / name) != checksum:
                raise ValueError(f"Frozen run input changed: {name}")
    else:
        if any(p.name != ".run.lock" for p in output.iterdir()):
            raise ValueError("Output must be empty for a new API run")
        for name in files:
            dest = output / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, dest)
        (output / "responses").mkdir()
        atomic_json(config_path, config)
    return manifest


def request_payload(config, prompt, case_text):
    payload = {k: config[k] for k in
               ("model", "temperature", "top_p", "max_tokens", "enable_thinking")}
    if config["enable_thinking"]:
        payload["thinking_budget"] = config["thinking_budget"]
    payload.update(enable_search=False, stream=True, stream_options={"include_usage": True},
                   messages=[{"role": "system", "content": prompt},
                             {"role": "user", "content": case_text}])
    return payload


def read_stream(lines, raw, redact=lambda text: text):
    parts, usage, model, request_id, finish = [], {}, None, None, None
    done = False
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        raw.write(redact(line) + "\n")
        raw.flush()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        chunk = json.loads(data)
        if chunk.get("error"):
            raise ValueError("API error in stream; inspect raw.sse")
        usage = chunk.get("usage") or usage
        model = chunk.get("model") or model
        request_id = chunk.get("id") or request_id
        for choice in chunk.get("choices", []):
            if choice.get("index", 0) == 0:
                parts.append(choice.get("delta", {}).get("content") or "")
                finish = choice.get("finish_reason") or finish
    return {"content": "".join(parts), "usage": usage, "model": model,
            "request_id": request_id, "finish_reason": finish, "stream_done": done}


def run_case(entry, output, config, key, retries=2, retry_failed=False):
    case_id = entry["case_id"]
    folder = output / "api_calls" / case_id
    folder.mkdir(parents=True, exist_ok=True)
    status_file = folder / "status.json"
    answer_file = output / entry["response_file"]
    previous = strict_json(status_file) if status_file.exists() else None
    if previous:
        if previous["status"] == "completed":
            if digest(answer_file) != previous["response_sha256"]:
                raise ValueError(f"Saved response changed: {case_id}")
            return previous
        if not retry_failed:
            return previous
    if answer_file.exists():
        raise ValueError(f"Untracked response exists; will not overwrite: {case_id}")
    payload = request_payload(config, (output / "F_prompt.md").read_text(encoding="utf-8"),
                              (output / entry["input_file"]).read_text(encoding="utf-8"))
    redact = lambda text: text.replace(key, "[REDACTED]")
    record = {"case_id": case_id, "status": "started",
              "attempts": previous["attempts"] if previous else [],
              "started_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(status_file, record)
    for attempt in range(retries + 1):
        attempt_dir = folder / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8])
        attempt_dir.mkdir()
        write_json(attempt_dir / "request.json", payload)
        start = time.monotonic()
        info = {"directory": str(attempt_dir.relative_to(output)), "attempt": len(record["attempts"]) + 1}
        retry = False
        try:
            with requests.post(config["base_url"] + "/chat/completions", json=payload,
                               headers={"Authorization": "Bearer " + key}, stream=True,
                               timeout=(20, config["timeout_seconds"]), allow_redirects=False) as response:
                info["http_status"] = response.status_code
                if response.status_code != 200:
                    (attempt_dir / "error.txt").write_text(redact(response.text), encoding="utf-8")
                    retry = response.status_code in (429, 500, 502, 503, 504)
                    info["error"] = f"HTTP {response.status_code}"
                else:
                    with (attempt_dir / "raw.sse").open("w", encoding="utf-8") as raw:
                        result = read_stream(response.iter_lines(), raw, redact)
                    info.update({k: v for k, v in result.items() if k != "content"})
                    content = redact(result["content"])
                    (attempt_dir / "answer.txt").write_text(content, encoding="utf-8")
                    if not result["stream_done"] or result["finish_reason"] != "stop" or not content.strip():
                        info["error"] = "Incomplete or refused response; not automatically retried"
                    else:
                        # Malformed JSON is also an outcome, never silently regenerate it.
                        with answer_file.open("x", encoding="utf-8") as handle:
                            handle.write(content)
                        record.update(status="completed", response_sha256=digest(answer_file))
        except (requests.RequestException, ValueError, OSError) as exc:
            # Transport failure may already be billed; require an explicit retry.
            info["error"] = type(exc).__name__
        info["elapsed_seconds"] = round(time.monotonic() - start, 3)
        write_json(attempt_dir / "metadata.json", info)
        record["attempts"].append(info)
        atomic_json(status_file, record)
        if record["status"] == "completed":
            break
        if not retry or attempt == retries:
            record["status"] = "failed"
            break
        time.sleep(min(60, 2 ** (attempt + 1)))
    atomic_json(status_file, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=PROJECT / "f_group_experiment/first3_five_students")
    parser.add_argument("--output-root", type=Path, default=PROJECT / "f_group_experiment/first3_five_students_api")
    parser.add_argument("--base-url", default=os.environ.get("DASHSCOPE_BASE_URL"))
    parser.add_argument("--model", default="qwen3.8-max")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--thinking-budget", type=int, default=4000)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.base_url:
        parser.error("Set --base-url or DASHSCOPE_BASE_URL from your Bailian console")
    base_url = validate_base_url(args.base_url)
    if (not 1 <= args.workers <= 8 or not 0 <= args.retries <= 5
            or not 0 <= args.temperature < 2 or not 0 < args.top_p <= 1
            or args.max_tokens <= 0 or args.thinking_budget <= 0 or args.timeout_seconds <= 0):
        parser.error("Invalid sampling, concurrency, retry or timeout setting")
    config = {"base_url": base_url, "model": args.model, "temperature": args.temperature,
              "top_p": args.top_p, "max_tokens": args.max_tokens,
              "enable_thinking": not args.no_thinking, "thinking_budget": args.thinking_budget,
              "timeout_seconds": args.timeout_seconds, "enable_search": False,
              "transport": "chat_completions_sse", "runner_sha256": digest(Path(__file__))}
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = freeze_experiment(args.experiment_root, args.output_root, config)
        entries = manifest["cases"]
        if args.case_id:
            unknown = set(args.case_id) - {e["case_id"] for e in entries}
            if unknown:
                parser.error(f"Unknown case IDs: {sorted(unknown)}")
            entries = [e for e in entries if e["case_id"] in args.case_id]
        if not entries:
            parser.error("No cases selected")
        if args.dry_run:
            print(f"Prepared {len(entries)} cases; no API requests sent.")
            return
        key = os.environ.get("DASHSCOPE_API_KEY") or getpass.getpass("DASHSCOPE_API_KEY (not saved): ")
        if not key.strip():
            parser.error("API key is empty")
        first = run_case(entries[0], args.output_root, config, key, args.retries, args.retry_failed)
        print(f"{first['case_id']}: {first['status']}", flush=True)
        if first["status"] == "completed":
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(run_case, e, args.output_root, config, key,
                                           args.retries, args.retry_failed) for e in entries[1:]]
                for future in as_completed(futures):
                    r = future.result()
                    print(f"{r['case_id']}: {r['status']}", flush=True)
        validation_dir = args.output_root / ("validation_" + uuid.uuid4().hex[:8])
        report = validate(args.output_root, validation_dir)
        statuses = [strict_json(p) for p in sorted((args.output_root / "api_calls").glob("*/status.json"))]
        summary = {"completed": sum(r["status"] == "completed" for r in statuses),
                   "failed": sum(r["status"] != "completed" for r in statuses),
                   "expected": len(manifest["cases"]), "validation_dir": str(validation_dir),
                   "automatic_error_count": len(report["automatic_errors"]),
                   "diagnosis_accuracy": None}
        atomic_json(args.output_root / "api_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if summary["failed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
