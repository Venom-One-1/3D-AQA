from __future__ import annotations

import argparse
import json
import mimetypes
import re
import shutil
import sys
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from temporal_query_annotator.domain import ValidationError
    from temporal_query_annotator.llm import (
        OpenAICompatibleProvider,
        build_user_prompt,
        config_from_environment,
        validate_candidate,
    )
    from temporal_query_annotator.project import AnnotationProject
else:
    from .domain import ValidationError
    from .llm import (
        OpenAICompatibleProvider,
        build_user_prompt,
        config_from_environment,
        validate_candidate,
    )
    from .project import AnnotationProject


STATIC_ROOT = Path(__file__).resolve().parent / "static"
FRAME_NAME_RE = re.compile(r"^\d{5}\.jpg$")


class ServerContext:
    def __init__(self, project: AnnotationProject, provider: OpenAICompatibleProvider):
        self.project = project
        self.provider = provider


class AnnotationHandler(BaseHTTPRequestHandler):
    server_version = "TemporalQueryAnnotator/0.1"
    context: ServerContext

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

    def do_GET(self) -> None:
        try:
            path = urllib.parse.urlparse(self.path).path
            if path == "/api/project":
                self._json(
                    self.context.project.public_project(
                        self.context.provider.config.public_status()
                    )
                )
            elif path == "/api/annotations":
                self._json(self.context.project.annotations())
            elif path.startswith("/api/prompt/"):
                technique_id = urllib.parse.unquote(path.removeprefix("/api/prompt/"))
                task = self.context.project.task_by_id[technique_id]
                self._json(
                    {
                        "system_prompt": __import__(
                            "temporal_query_annotator.llm", fromlist=["SYSTEM_PROMPT"]
                        ).SYSTEM_PROMPT,
                        "user_prompt": build_user_prompt(task, self.context.project.metrics),
                    }
                )
            elif path == "/api/export":
                payload = json.dumps(
                    self.context.project.export_payload(), ensure_ascii=False, indent=2
                ).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="{self.context.project.config["project_id"]}.gold.json"',
                )
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            elif path == "/media/video":
                self._file(self.context.project.paths.video, allow_range=True)
            elif path.startswith("/media/frames/"):
                frame_name = urllib.parse.unquote(path.removeprefix("/media/frames/"))
                if not FRAME_NAME_RE.fullmatch(frame_name):
                    raise FileNotFoundError(frame_name)
                self._file(self.context.project.frame_path(frame_name))
            else:
                self._static(path)
        except KeyError as error:
            self._error(HTTPStatus.NOT_FOUND, f"Unknown resource: {error}")
        except FileNotFoundError as error:
            self._error(HTTPStatus.NOT_FOUND, str(error))
        except Exception as error:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))

    def do_PUT(self) -> None:
        try:
            path = urllib.parse.urlparse(self.path).path
            if not path.startswith("/api/annotations/"):
                self._error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
                return
            technique_id = urllib.parse.unquote(path.removeprefix("/api/annotations/"))
            annotation = self._read_json()
            saved = self.context.project.save_annotation(technique_id, annotation)
            self._json(saved)
        except KeyError:
            self._error(HTTPStatus.NOT_FOUND, "Unknown technique_id")
        except ValidationError as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
        except Exception as error:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))

    def do_POST(self) -> None:
        try:
            path = urllib.parse.urlparse(self.path).path
            if not path.startswith("/api/generate/"):
                self._error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
                return
            technique_id = urllib.parse.unquote(path.removeprefix("/api/generate/"))
            task = self.context.project.task_by_id[technique_id]
            prompt = build_user_prompt(task, self.context.project.metrics)
            if not self.context.provider.config.enabled:
                self._json(
                    {
                        "error": "LLM provider is not configured",
                        "manual_mode": True,
                        "prompt": prompt,
                    },
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            candidate = self.context.provider.generate(prompt)
            validate_candidate(candidate, self.context.project.metric_ids)
            annotation = self.context.project.save_candidate(technique_id, candidate)
            self._json({"candidate": candidate, "annotation": annotation})
        except KeyError:
            self._error(HTTPStatus.NOT_FOUND, "Unknown technique_id")
        except ValidationError as error:
            self._error(HTTPStatus.BAD_GATEWAY, str(error))
        except Exception as error:
            self._error(HTTPStatus.BAD_GATEWAY, str(error))

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValidationError("Invalid Content-Length") from error
        if length <= 0 or length > 2_000_000:
            raise ValidationError("Request body must be between 1 byte and 2 MB")
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValidationError(f"Invalid JSON request: {error}") from error
        if not isinstance(value, dict):
            raise ValidationError("Request body must be a JSON object")
        return value

    def _json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"error": message}, status=status)

    def _static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        candidate = (STATIC_ROOT / relative).resolve()
        if STATIC_ROOT.resolve() not in candidate.parents and candidate != STATIC_ROOT.resolve():
            raise FileNotFoundError(relative)
        if not candidate.is_file():
            raise FileNotFoundError(relative)
        self._file(candidate)

    def _file(self, path: Path, allow_range: bool = False) -> None:
        size = path.stat().st_size
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        start = 0
        end = size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range") if allow_range else None
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            if match.group(1):
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else end
            elif match.group(2):
                suffix = int(match.group(2))
                start = max(0, size - suffix)
            if start > end or start >= size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            end = min(end, size - 1)
            status = HTTPStatus.PARTIAL_CONTENT

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Annotate Gold-Standard temporal queries on a 5 FPS video grid."
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=root / "projects" / "bv1we411w7jb_first3.json",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--llm-api-url")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-api-key-env", default="TQA_LLM_API_KEY")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = AnnotationProject(args.project)
    llm_config = config_from_environment(
        api_url=args.llm_api_url,
        model=args.llm_model,
        api_key_env=args.llm_api_key_env,
    )
    provider = OpenAICompatibleProvider(llm_config)
    context = ServerContext(project, provider)
    handler = type("ConfiguredAnnotationHandler", (AnnotationHandler,), {"context": context})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Temporal Query Annotator: http://{args.host}:{args.port}")
    print(f"Project: {project.config['title']} ({len(project.tasks)} tasks)")
    print(f"Annotations: {project.paths.annotations}")
    if not llm_config.enabled:
        print("LLM: manual import mode (configure TQA_LLM_* variables to enable generation)")
    else:
        print(f"LLM: {llm_config.model}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
