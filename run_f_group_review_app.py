"""Local UI for paired Qwen F-group and frozen rule-baseline review."""

import argparse
import fcntl
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import secrets
from urllib.parse import urlparse, unquote

from aqa3d.f_group_review import ComparisonStore
from run_feedback_review_app import ReviewRequestHandler

PROJECT = Path(__file__).resolve().parent
STATIC = PROJECT / "f_group_review_static"


class ComparisonHandler(ReviewRequestHandler):
    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        assets = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                  "/styles.css": ("styles.css", "text/css; charset=utf-8"),
                  "/lucide.js": ("lucide.js", "text/javascript; charset=utf-8")}
        if path in assets:
            name, content_type = assets[path]
            self._send_bytes((STATIC / name).read_bytes(), content_type)
        elif path == "/api/data":
            self._send_json(dict(self.store.client_payload(), token=self.token))
        elif path == "/api/export":
            self._send_bytes(self.store.export_csv(), "text/csv; charset=utf-8",
                             extra={"Content-Disposition": 'attachment; filename="f_group_review.csv"'})
        else:
            super().do_GET()

    def do_POST(self):
        if self.path != "/api/review":
            self.send_error(404)
            return
        if self.headers.get("X-Review-Token") != self.token:
            self.send_error(403)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1048576:
                raise ValueError("Invalid request size")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Expected object")
            self._send_json(self.store.update_many(data.get("updates"), data.get("revision")))
        except (ValueError, TypeError, AttributeError) as exc:
            self._send_json({"error": str(exc)}, 400)
        except OSError:
            self._send_json({"error": "Cannot write review state; changes were not saved"}, 500)


def create_server(store, host="127.0.0.1", port=8502):
    handler = type("BoundComparisonHandler", (ComparisonHandler,), {"store": store, "token": secrets.token_urlsafe(32)})
    return ThreadingHTTPServer((host, port), handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=PROJECT / "f_group_experiment/first3_five_students_api")
    parser.add_argument("--metric-root", type=Path, default=PROJECT / "endpoint_metric_results/first3_five_students")
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8502)
    args = parser.parse_args()
    state = args.state_path or args.experiment_root / "ui_review/review_progress.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    with state.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(2, "A review server is already using this annotation state:\n"
                        f"  {state}\n"
                        "Open the existing server URL, or stop that server before restarting.\n"
                        "Changing --port does not release the annotation lock. "
                        "Do not delete the lock file while the server is running.\n")
        store = ComparisonStore(args.experiment_root, args.metric_root, state)
        server = create_server(store, args.host, args.port)
        print(f"{len(store.cases)} cases / {len(store.records)} review items: http://{args.host}:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
