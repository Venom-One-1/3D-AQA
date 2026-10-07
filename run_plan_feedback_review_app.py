"""Serve new-method/F feedback review without changing either experiment."""

import argparse
import fcntl
from http.server import ThreadingHTTPServer
from pathlib import Path
import secrets
from urllib.parse import unquote, urlparse

from aqa3d.plan_feedback_review import PlanFeedbackReviewStore
from run_f_group_review_app import ComparisonHandler

PROJECT = Path(__file__).resolve().parent
STATIC = PROJECT/"plan_feedback_review_static"


class PlanReviewHandler(ComparisonHandler):
    def _send_bytes(self, data, content_type, status=200, extra=None):
        if self.command == "HEAD":
            self._headers(status, content_type, len(data), extra)
        else:
            super()._send_bytes(data, content_type, status, extra)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        assets = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                  "/review.css": ("review.css", "text/css; charset=utf-8")}
        if path in assets:
            name, mime = assets[path]
            self._send_bytes((STATIC/name).read_bytes(), mime)
        elif path == "/api/export":
            self._send_bytes(self.store.export_csv(), "text/csv; charset=utf-8",
                             extra={"Content-Disposition": 'attachment; filename="plan_feedback_f_review.csv"'})
        else:
            super().do_GET()


def create_server(store, host="127.0.0.1", port=8505):
    handler = type("BoundPlanReviewHandler", (PlanReviewHandler,), {"store": store, "token": secrets.token_urlsafe(32)})
    return ThreadingHTTPServer((host, port), handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=PROJECT/"plan_feedback_results/first3_five_students_v1")
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8505)
    args = parser.parse_args()
    state = args.state_path or args.experiment_root/"ui_comparison/review_progress.json"
    if state.resolve().is_relative_to(args.experiment_root.resolve()) and not state.resolve().is_relative_to(args.experiment_root.resolve()/"ui_comparison"):
        parser.error("Inside the experiment, --state-path must be under ui_comparison/")
    state.parent.mkdir(parents=True, exist_ok=True)
    with state.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(2, f"Review server already using {state}. Open the existing server; do not delete its lock file.\n")
        store = PlanFeedbackReviewStore(args.experiment_root, state)
        try:
            server = create_server(store, args.host, args.port)
        except OSError as exc:
            parser.exit(2, f"Cannot bind {args.host}:{args.port}: {exc}. Use an unused --port.\n")
        print(f"{len(store.cases)} cases / {len(store.records)} review items: http://{args.host}:{server.server_port}", flush=True)
        print(f"Review state: {state}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
