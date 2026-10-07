"""Review repaired technique-to-measurement plans in a local browser."""

import argparse
import fcntl
from http.server import ThreadingHTTPServer
from pathlib import Path
import secrets
from urllib.parse import unquote, urlparse

from aqa3d.check_plan_review import PlanReviewStore
from run_f_group_review_app import ComparisonHandler
from run_feedback_review_app import ReviewRequestHandler


PROJECT = Path(__file__).resolve().parent
STATIC = PROJECT / "check_plan_review_static"


class PlanReviewHandler(ComparisonHandler):
    def _send_bytes(self, data, content_type, status=200, extra=None):
        self._headers(status, content_type, len(data), extra)
        if self.command != "HEAD":
            self.wfile.write(data)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        assets = {"/": (STATIC / "index.html", "text/html; charset=utf-8"),
                  "/app.js": (STATIC / "app.js", "text/javascript; charset=utf-8"),
                  "/styles.css": (STATIC / "styles.css", "text/css; charset=utf-8"),
                  "/lucide.js": (PROJECT / "f_group_review_static/lucide.js", "text/javascript; charset=utf-8")}
        try:
            if path in assets:
                file, mime = assets[path]
                self._send_bytes(file.read_bytes(), mime)
            elif path == "/api/data":
                self._send_json(dict(self.store.client_payload(), token=self.token))
            elif path == "/api/export":
                self._send_bytes(self.store.export_csv(), "text/csv; charset=utf-8",
                                 extra={"Content-Disposition": 'attachment; filename="check_plan_review.csv"'})
            else:
                ReviewRequestHandler.do_GET(self)
        except OSError:
            self._send_json({"error": "Cannot read review asset"}, 500)


def create_server(store, host="127.0.0.1", port=8504):
    handler = type("BoundPlanReviewHandler", (PlanReviewHandler,),
                   {"store": store, "token": secrets.token_urlsafe(32)})
    return ThreadingHTTPServer((host, port), handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path,
                        default=PROJECT / "check_plan_experiment/first3_v1_repaired_v1")
    parser.add_argument("--metric-root", type=Path,
                        default=PROJECT / "endpoint_metric_results/first3_five_students")
    parser.add_argument("--state-path", type=Path,
                        default=PROJECT / "check_plan_review_results/first3_v1/review_progress.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--judge-report", type=Path, help="Optional independent pre-review overlay; never replaces human labels")
    parser.add_argument("--port", type=int, default=8504)
    args = parser.parse_args()
    state = args.state_path.resolve()
    if any(state.is_relative_to(p.resolve()) for p in (args.experiment_root, args.metric_root)):
        parser.error("--state-path must be outside the frozen experiment/metric directories")
    state.parent.mkdir(parents=True, exist_ok=True)
    with state.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(2, f"Review state already in use: {state}\nOpen the existing server. Do not delete the lock file.\n")
        store = PlanReviewStore(args.experiment_root, args.metric_root, state, args.judge_report)
        try:
            server = create_server(store, args.host, args.port)
        except OSError as exc:
            parser.exit(2, f"Cannot bind {args.host}:{args.port}: {exc}. Choose another --port.\n")
        print(f"{len(store.cases)} plans / {len(store.records)} clauses: http://{args.host}:{server.server_port}", flush=True)
        print(f"Annotation state: {state}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
