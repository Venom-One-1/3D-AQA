import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from run_f_group_api import read_stream, request_payload, run_case, validate_base_url


class APIBatchTests(unittest.TestCase):
    def config(self):
        return dict(model="qwen3.8-max", temperature=0.2, top_p=0.8, max_tokens=16384,
                    enable_thinking=True, thinking_budget=4000, timeout_seconds=10,
                    base_url="https://example.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")

    def test_url_rejects_redirect_targets_and_credentials(self):
        for url in ["https://evil.com/compatible-mode/v1", "http://dashscope.aliyuncs.com/compatible-mode/v1",
                    "https://dashscope.aliyuncs.com.evil.com/compatible-mode/v1",
                    "https://key@dashscope.aliyuncs.com/compatible-mode/v1"]:
            with self.assertRaises(ValueError):
                validate_base_url(url)
        self.assertEqual(validate_base_url(self.config()["base_url"]), self.config()["base_url"])

    def test_payload_frozen_text_only(self):
        p = request_payload(self.config(), "prompt", '{"case_id":"F001"}')
        self.assertFalse(p["enable_search"])
        self.assertEqual(p["messages"][1]["content"], '{"case_id":"F001"}')
        self.assertEqual(len(p["messages"]), 2)
        self.assertEqual(p["thinking_budget"], 4000)
        self.assertNotIn("Authorization", p)

    def test_stream_keeps_reasoning_separate_and_usage(self):
        raw = io.StringIO()
        lines = [b'data: {"choices":[{"delta":{"reasoning_content":"private thought"}}]}',
                 b'data: {"choices":[{"delta":{"content":"{}"},"finish_reason":"stop"}]}',
                 b'data: {"choices":[],"usage":{"total_tokens":123}}', b'data: [DONE]']
        result = read_stream(lines, raw)
        self.assertEqual(result["content"], "{}")
        self.assertEqual(result["usage"]["total_tokens"], 123)
        self.assertTrue(result["stream_done"])
        self.assertIn("private thought", raw.getvalue())
        self.assertFalse(read_stream(lines[:-1], io.StringIO())["stream_done"])

    def setup_case(self, root):
        (root / "responses").mkdir()
        (root / "inputs").mkdir()
        (root / "F_prompt.md").write_text("prompt")
        (root / "inputs/F001.json").write_text('{"case_id":"F001"}')
        return {"case_id": "F001", "input_file": "inputs/F001.json", "response_file": "responses/F001.json"}

    def response(self, status=200):
        r = MagicMock()
        r.__enter__.return_value = r
        r.status_code = status
        r.text = "key-secret"
        r.iter_lines.return_value = iter([
            b'data: {"choices":[{"delta":{"content":"not JSON"},"finish_reason":"stop"}]}',
            b'data: [DONE]'])
        return r

    def test_malformed_answer_preserved_and_resume_skips(self):
        with tempfile.TemporaryDirectory() as tmp, patch("run_f_group_api.requests.post") as post:
            root = Path(tmp)
            e = self.setup_case(root)
            post.return_value = self.response()
            self.assertEqual(run_case(e, root, self.config(), "key-secret")["status"], "completed")
            run_case(e, root, self.config(), "key-secret")
            self.assertEqual(post.call_count, 1)
            self.assertEqual((root / e["response_file"]).read_text(), "not JSON")
            (root / e["response_file"]).write_text("edited")
            with self.assertRaises(ValueError):
                run_case(e, root, self.config(), "key-secret")

    def test_auth_error_not_retried_and_key_redacted(self):
        with tempfile.TemporaryDirectory() as tmp, patch("run_f_group_api.requests.post") as post:
            root = Path(tmp)
            e = self.setup_case(root)
            post.return_value = self.response(401)
            self.assertEqual(run_case(e, root, self.config(), "key-secret")["status"], "failed")
            self.assertEqual(post.call_count, 1)
            for p in root.rglob("*"):
                if p.is_file():
                    self.assertNotIn("key-secret", p.read_text())

    def test_rate_limit_retries_bounded(self):
        with tempfile.TemporaryDirectory() as tmp, patch("run_f_group_api.requests.post") as post, patch("run_f_group_api.time.sleep"):
            root = Path(tmp)
            e = self.setup_case(root)
            post.return_value = self.response(429)
            r = run_case(e, root, self.config(), "key-secret", retries=1)
            self.assertEqual(post.call_count, 2)
            self.assertEqual(len(r["attempts"]), 2)
            self.assertEqual(r["status"], "failed")


if __name__ == "__main__":
    unittest.main()
