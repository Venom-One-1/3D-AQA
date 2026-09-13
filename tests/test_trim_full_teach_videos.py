import csv
import tempfile
import unittest
from pathlib import Path

from trim_full_teach_videos import TrimJob, find_trim_jobs, write_manifest


def label(video_id: str, tag_id: int, start: int) -> dict[str, str]:
    return {
        "URLID": video_id,
        "URL": f"{video_id}.mp4",
        "TagID": str(tag_id),
        "Tag": "",
        "Start": str(start),
        "End": str(start),
        "State": "1",
    }


class TrimFullTeachVideoTests(unittest.TestCase):
    def test_boundary_text_repairs_stale_tag_id(self):
        first = label("teacher", 1, 5)
        first["Tag"] = "起势"
        last = label("teacher", 1, 268)
        last["Tag"] = "收势"

        jobs = find_trim_jobs(
            [first, last],
            Path("/missing"),
            Path("/output"),
            require_complete_24=False,
        )

        self.assertEqual(len(jobs), 1)
        self.assertEqual((jobs[0].start_time, jobs[0].end_time), (5.0, 268.0))

    def test_boundary_only_accepts_first_and_last_move_labels(self):
        rows = [label("teacher", 1, 5), label("teacher", 24, 305)]

        jobs = find_trim_jobs(
            rows,
            Path("/missing"),
            Path("/output"),
            require_complete_24=False,
        )

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].start_time, 5.0)
        self.assertEqual(jobs[0].end_time, 305.0)

    def test_complete_mode_still_rejects_partial_annotations(self):
        rows = [label("teacher", 1, 5), label("teacher", 24, 305)]

        jobs = find_trim_jobs(rows, Path("/missing"), Path("/output"))

        self.assertEqual(jobs, [])

    def test_manifest_update_keeps_previous_video_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = TrimJob(
                video_id="old",
                source_video=Path("old.mp4"),
                output_video=root / "old.mp4",
                start_time=0.0,
                end_time=10.0,
                duration=10.0,
                move_count=24,
                source_duration=12.0,
                status="trimmed",
            )
            new = TrimJob(
                video_id="new",
                source_video=Path("new.mp4"),
                output_video=root / "new.mp4",
                start_time=1.0,
                end_time=11.0,
                duration=10.0,
                move_count=2,
                source_duration=13.0,
                status="trimmed",
            )
            write_manifest(root, Path("old.txt"), [old])
            write_manifest(root, Path("new.txt"), [new])

            with (root / "trim_manifest.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["video_id"] for row in rows}, {"old", "new"})


if __name__ == "__main__":
    unittest.main()
