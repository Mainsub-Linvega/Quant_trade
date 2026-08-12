from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_data_release import compare


class AuditDataReleaseTest(unittest.TestCase):
    def test_compare_detects_added_and_modified_files(self) -> None:
        baseline = {"manifest": {"version": 1}, "sample_submission": {"sha256": "a"},
                    "splits": {"train": [{"path": "train/a.parquet", "rows": 10,
                                            "bytes": 100, "sha256": "x"}], "test": []}}
        current = {"manifest": {"version": 2}, "sample_submission": {"sha256": "a"},
                   "splits": {"train": [{"path": "train/a.parquet", "rows": 11,
                                           "bytes": 101, "sha256": "y"},
                                          {"path": "train/b.parquet", "rows": 5,
                                           "bytes": 50, "sha256": "z"}], "test": []}}
        result = compare(baseline, current)
        self.assertTrue(result["changed"])
        self.assertTrue(result["manifest_changed"])
        self.assertEqual(result["splits"]["train"]["added"], ["train/b.parquet"])
        self.assertEqual(result["splits"]["train"]["modified"], ["train/a.parquet"])
        self.assertEqual(result["splits"]["train"]["row_delta"], 6)

    def test_compare_identical_release(self) -> None:
        release = {"manifest": {"version": 1}, "sample_submission": None,
                   "splits": {"train": [], "test": []}}
        result = compare(release, release)
        self.assertFalse(result["changed"])
        self.assertIn("keep caches", result["cache_action"])


if __name__ == "__main__":
    unittest.main()
