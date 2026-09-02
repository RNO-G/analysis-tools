from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from season_report.generate_report import (
    extract_validation_rows,
    remove_temporary_result,
    verify_run_in_combined,
    write_csv_atomic,
)
from season_report.heatmaps import generate_health_heatmaps


class SeasonReportTests(unittest.TestCase):
    def test_extract_combine_and_plot_validation_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "result"
            summary_dir = result / "channel_health_summary"
            summary_dir.mkdir(parents=True)
            source = summary_dir / "validation_summary_station14_run_260080.csv"
            source.write_text(
                "Channel,SNR,Glitching,Channel Health (FORCE)\n"
                "0,OK,OK,OK\n"
                "1,!!,X,X\n",
                encoding="utf-8",
            )
            fields, rows = extract_validation_rows(result, 14, 260080)
            self.assertEqual(fields[:3], ["Station", "Run", "Channel"])
            self.assertEqual(rows[1]["Glitching"], "X")

            combined = root / "channel_health_by_run.csv"
            write_csv_atomic(combined, fields, rows)
            verify_run_in_combined(combined, 260080, 2)
            paths = generate_health_heatmaps(combined, root / "plots")
            self.assertEqual(set(paths), {"SNR", "Glitching", "Channel Health (FORCE)"})
            self.assertTrue(all(path.is_file() for path in paths.values()))

    def test_temporary_result_deletion_is_strictly_scoped(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            expected = base / "26-09-02_station-14_run260080-run260080_123456"
            expected.mkdir()
            remove_temporary_result(expected, base, 14, 260080)
            self.assertFalse(expected.exists())

            unexpected = base / "unrelated"
            unexpected.mkdir()
            with self.assertRaises(RuntimeError):
                remove_temporary_result(unexpected, base, 14, 260080)
            self.assertTrue(unexpected.exists())


if __name__ == "__main__":
    unittest.main()
