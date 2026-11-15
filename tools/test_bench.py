"""CPU-only tests for workload parsing, SM threshold selection, and saved results."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bench


class Expressions(unittest.TestCase):
    def test_workload_sizes(self):
        for expression, expected in (
            ("1024 * 1024 * 1024 * 2 + 1", 2147483649),
            ("(1 << 24) + 1", 16777217),
            ("2**30", 1073741824),
            ("0x1000 / 4", 1024),
            ("7 // 2", 3),
        ):
            with self.subTest(expression=expression):
                self.assertEqual(bench.positive(expression), expected)

    def test_invalid_or_unsafe_expressions(self):
        for expression in ("__import__('os').getcwd()", "x * 2", "2.5", "True", "3 / 2",
                           "2**1000000000", "1 << 64", "1 / 0", "2**63", "[1]", "0", "-1"):
            with self.subTest(expression=expression):
                with self.assertRaises(argparse.ArgumentTypeError):
                    bench.positive(expression)

    def test_sm_ranges(self):
        self.assertEqual(bench.sm_counts("8,1:4,2,10:14:2"), [1, 2, 3, 4, 8, 10, 12, 14])
        for value in ("", "0", "4:1", "1:4:0", "1:1000000", "1:2:3:4"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                bench.sm_counts(value)


class Sweep(unittest.TestCase):
    def args(self, path=None):
        return argparse.Namespace(sweep_sms=[1, 2, 3, 4], within_percent=3.0,
                                  output=path, problem="copy", arch="80", opt=3,
                                  numel=1048577, peak_sms=32, threads=256)

    def measurement(self, bandwidth, peak=False, numel=1048577):
        return {"numel": numel, "available_sms": 108, "valid": True,
                "kernel_gbps": 0 if peak else bandwidth, "peak_gbps": bandwidth if peak else 0}

    @patch.object(bench, "inspect")
    @patch.object(bench, "build")
    @patch.object(bench, "measure")
    def test_stops_at_first_point_within_three_percent(self, measure, build, inspect):
        build.return_value = (bench.ROOT / "build/test", {})
        measure.side_effect = [self.measurement(1000, peak=True), self.measurement(700),
                               self.measurement(969), self.measurement(970)]
        result = bench.sweep(self.args(), Path("k0.cu"))
        self.assertEqual(result["minimum_sms"], 3)
        self.assertEqual([p["sms"] for p in result["points"]], [1, 2, 3])
        self.assertEqual(measure.call_count, 4)

    @patch.object(bench, "inspect")
    @patch.object(bench, "build")
    @patch.object(bench, "measure")
    def test_reports_no_match_and_saves_bandwidth(self, measure, build, inspect):
        build.return_value = (bench.ROOT / "build/test", {})
        measure.side_effect = [self.measurement(1000, peak=True)] + [self.measurement(900)] * 4
        result = bench.sweep(self.args(), Path("k0.cu"))
        self.assertIsNone(result["minimum_sms"])
        with tempfile.TemporaryDirectory() as directory:
            path = bench.save_sweeps(self.args(Path(directory) / "results.json"), [result])
            self.assertEqual(json.loads(path.read_text())["results"][0]["points"][0]["gbps"], 900)
            self.assertIn("GB/s", path.with_suffix(".csv").read_text())

    @patch.object(bench, "inspect")
    @patch.object(bench, "build")
    @patch.object(bench, "measure")
    def test_rejects_a_changed_workload(self, measure, build, inspect):
        build.return_value = (bench.ROOT / "build/test", {})
        measure.side_effect = [self.measurement(1000, peak=True), self.measurement(900, numel=5)]
        with self.assertRaisesRegex(RuntimeError, "workload changed"):
            bench.sweep(self.args(), Path("k0.cu"))


if __name__ == "__main__":
    unittest.main()
