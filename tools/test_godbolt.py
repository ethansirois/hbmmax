"""Check that exported kernels carry their settings and need no project files."""
from pathlib import Path
import tempfile
import unittest

import bench
from godbolt import render_snippet


class Snippets(unittest.TestCase):
    def settings(self, **overrides):
        return {"arch": "sm_80", "opt": 3, "sms": 3, "peak_sms": 32,
                "numel": 4194305, "threads": 256, "main_loops": None, **overrides}

    def test_all_variants_are_standalone_and_device_only(self):
        for name, kernel in bench.available("copy").items():
            with self.subTest(kernel=name):
                snippet = render_snippet(bench.ROOT, kernel, self.settings())
                self.assertNotRegex(snippet, r'#\s*include\s+"')
                self.assertNotIn("#pragma once", snippet)
                self.assertNotIn("struct RunConfig", snippet)
                self.assertNotIn("large_SMs(", snippet)
                self.assertNotIn("int main(", snippet)
                self.assertEqual(snippet.count("small_SMs("), 1)
                self.assertIn("#define HBM_SMS 3\n", snippet)
                self.assertIn("#define HBM_NUMEL 4194305LL", snippet)
                self.assertIn("-std=c++17 -O3 -arch=sm_80 -Xptxas=-O3 -lineinfo", snippet)

    def test_helpers_and_pipeline_headers_are_preserved(self):
        snippet = render_snippet(bench.ROOT, bench.available("copy")["k3_1"], self.settings())
        self.assertIn("#include <cuda_pipeline.h>", snippet)
        self.assertIn("void load_group(", snippet)
        self.assertIn("void copy_tail(", snippet)
        self.assertIn("asm volatile", snippet)

    def test_loop_count_and_compiler_versions(self):
        snippet = render_snippet(
            bench.ROOT, bench.available("copy")["k3"],
            self.settings(numel=None, main_loops=512, arch="sm_86", opt=0),
            {"nvcc": {"version": "header\nCuda compilation tools, release 12.4, V12.4.131\n"}})
        self.assertIn("#define HBM_MAIN_LOOPS 512", snippet)
        self.assertNotIn("#define HBM_NUMEL", snippet)
        self.assertIn("-O0 -arch=sm_86 -Xptxas=-O0", snippet)
        self.assertIn("// Local nvcc: Cuda compilation tools, release 12.4, V12.4.131", snippet)

    def test_repeated_headers_are_included_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "shared.cuh").write_text("#pragma once\nconstexpr int value = 3;\n")
            (root / "wrapper.cuh").write_text('#pragma once\n#include "shared.cuh"\n')
            (root / "kernel.cu").write_text('#include "shared.cuh"\n#include "wrapper.cuh"\n')
            snippet = render_snippet(root, root / "kernel.cu", self.settings())
            self.assertEqual(snippet.count("constexpr int value = 3;"), 1)

    def test_unresolved_local_header_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "kernel.cu").write_text('#include "missing.cuh"\n')
            with self.assertRaisesRegex(ValueError, "missing.cuh"):
                render_snippet(root, root / "kernel.cu", self.settings())


if __name__ == "__main__":
    unittest.main()
