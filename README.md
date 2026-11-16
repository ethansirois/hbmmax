# A100 HBM Saturation with Minimal SMs

Writeup: **https://ethansirois.github.io/hbmmax/**

The article source is `docs/a100.qmd`. The site is published with Quarto to GitHub Pages.

## Layout

```text
kernels/
  common/             shared device helpers (mem_ops.cuh)
  copy/
    k0/ ... k3/       kernel variants, with the original filenames
    problem.cuh       copy buffers, launches, validation, byte accounting
    baseline.cuh      large-SM bandwidth reference
    *_config.cuh      family-specific compile-time constants
    tail.cuh          remainder handling for arbitrary element counts
  sum/                reserved for the next problem; no implementation yet
bench/                one host driver, device translation unit, timing, CUDA helpers
tools/                build/run/inspect/sweep CLI and SASS formatter
config.json           editable machine and benchmark defaults
docs/                 article and images
snapshots/legacy/     original source, binaries, and assembly, preserved byte-for-byte
snapshots/site/       previous rendered article and its assets
results/legacy.txt    previous measurements
build/                generated binaries, assembly, manifests, comparison views (ignored)
```

The Python CLI calls `nvcc`, `ptxas`, and `cuobjdump` directly; CMake is not required.
Run the commands below from this directory. Python 3.10+ and a CUDA toolkit supporting
the chosen architecture are required; running benchmarks also requires a working NVIDIA driver/GPU.

## Configuration

Edit `config.json` to set the architecture, optimization level, element count, SM budgets,
and timing defaults for your machine. Command-line options override these settings.
For a private machine configuration, copy it to `config.local.json` (ignored) and use
`--config config.local.json`.

`--sms` is the small-kernel block/SM budget; `--peak-sms` is the budget for the bandwidth
reference. These experiments launch one persistent block per intended SM; the budget
does not itself guarantee that blocks occupy distinct SMs. Choose budgets within the
device's actual SM count. The reference is measured bandwidth at the configured budget,
not the GPU's theoretical peak.

`--numel` counts **float elements**, not bytes. It accepts quoted integer arithmetic,
including `+`, `-`, `*`, `/` (exact division), `//`, `%`, `**`, and shifts:

```bash
python3 tools/bench.py run --kernel k0 \
  -arch=sm_80 -O3 --sms 4 --peak-sms 32 \
  --numel "1024 * 1024 * 1024 * 2 + 1"
```

Expressions are parsed as bounded integer arithmetic; they cannot call functions or execute code.
This example needs roughly 16 GiB of host RAM and 16 GiB of device memory for copy's
two buffers on each side. Kernel vector indices currently limit sizes to `numel / 4 <= INT32_MAX`.

SM budgets, element count, threads, and optimization are compile-time settings so that
constant strides, offsets, and unrolling remain visible in PTX/SASS. Changed settings
automatically select a new cached build. Device, warmup, and iteration counts are runtime settings.

## Build and inspect

```bash
python3 tools/bench.py list
python3 tools/bench.py inspect --kernels k0,k1,k2,k3 -arch=sm_80 -O3
python3 tools/bench.py build --kernel k3_1 --sms 3 --numel "(1 << 24) + 1"
```

Both `build` and `inspect` perform the same explicit pipeline:

1. Compile `bench/device.cu`, including the selected kernel and reference, to `kernel.ptx`.
2. Assemble that PTX with `ptxas -arch=sm_XX -OX` into `kernel.cubin`.
3. Dump `kernel.sass` from that cubin and run `collapse_sass` to produce `kernel.gist.sass`.
4. Compile the common host driver, which loads and launches **that exact cubin**.

Every build keeps these artifacts, a compiler log, source snapshots, and a `manifest.json`
recording commands, tool versions, settings, and source/artifact hashes.
The CLI prints a comparison index under `build/inspect/<settings>/README.md`. Its flat
`source/`, `built-source/`, `ptx/`, and `sass/` directories make comparisons convenient.
`source/` links to editable code; `built-source/` links to the exact source used for inspection.
Each assembly file includes both `small_SMs` and `large_SMs`.

### Compiler Explorer / Godbolt

Every build, run, and sweep point also generates `kernel.godbolt.cu`: a standalone
copy-paste input containing the selected kernel, its settings, and required device
helpers. Local includes are expanded; CUDA/standard-library includes stay intact.
The host driver and large-SM baseline are omitted. The comparison index links to it
in `godbolt/`, and `run` prints the snippet path before launching the benchmark.

You can generate snippets without building or running anything, even without an
installed CUDA toolkit:

```bash
python3 tools/bench.py snippet --kernels k0,k3_1 \
  --sms 4 --numel "(1 << 24) + 1" -arch=sm_80 -O3
```

These files appear under `build/snippets/<settings>/`. Paste the whole file into a
CUDA C++ editor on [Compiler Explorer](https://godbolt.org/), choose NVCC, copy the
compiler options from its opening comment, and open the Device pane to view SASS
or PTX. No `main()` or execution is needed. The comment records the local NVCC and
ptxas versions when available; choose matching versions for a close comparison.
The options include `-Xptxas=-OX` to match the local assembler optimization level.
Compiler versions, additional flags, and the compilation context can change SASS;
the local cubin remains the record of the code actually benchmarked.

Use `--verbose` to see commands and compiler diagnostics, or `--rebuild` to bypass the cache.
`NVCC`, `PTXAS`, and `CUOBJDUMP` environment variables (or matching CLI options) select tools.

To format an existing dump independently:

```bash
python3 tools/collapse_sass.py input.sass -o output.gist.sass
```

## Run and sweep

```bash
python3 tools/bench.py run --kernel k3 --sms 4 --iterations 100
python3 tools/bench.py sweep --kernels k0,k1,k2,k3 \
  --sweep-sms "1:32" --peak-sms 32 --within-percent 3 \
  --numel "1 << 24"
```

Sweep selections support individual values and inclusive ranges, such as `1,2,4,8,16`
or `1:16,20:32:4`. Points run in ascending order. For each kernel, the tool first measures
the large-SM reference on the same input size. It then stops at the first tested budget
with at least 97% of that reference bandwidth (configurable with `--within-percent`).
The console reports the SM count and GB/s for points before it. With a sparse selection,
this is the minimum **tested** budget. If no point reaches the target, it says so and
reports all tested bandwidths. JSON and CSV measurements are saved under `results/`,
or at `--output path.json`; completed kernels are saved as the sweep progresses.

Bandwidth uses decimal GB/s. Copy counts input reads plus output writes (`2 * numel * 4`
bytes). Allocation, initialization, validation, and transfers are outside CUDA-event timing.
Each run validates a single launch and the timed result. Timing uses the same buffers
repeatedly; choose a working set larger than the GPU's cache when measuring HBM bandwidth.

## Kernel migration notes

The active kernels contain their implementation and variant-specific helpers; common
configuration and host code were extracted. The original files in `snapshots/legacy`
are historical artifacts, and their old SASS is not claimed to match new builds.

The active `k1_1` and `k2_t` variants fix an offset counted twice after advancing pointers.
Grouped kernels now process only complete groups and handle the remainder separately.
Async kernels retain their prologue/main-loop/epilogue and use a direct-copy remainder;
inputs smaller than the pipeline prologue use the direct path. Scalar tails handle
element counts not divisible by four. These paths let all sweep points use the same
requested element count, but can affect assembly/performance, especially for small inputs.

For an exactly aligned historical k3 pipeline size, use `--main-loops 512` in place of
`--numel`. It derives the size from the SM budget and cannot be used for a fixed-size sweep.

To add another problem, provide its configuration, `problem.cuh`, `baseline.cuh`, and
kernel variants using the same driver interface. Only existing variants are discovered;
sum can start with k0 and k3 without intermediate versions.

## Verification and article

```bash
python3 -m unittest discover -s tools -p 'test_*.py'
quarto render docs/a100.qmd
```

Quarto writes `docs/index.html`; serve/open it with the repository directory available
so links to `snapshots/legacy` resolve. The historical article's source/assembly references
now point to the preserved snapshots.
