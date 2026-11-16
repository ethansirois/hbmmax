#pragma once

#include "kernels/common/config.cuh"

constexpr int MAX_REGS = 61;
constexpr int SMEM_LEN = 20;
constexpr int SMALL_REGS = MAX_REGS - SMEM_LEN;
constexpr int SMALL_STEP = SMALL_SMs_USED * THREADS_PER_BLOCK;
constexpr int SMALL_BYTE_STEP = SMALL_STEP * sizeof(float4);
// Preserve the original prologue/main-loop/epilogue layout exactly.
constexpr int PIPELINE_OVERHEAD = 2 * MAX_REGS + SMEM_LEN;
#ifdef HBM_NUMEL
constexpr int64_t NUMEL = HBM_NUMEL;
#elif defined(HBM_MAIN_LOOPS)
constexpr int64_t NUMEL = int64_t(4) * SMALL_STEP *
    (int64_t(HBM_MAIN_LOOPS) * (MAX_REGS + SMEM_LEN) + PIPELINE_OVERHEAD);
#else
constexpr int64_t NUMEL = 1 << 24;
#endif
constexpr int NUM_MAIN_LOOPS = NUMEL / (4 * SMALL_STEP) < PIPELINE_OVERHEAD ? 0 :
    (NUMEL / (4 * SMALL_STEP) - PIPELINE_OVERHEAD) / (MAX_REGS + SMEM_LEN);
constexpr int PIPELINE_BULK_VECTORS = NUMEL / 4 < PIPELINE_OVERHEAD * SMALL_STEP ? 0 :
    SMALL_STEP * (NUM_MAIN_LOOPS * (MAX_REGS + SMEM_LEN) + PIPELINE_OVERHEAD);
static_assert(NUMEL > 0 && NUMEL / 4 <= INT32_MAX, "numel exceeds the kernel's vector index range");
#include "kernels/copy/tail.cuh"
