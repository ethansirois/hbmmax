#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// Build-time settings: these affect strides, unrolling, and generated assembly.
#ifndef HBM_SMS
#define HBM_SMS 4
#endif
#ifndef HBM_PEAK_SMS
#define HBM_PEAK_SMS 32
#endif
#ifndef HBM_THREADS
#define HBM_THREADS 256
#endif

constexpr int SMALL_SMs_USED = HBM_SMS;
constexpr int LARGE_SMs_USED = HBM_PEAK_SMS;
constexpr int THREADS_PER_BLOCK = HBM_THREADS;
constexpr int THREADS_PER_SM = THREADS_PER_BLOCK;
constexpr int BLOCKS_PER_SM = 1;
static_assert(SMALL_SMs_USED > 0 && LARGE_SMs_USED > 0, "SM budgets must be positive");
static_assert(THREADS_PER_BLOCK > 0 && THREADS_PER_BLOCK <= 1024 &&
              THREADS_PER_BLOCK % 32 == 0, "threads must be a warp multiple, up to 1024");
