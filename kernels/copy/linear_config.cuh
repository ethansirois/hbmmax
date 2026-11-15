#pragma once

#include "bench/config.hpp"
#ifndef HBM_NUMEL
#define HBM_NUMEL (1 << 24)
#endif

constexpr int64_t NUMEL = HBM_NUMEL;
constexpr int NUM_THREAD_BLOCKS = NUMEL / THREADS_PER_BLOCK / 4;
constexpr int SMALL_NUM_STEPS = NUM_THREAD_BLOCKS / SMALL_SMs_USED;
constexpr int SMALL_STEP = SMALL_SMs_USED * THREADS_PER_BLOCK;
constexpr int SMALL_BYTE_STEP = SMALL_STEP * sizeof(float4);
constexpr int SMEM_LEN = 0;
static_assert(NUMEL > 0 && NUMEL / 4 <= INT32_MAX, "numel exceeds the kernel's vector index range");
#include "kernels/copy/tail.cuh"
