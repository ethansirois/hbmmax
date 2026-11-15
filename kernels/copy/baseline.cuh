#pragma once

// Use the same input length as the selected kernel, including a possible tail.
extern "C" __global__ void large_SMs(const float4* __restrict__ dA, float4* __restrict__ dB) {
    constexpr int step = LARGE_SMs_USED * THREADS_PER_BLOCK;
    const int idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    constexpr int vectors = NUMEL / 4;
    #pragma unroll
    for (int i = 0; i < vectors / step; ++i) {
        dB[i * step + idx] = dA[i * step + idx];
    }
    if constexpr (vectors % step != 0) {
        if (idx < vectors % step) {
            dB[(vectors / step) * step + idx] = dA[(vectors / step) * step + idx];
        }
    }
    if constexpr (NUMEL % 4 != 0) {
        if (idx < NUMEL % 4) {
            const int64_t i = (NUMEL / 4) * 4 + idx;
            reinterpret_cast<float*>(dB)[i] = reinterpret_cast<const float*>(dA)[i];
        }
    }
}
