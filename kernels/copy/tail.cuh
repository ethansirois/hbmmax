#pragma once

// Compile away for aligned sizes. Keep every SM budget on exactly the requested workload.
template <int BulkVectors>
__device__ __forceinline__
void copy_tail(const float4* input, float4* output) {
    const int idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    if constexpr (BulkVectors < NUMEL / 4) {
        for (int64_t i = int64_t(BulkVectors) + idx; i < NUMEL / 4; i += SMALL_STEP)
            output[i] = input[i];
    }
    if constexpr (NUMEL % 4 != 0) {
        if (idx < NUMEL % 4) {
            const int64_t i = (NUMEL / 4) * 4 + idx;
            reinterpret_cast<float*>(output)[i] = reinterpret_cast<const float*>(input)[i];
        }
    }
}
