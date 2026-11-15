#include "kernels/copy/linear_config.cuh"

extern "C"
__global__ void 
small_SMs(const float4* __restrict__ dA, float4* __restrict__ dB) {
    copy_tail<SMALL_NUM_STEPS * SMALL_STEP>(dA, dB);

    constexpr int step = BLOCKS_PER_SM * SMALL_SMs_USED * THREADS_PER_BLOCK;
    const int idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < SMALL_NUM_STEPS; ++i) {
        dB[i * step + idx] = dA[i * step + idx];
    }
}
